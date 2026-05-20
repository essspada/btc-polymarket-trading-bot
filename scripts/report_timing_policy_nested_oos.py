#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_balance_growth import _date_key, _read_jsonl, run_balance_backtest
from scripts.compare_balance_profiles import _profile_config
from scripts.report_timing_policy_oos import build_timing_policy_rows, common_days_across_delays, filter_rows_by_days
from src.polymarket.fees import FeeModelConfig


def _parse_dataset_specs(values: Iterable[str]) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for raw in values:
        label, sep, path_raw = str(raw).partition("=")
        if not sep:
            raise SystemExit(f"dataset spec must be DELAY=PATH, got: {raw}")
        delay = int(label.strip().rstrip("s"))
        out.append((delay, Path(path_raw.strip()).resolve()))
    return sorted(out, key=lambda item: item[0])


def _parse_stage(raw: str) -> tuple[int, str]:
    delay_raw, sep, candidate = str(raw).partition(":")
    if not sep:
        raise SystemExit(f"stage must be DELAY:CANDIDATE, got: {raw}")
    return int(delay_raw.strip().rstrip("s")), candidate.strip()


def _parse_margin_grid(raw: str) -> list[float]:
    values = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not values:
        raise SystemExit("margin grid must not be empty")
    return values


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "final_balance": float(result.get("final_balance", 0.0)),
        "target_hit": bool(result.get("target_hit", False)),
        "trades_taken": int(result.get("trades_taken", 0)),
        "win_rate": float(result.get("win_rate", 0.0)),
        "max_drawdown_pct": float(result.get("max_drawdown_pct", 0.0)),
        "positive_days": int(result.get("positive_days", 0)),
        "negative_days": int(result.get("negative_days", 0)),
        "worst_day_pnl": float((result.get("worst_day") or {}).get("pnl", 0.0)),
        "best_day_pnl": float((result.get("best_day") or {}).get("pnl", 0.0)),
    }


def select_best_margin_result(results: dict[float, dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    if not results:
        raise ValueError("results must not be empty")
    zero_neg = [(margin, result) for margin, result in results.items() if int(result.get("negative_days", 0)) == 0]
    pool = zero_neg if zero_neg else list(results.items())
    pool.sort(
        key=lambda item: (
            float(item[1].get("final_balance", 0.0)),
            -float(item[1].get("max_drawdown_pct", 0.0)),
            -int(item[1].get("negative_days", 0)),
        ),
        reverse=True,
    )
    return pool[0]


def _rows_for_days(rows: list[dict[str, Any]], day_set: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if _date_key(row) in day_set]


def _build_policy_result(
    rows_by_delay: dict[int, list[dict[str, Any]]],
    *,
    day_set: set[str],
    stages: list[tuple[int, str, float]],
    fee_cfg: FeeModelConfig,
    slippage_bps: float,
    min_edge_to_trade: float,
    allow_late_fresh_start: bool,
    late_start_grace_seconds: float,
    risk_cfg: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    subset_by_delay = {delay: _rows_for_days(rows, day_set) for delay, rows in rows_by_delay.items()}
    policy_rows, diagnostics = build_timing_policy_rows(
        subset_by_delay,
        stages=stages,
        fee_cfg=fee_cfg,
        slippage_bps=slippage_bps,
        min_edge_to_trade=min_edge_to_trade,
        allow_late_fresh_start=allow_late_fresh_start,
        late_start_grace_seconds=late_start_grace_seconds,
    )
    result = run_balance_backtest(
        policy_rows,
        candidate_key="spot_timing_policy",
        fee_cfg=fee_cfg,
        slippage_bps=slippage_bps,
        risk_cfg=risk_cfg,
    )
    return policy_rows, diagnostics, result


def main() -> int:
    ap = argparse.ArgumentParser(description="Nested forward OOS report for sequential timing policy.")
    ap.add_argument("--datasets", nargs="+", required=True, help="One or more DELAY=PATH specs")
    ap.add_argument("--primary-stage", default="180:spot_consensus_blend")
    ap.add_argument("--fallback-stage", default="240:spot_logistic_online")
    ap.add_argument("--margin-grid", default="0.00,0.01,0.02,0.03,0.04,0.05,0.08,0.09,0.10")
    ap.add_argument("--block-days", type=int, default=5)
    ap.add_argument("--common-days-only", action="store_true", help="Restrict all delay datasets to the intersection of observed days before building folds.")
    ap.add_argument("--profile", default="steady_compound_v1")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dataset_specs = _parse_dataset_specs(list(args.datasets))
    primary_delay, primary_candidate = _parse_stage(args.primary_stage)
    fallback_delay, fallback_candidate = _parse_stage(args.fallback_stage)
    margins = _parse_margin_grid(args.margin_grid)

    cfg_raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    fee_cfg = FeeModelConfig(**cfg_raw["fees"])
    slippage_bps = float(cfg_raw.get("execution", {}).get("slippage_bps_taker", 12.0))
    min_edge_to_trade = float(cfg_raw.get("execution", {}).get("min_edge_to_trade", 0.004))
    timing_runtime_cfg = (cfg_raw.get("model") or {}).get("timing_policy_runtime", {})
    allow_late_fresh_start = bool(timing_runtime_cfg.get("allow_late_fresh_start", False))
    late_start_grace_seconds = max(0.0, float(timing_runtime_cfg.get("late_start_grace_seconds", 0.0)))
    risk_cfg = _profile_config(str(args.profile))
    rows_by_delay = {delay: _read_jsonl(path) for delay, path in dataset_specs}
    coverage_days: list[str] | None = None
    if bool(args.common_days_only):
        coverage_days = common_days_across_delays(rows_by_delay)
        rows_by_delay = filter_rows_by_days(rows_by_delay, set(coverage_days))

    all_days = sorted({_date_key(row) for rows in rows_by_delay.values() for row in rows if _date_key(row)})
    block_days = max(1, int(args.block_days))
    blocks = [all_days[i : i + block_days] for i in range(0, len(all_days), block_days) if all_days[i : i + block_days]]

    folds: list[dict[str, Any]] = []
    selected_margin_counts: dict[str, int] = {}

    for fold_idx in range(1, len(blocks)):
        train_days = [day for block in blocks[:fold_idx] for day in block]
        test_days = list(blocks[fold_idx])
        train_set = set(train_days)
        test_set = set(test_days)

        margin_train_results: dict[float, dict[str, Any]] = {}
        for margin in margins:
            _, _, train_result = _build_policy_result(
                rows_by_delay,
                day_set=train_set,
                stages=[
                    (primary_delay, primary_candidate, float(margin)),
                    (fallback_delay, fallback_candidate, 0.0),
                ],
                fee_cfg=fee_cfg,
                slippage_bps=slippage_bps,
                min_edge_to_trade=min_edge_to_trade,
                allow_late_fresh_start=allow_late_fresh_start,
                late_start_grace_seconds=late_start_grace_seconds,
                risk_cfg=risk_cfg,
            )
            margin_train_results[float(margin)] = _summary(train_result)

        chosen_margin, chosen_train_summary = select_best_margin_result(margin_train_results)
        selected_margin_counts[str(chosen_margin)] = selected_margin_counts.get(str(chosen_margin), 0) + 1

        _, chosen_diag, chosen_test_result = _build_policy_result(
            rows_by_delay,
            day_set=test_set,
            stages=[
                (primary_delay, primary_candidate, float(chosen_margin)),
                (fallback_delay, fallback_candidate, 0.0),
            ],
            fee_cfg=fee_cfg,
            slippage_bps=slippage_bps,
            min_edge_to_trade=min_edge_to_trade,
            allow_late_fresh_start=allow_late_fresh_start,
            late_start_grace_seconds=late_start_grace_seconds,
            risk_cfg=risk_cfg,
        )
        _, _, fixed008_test_result = _build_policy_result(
            rows_by_delay,
            day_set=test_set,
            stages=[
                (primary_delay, primary_candidate, 0.08),
                (fallback_delay, fallback_candidate, 0.0),
            ],
            fee_cfg=fee_cfg,
            slippage_bps=slippage_bps,
            min_edge_to_trade=min_edge_to_trade,
            allow_late_fresh_start=allow_late_fresh_start,
            late_start_grace_seconds=late_start_grace_seconds,
            risk_cfg=risk_cfg,
        )
        _, _, log240_test_result = _build_policy_result(
            rows_by_delay,
            day_set=test_set,
            stages=[(fallback_delay, fallback_candidate, 0.0)],
            fee_cfg=fee_cfg,
            slippage_bps=slippage_bps,
            min_edge_to_trade=min_edge_to_trade,
            allow_late_fresh_start=allow_late_fresh_start,
            late_start_grace_seconds=late_start_grace_seconds,
            risk_cfg=risk_cfg,
        )
        _, _, cons180_test_result = _build_policy_result(
            rows_by_delay,
            day_set=test_set,
            stages=[(primary_delay, primary_candidate, 0.0)],
            fee_cfg=fee_cfg,
            slippage_bps=slippage_bps,
            min_edge_to_trade=min_edge_to_trade,
            allow_late_fresh_start=allow_late_fresh_start,
            late_start_grace_seconds=late_start_grace_seconds,
            risk_cfg=risk_cfg,
        )

        folds.append(
            {
                "fold_index": fold_idx,
                "train_days": train_days,
                "test_days": test_days,
                "selected_margin": float(chosen_margin),
                "train_margin_sweep": {str(k): v for k, v in sorted(margin_train_results.items())},
                "selected_train_summary": chosen_train_summary,
                "selected_test_summary": _summary(chosen_test_result),
                "selected_test_diagnostics": chosen_diag,
                "fixed_0.08_test_summary": _summary(fixed008_test_result),
                "logistic_240_only_test_summary": _summary(log240_test_result),
                "consensus_180_only_test_summary": _summary(cons180_test_result),
            }
        )

    def _agg(key: str) -> dict[str, Any]:
        vals = [fold[key] for fold in folds]
        return {
            "sum_final_balance": float(sum(v["final_balance"] for v in vals)),
            "min_final_balance": float(min(v["final_balance"] for v in vals)) if vals else 0.0,
            "max_drawdown_pct": float(max(v["max_drawdown_pct"] for v in vals)) if vals else 0.0,
            "negative_days_max": int(max(v["negative_days"] for v in vals)) if vals else 0,
            "avg_win_rate": float(sum(v["win_rate"] for v in vals) / len(vals)) if vals else 0.0,
        }

    payload = {
        "profile": str(args.profile),
        "primary_stage": {"delay_seconds": primary_delay, "candidate_key": primary_candidate},
        "fallback_stage": {"delay_seconds": fallback_delay, "candidate_key": fallback_candidate},
        "margin_grid": margins,
        "block_days": block_days,
        "datasets": {str(delay): str(path) for delay, path in dataset_specs},
        "timing_policy_runtime": {
            "allow_late_fresh_start": allow_late_fresh_start,
            "late_start_grace_seconds": late_start_grace_seconds,
        },
        "common_days_only": bool(args.common_days_only),
        "coverage_days": coverage_days,
        "selected_margin_counts": selected_margin_counts,
        "folds": folds,
        "aggregate": {
            "selected_policy": _agg("selected_test_summary"),
            "fixed_0.08_policy": _agg("fixed_0.08_test_summary"),
            "logistic_240_only": _agg("logistic_240_only_test_summary"),
            "consensus_180_only": _agg("consensus_180_only_test_summary"),
        },
    }

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
