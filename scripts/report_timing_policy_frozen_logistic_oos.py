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
from src.strategy.spot_logistic import SpotLogisticConfig, fit_rolling_spot_logistic, predict_spot_logistic_probability


def _parse_dataset_specs(values: Iterable[str]) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for raw in values:
        label, sep, path_raw = str(raw).partition("=")
        if not sep:
            raise SystemExit(f"dataset spec must be DELAY=PATH, got: {raw}")
        delay = int(label.strip().rstrip("s"))
        out.append((delay, Path(path_raw.strip()).resolve()))
    return sorted(out, key=lambda item: item[0])


def _rows_for_days(rows: list[dict[str, Any]], day_set: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if _date_key(row) in day_set]


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


def _with_frozen_logistic(
    train_rows_240: list[dict[str, Any]],
    test_rows_240: list[dict[str, Any]],
    cfg: SpotLogisticConfig,
) -> list[dict[str, Any]]:
    model = fit_rolling_spot_logistic(train_rows_240, cfg)
    out: list[dict[str, Any]] = []
    for row in test_rows_240:
        row_local = dict(row)
        row_local["spot_logistic_online_p_up"] = predict_spot_logistic_probability(model, row_local)
        out.append(row_local)
    return out


def _aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "sum_final_balance": 0.0,
            "min_final_balance": 0.0,
            "max_drawdown_pct": 0.0,
            "negative_days_max": 0,
            "avg_win_rate": 0.0,
        }
    return {
        "sum_final_balance": float(sum(x["final_balance"] for x in items)),
        "min_final_balance": float(min(x["final_balance"] for x in items)),
        "max_drawdown_pct": float(max(x["max_drawdown_pct"] for x in items)),
        "negative_days_max": int(max(x["negative_days"] for x in items)),
        "avg_win_rate": float(sum(x["win_rate"] for x in items) / len(items)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare timing policy with online vs frozen logistic fallback across OOS blocks.")
    ap.add_argument("--datasets", nargs="+", required=True, help="One or more DELAY=PATH specs")
    ap.add_argument("--primary-delay", type=int, default=180)
    ap.add_argument("--fallback-delay", type=int, default=240)
    ap.add_argument("--primary-candidate", default="spot_consensus_blend")
    ap.add_argument("--margins", default="0.08,0.10")
    ap.add_argument("--block-days", type=int, default=5)
    ap.add_argument("--common-days-only", action="store_true", help="Restrict all delay datasets to the intersection of observed days before building folds.")
    ap.add_argument("--profile", default="steady_compound_v1")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dataset_specs = _parse_dataset_specs(list(args.datasets))
    cfg_raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    fee_cfg = FeeModelConfig(**cfg_raw["fees"])
    slippage_bps = float(cfg_raw.get("execution", {}).get("slippage_bps_taker", 12.0))
    min_edge_to_trade = float(cfg_raw.get("execution", {}).get("min_edge_to_trade", 0.004))
    timing_runtime_cfg = (cfg_raw.get("model") or {}).get("timing_policy_runtime", {})
    allow_late_fresh_start = bool(timing_runtime_cfg.get("allow_late_fresh_start", False))
    late_start_grace_seconds = max(0.0, float(timing_runtime_cfg.get("late_start_grace_seconds", 0.0)))
    risk_cfg = _profile_config(str(args.profile))
    log_cfg = SpotLogisticConfig.from_dict((cfg_raw.get("model") or {}).get("spot_logistic", {}))
    rows_by_delay = {delay: _read_jsonl(path) for delay, path in dataset_specs}
    margins = [float(x.strip()) for x in str(args.margins).split(",") if x.strip()]

    coverage_days: list[str] | None = None
    if bool(args.common_days_only):
        coverage_days = common_days_across_delays(rows_by_delay)
        rows_by_delay = filter_rows_by_days(rows_by_delay, set(coverage_days))

    all_days = sorted({_date_key(row) for rows in rows_by_delay.values() for row in rows if _date_key(row)})
    block_days = max(1, int(args.block_days))
    blocks = [all_days[i : i + block_days] for i in range(0, len(all_days), block_days) if all_days[i : i + block_days]]

    payload: dict[str, Any] = {
        "profile": str(args.profile),
        "primary_delay": int(args.primary_delay),
        "primary_candidate": str(args.primary_candidate),
        "fallback_delay": int(args.fallback_delay),
        "margins": margins,
        "block_days": block_days,
        "datasets": {str(delay): str(path) for delay, path in dataset_specs},
        "timing_policy_runtime": {
            "allow_late_fresh_start": allow_late_fresh_start,
            "late_start_grace_seconds": late_start_grace_seconds,
        },
        "common_days_only": bool(args.common_days_only),
        "coverage_days": coverage_days,
        "folds": [],
        "aggregate": {},
    }

    aggregate_store: dict[str, list[dict[str, Any]]] = {str(m): [] for m in margins}
    aggregate_store_frozen: dict[str, list[dict[str, Any]]] = {str(m): [] for m in margins}

    for fold_idx in range(1, len(blocks)):
        train_days = [day for block in blocks[:fold_idx] for day in block]
        test_days = list(blocks[fold_idx])
        train_set = set(train_days)
        test_set = set(test_days)

        train_rows_240 = _rows_for_days(rows_by_delay[int(args.fallback_delay)], train_set)
        test_rows_240_frozen = _with_frozen_logistic(train_rows_240, _rows_for_days(rows_by_delay[int(args.fallback_delay)], test_set), log_cfg)

        fold_payload: dict[str, Any] = {
            "fold_index": fold_idx,
            "train_days": train_days,
            "test_days": test_days,
            "margins": {},
        }

        for margin in margins:
            online_rows_by_delay = {delay: _rows_for_days(rows, test_set) for delay, rows in rows_by_delay.items()}
            frozen_rows_by_delay = dict(online_rows_by_delay)
            frozen_rows_by_delay[int(args.fallback_delay)] = test_rows_240_frozen

            online_policy_rows, online_diag = build_timing_policy_rows(
                online_rows_by_delay,
                stages=[
                    (int(args.primary_delay), str(args.primary_candidate), float(margin)),
                    (int(args.fallback_delay), "spot_logistic_online", 0.0),
                ],
                fee_cfg=fee_cfg,
                slippage_bps=slippage_bps,
                min_edge_to_trade=min_edge_to_trade,
                allow_late_fresh_start=allow_late_fresh_start,
                late_start_grace_seconds=late_start_grace_seconds,
            )
            online_balance = run_balance_backtest(
                online_policy_rows,
                candidate_key="spot_timing_policy",
                fee_cfg=fee_cfg,
                slippage_bps=slippage_bps,
                risk_cfg=risk_cfg,
            )

            frozen_policy_rows, frozen_diag = build_timing_policy_rows(
                frozen_rows_by_delay,
                stages=[
                    (int(args.primary_delay), str(args.primary_candidate), float(margin)),
                    (int(args.fallback_delay), "spot_logistic_online", 0.0),
                ],
                fee_cfg=fee_cfg,
                slippage_bps=slippage_bps,
                min_edge_to_trade=min_edge_to_trade,
                allow_late_fresh_start=allow_late_fresh_start,
                late_start_grace_seconds=late_start_grace_seconds,
            )
            frozen_balance = run_balance_backtest(
                frozen_policy_rows,
                candidate_key="spot_timing_policy",
                fee_cfg=fee_cfg,
                slippage_bps=slippage_bps,
                risk_cfg=risk_cfg,
            )

            online_summary = _summary(online_balance)
            frozen_summary = _summary(frozen_balance)
            aggregate_store[str(margin)].append(online_summary)
            aggregate_store_frozen[str(margin)].append(frozen_summary)
            fold_payload["margins"][str(margin)] = {
                "online": online_summary,
                "online_selection": online_diag,
                "frozen": frozen_summary,
                "frozen_selection": frozen_diag,
            }

        payload["folds"].append(fold_payload)

    payload["aggregate"] = {
        str(margin): {
            "online": _aggregate(aggregate_store[str(margin)]),
            "frozen": _aggregate(aggregate_store_frozen[str(margin)]),
        }
        for margin in margins
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
