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
from scripts.report_timing_policy_nested_oos import select_best_margin_result
from scripts.report_timing_policy_oos import build_timing_policy_rows, common_days_across_delays, filter_rows_by_days
from src.polymarket.fees import FeeModelConfig


def _parse_dataset_specs(values: Iterable[str]) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for raw in values:
        label, sep, path_raw = str(raw).partition("=")
        if not sep:
            raise SystemExit(f"dataset spec must be DELAY=PATH, got: {raw}")
        delay = int(label.strip().rstrip("s"))
        out[delay] = Path(path_raw.strip()).resolve()
    return out


def _parse_families(raw: str) -> dict[str, list[tuple[str, int, str]]]:
    families: dict[str, list[tuple[str, int, str]]] = {}
    for chunk in str(raw).split(";"):
        item = chunk.strip()
        if not item:
            continue
        name, sep, spec = item.partition("=")
        if not sep:
            raise SystemExit(f"family spec must be NAME=STAGES, got: {item}")
        stages: list[tuple[str, int, str]] = []
        for stage_raw in spec.split(","):
            stage = stage_raw.strip()
            if not stage:
                continue
            parts = stage.split(":")
            if len(parts) not in {2, 3}:
                raise SystemExit(f"stage must be DELAY:CANDIDATE or train:DELAY:CANDIDATE, got: {stage}")
            if len(parts) == 2:
                delay = int(parts[0].strip().rstrip("s"))
                candidate = parts[1].strip()
                stages.append(("fixed", delay, candidate))
            else:
                mode = parts[0].strip().lower()
                delay = int(parts[1].strip().rstrip("s"))
                candidate = parts[2].strip()
                if mode != "train":
                    raise SystemExit(f"only 'train' stage-mode supported, got: {stage}")
                stages.append(("train", delay, candidate))
        if not stages:
            raise SystemExit(f"family has no stages: {item}")
        families[name.strip()] = stages
    if not families:
        raise SystemExit("at least one family is required")
    return families


def filter_families_by_available_delays(
    families: dict[str, list[tuple[str, int, str]]],
    available_delays: Iterable[int],
) -> tuple[dict[str, list[tuple[str, int, str]]], dict[str, list[int]]]:
    available = {int(delay) for delay in available_delays}
    valid: dict[str, list[tuple[str, int, str]]] = {}
    skipped: dict[str, list[int]] = {}
    for name, family in families.items():
        missing = sorted({int(delay) for _, delay, _ in family if int(delay) not in available})
        if missing:
            skipped[name] = missing
            continue
        valid[name] = family
    return valid, skipped


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
    }


def _stages_with_margin(family: list[tuple[str, int, str]], margin: float) -> list[tuple[int, str, float]]:
    out: list[tuple[int, str, float]] = []
    train_used = False
    for mode, delay, candidate in family:
        extra_edge = float(margin) if mode == "train" and not train_used else 0.0
        if mode == "train" and not train_used:
            train_used = True
        out.append((int(delay), candidate, float(extra_edge)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Nested OOS compare across multiple simple timing-policy families.")
    ap.add_argument("--datasets", nargs="+", required=True, help="One or more DELAY=PATH specs")
    ap.add_argument(
        "--families",
        default="cons180_then_log240=train:180:spot_consensus_blend,240:spot_logistic_online;"
        "log180_then_log240=train:180:spot_logistic_online,240:spot_logistic_online;"
        "cons120_then_log240=train:120:spot_consensus_blend,240:spot_logistic_online;"
        "cons120_then_cons180_then_log240=train:120:spot_consensus_blend,180:spot_consensus_blend,240:spot_logistic_online;"
        "cons180_then_cons240=train:180:spot_consensus_blend,240:spot_consensus_blend;"
        "path180_then_log240=train:180:spot_window_path,240:spot_logistic_online",
    )
    ap.add_argument("--margin-grid", default="0.00,0.01,0.02,0.03,0.04,0.05,0.08,0.09,0.10")
    ap.add_argument("--block-days", type=int, default=5)
    ap.add_argument("--common-days-only", action="store_true", help="Restrict all delay datasets to the intersection of observed days before building folds.")
    ap.add_argument("--profile", default="steady_compound_v1")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dataset_paths = _parse_dataset_specs(list(args.datasets))
    rows_by_delay = {delay: _read_jsonl(path) for delay, path in dataset_paths.items()}
    families = _parse_families(args.families)
    available_delays = sorted(rows_by_delay)
    families, skipped_families = filter_families_by_available_delays(families, available_delays)
    if not families:
        skipped_text = ", ".join(f"{name}:{delays}" for name, delays in sorted(skipped_families.items()))
        raise SystemExit(f"no valid families remain for available delays {available_delays}; skipped={skipped_text}")
    margins = [float(x.strip()) for x in str(args.margin_grid).split(",") if x.strip()]

    cfg_raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    fee_cfg = FeeModelConfig(**cfg_raw["fees"])
    slippage_bps = float(cfg_raw.get("execution", {}).get("slippage_bps_taker", 12.0))
    min_edge_to_trade = float(cfg_raw.get("execution", {}).get("min_edge_to_trade", 0.004))
    timing_runtime_cfg = (cfg_raw.get("model") or {}).get("timing_policy_runtime", {})
    allow_late_fresh_start = bool(timing_runtime_cfg.get("allow_late_fresh_start", False))
    late_start_grace_seconds = max(0.0, float(timing_runtime_cfg.get("late_start_grace_seconds", 0.0)))
    risk_cfg = _profile_config(str(args.profile))

    coverage_days: list[str] | None = None
    if bool(args.common_days_only):
        coverage_days = common_days_across_delays(rows_by_delay)
        rows_by_delay = filter_rows_by_days(rows_by_delay, set(coverage_days))

    all_days = sorted({_date_key(row) for rows in rows_by_delay.values() for row in rows if _date_key(row)})
    block_days = max(1, int(args.block_days))
    blocks = [all_days[i : i + block_days] for i in range(0, len(all_days), block_days) if all_days[i : i + block_days]]

    payload: dict[str, Any] = {
        "profile": str(args.profile),
        "block_days": block_days,
        "margin_grid": margins,
        "datasets": {str(delay): str(path) for delay, path in dataset_paths.items()},
        "available_delays": available_delays,
        "timing_policy_runtime": {
            "allow_late_fresh_start": allow_late_fresh_start,
            "late_start_grace_seconds": late_start_grace_seconds,
        },
        "common_days_only": bool(args.common_days_only),
        "coverage_days": coverage_days,
        "skipped_families_missing_delays": {name: delays for name, delays in sorted(skipped_families.items())},
        "families": {},
    }

    for family_name, family_spec in families.items():
        folds: list[dict[str, Any]] = []
        for fold_idx in range(1, len(blocks)):
            train_days = [day for block in blocks[:fold_idx] for day in block]
            test_days = list(blocks[fold_idx])
            train_set = set(train_days)
            test_set = set(test_days)

            train_rows_by_delay = {delay: _rows_for_days(rows, train_set) for delay, rows in rows_by_delay.items()}
            test_rows_by_delay = {delay: _rows_for_days(rows, test_set) for delay, rows in rows_by_delay.items()}

            margin_train_results: dict[float, dict[str, Any]] = {}
            for margin in margins:
                stages = _stages_with_margin(family_spec, float(margin))
                policy_rows, _ = build_timing_policy_rows(
                    train_rows_by_delay,
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
                margin_train_results[float(margin)] = _summary(result)

            selected_margin, _ = select_best_margin_result(margin_train_results)
            selected_stages = _stages_with_margin(family_spec, float(selected_margin))
            test_policy_rows, diag = build_timing_policy_rows(
                test_rows_by_delay,
                stages=selected_stages,
                fee_cfg=fee_cfg,
                slippage_bps=slippage_bps,
                min_edge_to_trade=min_edge_to_trade,
                allow_late_fresh_start=allow_late_fresh_start,
                late_start_grace_seconds=late_start_grace_seconds,
            )
            test_result = run_balance_backtest(
                test_policy_rows,
                candidate_key="spot_timing_policy",
                fee_cfg=fee_cfg,
                slippage_bps=slippage_bps,
                risk_cfg=risk_cfg,
            )
            folds.append(
                {
                    "fold_index": fold_idx,
                    "train_days": train_days,
                    "test_days": test_days,
                    "selected_margin": float(selected_margin),
                    "test_summary": _summary(test_result),
                    "selection": diag,
                }
            )

        payload["families"][family_name] = {
            "spec": [
                {"mode": mode, "delay_seconds": delay, "candidate_key": cand}
                for mode, delay, cand in family_spec
            ],
            "folds": folds,
            "aggregate": {
                "sum_final_balance": float(sum(f["test_summary"]["final_balance"] for f in folds)),
                "min_final_balance": float(min(f["test_summary"]["final_balance"] for f in folds)) if folds else 0.0,
                "negative_days_max": int(max(f["test_summary"]["negative_days"] for f in folds)) if folds else 0,
                "max_drawdown_pct": float(max(f["test_summary"]["max_drawdown_pct"] for f in folds)) if folds else 0.0,
                "avg_win_rate": float(sum(f["test_summary"]["win_rate"] for f in folds) / len(folds)) if folds else 0.0,
                "selected_margins": [f["selected_margin"] for f in folds],
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
