#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_balance_growth import _candidate_prob, _chosen_trade, _date_key, _read_jsonl, run_balance_backtest
from scripts.compare_balance_profiles import _profile_config
from src.polymarket.fees import FeeModelConfig
from src.strategy.timing_policy import TimingPolicyStage, next_future_stage_index, timing_policy_entry_mode


def _parse_dataset_specs(values: Iterable[str]) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for raw in values:
        label, sep, path_raw = str(raw).partition("=")
        if not sep:
            raise SystemExit(f"dataset spec must be DELAY=PATH, got: {raw}")
        delay = int(label.strip().rstrip("s"))
        out.append((delay, Path(path_raw.strip()).resolve()))
    return sorted(out, key=lambda item: item[0])


def _parse_stage_specs(raw: str) -> list[tuple[int, str, float]]:
    out: list[tuple[int, str, float]] = []
    for chunk in str(raw).split(","):
        item = chunk.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) not in {2, 3}:
            raise SystemExit(f"stage spec must be DELAY:CANDIDATE[:EXTRA_EDGE], got: {item}")
        delay = int(parts[0].strip().rstrip("s"))
        candidate = parts[1].strip()
        extra_edge = float(parts[2].strip()) if len(parts) == 3 else 0.0
        out.append((delay, candidate, extra_edge))
    if not out:
        raise SystemExit("at least one stage is required")
    return sorted(out, key=lambda item: item[0])


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "final_balance": float(result.get("final_balance", 0.0)),
        "target_hit": bool(result.get("target_hit", False)),
        "target_hit_at": result.get("target_hit_at"),
        "trades_taken": int(result.get("trades_taken", 0)),
        "win_rate": float(result.get("win_rate", 0.0)),
        "max_drawdown_pct": float(result.get("max_drawdown_pct", 0.0)),
        "active_days": int(result.get("active_days", 0)),
        "positive_days": int(result.get("positive_days", 0)),
        "negative_days": int(result.get("negative_days", 0)),
        "days_blocked_by_hard_drawdown": int(result.get("days_blocked_by_hard_drawdown", 0)),
        "best_day_pnl": float((result.get("best_day") or {}).get("pnl", 0.0)),
        "worst_day_pnl": float((result.get("worst_day") or {}).get("pnl", 0.0)),
    }


def _split_rows(rows: list[dict[str, Any]], holdout_days: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    dates = sorted({_date_key(row) for row in rows if _date_key(row)})
    if len(dates) <= holdout_days:
        raise SystemExit(f"not enough unique days for holdout={holdout_days}: only {len(dates)}")
    train_days = dates[:-holdout_days]
    test_days = dates[-holdout_days:]
    train_set = set(train_days)
    test_set = set(test_days)
    train_rows = [row for row in rows if _date_key(row) in train_set]
    test_rows = [row for row in rows if _date_key(row) in test_set]
    return train_rows, test_rows, train_days, test_days


def common_days_across_delays(rows_by_delay: dict[int, list[dict[str, Any]]]) -> list[str]:
    day_sets = []
    for rows in rows_by_delay.values():
        day_sets.append({_date_key(row) for row in rows if _date_key(row)})
    if not day_sets:
        return []
    return sorted(set.intersection(*day_sets))


def filter_rows_by_days(rows_by_delay: dict[int, list[dict[str, Any]]], allowed_days: set[str]) -> dict[int, list[dict[str, Any]]]:
    return {
        delay: [row for row in rows if _date_key(row) in allowed_days]
        for delay, rows in rows_by_delay.items()
    }


def _row_seconds_from_start(row: dict[str, Any], default_delay: int) -> float:
    value = row.get("seconds_from_window_start")
    if value is None:
        value = row.get("entry_delay_seconds")
    try:
        return float(value)
    except Exception:
        return float(default_delay)


def build_timing_policy_rows(
    rows_by_delay: dict[int, list[dict[str, Any]]],
    *,
    stages: list[tuple[int, str, float]],
    fee_cfg: FeeModelConfig,
    slippage_bps: float,
    min_edge_to_trade: float,
    allow_late_fresh_start: bool = True,
    late_start_grace_seconds: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_by_market: dict[str, dict[int, dict[str, Any]]] = {}
    available_delay_counts: Counter[int] = Counter()
    timing_stages = tuple(
        TimingPolicyStage(delay_seconds=int(delay), candidate_key=str(candidate_key), extra_edge=float(extra_edge))
        for delay, candidate_key, extra_edge in stages
    )
    for delay, rows in rows_by_delay.items():
        for row in rows:
            market_id = str(row.get("market_id") or row.get("market_slug") or "")
            if not market_id:
                continue
            rows_by_market.setdefault(market_id, {})[delay] = row
            available_delay_counts[delay] += 1

    selected_rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    delay_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    missing_stage_counts: Counter[str] = Counter()
    skipped_market_reason_counts: Counter[str] = Counter()

    for _market_id, delay_map in rows_by_market.items():
        selected: dict[str, Any] | None = None
        last_candidate_row: dict[str, Any] | None = None
        last_any_row: dict[str, Any] | None = None
        skip_market_reason: str | None = None
        for stage_idx, (delay, candidate_key, extra_edge) in enumerate(stages):
            row = delay_map.get(delay)
            if row is None:
                missing_stage_counts[f"{delay}:{candidate_key}"] += 1
                continue
            last_any_row = row
            seconds_from_start = _row_seconds_from_start(row, int(delay))
            entry_mode = timing_policy_entry_mode(
                stage=timing_stages[stage_idx],
                current_stage_index=None,
                seconds_from_start=seconds_from_start,
                late_start_grace_seconds=late_start_grace_seconds,
            )
            if entry_mode == "late_fresh_start" and not allow_late_fresh_start:
                future_stage_index = next_future_stage_index(
                    timing_stages,
                    seconds_from_start=seconds_from_start,
                    current_stage_index=stage_idx,
                )
                if future_stage_index is not None:
                    continue
                skip_market_reason = "late_fresh_start_disabled"
                break
            p_up = _candidate_prob(row, candidate_key)
            trade = _chosen_trade(row, p_up, fee_cfg, slippage_bps) if p_up is not None else None
            row_local = dict(row)
            row_local["spot_timing_policy_p_up"] = p_up
            row_local["spot_timing_policy_source"] = candidate_key
            row_local["spot_timing_policy_stage_delay"] = int(delay)
            row_local["spot_timing_policy_stage_extra_edge"] = float(extra_edge)
            row_local["spot_timing_policy_reason"] = "stage_fallback"
            row_local["spot_timing_policy_entry_mode"] = entry_mode
            row_local["spot_timing_policy_selected_side"] = trade.get("side") if trade is not None else None
            row_local["spot_timing_policy_edge"] = float(trade.get("edge")) if trade is not None else None
            last_candidate_row = row_local
            required_edge = float(min_edge_to_trade) + max(0.0, float(extra_edge))
            if trade is not None and float(trade.get("edge", 0.0)) >= required_edge:
                row_local["spot_timing_policy_reason"] = "edge_hit"
                selected = row_local
                break

        if skip_market_reason is not None:
            skipped_market_reason_counts[skip_market_reason] += 1
            continue

        if selected is None:
            if last_candidate_row is not None:
                selected = last_candidate_row
                reason_counts[str(selected.get("spot_timing_policy_reason") or "stage_fallback")] += 1
            else:
                fallback_row = last_any_row
                if fallback_row is None and delay_map:
                    fallback_row = delay_map[max(delay_map)]
                selected = dict(fallback_row or {})
                selected["spot_timing_policy_p_up"] = None
                selected["spot_timing_policy_source"] = None
                selected["spot_timing_policy_stage_delay"] = None
                selected["spot_timing_policy_stage_extra_edge"] = None
                selected["spot_timing_policy_reason"] = "no_candidate_probability"
                selected["spot_timing_policy_selected_side"] = None
                selected["spot_timing_policy_edge"] = None
                reason_counts["no_candidate_probability"] += 1
        else:
            reason_counts[str(selected.get("spot_timing_policy_reason") or "edge_hit")] += 1

        source = str(selected.get("spot_timing_policy_source") or "none")
        stage_delay = selected.get("spot_timing_policy_stage_delay")
        source_counts[source] += 1
        delay_counts[str(stage_delay) if stage_delay is not None else "none"] += 1
        selected_rows.append(selected)

    selected_rows.sort(key=lambda row: str(row.get("created_at") or row.get("ts_utc") or ""))
    diagnostics = {
        "markets_total": len(rows_by_market),
        "rows_selected": len(selected_rows),
        "available_delay_counts": {str(k): int(v) for k, v in sorted(available_delay_counts.items())},
        "selected_source_counts": dict(source_counts),
        "selected_delay_counts": dict(delay_counts),
        "selection_reason_counts": dict(reason_counts),
        "missing_stage_counts": dict(missing_stage_counts),
        "skipped_market_reason_counts": dict(skipped_market_reason_counts),
    }
    return selected_rows, diagnostics


def main() -> int:
    ap = argparse.ArgumentParser(description="Build and evaluate a sequential timing policy over multiple entry-delay datasets.")
    ap.add_argument("--datasets", nargs="+", required=True, help="One or more DELAY=PATH specs, e.g. 120=foo.jsonl")
    ap.add_argument("--stages", default="120:spot_consensus_blend,180:spot_consensus_blend,240:spot_logistic_online")
    ap.add_argument("--profile", default="steady_compound_v1")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--holdout-days", type=int, default=5)
    ap.add_argument("--common-days-only", action="store_true", help="Restrict all delay datasets to the intersection of observed days before building the policy.")
    ap.add_argument("--out-dataset", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dataset_specs = _parse_dataset_specs(list(args.datasets))
    stages = _parse_stage_specs(str(args.stages))
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
    policy_rows, diagnostics = build_timing_policy_rows(
        rows_by_delay,
        stages=stages,
        fee_cfg=fee_cfg,
        slippage_bps=slippage_bps,
        min_edge_to_trade=min_edge_to_trade,
        allow_late_fresh_start=allow_late_fresh_start,
        late_start_grace_seconds=late_start_grace_seconds,
    )
    train_rows, test_rows, train_days, test_days = _split_rows(policy_rows, int(args.holdout_days))
    full_result = run_balance_backtest(policy_rows, candidate_key="spot_timing_policy", fee_cfg=fee_cfg, slippage_bps=slippage_bps, risk_cfg=risk_cfg)
    train_result = run_balance_backtest(train_rows, candidate_key="spot_timing_policy", fee_cfg=fee_cfg, slippage_bps=slippage_bps, risk_cfg=risk_cfg)
    test_result = run_balance_backtest(test_rows, candidate_key="spot_timing_policy", fee_cfg=fee_cfg, slippage_bps=slippage_bps, risk_cfg=risk_cfg)

    payload = {
        "profile": str(args.profile),
        "stages": [
            {
                "delay_seconds": int(delay),
                "candidate_key": cand,
                "extra_edge": float(extra_edge),
            }
            for delay, cand, extra_edge in stages
        ],
        "datasets": {str(delay): str(path) for delay, path in dataset_specs},
        "timing_policy_runtime": {
            "allow_late_fresh_start": allow_late_fresh_start,
            "late_start_grace_seconds": late_start_grace_seconds,
        },
        "common_days_only": bool(args.common_days_only),
        "coverage_days": coverage_days,
        "selection": diagnostics,
        "full": _summary(full_result),
        "oos_split": {
            "holdout_days": int(args.holdout_days),
            "train_days": train_days,
            "test_days": test_days,
            "rows_train": len(train_rows),
            "rows_test": len(test_rows),
            "train": _summary(train_result),
            "test": _summary(test_result),
        },
    }

    if args.out_dataset:
        out_dataset = Path(args.out_dataset)
        out_dataset.parent.mkdir(parents=True, exist_ok=True)
        with out_dataset.open("w", encoding="utf-8") as f:
            for row in policy_rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
