#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest.redecision import RedecisionConfig, RedecisionPolicy, load_rollout_rows, run_redecision_replay
from src.backtest.regime_rule_diagnostics import build_trade_features
from src.backtest.regime_sensitivity import (
    DEFAULT_ANCHOR_PARAMS,
    build_quiet_consensus_chase_grid,
    evaluate_rule_grid,
    rank_results_for_robustness,
    summarize_sensitivity,
    summarize_trades_by_run,
    train_test_validate_grid,
)
from src.polymarket.fees import FeeModelConfig

DEFAULT_RUNS = (
    "outputs/live_rollout/live_rollout_log180_then_log240_20260315_081103",
    "outputs/live_rollout/live_rollout_log180_then_log240_20260316_235922",
    "outputs/live_rollout/live_rollout_safe_20260406_000311",
    "outputs/live_rollout/live_rollout_safe_postfix_20260422_manual",
    "outputs/live_rollout/live_rollout_safe_sizingcap_phase7_20260424_runtime_cap12",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def _fee_cfg(cfg: Mapping[str, Any]) -> FeeModelConfig:
    fees = cfg.get("fees") if isinstance(cfg.get("fees"), dict) else {}
    return FeeModelConfig(
        maker_fee_bps=float(fees.get("maker_fee_bps", 0.0)),
        taker_fee_bps=float(fees.get("taker_fee_bps", 0.0)),
        curve_rate=float(fees.get("curve_rate", 0.0)),
        curve_exponent=float(fees.get("curve_exponent", 1.0)),
        min_fee=float(fees.get("min_fee", 0.0)),
    )


def _parse_optional_float(raw: str) -> float | None:
    value = str(raw).strip().lower()
    if value in {"none", "null"}:
        return None
    return float(value)


def _discover_run_dirs(root: Path) -> list[Path]:
    return sorted({path.parent for path in root.rglob("paper_outcomes_5m.jsonl")}, key=lambda path: str(path))


def _run_order(trades: Iterable[Mapping[str, Any]]) -> list[str]:
    earliest: dict[str, str] = {}
    for trade in trades:
        run = str(trade.get("run") or "unknown")
        ts = str(trade.get("created_at") or "")
        if not ts:
            ts = run
        if run not in earliest or ts < earliest[run]:
            earliest[run] = ts
    return [run for run, _ in sorted(earliest.items(), key=lambda item: (item[1], item[0]))]


def _target_run(run_order: list[str], explicit: str | None) -> str | None:
    if explicit:
        for name in run_order:
            if explicit == name or explicit in name:
                return name
    for name in run_order:
        if "phase7" in name.lower():
            return name
    return run_order[-1] if run_order else None


def _load_trade_features(
    run_dir: Path,
    *,
    candidate_key: str,
    min_expected_roi_cash: float | None,
    fill_mode: str,
    initial_balance: float,
    min_trade_notional: float,
) -> list[dict[str, Any]]:
    cfg = _load_yaml(run_dir / "paper_config.yaml")
    rows = load_rollout_rows(run_dir)
    overrides: dict[str, Any] = {}
    if min_expected_roi_cash is not None:
        overrides["min_expected_roi_cash"] = float(min_expected_roi_cash)
    policy = RedecisionPolicy(
        name=str(candidate_key),
        candidate_key=str(candidate_key),
        exec_overrides=overrides,
    )
    replay = run_redecision_replay(
        rows,
        policies=[policy],
        fee_cfg=_fee_cfg(cfg),
        exec_cfg=cfg.get("execution") if isinstance(cfg.get("execution"), dict) else {},
        risk_cfg=cfg.get("risk") if isinstance(cfg.get("risk"), dict) else {},
        config=RedecisionConfig(
            initial_balance=float(initial_balance),
            min_trade_notional=float(min_trade_notional),
            assume_filled=str(fill_mode) == "assume_filled",
            fill_mode=str(fill_mode),
            include_decisions=True,
        ),
    )
    decisions = replay["policies"][str(candidate_key)]["decisions"]
    return build_trade_features(rows, decisions, run_name=run_dir.name, fill_mode=str(fill_mode))


def _train_runs(run_order: list[str], target_run: str | None, mode: str) -> list[str]:
    if target_run is None:
        return run_order[:-1]
    if mode == "not_target":
        return [run for run in run_order if run != target_run]
    if target_run in run_order:
        idx = run_order.index(target_run)
        before = run_order[:idx]
        return before if before else [run for run in run_order if run != target_run]
    return [run for run in run_order if run != target_run]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate quiet-consensus-chase robustness with threshold sensitivity and train/test split.")
    parser.add_argument("run_dir", nargs="*", help="Rollout directories. Defaults to the five key rollouts unless --discover-root is used.")
    parser.add_argument("--discover-root", help="Recursively discover rollout dirs containing paper_outcomes_5m.jsonl.")
    parser.add_argument("--candidate-key", default="spot_consensus_blend")
    parser.add_argument("--min-expected-roi-cash", default="0.02", help="Float threshold or 'none'.")
    parser.add_argument("--fill-mode", choices=("assume_filled", "paper_sim"), default="assume_filled")
    parser.add_argument("--target-run", help="Run name/substr used as holdout target; defaults to the Phase7 run.")
    parser.add_argument("--train-mode", choices=("before_target", "not_target"), default="before_target")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--initial-balance", type=float, default=100.0)
    parser.add_argument("--min-trade-notional", type=float, default=0.0)
    parser.add_argument("--min-train-skipped-filled", type=int, default=3)
    parser.add_argument("--include-all-rules", action="store_true")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    if args.discover_root:
        run_dirs = _discover_run_dirs(Path(args.discover_root))
    elif args.run_dir:
        run_dirs = [Path(value) for value in args.run_dir]
    else:
        run_dirs = [Path(value) for value in DEFAULT_RUNS]
    if not run_dirs:
        raise SystemExit("No rollout directories found.")
    min_expected_roi_cash = _parse_optional_float(str(args.min_expected_roi_cash))

    trades: list[dict[str, Any]] = []
    load_errors: list[dict[str, str]] = []
    for run_dir in run_dirs:
        try:
            trades.extend(
                _load_trade_features(
                    run_dir,
                    candidate_key=str(args.candidate_key),
                    min_expected_roi_cash=min_expected_roi_cash,
                    fill_mode=str(args.fill_mode),
                    initial_balance=float(args.initial_balance),
                    min_trade_notional=float(args.min_trade_notional),
                )
            )
        except Exception as exc:
            load_errors.append({"run_dir": str(run_dir), "error": str(exc)})

    grid = build_quiet_consensus_chase_grid()
    order = _run_order(trades)
    target = _target_run(order, args.target_run)
    train_runs = _train_runs(order, target, str(args.train_mode))
    test_runs = [target] if target else order[-1:]

    all_results = evaluate_rule_grid(trades, grid)
    ranked_all = rank_results_for_robustness(all_results, target_run=target)
    output: dict[str, Any] = {
        "candidate_key": str(args.candidate_key),
        "min_expected_roi_cash": min_expected_roi_cash,
        "fill_mode": str(args.fill_mode),
        "anchor_params": {
            "agreement_min": DEFAULT_ANCHOR_PARAMS.agreement_min,
            "chase_absret_min": DEFAULT_ANCHOR_PARAMS.chase_absret_min,
            "recent_flat_max": DEFAULT_ANCHOR_PARAMS.recent_flat_max,
            "vol5m_max": DEFAULT_ANCHOR_PARAMS.vol5m_max,
        },
        "run_dirs": [str(path) for path in run_dirs],
        "load_errors": load_errors,
        "run_order": order,
        "target_run": target,
        "train_mode": str(args.train_mode),
        "baseline_by_run": summarize_trades_by_run(trades),
        "trade_count": len(trades),
        "grid_rule_count": len(grid),
        "overall_sensitivity": summarize_sensitivity(
            all_results,
            target_run=target,
            min_delta=0.0,
            max_negative_delta_runs=0,
            min_skipped_filled=1,
            require_target_overlay_nonnegative=bool(target),
        ),
        "top_overall_rules": ranked_all[: int(args.top_n)],
        "train_test": train_test_validate_grid(
            trades,
            grid,
            train_runs=train_runs,
            test_runs=test_runs,
            target_run=target,
            top_n=int(args.top_n),
            min_train_delta=0.0,
            max_train_negative_delta_runs=0,
            min_train_skipped_filled=int(args.min_train_skipped_filled),
        ),
    }
    if args.include_all_rules:
        output["all_rules"] = ranked_all

    text = json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
