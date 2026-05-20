#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.backtest_balance_growth import BalanceRiskConfig, _read_jsonl, run_balance_backtest
from src.polymarket.fees import FeeModelConfig

PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "default": {},
    "target_guard_v2": {
        "kelly_scale": 0.30,
        "max_fraction": 0.14,
        "daily_loss_limit_frac": 0.12,
        "soft_drawdown_frac": 0.12,
        "hard_drawdown_frac": 0.28,
        "drawdown_size_mult": 0.50,
        "hard_drawdown_mode": "throttle",
        "hard_drawdown_size_mult": 0.18,
        "hard_drawdown_cooldown_windows": 6,
        "target_size_mult": 0.0,
        "stop_after_target": True,
        "balance_growth_exponent": 0.65,
        "max_trade_notional_abs": 100.0,
    },
    "steady_compound_v1": {
        "kelly_scale": 0.22,
        "max_fraction": 0.12,
        "daily_loss_limit_frac": 0.10,
        "soft_drawdown_frac": 0.10,
        "hard_drawdown_frac": 0.25,
        "drawdown_size_mult": 0.45,
        "hard_drawdown_mode": "throttle",
        "hard_drawdown_size_mult": 0.15,
        "hard_drawdown_cooldown_windows": 8,
        "target_size_mult": 0.15,
        "stop_after_target": False,
        "balance_growth_exponent": 0.50,
        "max_trade_notional_abs": 90.0,
    },
    "balanced_growth_v2": {
        "kelly_scale": 0.28,
        "max_fraction": 0.14,
        "daily_loss_limit_frac": 0.12,
        "soft_drawdown_frac": 0.12,
        "hard_drawdown_frac": 0.28,
        "drawdown_size_mult": 0.50,
        "hard_drawdown_mode": "throttle",
        "hard_drawdown_size_mult": 0.18,
        "hard_drawdown_cooldown_windows": 6,
        "target_size_mult": 0.20,
        "stop_after_target": False,
        "balance_growth_exponent": 0.60,
        "max_trade_notional_abs": 120.0,
    },
}


def _parse_dataset_specs(values: list[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for raw in values:
        label, sep, path_raw = raw.partition("=")
        if not sep:
            raise SystemExit(f"dataset spec must be LABEL=PATH, got: {raw}")
        out.append((label.strip(), Path(path_raw.strip()).resolve()))
    return out


def _profile_config(name: str) -> BalanceRiskConfig:
    if name not in PROFILE_PRESETS:
        raise SystemExit(f"unknown profile: {name}")
    return BalanceRiskConfig(
        initial_balance=100.0,
        target_balance=1000.0,
        min_edge_to_trade=0.004,
        min_fraction=0.01,
        min_trade_notional=5.0,
        loss_streak_size_mult=0.65,
        loss_streak_reduce_after=2,
        cooldown_after_losses=3,
        cooldown_windows=3,
        **PROFILE_PRESETS[name],
    )


def _profile_summary(result: dict[str, Any]) -> dict[str, Any]:
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
        "post_target_trades": int(result.get("post_target_trades", 0)),
        "post_target_pnl": float(result.get("post_target_pnl", 0.0)),
        "best_day_pnl": float((result.get("best_day") or {}).get("pnl", 0.0)),
        "worst_day_pnl": float((result.get("worst_day") or {}).get("pnl", 0.0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare bankroll-risk profiles on historical BTC 5m datasets.")
    ap.add_argument("--datasets", nargs="+", required=True, help="One or more LABEL=PATH specs")
    ap.add_argument("--profiles", default="default,target_guard_v2,steady_compound_v1,balanced_growth_v2")
    ap.add_argument("--candidate-key", default="spot_consensus_blend")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dataset_specs = _parse_dataset_specs(list(args.datasets))
    cfg_raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    fee_cfg = FeeModelConfig(**cfg_raw["fees"])
    slippage_bps = float(cfg_raw.get("execution", {}).get("slippage_bps_taker", 12.0))
    profile_names = [x.strip() for x in str(args.profiles).split(",") if x.strip()]

    rows_by_label = {label: _read_jsonl(path) for label, path in dataset_specs}
    payload: dict[str, Any] = {
        "candidate_key": str(args.candidate_key),
        "slippage_bps": slippage_bps,
        "datasets": {label: str(path) for label, path in dataset_specs},
        "profiles": {},
    }

    for name in profile_names:
        risk_cfg = _profile_config(name)
        per_dataset: dict[str, Any] = {}
        for label, _ in dataset_specs:
            result = run_balance_backtest(
                rows_by_label[label],
                candidate_key=str(args.candidate_key),
                fee_cfg=fee_cfg,
                slippage_bps=slippage_bps,
                risk_cfg=risk_cfg,
            )
            per_dataset[label] = _profile_summary(result)
        values = list(per_dataset.values())
        payload["profiles"][name] = {
            "risk_config": risk_cfg.__dict__,
            "aggregate": {
                "all_hit_target": all(bool(x["target_hit"]) for x in values),
                "min_final_balance": float(min(x["final_balance"] for x in values)),
                "avg_final_balance": float(sum(x["final_balance"] for x in values) / len(values)),
                "max_drawdown_pct": float(max(x["max_drawdown_pct"] for x in values)),
                "min_active_days": int(min(x["active_days"] for x in values)),
                "positive_days_min": int(min(x["positive_days"] for x in values)),
                "negative_days_max": int(max(x["negative_days"] for x in values)),
            },
            "datasets": per_dataset,
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
