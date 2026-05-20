#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.polymarket.fees import FeeModelConfig, compute_trade_fee


@dataclass
class BalanceRiskConfig:
    initial_balance: float = 100.0
    target_balance: float = 1000.0
    min_edge_to_trade: float = 0.004
    kelly_scale: float = 0.75
    max_fraction: float = 0.35
    min_fraction: float = 0.02
    min_trade_notional: float = 5.0
    daily_loss_limit_frac: float = 0.20
    soft_drawdown_frac: float = 0.18
    hard_drawdown_frac: float = 0.45
    drawdown_size_mult: float = 0.55
    loss_streak_size_mult: float = 0.70
    loss_streak_reduce_after: int = 2
    cooldown_after_losses: int = 4
    cooldown_windows: int = 3
    hard_drawdown_mode: str = "stop"
    hard_drawdown_size_mult: float = 0.25
    hard_drawdown_cooldown_windows: int = 0
    target_size_mult: float = 1.0
    stop_after_target: bool = False
    balance_growth_exponent: float = 1.0
    max_trade_notional_abs: float = 0.0
    kelly_probability_temperature: float = 1.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def _clip_prob(value: Any) -> float | None:
    p = _to_float(value)
    if p is None:
        return None
    return max(1e-6, min(1.0 - 1e-6, float(p)))


def _candidate_prob(row: dict[str, Any], candidate_key: str) -> float | None:
    aliases = {
        "market_price_baseline": "market_p_up",
        "market_p_up": "market_p_up",
        "spot_window_path": "spot_window_path_p_up",
        "spot_window_path_p_up": "spot_window_path_p_up",
        "spot_market_blend": "spot_market_blend_p_up",
        "spot_market_blend_p_up": "spot_market_blend_p_up",
        "spot_logistic_online": "spot_logistic_online_p_up",
        "spot_logistic_online_p_up": "spot_logistic_online_p_up",
        "spot_consensus_blend": "spot_consensus_blend_p_up",
        "spot_consensus_blend_p_up": "spot_consensus_blend_p_up",
        "spot_timing_policy": "spot_timing_policy_p_up",
        "spot_timing_policy_p_up": "spot_timing_policy_p_up",
    }
    key = aliases.get(str(candidate_key).strip(), str(candidate_key).strip())
    return _clip_prob(row.get(key))


def _date_key(row: dict[str, Any]) -> str:
    return str(row.get("created_at") or row.get("ts_utc") or "")[:10]


def _chosen_trade(
    row: dict[str, Any],
    p_up: float,
    fee_cfg: FeeModelConfig,
    slippage_bps: float,
) -> dict[str, Any] | None:
    up_price = _to_float(row.get("up_entry_price"))
    down_price = _to_float(row.get("down_entry_price"))
    if up_price is None or down_price is None:
        return None
    if not (0.0 < up_price < 1.0 and 0.0 < down_price < 1.0):
        return None

    up_fee = compute_trade_fee(price=float(up_price), size=1.0, is_taker=True, cfg=fee_cfg)
    down_fee = compute_trade_fee(price=float(down_price), size=1.0, is_taker=True, cfg=fee_cfg)
    up_slip = float(up_price) * (slippage_bps / 10_000.0)
    down_slip = float(down_price) * (slippage_bps / 10_000.0)

    cost_up = float(up_price) + up_fee + up_slip
    cost_down = float(down_price) + down_fee + down_slip
    if not (0.0 < cost_up < 1.0 and 0.0 < cost_down < 1.0):
        return None

    edge_up = p_up - cost_up
    edge_down = (1.0 - p_up) - cost_down
    if edge_up >= edge_down:
        return {
            "side": "up",
            "price": float(up_price),
            "cost": float(cost_up),
            "edge": float(edge_up),
            "p_side": float(p_up),
        }
    return {
        "side": "down",
        "price": float(down_price),
        "cost": float(cost_down),
        "edge": float(edge_down),
        "p_side": float(1.0 - p_up),
    }


def _kelly_fraction(p_side: float, cost: float) -> float:
    if not (0.0 < cost < 1.0):
        return 0.0
    return max(0.0, float((p_side - cost) / max(1e-9, 1.0 - cost)))


def _temperature_scaled_probability(p: float, temperature: float) -> float:
    p_clipped = max(1e-6, min(1.0 - 1e-6, float(p)))
    temp = max(1e-6, float(temperature))
    if abs(temp - 1.0) < 1e-9:
        return p_clipped
    logit = math.log(p_clipped / (1.0 - p_clipped))
    scaled = logit / temp
    return 1.0 / (1.0 + math.exp(-scaled))


def _sizing_balance(balance: float, risk_cfg: BalanceRiskConfig) -> float:
    if balance <= 0.0:
        return 0.0
    initial = max(1e-9, float(risk_cfg.initial_balance))
    exponent = max(0.0, float(risk_cfg.balance_growth_exponent))
    relative_growth = max(float(balance) / initial, 0.0)
    scaled = initial * (relative_growth**exponent)
    return max(0.0, min(float(balance), float(scaled)))


def _risk_config_from_args(args: argparse.Namespace) -> BalanceRiskConfig:
    return BalanceRiskConfig(
        initial_balance=float(args.initial_balance),
        target_balance=float(args.target_balance),
        min_edge_to_trade=float(args.min_edge_to_trade),
        kelly_scale=float(args.kelly_scale),
        max_fraction=float(args.max_fraction),
        min_fraction=float(args.min_fraction),
        min_trade_notional=float(args.min_trade_notional),
        daily_loss_limit_frac=float(args.daily_loss_limit_frac),
        soft_drawdown_frac=float(args.soft_drawdown_frac),
        hard_drawdown_frac=float(args.hard_drawdown_frac),
        drawdown_size_mult=float(args.drawdown_size_mult),
        loss_streak_size_mult=float(args.loss_streak_size_mult),
        loss_streak_reduce_after=int(args.loss_streak_reduce_after),
        cooldown_after_losses=int(args.cooldown_after_losses),
        cooldown_windows=int(args.cooldown_windows),
        hard_drawdown_mode=str(args.hard_drawdown_mode).strip().lower(),
        hard_drawdown_size_mult=float(args.hard_drawdown_size_mult),
        hard_drawdown_cooldown_windows=int(args.hard_drawdown_cooldown_windows),
        target_size_mult=float(args.target_size_mult),
        stop_after_target=bool(args.stop_after_target),
        balance_growth_exponent=float(args.balance_growth_exponent),
        max_trade_notional_abs=float(args.max_trade_notional_abs),
        kelly_probability_temperature=float(args.kelly_probability_temperature),
    )


def run_balance_backtest(
    rows: list[dict[str, Any]],
    *,
    candidate_key: str,
    fee_cfg: FeeModelConfig,
    slippage_bps: float,
    risk_cfg: BalanceRiskConfig,
    include_trade_log: bool = False,
) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: str(row.get("created_at") or row.get("ts_utc") or ""))
    balance = float(risk_cfg.initial_balance)
    peak_balance = balance
    max_drawdown_pct = 0.0
    target_hit_at: str | None = None
    target_hit_trade_index: int | None = None

    current_day = ""
    day_start_balance = balance
    day_low_balance = balance
    loss_streak = 0
    cooldown_left = 0
    hard_drawdown_engaged = False
    reason_counts: Counter[str] = Counter()
    trade_logs: list[dict[str, Any]] = []
    daily: dict[str, dict[str, Any]] = {}

    def _ensure_day(day: str) -> None:
        nonlocal current_day, day_start_balance, day_low_balance, cooldown_left, loss_streak
        if day == current_day:
            return
        current_day = day
        day_start_balance = balance
        day_low_balance = balance
        cooldown_left = 0
        loss_streak = 0
        daily.setdefault(
            day,
            {
                "start_balance": float(balance),
                "end_balance": float(balance),
                "pnl": 0.0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "skips": Counter(),
                "worst_trade_pnl": 0.0,
                "best_trade_pnl": 0.0,
                "target_hit": False,
            },
        )

    for idx, row in enumerate(ordered, start=1):
        day = _date_key(row)
        _ensure_day(day)
        daily_entry = daily[day]

        p_up = _candidate_prob(row, candidate_key)
        if p_up is None:
            reason_counts["missing_probability"] += 1
            daily_entry["skips"]["missing_probability"] += 1
            continue

        trade = _chosen_trade(row, p_up, fee_cfg, slippage_bps)
        if trade is None:
            reason_counts["missing_price"] += 1
            daily_entry["skips"]["missing_price"] += 1
            continue

        if trade["edge"] < float(risk_cfg.min_edge_to_trade):
            reason_counts["edge_below_threshold"] += 1
            daily_entry["skips"]["edge_below_threshold"] += 1
            continue

        daily_drawdown_frac = 0.0
        if day_start_balance > 0:
            daily_drawdown_frac = max(0.0, (day_start_balance - balance) / day_start_balance)
        if daily_drawdown_frac >= float(risk_cfg.daily_loss_limit_frac):
            reason_counts["daily_loss_limit"] += 1
            daily_entry["skips"]["daily_loss_limit"] += 1
            continue

        peak_drawdown_frac = 0.0
        if peak_balance > 0:
            peak_drawdown_frac = max(0.0, (peak_balance - balance) / peak_balance)
        if peak_drawdown_frac >= float(risk_cfg.hard_drawdown_frac):
            hard_drawdown_engaged = True
        elif peak_drawdown_frac < float(risk_cfg.hard_drawdown_frac) * 0.8:
            hard_drawdown_engaged = False

        if bool(risk_cfg.stop_after_target) and target_hit_at is not None:
            reason_counts["after_target_stop"] += 1
            daily_entry["skips"]["after_target_stop"] += 1
            continue

        if peak_drawdown_frac >= float(risk_cfg.hard_drawdown_frac) and str(risk_cfg.hard_drawdown_mode) == "stop":
            reason_counts["hard_drawdown_stop"] += 1
            daily_entry["skips"]["hard_drawdown_stop"] += 1
            continue

        if cooldown_left > 0:
            cooldown_left -= 1
            reason_counts["cooldown"] += 1
            daily_entry["skips"]["cooldown"] += 1
            continue

        sizing_p_side = _temperature_scaled_probability(
            float(trade["p_side"]),
            float(risk_cfg.kelly_probability_temperature),
        )
        kelly_fraction = _kelly_fraction(float(sizing_p_side), float(trade["cost"]))
        if kelly_fraction <= 0.0:
            reason_counts["non_positive_kelly"] += 1
            daily_entry["skips"]["non_positive_kelly"] += 1
            continue

        size_mult = 1.0
        if peak_drawdown_frac >= float(risk_cfg.soft_drawdown_frac):
            size_mult *= float(risk_cfg.drawdown_size_mult)
        if peak_drawdown_frac >= float(risk_cfg.hard_drawdown_frac) and str(risk_cfg.hard_drawdown_mode) == "throttle":
            size_mult *= float(risk_cfg.hard_drawdown_size_mult)
            if hard_drawdown_engaged and cooldown_left <= 0 and int(risk_cfg.hard_drawdown_cooldown_windows) > 0:
                cooldown_left = int(risk_cfg.hard_drawdown_cooldown_windows)
        if loss_streak >= int(risk_cfg.loss_streak_reduce_after):
            size_mult *= float(risk_cfg.loss_streak_size_mult)
        if target_hit_at is not None:
            size_mult *= float(risk_cfg.target_size_mult)

        spend_fraction = float(risk_cfg.kelly_scale) * float(kelly_fraction) * size_mult
        spend_fraction = min(float(risk_cfg.max_fraction), spend_fraction)
        if spend_fraction < float(risk_cfg.min_fraction):
            reason_counts["fraction_below_min"] += 1
            daily_entry["skips"]["fraction_below_min"] += 1
            continue

        spend_notional = spend_fraction * _sizing_balance(balance, risk_cfg)
        if float(risk_cfg.max_trade_notional_abs) > 0.0:
            spend_notional = min(float(spend_notional), float(risk_cfg.max_trade_notional_abs))
        if spend_notional < float(risk_cfg.min_trade_notional):
            reason_counts["trade_notional_too_small"] += 1
            daily_entry["skips"]["trade_notional_too_small"] += 1
            continue

        actual_side = str(row.get("actual_side") or "").strip().lower()
        win = actual_side == str(trade["side"])
        payout_multiple = (1.0 - float(trade["cost"])) / float(trade["cost"])
        pnl = spend_notional * payout_multiple if win else -spend_notional
        balance += pnl
        peak_balance = max(peak_balance, balance)
        day_low_balance = min(day_low_balance, balance)
        max_drawdown_pct = max(
            max_drawdown_pct,
            max(0.0, (peak_balance - balance) / peak_balance) if peak_balance > 0 else 0.0,
        )

        if win:
            loss_streak = 0
        else:
            loss_streak += 1
            if loss_streak >= int(risk_cfg.cooldown_after_losses):
                cooldown_left = int(risk_cfg.cooldown_windows)

        daily_entry["trades"] += 1
        daily_entry["wins"] += int(win)
        daily_entry["losses"] += int(not win)
        daily_entry["pnl"] += float(pnl)
        daily_entry["end_balance"] = float(balance)
        daily_entry["worst_trade_pnl"] = min(float(daily_entry["worst_trade_pnl"]), float(pnl))
        daily_entry["best_trade_pnl"] = max(float(daily_entry["best_trade_pnl"]), float(pnl))

        trade_logs.append(
            {
                "trade_index": idx,
                "market_id": row.get("market_id"),
                "created_at": row.get("created_at"),
                "date": day,
                "side": trade["side"],
                "actual_side": actual_side,
                "win": win,
                "p_side": float(trade["p_side"]),
                "sizing_p_side": float(sizing_p_side),
                "edge": float(trade["edge"]),
                "cost": float(trade["cost"]),
                "balance_before": float(balance - pnl),
                "balance_after": float(balance),
                "spend_fraction": float(spend_fraction),
                "spend_notional": float(spend_notional),
                "kelly_fraction": float(kelly_fraction),
                "size_mult": float(size_mult),
                "pnl": float(pnl),
            }
        )

        if target_hit_at is None and balance >= float(risk_cfg.target_balance):
            target_hit_at = str(row.get("created_at") or "")
            target_hit_trade_index = idx
            daily_entry["target_hit"] = True

    for _day_key, entry in daily.items():
        entry["start_balance"] = float(entry["start_balance"])
        entry["end_balance"] = float(entry["end_balance"])
        entry["pnl"] = float(entry["pnl"])
        entry["win_rate"] = float(entry["wins"] / entry["trades"]) if entry["trades"] > 0 else 0.0
        entry["skips"] = dict(entry["skips"])

    total_trades = len(trade_logs)
    wins = sum(1 for row in trade_logs if row["win"])
    losses = total_trades - wins
    post_target_trades = 0
    post_target_pnl = 0.0
    if target_hit_at is not None:
        for row in trade_logs:
            if str(row.get("created_at") or "") > str(target_hit_at):
                post_target_trades += 1
                post_target_pnl += float(row.get("pnl", 0.0))
    worst_day = None
    best_day = None
    if daily:
        ordered_daily = [{"date": day, **daily[day]} for day in sorted(daily)]
        worst_day = min(ordered_daily, key=lambda item: item["pnl"])
        best_day = max(ordered_daily, key=lambda item: item["pnl"])
    else:
        ordered_daily = []
    positive_days = int(sum(1 for entry in daily.values() if float(entry.get("pnl", 0.0)) > 0.0))
    negative_days = int(sum(1 for entry in daily.values() if float(entry.get("pnl", 0.0)) < 0.0))

    return {
        "candidate_key": candidate_key,
        "initial_balance": float(risk_cfg.initial_balance),
        "final_balance": float(balance),
        "target_balance": float(risk_cfg.target_balance),
        "target_hit": bool(target_hit_at is not None),
        "target_hit_at": target_hit_at,
        "target_hit_trade_index": target_hit_trade_index,
        "return_multiple": float(balance / risk_cfg.initial_balance) if risk_cfg.initial_balance > 0 else 0.0,
        "trades_taken": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": float(wins / total_trades) if total_trades > 0 else 0.0,
        "max_drawdown_pct": float(max_drawdown_pct),
        "peak_balance": float(peak_balance),
        "reason_counts": dict(reason_counts),
        "best_day": best_day,
        "worst_day": worst_day,
        "active_days": int(sum(1 for entry in daily.values() if int(entry.get("trades", 0)) > 0)),
        "positive_days": positive_days,
        "negative_days": negative_days,
        "days_blocked_by_hard_drawdown": int(
            sum(1 for entry in daily.values() if int(entry.get("skips", {}).get("hard_drawdown_stop", 0)) > 0 and int(entry.get("trades", 0)) == 0)
        ),
        "post_target_trades": int(post_target_trades),
        "post_target_pnl": float(post_target_pnl),
        "daily": ordered_daily,
        "trade_log_preview": trade_logs[:20],
        "last_trade": trade_logs[-1] if trade_logs else None,
        "trade_log": trade_logs if include_trade_log else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a bankroll growth backtest on historical BTC 5m replay rows.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--candidate-key", default="spot_consensus_blend")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--initial-balance", type=float, default=100.0)
    ap.add_argument("--target-balance", type=float, default=1000.0)
    ap.add_argument("--min-edge-to-trade", type=float, default=0.004)
    ap.add_argument("--kelly-scale", type=float, default=0.75)
    ap.add_argument("--max-fraction", type=float, default=0.35)
    ap.add_argument("--min-fraction", type=float, default=0.02)
    ap.add_argument("--min-trade-notional", type=float, default=5.0)
    ap.add_argument("--daily-loss-limit-frac", type=float, default=0.20)
    ap.add_argument("--soft-drawdown-frac", type=float, default=0.18)
    ap.add_argument("--hard-drawdown-frac", type=float, default=0.45)
    ap.add_argument("--drawdown-size-mult", type=float, default=0.55)
    ap.add_argument("--loss-streak-size-mult", type=float, default=0.70)
    ap.add_argument("--loss-streak-reduce-after", type=int, default=2)
    ap.add_argument("--cooldown-after-losses", type=int, default=4)
    ap.add_argument("--cooldown-windows", type=int, default=3)
    ap.add_argument("--hard-drawdown-mode", choices=["stop", "throttle"], default="stop")
    ap.add_argument("--hard-drawdown-size-mult", type=float, default=0.25)
    ap.add_argument("--hard-drawdown-cooldown-windows", type=int, default=0)
    ap.add_argument("--target-size-mult", type=float, default=1.0)
    ap.add_argument("--stop-after-target", action="store_true")
    ap.add_argument("--balance-growth-exponent", type=float, default=1.0)
    ap.add_argument("--max-trade-notional-abs", type=float, default=0.0)
    ap.add_argument("--kelly-probability-temperature", type=float, default=1.0)
    ap.add_argument("--slippage-bps", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = _read_jsonl(Path(args.dataset))
    cfg_raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    fee_cfg = FeeModelConfig(**cfg_raw["fees"])
    exec_cfg = cfg_raw.get("execution", {})
    slippage_bps = float(args.slippage_bps if args.slippage_bps is not None else exec_cfg.get("slippage_bps_taker", 12.0))
    risk_cfg = _risk_config_from_args(args)

    payload = run_balance_backtest(
        rows,
        candidate_key=str(args.candidate_key),
        fee_cfg=fee_cfg,
        slippage_bps=slippage_bps,
        risk_cfg=risk_cfg,
    )
    payload["slippage_bps"] = slippage_bps
    payload["risk_config"] = {
        "initial_balance": risk_cfg.initial_balance,
        "target_balance": risk_cfg.target_balance,
        "min_edge_to_trade": risk_cfg.min_edge_to_trade,
        "kelly_scale": risk_cfg.kelly_scale,
        "max_fraction": risk_cfg.max_fraction,
        "min_fraction": risk_cfg.min_fraction,
        "min_trade_notional": risk_cfg.min_trade_notional,
        "daily_loss_limit_frac": risk_cfg.daily_loss_limit_frac,
        "soft_drawdown_frac": risk_cfg.soft_drawdown_frac,
        "hard_drawdown_frac": risk_cfg.hard_drawdown_frac,
        "drawdown_size_mult": risk_cfg.drawdown_size_mult,
        "loss_streak_size_mult": risk_cfg.loss_streak_size_mult,
        "loss_streak_reduce_after": risk_cfg.loss_streak_reduce_after,
        "cooldown_after_losses": risk_cfg.cooldown_after_losses,
        "cooldown_windows": risk_cfg.cooldown_windows,
        "hard_drawdown_mode": risk_cfg.hard_drawdown_mode,
        "hard_drawdown_size_mult": risk_cfg.hard_drawdown_size_mult,
        "hard_drawdown_cooldown_windows": risk_cfg.hard_drawdown_cooldown_windows,
        "target_size_mult": risk_cfg.target_size_mult,
        "stop_after_target": risk_cfg.stop_after_target,
        "balance_growth_exponent": risk_cfg.balance_growth_exponent,
        "max_trade_notional_abs": risk_cfg.max_trade_notional_abs,
        "kelly_probability_temperature": risk_cfg.kelly_probability_temperature,
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
