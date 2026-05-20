from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any

from src.polymarket.fees import FeeModelConfig, compute_trade_fee
from src.strategy.sizing import cap_exposure_by_balance, size_from_edge


@dataclass
class PredictedSideBacktestConfig:
    initial_balance: float = 100.0
    min_edge_to_trade: float = 0.004
    max_exposure_per_window_usd: float = 150.0
    max_balance_fraction_per_trade: float = 0.12
    flat_trade_notional: float | None = None
    min_trade_notional: float = 0.0
    use_normalized_entry_prices: bool = True
    max_raw_entry_sum_deviation: float | None = None
    include_trade_log: bool = False


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def _valid_price(value: Any) -> bool:
    value_f = _to_float(value)
    return value_f is not None and 0.0 < value_f < 1.0


def _date_key(row: dict[str, Any]) -> str:
    return str(row.get("created_at") or row.get("ts_utc") or "")[:10]


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
    value = _to_float(row.get(key))
    if value is None:
        return None
    return max(1e-6, min(1.0 - 1e-6, float(value)))


def _selected_side_price(
    row: dict[str, Any],
    *,
    predicted_side: str,
    use_normalized_entry_prices: bool,
) -> float | None:
    if use_normalized_entry_prices:
        up_price = _to_float(row.get("up_entry_price_norm"))
        down_price = _to_float(row.get("down_entry_price_norm"))
    else:
        up_price = None
        down_price = None
    if not (_valid_price(up_price) and _valid_price(down_price)):
        up_price = _to_float(row.get("up_entry_price"))
        down_price = _to_float(row.get("down_entry_price"))
    if not (_valid_price(up_price) and _valid_price(down_price)):
        return None
    return float(up_price if predicted_side == "up" else down_price)


def _cash_cost_per_share(
    *,
    price: float,
    fee_cfg: FeeModelConfig,
    slippage_bps: float,
) -> float:
    fee_per_share = compute_trade_fee(price=price, size=1.0, is_taker=True, cfg=fee_cfg)
    slip_per_share = float(price) * (float(slippage_bps) / 10_000.0)
    return float(price) + float(fee_per_share) + float(slip_per_share)


def _summary_stats(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def run_predicted_side_backtest(
    rows: Iterable[dict[str, Any]],
    *,
    candidate_key: str,
    fee_cfg: FeeModelConfig,
    slippage_bps: float,
    config: PredictedSideBacktestConfig,
) -> dict[str, Any]:
    ordered = sorted(list(rows), key=lambda row: str(row.get("created_at") or row.get("ts_utc") or ""))

    balance = float(config.initial_balance)
    peak_balance = float(balance)
    max_drawdown_pct = 0.0

    daily: dict[str, dict[str, Any]] = {}
    trade_logs: list[dict[str, Any]] = []
    skip_reason_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()
    signal_stage_counts: Counter[str] = Counter()
    signal_entry_mode_counts: Counter[str] = Counter()
    trade_entry_mode_counts: Counter[str] = Counter()

    signal_total = 0
    signal_correct = 0
    trades_taken = 0
    trade_wins = 0

    def _ensure_day(day: str) -> dict[str, Any]:
        daily.setdefault(
            day,
            {
                "start_balance": float(balance),
                "end_balance": float(balance),
                "pnl": 0.0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "signal_rows": 0,
                "signal_correct": 0,
                "skips": Counter(),
                "best_trade_pnl": 0.0,
                "worst_trade_pnl": 0.0,
            },
        )
        return daily[day]

    for idx, row in enumerate(ordered, start=1):
        day = _date_key(row)
        daily_entry = _ensure_day(day)

        p_up = _candidate_prob(row, candidate_key)
        if p_up is None:
            skip_reason_counts["missing_probability"] += 1
            daily_entry["skips"]["missing_probability"] += 1
            continue

        predicted_side = "up" if float(p_up) >= 0.5 else "down"
        actual_side = str(row.get("actual_side") or "").strip().lower()
        signal_total += 1
        daily_entry["signal_rows"] += 1
        if actual_side == predicted_side:
            signal_correct += 1
            daily_entry["signal_correct"] += 1

        stage_delay = row.get("spot_timing_policy_stage_delay")
        entry_mode = str(row.get("spot_timing_policy_entry_mode") or "").strip()
        signal_stage_counts[str(stage_delay) if stage_delay is not None else "none"] += 1
        signal_entry_mode_counts[entry_mode or "none"] += 1

        raw_sum = _to_float(row.get("entry_price_sum_raw"))
        if (
            config.max_raw_entry_sum_deviation is not None
            and raw_sum is not None
            and abs(float(raw_sum) - 1.0) > float(config.max_raw_entry_sum_deviation)
        ):
            skip_reason_counts["entry_price_sum_out_of_bounds"] += 1
            daily_entry["skips"]["entry_price_sum_out_of_bounds"] += 1
            continue

        price = _selected_side_price(
            row,
            predicted_side=predicted_side,
            use_normalized_entry_prices=bool(config.use_normalized_entry_prices),
        )
        if price is None:
            skip_reason_counts["missing_entry_price"] += 1
            daily_entry["skips"]["missing_entry_price"] += 1
            continue

        p_side = float(p_up if predicted_side == "up" else (1.0 - p_up))
        cost_per_share = _cash_cost_per_share(price=price, fee_cfg=fee_cfg, slippage_bps=slippage_bps)
        edge_per_share = float(p_side - cost_per_share)
        if edge_per_share < float(config.min_edge_to_trade):
            skip_reason_counts["net_edge_below_threshold"] += 1
            daily_entry["skips"]["net_edge_below_threshold"] += 1
            continue

        if config.flat_trade_notional is not None:
            cash_required = min(float(balance), max(0.0, float(config.flat_trade_notional)))
        else:
            max_exposure = cap_exposure_by_balance(
                max_exposure_usd=float(config.max_exposure_per_window_usd),
                balance_usd=float(balance),
                max_balance_fraction_per_trade=float(config.max_balance_fraction_per_trade),
            )
            cash_required = size_from_edge(
                max_exposure_usd=float(max_exposure),
                edge=float(edge_per_share),
                min_edge=float(config.min_edge_to_trade),
            )
        if cash_required <= 0.0:
            skip_reason_counts["edge_below_threshold"] += 1
            daily_entry["skips"]["edge_below_threshold"] += 1
            continue
        if cash_required < float(config.min_trade_notional):
            skip_reason_counts["trade_notional_too_small"] += 1
            daily_entry["skips"]["trade_notional_too_small"] += 1
            continue

        pnl = float(cash_required * ((1.0 - cost_per_share) / cost_per_share)) if predicted_side == actual_side else -float(cash_required)
        balance_before = float(balance)
        balance = float(balance + pnl)
        peak_balance = max(float(peak_balance), float(balance))
        max_drawdown_pct = max(
            float(max_drawdown_pct),
            max(0.0, (float(peak_balance) - float(balance)) / float(peak_balance)) if peak_balance > 0 else 0.0,
        )

        trades_taken += 1
        trade_wins += int(predicted_side == actual_side)
        stage_counts[str(stage_delay) if stage_delay is not None else "none"] += 1
        trade_entry_mode_counts[entry_mode or "none"] += 1
        daily_entry["trades"] += 1
        daily_entry["wins"] += int(predicted_side == actual_side)
        daily_entry["losses"] += int(predicted_side != actual_side)
        daily_entry["pnl"] += float(pnl)
        daily_entry["end_balance"] = float(balance)
        daily_entry["best_trade_pnl"] = max(float(daily_entry["best_trade_pnl"]), float(pnl))
        daily_entry["worst_trade_pnl"] = min(float(daily_entry["worst_trade_pnl"]), float(pnl))

        trade_logs.append(
            {
                "trade_index": idx,
                "market_id": row.get("market_id"),
                "created_at": row.get("created_at"),
                "date": day,
                "predicted_side": predicted_side,
                "actual_side": actual_side,
                "signal_correct": bool(predicted_side == actual_side),
                "p_up": float(p_up),
                "p_side": float(p_side),
                "price": float(price),
                "cost_per_share": float(cost_per_share),
                "expected_edge": float(edge_per_share),
                "cash_required": float(cash_required),
                "balance_before": float(balance_before),
                "balance_after": float(balance),
                "net_pnl": float(pnl),
                "timing_policy_stage_delay": stage_delay,
                "timing_policy_entry_mode": entry_mode or None,
            }
        )

    ordered_daily = [{"date": day, **daily[day], "skips": dict(daily[day]["skips"])} for day in sorted(daily)]
    positive_days = int(sum(1 for entry in daily.values() if float(entry.get("pnl", 0.0)) > 0.0))
    negative_days = int(sum(1 for entry in daily.values() if float(entry.get("pnl", 0.0)) < 0.0))
    best_day = max(ordered_daily, key=lambda item: item["pnl"]) if ordered_daily else None
    worst_day = min(ordered_daily, key=lambda item: item["pnl"]) if ordered_daily else None

    cash_requireds = [float(item["cash_required"]) for item in trade_logs]
    costs = [float(item["cost_per_share"]) for item in trade_logs]
    expected_edges = [float(item["expected_edge"]) for item in trade_logs]

    return {
        "candidate_key": str(candidate_key),
        "initial_balance": float(config.initial_balance),
        "final_balance": float(balance),
        "return_multiple": float(balance / config.initial_balance) if config.initial_balance > 0 else 0.0,
        "signal_rows": int(signal_total),
        "signal_accuracy": float(signal_correct / signal_total) if signal_total else 0.0,
        "trades_taken": int(trades_taken),
        "trade_win_rate": float(trade_wins / trades_taken) if trades_taken else 0.0,
        "max_drawdown_pct": float(max_drawdown_pct),
        "peak_balance": float(peak_balance),
        "positive_days": positive_days,
        "negative_days": negative_days,
        "best_day": best_day,
        "worst_day": worst_day,
        "daily": ordered_daily,
        "skip_reason_counts": dict(skip_reason_counts),
        "signal_stage_counts": dict(signal_stage_counts),
        "trade_stage_counts": dict(stage_counts),
        "signal_entry_mode_counts": dict(signal_entry_mode_counts),
        "trade_entry_mode_counts": dict(trade_entry_mode_counts),
        "cash_required_stats": _summary_stats(cash_requireds),
        "cost_per_share_stats": _summary_stats(costs),
        "expected_edge_stats": _summary_stats(expected_edges),
        "trade_log_preview": trade_logs[:20],
        "last_trade": trade_logs[-1] if trade_logs else None,
        "trade_log": trade_logs if bool(config.include_trade_log) else None,
    }
