"""Misc runtime helpers shared across sim/monitor/paper/backtest entrypoints."""
from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.execution.paper_execution import estimate_maker_fill_probability
from src.polymarket.clients.clob_client import ClobClient
from src.polymarket.clients.gamma_client import GammaClient
from src.polymarket.fees import FeeModelConfig, compute_trade_fee
from src.polymarket.orderbook import MarketBooks
from src.runtime.state import PendingPrediction
from src.strategy.timing_policy import DeferredTimingDecision


def to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def to_optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def parse_iso_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def resolve_output_path(outputs_dir: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return outputs_dir / p


def gamma_error_payload(gamma: GammaClient) -> dict[str, Any]:
    return {
        "error": gamma.last_error,
        "error_type": gamma.last_error_type,
        "url": gamma.last_error_url,
    }


def build_monitor_summary(
    now: datetime,
    started_at: datetime,
    pending: dict[str, PendingPrediction],
    deferred_timing: dict[str, DeferredTimingDecision],
    resolved_ids: set[str],
    total: int,
    correct: int,
) -> dict[str, Any]:
    accuracy = (correct / total) if total > 0 else 0.0
    return {
        "ts_utc": now.isoformat(),
        "mode": "MONITOR_5M",
        "runtime_minutes": round((now - started_at).total_seconds() / 60.0, 3),
        "predictions_total": total,
        "predictions_correct": correct,
        "accuracy": accuracy,
        "pending_count": len(pending),
        "deferred_timing_count": len(deferred_timing),
        "resolved_count": len(resolved_ids),
        "pending": [asdict(x) for x in pending.values()],
        "deferred_timing": [asdict(x) for x in deferred_timing.values()],
    }


def compute_trade_pnl_per_share(
    predicted_side: str,
    actual_side: str,
    p_up: float,
    up_ask: float,
    down_ask: float,
    fee_cfg: FeeModelConfig,
    taker_slippage_bps: float,
    taker_fee_bps_override: float | None = None,
) -> dict[str, float]:
    side = str(predicted_side).strip().lower()
    if side not in {"up", "down"}:
        return {"edge": 0.0, "net_pnl": 0.0, "entry_price": 0.0, "fee": 0.0, "slippage": 0.0}

    p_down = 1.0 - p_up
    entry_price = up_ask if side == "up" else down_ask
    p_side = p_up if side == "up" else p_down
    edge = p_side - entry_price

    cfg_for_fee = fee_cfg
    if taker_fee_bps_override is not None:
        cfg_for_fee = FeeModelConfig(
            maker_fee_bps=fee_cfg.maker_fee_bps,
            taker_fee_bps=float(taker_fee_bps_override),
            curve_rate=fee_cfg.curve_rate,
            curve_exponent=fee_cfg.curve_exponent,
            min_fee=fee_cfg.min_fee,
            maker_fee_rate=fee_cfg.maker_fee_rate,
            taker_fee_rate=float(taker_fee_bps_override) / 10_000.0,
        )
    fee = compute_trade_fee(price=entry_price, size=1.0, is_taker=True, cfg=cfg_for_fee)
    slippage = entry_price * (taker_slippage_bps / 10_000.0)
    gross = (1.0 - entry_price) if actual_side == side else (-entry_price)
    net = gross - fee - slippage
    return {
        "edge": float(edge),
        "net_pnl": float(net),
        "entry_price": float(entry_price),
        "fee": float(fee),
        "slippage": float(slippage),
    }


def fetch_taker_fee_bps_by_token(clob: ClobClient, books: MarketBooks) -> dict[str, float]:
    out: dict[str, float] = {}
    for token_id in (books.up.token_id, books.down.token_id):
        if token_id in out:
            continue
        try:
            out[token_id] = float(clob.get_fee_rate_bps(token_id))
        except Exception:
            continue
    return out


def maker_fill_probability_by_token(
    *,
    books: MarketBooks,
    seconds_to_expiry: float,
    exec_cfg: dict[str, Any],
) -> dict[str, float]:
    kwargs = {
        "maker_fill_floor": float(exec_cfg.get("maker_fill_floor", 0.05)),
        "maker_fill_cap": float(exec_cfg.get("maker_fill_cap", 0.9)),
        "maker_fill_base": float(exec_cfg.get("maker_fill_base", 0.78)),
        "maker_fill_spread_penalty": float(exec_cfg.get("maker_fill_spread_penalty", 7.0)),
        "maker_fill_late_penalty_90": float(exec_cfg.get("maker_fill_late_penalty_90", 0.15)),
        "maker_fill_late_penalty_45": float(exec_cfg.get("maker_fill_late_penalty_45", 0.10)),
    }
    return {
        books.up.token_id: estimate_maker_fill_probability(
            spread=float(books.up.spread),
            seconds_to_expiry=float(seconds_to_expiry),
            **kwargs,
        ),
        books.down.token_id: estimate_maker_fill_probability(
            spread=float(books.down.spread),
            seconds_to_expiry=float(seconds_to_expiry),
            **kwargs,
        ),
    }
