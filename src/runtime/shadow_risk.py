"""Adaptive-risk shadow payload construction.

Runs the adaptive-risk decision against the current account / window state and
returns a structured payload that the monitor and paper modes log alongside
each trade decision. The shadow never modifies the live order.
"""
from __future__ import annotations

from typing import Any

from src.polymarket.orderbook import MarketBooks
from src.runtime.x3 import selected_spread_for_side
from src.strategy.adaptive_risk import resolve_adaptive_risk_shadow


def build_adaptive_risk_shadow_payload(
    *,
    risk_cfg: dict[str, Any],
    decision: Any,
    candidate_cash_required: float,
    current_equity_usd: float,
    available_cash_usd: float,
    reserved_cash_usd: float,
    peak_equity_usd: float,
    daily_pnl: float,
    loss_streak: int,
    consecutive_wins: int,
    cooldown_left: int,
    recent_accuracy: float,
    open_positions: int,
    max_open_positions: int,
    risk_budget_cash_usd: float,
    confidence: float | None,
    spread_norm: float | None,
    spot_recent_vol_5m_bps: float | None,
    seconds_to_expiry: float | None,
) -> dict[str, Any]:
    cfg = risk_cfg.get("adaptive_risk", {})
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return {}
    decision_payload = resolve_adaptive_risk_shadow(
        cfg=cfg,
        candidate_cash_usd=float(candidate_cash_required),
        available_cash_usd=float(available_cash_usd),
        reserved_cash_usd=float(reserved_cash_usd),
        current_equity_usd=float(current_equity_usd),
        peak_equity_usd=float(peak_equity_usd),
        daily_pnl=float(daily_pnl),
        loss_streak=int(loss_streak),
        consecutive_wins=int(consecutive_wins),
        cooldown_left=int(cooldown_left),
        recent_accuracy=float(recent_accuracy),
        open_positions=int(open_positions),
        max_open_positions=int(max_open_positions),
        risk_budget_cash_usd=float(risk_budget_cash_usd),
        expected_roi_cash=getattr(decision, "expected_roi_cash", None),
        breakeven_margin=getattr(decision, "breakeven_margin", None),
        confidence=confidence,
        spread_norm=spread_norm,
        spot_recent_vol_5m_bps=spot_recent_vol_5m_bps,
        seconds_to_expiry=seconds_to_expiry,
    )
    return decision_payload.to_payload()


def build_monitor_adaptive_risk_shadow_payload(
    *,
    risk_cfg: dict[str, Any],
    paper_cfg: dict[str, Any],
    decision: Any,
    books: MarketBooks,
    candidate_side: str | None,
    confidence: float | None,
    spot_recent_vol_5m_bps: float | None,
    seconds_to_expiry: float | None,
) -> dict[str, Any]:
    cfg = risk_cfg.get("adaptive_risk", {})
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return {}
    try:
        equity = float(cfg.get("monitor_shadow_equity_usd") or paper_cfg.get("initial_balance_usd", 100.0))
    except (TypeError, ValueError):
        equity = 100.0
    equity = max(0.0, equity)
    try:
        max_exposure = float(risk_cfg.get("max_exposure_per_window_usd", 0.0))
    except (TypeError, ValueError):
        max_exposure = 0.0
    try:
        max_fraction = float(risk_cfg.get("max_balance_fraction_per_trade", 1.0))
    except (TypeError, ValueError):
        max_fraction = 1.0
    risk_budget = min(max_exposure, equity * max(0.0, max_fraction)) if equity > 0.0 else max(0.0, max_exposure)
    payload = build_adaptive_risk_shadow_payload(
        risk_cfg=risk_cfg,
        decision=decision,
        candidate_cash_required=float(getattr(decision, "cash_required", 0.0) or 0.0),
        current_equity_usd=float(equity),
        available_cash_usd=float(equity),
        reserved_cash_usd=0.0,
        peak_equity_usd=float(equity),
        daily_pnl=0.0,
        loss_streak=0,
        consecutive_wins=0,
        cooldown_left=0,
        recent_accuracy=0.0,
        open_positions=0,
        max_open_positions=int(risk_cfg.get("max_open_positions", 0) or 0),
        risk_budget_cash_usd=float(risk_budget),
        confidence=confidence,
        spread_norm=selected_spread_for_side(books, candidate_side),
        spot_recent_vol_5m_bps=spot_recent_vol_5m_bps,
        seconds_to_expiry=seconds_to_expiry,
    )
    if payload:
        payload["adaptive_risk_monitor_only"] = True
        payload["adaptive_risk_state_source"] = "monitor_only_synthetic_bankroll"
    return payload
