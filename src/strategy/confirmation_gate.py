from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from src.polymarket.orderbook import MarketBooks
from src.strategy.signals import SignalDecision


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def _candidate_p_up(candidate_models: Mapping[str, Any] | None, candidate_key: str) -> float | None:
    item = (candidate_models or {}).get(str(candidate_key)) if isinstance(candidate_models, Mapping) else None
    if not isinstance(item, Mapping):
        return None
    for key in ("p_up", "p_up_calibrated", "p_up_pre_calibration", "p_up_raw"):
        value = _to_float(item.get(key))
        if value is not None:
            return max(1e-6, min(1.0 - 1e-6, float(value)))
    return None


def _side_for_token(books: MarketBooks, token_id: str | None) -> str | None:
    token = str(token_id or "").strip()
    if token and token == books.up.token_id:
        return "up"
    if token and token == books.down.token_id:
        return "down"
    return None


def _spread_for_side(books: MarketBooks, side: str | None) -> float | None:
    if side == "up":
        return float(books.up.spread)
    if side == "down":
        return float(books.down.spread)
    return None


def _top3_size_for_decision(books: MarketBooks, side: str | None, order_type: str | None) -> float | None:
    order = str(order_type or "").strip().upper()
    if side == "up":
        return float(books.up.top3_bid_size if order == "MAKER" else books.up.top3_ask_size)
    if side == "down":
        return float(books.down.top3_bid_size if order == "MAKER" else books.down.top3_ask_size)
    return None


def _candidate_side(p_up: float) -> str:
    return "up" if float(p_up) >= 0.5 else "down"


def _p_for_side(p_up: float, side: str) -> float:
    return float(p_up if side == "up" else 1.0 - p_up)


def apply_confirmation_gate(
    *,
    decision: SignalDecision,
    books: MarketBooks,
    candidate_models: Mapping[str, Any] | None,
    gate_cfg: Mapping[str, Any] | None,
) -> tuple[SignalDecision, dict[str, Any]]:
    cfg = gate_cfg if isinstance(gate_cfg, Mapping) else {}
    enabled = bool(cfg.get("enabled", False))
    candidate_key = str(cfg.get("candidate_key", "proxy_logistic_market_blend") or "proxy_logistic_market_blend")
    payload: dict[str, Any] = {
        "confirmation_gate_enabled": enabled,
        "confirmation_gate_candidate_key": candidate_key,
        "confirmation_gate_passed": None,
        "confirmation_gate_reason": "disabled" if not enabled else None,
        "confirmation_gate_side": None,
        "confirmation_gate_candidate_side": None,
        "confirmation_gate_price": None,
        "confirmation_gate_spread": None,
        "confirmation_gate_top3_size": None,
        "confirmation_gate_edge": None,
        "confirmation_gate_min_edge": _to_float(cfg.get("min_edge"), 0.0),
        "confirmation_gate_min_top3_ask_size": _to_float(cfg.get("min_top3_ask_size")),
    }
    if not enabled:
        return decision, payload
    if decision.intent is None:
        payload["confirmation_gate_passed"] = True
        payload["confirmation_gate_reason"] = "no_intent"
        return decision, payload

    side = _side_for_token(books, decision.intent.token_id)
    price = float(decision.intent.price)
    spread = _spread_for_side(books, side)
    top3_size = _top3_size_for_decision(books, side, decision.intent.order_type)
    payload.update(
        {
            "confirmation_gate_side": side,
            "confirmation_gate_price": price,
            "confirmation_gate_spread": spread,
            "confirmation_gate_top3_size": top3_size,
        }
    )
    if side not in {"up", "down"}:
        payload["confirmation_gate_passed"] = False
        payload["confirmation_gate_reason"] = "unknown_decision_side"
        return replace(decision, action="no_trade", reason="confirmation_gate_unknown_decision_side", intent=None), payload

    min_price = _to_float(cfg.get("min_price"), 0.0)
    max_price = _to_float(cfg.get("max_price"), 1.0)
    if min_price is not None and price < float(min_price):
        payload["confirmation_gate_passed"] = False
        payload["confirmation_gate_reason"] = "price_below_min"
        return replace(decision, action="no_trade", reason="confirmation_gate_price_below_min", intent=None), payload
    if max_price is not None and price > float(max_price):
        payload["confirmation_gate_passed"] = False
        payload["confirmation_gate_reason"] = "price_above_max"
        return replace(decision, action="no_trade", reason="confirmation_gate_price_above_max", intent=None), payload

    max_spread = _to_float(cfg.get("max_spread"), None)
    if max_spread is not None and spread is not None and float(spread) > float(max_spread):
        payload["confirmation_gate_passed"] = False
        payload["confirmation_gate_reason"] = "spread_above_max"
        return replace(decision, action="no_trade", reason="confirmation_gate_spread_above_max", intent=None), payload

    min_top3_ask_size = _to_float(cfg.get("min_top3_ask_size"), None)
    if min_top3_ask_size is not None and (top3_size is None or float(top3_size) < float(min_top3_ask_size)):
        payload["confirmation_gate_passed"] = False
        payload["confirmation_gate_reason"] = "top3_ask_size_below_min"
        return replace(decision, action="no_trade", reason="confirmation_gate_top3_ask_size_below_min", intent=None), payload

    confirm_p_up = _candidate_p_up(candidate_models, candidate_key)
    if confirm_p_up is None:
        payload["confirmation_gate_passed"] = False
        payload["confirmation_gate_reason"] = "missing_candidate_probability"
        return replace(decision, action="no_trade", reason="confirmation_gate_missing_candidate", intent=None), payload

    confirm_side = _candidate_side(confirm_p_up)
    payload["confirmation_gate_candidate_side"] = confirm_side
    if bool(cfg.get("require_same_side", True)) and confirm_side != side:
        payload["confirmation_gate_passed"] = False
        payload["confirmation_gate_reason"] = "side_disagreement"
        return replace(decision, action="no_trade", reason="confirmation_gate_side_disagreement", intent=None), payload

    confirm_edge = float(_p_for_side(confirm_p_up, side) - price)
    min_edge = float(_to_float(cfg.get("min_edge"), 0.0) or 0.0)
    payload["confirmation_gate_edge"] = confirm_edge
    if confirm_edge < min_edge:
        payload["confirmation_gate_passed"] = False
        payload["confirmation_gate_reason"] = "edge_below_min"
        return replace(decision, action="no_trade", reason="confirmation_gate_edge_below_min", intent=None), payload

    payload["confirmation_gate_passed"] = True
    payload["confirmation_gate_reason"] = "ok"
    return decision, payload
