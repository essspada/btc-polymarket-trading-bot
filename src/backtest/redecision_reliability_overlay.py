from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def synthetic_row_from_redecision(row: Mapping[str, Any], decision: Mapping[str, Any], *, fill_mode: str) -> dict[str, Any]:
    """Convert a redecision output into a row usable by reliability-memory.

    Reliability memory expects the same field shape as `paper_outcomes_5m.jsonl`.
    Redecision policies may use a different probability source than the original
    runtime row, so this function rewrites `p_up`, `predicted_side`, decision
    economics, and hypothetical fill/PnL while preserving orderbook/regime fields.
    """

    out = dict(row)
    p_up = _to_float(decision.get("p_up"), _to_float(row.get("p_up"), 0.5))
    selected_side = str(decision.get("selected_side") or ("up" if p_up >= 0.5 else "down")).strip().lower()
    if selected_side not in {"up", "down"}:
        selected_side = "up" if p_up >= 0.5 else "down"

    action = str(decision.get("action") or "no_trade").strip().lower()
    if action not in {"trade", "no_trade"}:
        action = "no_trade"
    fill_mode_value = str(fill_mode or "assume_filled")
    if fill_mode_value == "paper_sim":
        filled = bool(decision.get("paper_sim_filled")) if action == "trade" else False
        pnl = _to_float(decision.get("paper_sim_pnl"), 0.0) if filled else 0.0
    else:
        filled = bool(action == "trade" and decision.get("pnl_if_filled") is not None)
        pnl = _to_float(decision.get("pnl_if_filled"), 0.0) if filled else 0.0

    out.update(
        {
            "p_up": float(p_up),
            "p_down": float(1.0 - p_up),
            "confidence": float(max(p_up, 1.0 - p_up)),
            "predicted_side": selected_side,
            "decision_action": action,
            "decision_reason": decision.get("reason"),
            "decision_best_edge": decision.get("best_edge"),
            "decision_expected_edge": decision.get("expected_edge") or decision.get("best_edge"),
            "decision_order_type": str(decision.get("order_type") or "").upper() if action == "trade" else None,
            "decision_token_id": decision.get("token_id"),
            "decision_cash_required": decision.get("cash_required_if_filled") or decision.get("decision_cash_required"),
            "trade_filled": bool(filled),
            "trade_net_pnl": float(pnl),
            "trade_order_type": str(decision.get("order_type") or "").upper() if action == "trade" else None,
            "trade_fill_size": decision.get("size"),
            "source_policy": decision.get("policy"),
            "source_candidate_key": decision.get("candidate_key"),
        }
    )
    return out


def synthetic_rows_from_redecision(
    ordered_rows: Iterable[Mapping[str, Any]],
    decisions: Iterable[Mapping[str, Any]],
    *,
    fill_mode: str,
) -> list[dict[str, Any]]:
    return [
        synthetic_row_from_redecision(row, decision, fill_mode=fill_mode)
        for row, decision in zip(list(ordered_rows), list(decisions), strict=True)
    ]
