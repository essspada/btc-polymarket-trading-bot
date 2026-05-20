from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from src.strategy.reliability_memory import score_row_reliability


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _is_trade(row: Mapping[str, Any]) -> bool:
    return str(row.get("decision_action") or "").strip().lower() == "trade"


def _is_actionable_match(match: Mapping[str, Any], *, max_shrunk_roi: float, min_entry_filled: int) -> bool:
    return (
        str(match.get("risk_label")) == "toxic"
        and int(match.get("filled", 0)) >= int(min_entry_filled)
        and float(match.get("shrunk_roi", 0.0)) <= float(max_shrunk_roi)
    )


def evaluate_reliability_overlay(
    rows: Iterable[Mapping[str, Any]],
    memory: Mapping[str, Any],
    *,
    max_shrunk_roi: float = -0.05,
    min_entry_filled: int = 3,
    allowed_experts: Iterable[str] | None = None,
    blocked_experts: Iterable[str] | None = None,
    include_rows: bool = False,
) -> dict[str, Any]:
    rows_list = list(rows)
    allowed = {str(value) for value in allowed_experts} if allowed_experts is not None else None
    blocked = {str(value) for value in blocked_experts} if blocked_experts is not None else set()
    original_pnl = 0.0
    overlay_pnl = 0.0
    proposed = 0
    filled = 0
    skipped_proposed = 0
    skipped_filled = 0
    skipped_wins = 0
    skipped_losses = 0
    skipped_pnl = 0.0
    saved_losses = 0.0
    missed_profit = 0.0
    skip_reasons: Counter[str] = Counter()
    row_outputs: list[dict[str, Any]] = []

    for row in rows_list:
        trade = _is_trade(row)
        row_filled = bool(row.get("trade_filled"))
        pnl = _to_float(row.get("trade_net_pnl"), 0.0) if row_filled else 0.0
        original_pnl += pnl
        proposed += int(trade)
        filled += int(row_filled)

        score = score_row_reliability(row, memory)
        actionable_matches = [
            match
            for match in score.get("veto_matches", [])
            if _is_actionable_match(match, max_shrunk_roi=max_shrunk_roi, min_entry_filled=min_entry_filled)
            and (allowed is None or str(match.get("expert")) in allowed)
            and str(match.get("expert")) not in blocked
        ]
        skip = bool(trade and actionable_matches)
        if skip:
            skipped_proposed += 1
            skipped_filled += int(row_filled)
            if row_filled:
                skipped_pnl += pnl
                skipped_wins += int(pnl > 0.0)
                skipped_losses += int(pnl < 0.0)
                if pnl < 0.0:
                    saved_losses += -pnl
                elif pnl > 0.0:
                    missed_profit += pnl
            for match in actionable_matches:
                reason = f"{match.get('expert')}|{match.get('matched_scope')}|roi={float(match.get('shrunk_roi', 0.0)):.3f}"
                skip_reasons[reason] += 1
        else:
            overlay_pnl += pnl

        if include_rows:
            row_outputs.append(
                {
                    "market_slug": row.get("market_slug"),
                    "created_at": row.get("created_at"),
                    "skip": skip,
                    "trade_filled": row_filled,
                    "trade_net_pnl": pnl,
                    "score": score,
                }
            )

    delta = overlay_pnl - original_pnl
    return {
        "rows": len(rows_list),
        "proposed": int(proposed),
        "filled": int(filled),
        "original_pnl": float(original_pnl),
        "overlay_pnl": float(overlay_pnl),
        "delta_pnl": float(delta),
        "skipped_proposed": int(skipped_proposed),
        "skipped_filled": int(skipped_filled),
        "skipped_wins": int(skipped_wins),
        "skipped_losses": int(skipped_losses),
        "skipped_pnl": float(skipped_pnl),
        "saved_losses": float(saved_losses),
        "missed_profit": float(missed_profit),
        "skip_reasons": dict(skip_reasons),
        "rows_detail": row_outputs if include_rows else [],
    }
