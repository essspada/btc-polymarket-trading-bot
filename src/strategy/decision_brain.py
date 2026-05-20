from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExpertSignal:
    name: str
    value: float | None = None
    confidence: float = 0.0
    uncertainty: float = 1.0
    veto: bool = False
    veto_reason: str | None = None
    regime_tags: dict[str, str | float | bool] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except Exception:
        return default


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _predicted_side(row: dict[str, Any]) -> str | None:
    side = str(row.get("predicted_side") or "").strip().lower()
    if side in {"up", "down"}:
        return side
    p_up = _to_float(row.get("p_up"))
    if p_up is None:
        return None
    return "up" if float(p_up) >= 0.5 else "down"


def _selected_entry_price(row: dict[str, Any]) -> float | None:
    side = _predicted_side(row)
    order_type = str(row.get("decision_order_type") or row.get("trade_order_type") or "").strip().upper()
    if side == "up":
        return _to_float(row.get("up_best_ask" if order_type == "TAKER" else "up_best_bid"))
    if side == "down":
        return _to_float(row.get("down_best_ask" if order_type == "TAKER" else "down_best_bid"))
    return None


def _candidate_agreement(row: dict[str, Any]) -> dict[str, Any]:
    target = _predicted_side(row)
    candidates = row.get("candidate_models")
    if target not in {"up", "down"} or not isinstance(candidates, dict):
        return {"agreement_ratio": None, "total": 0, "agree": 0, "disagree": 0}
    total = 0
    agree = 0
    details: dict[str, str] = {}
    for name, payload in candidates.items():
        if not isinstance(payload, dict):
            continue
        p_up = _to_float(payload.get("p_up"))
        if p_up is None:
            continue
        side = "up" if float(p_up) >= 0.5 else "down"
        details[str(name)] = side
        total += 1
        agree += int(side == target)
    ratio = float(agree / total) if total else None
    return {"agreement_ratio": ratio, "total": total, "agree": agree, "disagree": total - agree, "details": details}


def build_shadow_expert_signals(row: dict[str, Any]) -> list[ExpertSignal]:
    signals: list[ExpertSignal] = []

    book_ok = row.get("book_quality_ok")
    book_score = _to_float(row.get("book_quality_score"), 0.0) or 0.0
    signals.append(
        ExpertSignal(
            name="book_quality_expert",
            value=float(book_score),
            confidence=_clip01(float(book_score)),
            uncertainty=1.0 - _clip01(float(book_score)),
            veto=book_ok is False,
            veto_reason=None if book_ok is not False else str(row.get("book_quality_reason") or "book_quality_failed"),
            evidence={"book_quality_ok": book_ok, "book_quality_reason": row.get("book_quality_reason")},
        )
    )

    edge = _to_float(row.get("decision_best_edge"))
    edge_conf = _clip01(abs(float(edge or 0.0)) / 0.15)
    signals.append(
        ExpertSignal(
            name="edge_quality_expert",
            value=edge,
            confidence=edge_conf,
            uncertainty=1.0 - edge_conf,
            veto=edge is None or float(edge) <= 0.0,
            veto_reason="missing_or_nonpositive_edge" if edge is None or float(edge) <= 0.0 else None,
            evidence={"decision_action": row.get("decision_action"), "decision_reason": row.get("decision_reason")},
        )
    )

    side = _predicted_side(row)
    price = _selected_entry_price(row)
    extreme_price = price is not None and (float(price) < 0.08 or float(price) > 0.92)
    signals.append(
        ExpertSignal(
            name="entry_price_risk_expert",
            value=price,
            confidence=_clip01(abs(float(price or 0.5) - 0.5) * 2.0),
            uncertainty=1.0 - _clip01(abs(float(price or 0.5) - 0.5) * 2.0),
            veto=bool(extreme_price),
            veto_reason="extreme_entry_price_shadow" if extreme_price else None,
            evidence={"order_type": row.get("decision_order_type") or row.get("trade_order_type")},
        )
    )

    agreement = _candidate_agreement(row)
    ratio = agreement.get("agreement_ratio")
    ratio_f = _to_float(ratio)
    signals.append(
        ExpertSignal(
            name="candidate_agreement_expert",
            value=ratio_f,
            confidence=float(ratio_f) if ratio_f is not None else 0.0,
            uncertainty=1.0 - float(ratio_f) if ratio_f is not None else 1.0,
            veto=ratio_f is not None and ratio_f < 0.50,
            veto_reason="candidate_disagreement_shadow" if ratio_f is not None and ratio_f < 0.50 else None,
            evidence=agreement,
        )
    )

    seconds_to_expiry = _to_float(row.get("seconds_to_expiry"))
    late = seconds_to_expiry is not None and float(seconds_to_expiry) < 45.0
    signals.append(
        ExpertSignal(
            name="timing_expert",
            value=seconds_to_expiry,
            confidence=1.0 if seconds_to_expiry is not None else 0.0,
            uncertainty=0.0 if seconds_to_expiry is not None else 1.0,
            veto=bool(late),
            veto_reason="late_window_shadow" if late else None,
            regime_tags={"timing_reason": str(row.get("timing_policy_reason") or "missing")},
            evidence={"seconds_from_window_start": row.get("seconds_from_window_start")},
        )
    )

    spot_return_from_open = _to_float(row.get("spot_return_bps_from_open"))
    recent_ret = _to_float(row.get("spot_recent_return_1m_bps"))
    vol = _to_float(row.get("spot_recent_vol_5m_bps"))
    chasing_window_move = False
    if side == "up" and spot_return_from_open is not None:
        chasing_window_move = float(spot_return_from_open) >= 1.0
    elif side == "down" and spot_return_from_open is not None:
        chasing_window_move = float(spot_return_from_open) <= -1.0
    late_chase = (
        bool(chasing_window_move)
        and seconds_to_expiry is not None
        and float(seconds_to_expiry) <= 75.0
        and price is not None
        and float(price) >= 0.55
    )
    signals.append(
        ExpertSignal(
            name="late_chase_reversal_expert",
            value=spot_return_from_open,
            confidence=_clip01(abs(float(spot_return_from_open or 0.0)) / 10.0),
            uncertainty=1.0 - _clip01(abs(float(spot_return_from_open or 0.0)) / 10.0),
            veto=bool(late_chase),
            veto_reason="late_chase_reversal_shadow" if late_chase else None,
            regime_tags={
                "selected_side": side or "missing",
                "entry_price": float(price) if price is not None else -1.0,
                "seconds_to_expiry": float(seconds_to_expiry) if seconds_to_expiry is not None else -1.0,
            },
            evidence={
                "spot_return_bps_from_open": spot_return_from_open,
                "spot_recent_return_1m_bps": row.get("spot_recent_return_1m_bps"),
            },
        )
    )

    quiet_consensus_chase = (
        ratio_f is not None
        and float(ratio_f) >= 0.80
        and bool(chasing_window_move)
        and spot_return_from_open is not None
        and abs(float(spot_return_from_open)) >= 1.0
        and recent_ret is not None
        and abs(float(recent_ret)) <= 0.5
        and vol is not None
        and float(vol) <= 2.0
    )
    signals.append(
        ExpertSignal(
            name="quiet_consensus_chase_expert",
            value=spot_return_from_open,
            confidence=_clip01(abs(float(spot_return_from_open or 0.0)) / 10.0),
            uncertainty=1.0 - _clip01(abs(float(spot_return_from_open or 0.0)) / 10.0),
            veto=bool(quiet_consensus_chase),
            veto_reason="quiet_consensus_chase_shadow" if quiet_consensus_chase else None,
            regime_tags={
                "selected_side": side or "missing",
                "low_volatility": bool(vol is not None and float(vol) <= 2.0),
                "recent_flat": bool(recent_ret is not None and abs(float(recent_ret)) <= 0.5),
            },
            evidence={
                "agreement_ratio": ratio_f,
                "spot_return_bps_from_open": spot_return_from_open,
                "spot_recent_return_1m_bps": recent_ret,
                "spot_recent_vol_5m_bps": vol,
            },
        )
    )

    high_vol = vol is not None and float(vol) >= 8.0
    signals.append(
        ExpertSignal(
            name="spot_regime_expert",
            value=vol,
            confidence=_clip01(float(vol or 0.0) / 12.0),
            uncertainty=1.0 - _clip01(float(vol or 0.0) / 12.0),
            veto=False,
            regime_tags={"high_volatility": bool(high_vol), "recent_return_1m_bps": float(recent_ret or 0.0)},
            evidence={"spot_price_now": row.get("spot_price_now"), "spot_window_open_price": row.get("spot_window_open_price")},
        )
    )

    return signals


def summarize_shadow_vetoes(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for row in rows:
        is_filled = bool(row.get("trade_filled"))
        pnl = float(_to_float(row.get("trade_net_pnl"), 0.0) or 0.0)
        for signal in build_shadow_expert_signals(row):
            item = summary.setdefault(
                signal.name,
                {"rows": 0, "veto_rows": 0, "filled_veto_rows": 0, "pnl_on_veto_rows": 0.0, "veto_reasons": {}},
            )
            item["rows"] += 1
            if signal.veto:
                item["veto_rows"] += 1
                item["filled_veto_rows"] += int(is_filled)
                item["pnl_on_veto_rows"] += pnl if is_filled else 0.0
                reason = str(signal.veto_reason or "veto")
                item["veto_reasons"][reason] = int(item["veto_reasons"].get(reason, 0)) + 1
    return summary
