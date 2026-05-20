"""X3 risk-layer shadow and active overrides.

The X3 layer is an optional post-decision risk resolver that can shadow the
base sizing decision (audit-only) or actively cap/veto trades. It supports
multiple variants run in parallel for research diagnostics.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from src.polymarket.orderbook import MarketBooks
from src.strategy.x3_resolver import resolve_x3_candidate


def x3_shadow_enabled(risk_cfg: dict[str, Any]) -> bool:
    cfg = risk_cfg.get("x3_resolver", {})
    if not isinstance(cfg, dict):
        return False
    mode = str(cfg.get("mode") or "off").strip().lower()
    return bool(cfg.get("enabled", False)) and mode == "shadow"


def x3_active_enabled(risk_cfg: dict[str, Any]) -> bool:
    cfg = risk_cfg.get("x3_resolver", {})
    if not isinstance(cfg, dict):
        return False
    mode = str(cfg.get("mode") or "off").strip().lower()
    return bool(cfg.get("enabled", False)) and mode == "active"


def clean_outcome_side(value: Any) -> str | None:
    side = str(value or "").strip().lower()
    return side if side in {"up", "down"} else None


def outcome_side_for_intent(intent: Any, *, up_token_id: Any, down_token_id: Any) -> str | None:
    token_id = str(getattr(intent, "token_id", "") or "")
    if token_id and token_id == str(up_token_id or ""):
        return "up"
    if token_id and token_id == str(down_token_id or ""):
        return "down"
    return clean_outcome_side(getattr(intent, "side", None))


def selected_spread_for_side(books: MarketBooks, side: str | None) -> float | None:
    side_clean = clean_outcome_side(side)
    if side_clean == "up":
        return float(books.up.spread)
    if side_clean == "down":
        return float(books.down.spread)
    return None


def _x3_variant_configs(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    variants = cfg.get("variants")
    if not isinstance(variants, list) or not variants:
        return [dict(cfg)]
    inherited_keys = {
        "enabled",
        "mode",
        "min_trade_usd_after_cap",
        "respect_advisor_cap",
        "respect_advisor_veto",
        "hard_max_cash_usd",
    }
    base = {key: value for key, value in cfg.items() if key in inherited_keys}
    out: list[dict[str, Any]] = []
    for item in variants:
        if not isinstance(item, dict):
            continue
        merged = dict(base)
        merged.update(item)
        merged["enabled"] = cfg.get("enabled", True)
        merged["mode"] = cfg.get("mode", "shadow")
        out.append(merged)
    return out or [dict(cfg)]


def _x3_variant_entry(
    *,
    variant_cfg: dict[str, Any],
    decision: Any,
    candidate_cash_required: float,
    candidate_side: str | None = None,
) -> dict[str, Any]:
    intent = getattr(decision, "intent", None)
    policy_name = str(variant_cfg.get("policy_name") or variant_cfg.get("policy") or "disabled")
    mode = str(variant_cfg.get("mode") or "off").strip().lower()
    side_for_policy = clean_outcome_side(candidate_side)
    if intent is None:
        return {
            "policy_name": policy_name,
            "mode": mode,
            "action": "no_candidate",
            "candidate_cash_usd": 0.0,
            "final_cash_usd": 0.0,
            "cash_multiplier": 0.0,
            "reason_codes": ["x1_no_trade"],
            "would_reduce_cash": False,
            "would_veto": False,
            "candidate_size": None,
            "shadow_final_size": None,
            "candidate_side": side_for_policy,
            "advisor_effective_action": None,
            "advisor_confidence": None,
            "hard_guard_applied": False,
        }

    side_for_policy = side_for_policy or clean_outcome_side(getattr(intent, "side", None))
    x3_decision = resolve_x3_candidate(
        cfg=variant_cfg,
        candidate_cash_usd=float(candidate_cash_required),
        side=side_for_policy,
        entry_price=getattr(intent, "price", None),
        p_up=getattr(decision, "p_up", None),
        p_down=getattr(decision, "p_down", None),
        expected_roi_cash=getattr(decision, "expected_roi_cash", None),
        breakeven_margin=getattr(decision, "breakeven_margin", None),
        advisor=None,
    )
    multiplier = x3_decision.cash_multiplier
    candidate_size = float(getattr(intent, "size", 0.0) or 0.0)
    shadow_final_size = candidate_size * float(multiplier) if multiplier is not None else None
    return {
        "policy_name": policy_name,
        "mode": mode,
        "action": x3_decision.action,
        "candidate_cash_usd": x3_decision.candidate_cash_usd,
        "final_cash_usd": x3_decision.final_cash_usd,
        "cash_multiplier": x3_decision.cash_multiplier,
        "reason_codes": x3_decision.reason_codes,
        "would_reduce_cash": bool(
            x3_decision.final_cash_usd is not None
            and x3_decision.candidate_cash_usd is not None
            and x3_decision.final_cash_usd < x3_decision.candidate_cash_usd - 0.01
        ),
        "would_veto": bool(x3_decision.vetoed),
        "candidate_size": candidate_size,
        "shadow_final_size": shadow_final_size,
        "candidate_side": side_for_policy,
        "advisor_effective_action": x3_decision.advisor_effective_action,
        "advisor_confidence": x3_decision.advisor_confidence,
        "hard_guard_applied": x3_decision.hard_guard_applied,
    }


def build_x3_shadow_payload(
    *,
    risk_cfg: dict[str, Any],
    decision: Any,
    candidate_cash_required: float,
    raw_cash_required: float,
    sizing_cap_payload: dict[str, Any],
    candidate_side: str | None = None,
) -> dict[str, Any]:
    cfg = risk_cfg.get("x3_resolver", {})
    if not isinstance(cfg, dict) or not x3_shadow_enabled(risk_cfg):
        return {}

    intent = getattr(decision, "intent", None)
    variant_entries = [
        _x3_variant_entry(
            variant_cfg=variant_cfg,
            decision=decision,
            candidate_cash_required=float(candidate_cash_required),
            candidate_side=candidate_side,
        )
        for variant_cfg in _x3_variant_configs(cfg)
    ]
    primary = variant_entries[0] if variant_entries else {}
    mode = str(primary.get("mode") or cfg.get("mode") or "off").strip().lower()
    policy_name = str(primary.get("policy_name") or cfg.get("policy_name") or cfg.get("policy") or "disabled")
    base: dict[str, Any] = {
        "x3_shadow_only": True,
        "x3_runtime_applied": False,
        "x3_runtime_note": "paper_runtime_logs_shadow_only",
        "x3_x1_action": getattr(decision, "action", None),
        "x3_x1_reason": getattr(decision, "reason", None),
        "x3_raw_cash_required": float(max(0.0, raw_cash_required)),
        "x3_post_sizing_candidate_cash_usd": float(max(0.0, candidate_cash_required)),
        "x3_sizing_cap_applied_before_x3": bool(sizing_cap_payload.get("sizing_cap_applied", False)),
        "x3_variant_count": len(variant_entries),
        "x3_variants": variant_entries,
    }

    if intent is None:
        base.update(
            {
                "x3_enabled": True,
                "x3_mode": mode,
                "x3_policy_name": policy_name,
                "x3_action": primary.get("action", "no_candidate"),
                "x3_candidate_cash_usd": primary.get("candidate_cash_usd", 0.0),
                "x3_final_cash_usd": primary.get("final_cash_usd", 0.0),
                "x3_cash_multiplier": primary.get("cash_multiplier", 0.0),
                "x3_reason_codes": primary.get("reason_codes", ["x1_no_trade"]),
                "x3_advisor_effective_action": primary.get("advisor_effective_action"),
                "x3_advisor_confidence": primary.get("advisor_confidence"),
                "x3_hard_guard_applied": bool(primary.get("hard_guard_applied", False)),
                "x3_would_reduce_cash": bool(primary.get("would_reduce_cash", False)),
                "x3_would_veto": bool(primary.get("would_veto", False)),
                "x3_candidate_size": primary.get("candidate_size"),
                "x3_shadow_final_size": primary.get("shadow_final_size"),
                "x3_candidate_side": primary.get("candidate_side"),
            }
        )
        return base

    base.update(
        {
            "x3_enabled": True,
            "x3_mode": mode,
            "x3_policy_name": policy_name,
            "x3_action": primary.get("action"),
            "x3_candidate_cash_usd": primary.get("candidate_cash_usd"),
            "x3_final_cash_usd": primary.get("final_cash_usd"),
            "x3_cash_multiplier": primary.get("cash_multiplier"),
            "x3_reason_codes": primary.get("reason_codes"),
            "x3_advisor_effective_action": primary.get("advisor_effective_action"),
            "x3_advisor_confidence": primary.get("advisor_confidence"),
            "x3_hard_guard_applied": bool(primary.get("hard_guard_applied", False)),
            "x3_would_reduce_cash": bool(primary.get("would_reduce_cash", False)),
            "x3_would_veto": bool(primary.get("would_veto", False)),
            "x3_candidate_size": primary.get("candidate_size"),
            "x3_shadow_final_size": primary.get("shadow_final_size"),
            "x3_candidate_side": primary.get("candidate_side"),
        }
    )
    return base


def build_x3_active_payload(
    *,
    risk_cfg: dict[str, Any],
    decision: Any,
    candidate_cash_required: float,
    raw_cash_required: float,
    sizing_cap_payload: dict[str, Any],
    candidate_side: str | None = None,
) -> dict[str, Any]:
    cfg = risk_cfg.get("x3_resolver", {})
    if not isinstance(cfg, dict) or not x3_active_enabled(risk_cfg):
        return {}

    variant_cfg = _x3_variant_configs(cfg)[0]
    primary = _x3_variant_entry(
        variant_cfg=variant_cfg,
        decision=decision,
        candidate_cash_required=float(candidate_cash_required),
        candidate_side=candidate_side,
    )
    policy_name = str(primary.get("policy_name") or cfg.get("policy_name") or cfg.get("policy") or "disabled")
    return {
        "x3_shadow_only": False,
        "x3_runtime_applied": True,
        "x3_runtime_note": "paper_runtime_applies_active_x3",
        "x3_x1_action": getattr(decision, "action", None),
        "x3_x1_reason": getattr(decision, "reason", None),
        "x3_raw_cash_required": float(max(0.0, raw_cash_required)),
        "x3_post_sizing_candidate_cash_usd": float(max(0.0, candidate_cash_required)),
        "x3_sizing_cap_applied_before_x3": bool(sizing_cap_payload.get("sizing_cap_applied", False)),
        "x3_enabled": True,
        "x3_mode": "active",
        "x3_policy_name": policy_name,
        "x3_action": primary.get("action"),
        "x3_candidate_cash_usd": primary.get("candidate_cash_usd"),
        "x3_final_cash_usd": primary.get("final_cash_usd"),
        "x3_cash_multiplier": primary.get("cash_multiplier"),
        "x3_reason_codes": primary.get("reason_codes"),
        "x3_advisor_effective_action": primary.get("advisor_effective_action"),
        "x3_advisor_confidence": primary.get("advisor_confidence"),
        "x3_hard_guard_applied": bool(primary.get("hard_guard_applied", False)),
        "x3_would_reduce_cash": bool(primary.get("would_reduce_cash", False)),
        "x3_would_veto": bool(primary.get("would_veto", False)),
        "x3_candidate_size": primary.get("candidate_size"),
        "x3_shadow_final_size": primary.get("shadow_final_size"),
        "x3_candidate_side": primary.get("candidate_side"),
    }


def apply_x3_active_to_decision(
    *,
    risk_cfg: dict[str, Any],
    decision: Any,
    candidate_cash_required: float,
    raw_cash_required: float,
    sizing_cap_payload: dict[str, Any],
    candidate_side: str | None = None,
) -> tuple[Any, float, dict[str, Any]]:
    payload = build_x3_active_payload(
        risk_cfg=risk_cfg,
        decision=decision,
        candidate_cash_required=float(candidate_cash_required),
        raw_cash_required=float(raw_cash_required),
        sizing_cap_payload=sizing_cap_payload,
        candidate_side=candidate_side,
    )
    if not payload or getattr(decision, "intent", None) is None:
        return decision, float(candidate_cash_required), payload

    action = str(payload.get("x3_action") or "allow").lower()
    if action == "veto":
        updated = replace(
            decision,
            action="no_trade",
            reason=f"x3_veto:{','.join(payload.get('x3_reason_codes') or [])}",
            intent=None,
            cash_required=0.0,
        )
        return updated, 0.0, payload

    final_cash = payload.get("x3_final_cash_usd")
    multiplier = payload.get("x3_cash_multiplier")
    try:
        final_cash_f = float(final_cash)
        multiplier_f = float(multiplier)
    except (TypeError, ValueError):
        return decision, float(candidate_cash_required), payload

    if action == "cap" and 0.0 < final_cash_f < float(candidate_cash_required) - 0.01 and 0.0 < multiplier_f <= 1.0:
        updated_intent = replace(decision.intent, size=float(decision.intent.size) * multiplier_f)
        updated = replace(decision, intent=updated_intent, cash_required=final_cash_f)
        return updated, final_cash_f, payload

    return decision, float(candidate_cash_required), payload
