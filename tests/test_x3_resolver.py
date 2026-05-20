from types import SimpleNamespace

from src.polymarket.execution import OrderIntent
from src.runtime.x3 import apply_x3_active_to_decision, build_x3_shadow_payload
from src.strategy.signals import SignalDecision
from src.strategy.x3_resolver import resolve_x3_candidate


def test_x3_disabled_keeps_candidate_cash() -> None:
    decision = resolve_x3_candidate(
        cfg={"enabled": False, "large_cash_cap_usd": 12.0},
        candidate_cash_usd=21.99,
        side="up",
        entry_price=0.66,
    )
    assert decision.action == "allow"
    assert decision.final_cash_usd == 21.99
    assert decision.cash_multiplier == 1.0
    assert decision.reason_codes == ["disabled"]


def test_x3_cap_large_cash12_matches_current_shadow_policy() -> None:
    decision = resolve_x3_candidate(
        cfg={
            "enabled": True,
            "mode": "shadow",
            "policy": "cap_large_cash12",
            "large_cash_threshold_usd": 16.0,
            "large_cash_cap_usd": 12.0,
        },
        candidate_cash_usd=21.99,
        side="up",
        entry_price=0.66,
    )
    assert decision.action == "cap"
    assert decision.final_cash_usd == 12.0
    assert round(decision.cash_multiplier or 0.0, 6) == round(12.0 / 21.99, 6)
    assert "large_cash_cap_usd" in decision.reason_codes


def test_x3_never_increases_candidate_cash_even_with_bad_config_or_advisor() -> None:
    decision = resolve_x3_candidate(
        cfg={
            "enabled": True,
            "mode": "shadow",
            "global_cap_usd": 99.0,
            "respect_advisor_cap": True,
        },
        candidate_cash_usd=10.0,
        side="down",
        entry_price=0.60,
        advisor={"advisor_effective_action": "cap", "advisor_max_cash_usd": 1000.0, "advisor_confidence": 0.9},
    )
    assert decision.action == "allow"
    assert decision.final_cash_usd == 10.0
    assert decision.cash_multiplier == 1.0
    assert decision.advisor_effective_action == "cap"
    assert all((decision.final_cash_usd or 0.0) <= 10.0 for _ in [0])


def test_x3_combines_large_cash_and_mid_price_by_strictest_cap() -> None:
    decision = resolve_x3_candidate(
        cfg={
            "enabled": True,
            "mode": "shadow",
            "large_cash_threshold_usd": 16.0,
            "large_cash_cap_usd": 15.0,
            "mid_price_min": 0.55,
            "mid_price_max": 0.75,
            "mid_price_cap_usd": 12.0,
        },
        candidate_cash_usd=19.36,
        side="up",
        entry_price=0.60,
    )
    assert decision.action == "cap"
    assert decision.final_cash_usd == 12.0
    assert "large_cash_cap_usd" in decision.reason_codes
    assert "mid_price_cap_usd" in decision.reason_codes


def test_x3_advisor_can_reduce_or_veto_when_explicitly_respected() -> None:
    capped = resolve_x3_candidate(
        cfg={"enabled": True, "mode": "shadow", "respect_advisor_cap": True},
        candidate_cash_usd=14.0,
        side="up",
        entry_price=0.70,
        advisor={"advisor_effective_action": "cap", "advisor_max_cash_usd": 9.5, "advisor_confidence": 0.7},
    )
    assert capped.action == "cap"
    assert capped.final_cash_usd == 9.5
    assert "advisor_cap" in capped.reason_codes

    vetoed = resolve_x3_candidate(
        cfg={"enabled": True, "mode": "shadow", "respect_advisor_veto": True},
        candidate_cash_usd=14.0,
        side="down",
        entry_price=0.60,
        advisor={"advisor_effective_action": "veto", "advisor_max_cash_usd": 0.0, "advisor_confidence": 0.8},
    )
    assert vetoed.action == "veto"
    assert vetoed.final_cash_usd == 0.0
    assert vetoed.cash_multiplier == 0.0
    assert "advisor_veto" in vetoed.reason_codes


def test_x3_min_trade_floor_converts_tiny_cap_to_veto() -> None:
    decision = resolve_x3_candidate(
        cfg={
            "enabled": True,
            "mode": "shadow",
            "global_cap_usd": 3.0,
            "min_trade_usd_after_cap": 5.0,
        },
        candidate_cash_usd=20.0,
        side="up",
        entry_price=0.80,
    )
    assert decision.action == "veto"
    assert decision.final_cash_usd == 0.0
    assert "below_min_trade_after_cap" in decision.reason_codes


def test_x3_piecewise_side_entry_policy_caps_growth_guard_zones() -> None:
    up_mid = resolve_x3_candidate(
        cfg={"enabled": True, "mode": "shadow", "policy": "piecewise_side_entry_v1"},
        candidate_cash_usd=22.0,
        side="up",
        entry_price=0.66,
    )
    assert up_mid.action == "cap"
    assert up_mid.final_cash_usd == 15.0
    assert "piecewise_up_entry_lt_075_cap_usd" in up_mid.reason_codes

    up_expensive = resolve_x3_candidate(
        cfg={"enabled": True, "mode": "shadow", "policy": "piecewise_side_entry_v1"},
        candidate_cash_usd=22.0,
        side="up",
        entry_price=0.83,
    )
    assert up_expensive.action == "cap"
    assert up_expensive.final_cash_usd == 18.0
    assert "piecewise_up_entry_gte_075_cap_usd" in up_expensive.reason_codes

    down_mid = resolve_x3_candidate(
        cfg={"enabled": True, "mode": "shadow", "policy": "piecewise_side_entry_v1"},
        candidate_cash_usd=18.0,
        side="down",
        entry_price=0.60,
    )
    assert down_mid.action == "cap"
    assert down_mid.final_cash_usd == 10.0
    assert "piecewise_down_mid_price_cap_usd" in down_mid.reason_codes


def test_x3_economics_core_caps_thin_roi_margin_and_high_entry() -> None:
    decision = resolve_x3_candidate(
        cfg={
            "enabled": True,
            "mode": "shadow",
            "policy": "economics_core_v1",
            "min_expected_roi": 0.08,
            "thin_roi_cap_usd": 11.0,
            "min_breakeven_margin": 0.06,
            "thin_margin_cap_usd": 9.0,
            "high_entry_min": 0.75,
            "high_entry_min_expected_roi": 0.12,
            "high_entry_cap_usd": 7.0,
        },
        candidate_cash_usd=22.0,
        side="up",
        entry_price=0.80,
        p_up=0.82,
    )

    assert decision.action == "cap"
    assert decision.final_cash_usd == 7.0
    assert "economics_thin_roi_cap_usd" in decision.reason_codes
    assert "economics_thin_margin_cap_usd" in decision.reason_codes
    assert "economics_high_entry_cap_usd" in decision.reason_codes


def test_x3_economics_core_composite_reducer_never_increases_cash() -> None:
    decision = resolve_x3_candidate(
        cfg={
            "enabled": True,
            "mode": "shadow",
            "policy": "economics_core_v1",
            "capital_usd": 100.0,
            "composite_reducer_enabled": True,
            "roi_good": 0.20,
            "roi_bad": 0.02,
            "roi_floor_multiplier": 0.50,
            "margin_good": 0.12,
            "margin_bad": 0.02,
            "margin_floor_multiplier": 0.60,
            "cash_fraction_soft_limit": 0.12,
            "cash_fraction_floor_multiplier": 0.65,
        },
        candidate_cash_usd=20.0,
        side="up",
        entry_price=0.60,
        p_up=0.72,
    )

    assert decision.action == "cap"
    assert 0.0 < (decision.final_cash_usd or 0.0) <= 20.0
    assert "economics_composite_cap_usd" in decision.reason_codes


def test_x3_economics_core_caps_weak_low_entry_geometry() -> None:
    decision = resolve_x3_candidate(
        cfg={
            "enabled": True,
            "mode": "shadow",
            "policy": "economics_core_v1",
            "low_entry_max": 0.45,
            "low_entry_min_expected_roi": 0.20,
            "low_entry_cap_usd": 6.0,
        },
        candidate_cash_usd=18.0,
        side="down",
        entry_price=0.40,
        p_up=0.55,
    )

    assert decision.action == "cap"
    assert decision.final_cash_usd == 6.0
    assert "economics_low_entry_cap_usd" in decision.reason_codes


def test_x3_payload_prefix_is_runtime_friendly() -> None:
    decision = resolve_x3_candidate(
        cfg={"enabled": True, "mode": "shadow", "hard_max_cash_usd": 12.0},
        candidate_cash_usd=20.0,
        side="up",
        entry_price=0.80,
    )
    payload = decision.to_payload()
    assert payload["x3_action"] == "cap"
    assert payload["x3_candidate_cash_usd"] == 20.0
    assert payload["x3_final_cash_usd"] == 12.0
    assert payload["x3_hard_guard_applied"] is True


def test_paper_x3_shadow_payload_does_not_apply_runtime_change() -> None:
    intent = SimpleNamespace(side="up", price=0.66, size=33.318)
    decision = SimpleNamespace(action="trade", reason="expected_roi_cash", intent=intent)
    payload = build_x3_shadow_payload(
        risk_cfg={
            "x3_resolver": {
                "enabled": True,
                "mode": "shadow",
                "policy": "cap_large_cash12",
                "large_cash_threshold_usd": 16.0,
                "large_cash_cap_usd": 12.0,
            }
        },
        decision=decision,
        candidate_cash_required=21.99,
        raw_cash_required=21.99,
        sizing_cap_payload={"sizing_cap_applied": False},
    )
    assert payload["x3_shadow_only"] is True
    assert payload["x3_runtime_applied"] is False
    assert payload["x3_action"] == "cap"
    assert payload["x3_final_cash_usd"] == 12.0
    assert payload["x3_would_reduce_cash"] is True


def test_paper_x3_shadow_payload_logs_multiple_variants() -> None:
    intent = SimpleNamespace(side="up", price=0.83, size=26.0)
    decision = SimpleNamespace(action="trade", reason="expected_roi_cash", intent=intent)
    payload = build_x3_shadow_payload(
        risk_cfg={
            "x3_resolver": {
                "enabled": True,
                "mode": "shadow",
                "large_cash_threshold_usd": 16.0,
                "large_cash_cap_usd": 12.0,
                "variants": [
                    {
                        "policy": "cap_large_cash12",
                        "large_cash_threshold_usd": 16.0,
                        "large_cash_cap_usd": 12.0,
                    },
                    {"policy": "piecewise_side_entry_v1"},
                ],
            }
        },
        decision=decision,
        candidate_cash_required=21.5,
        raw_cash_required=21.5,
        sizing_cap_payload={"sizing_cap_applied": False},
    )

    assert payload["x3_shadow_only"] is True
    assert payload["x3_runtime_applied"] is False
    assert payload["x3_policy_name"] == "cap_large_cash12"
    assert payload["x3_action"] == "cap"
    assert payload["x3_final_cash_usd"] == 12.0
    assert payload["x3_variant_count"] == 2
    assert payload["x3_variants"][0]["policy_name"] == "cap_large_cash12"
    assert payload["x3_variants"][0]["final_cash_usd"] == 12.0
    assert payload["x3_variants"][1]["policy_name"] == "piecewise_side_entry_v1"
    assert payload["x3_variants"][1]["final_cash_usd"] == 18.0
    assert "large_cash_cap_usd" not in payload["x3_variants"][1]["reason_codes"]


def test_paper_x3_shadow_payload_uses_outcome_side_for_piecewise_policy() -> None:
    intent = SimpleNamespace(side="BUY", price=0.83, size=26.0)
    decision = SimpleNamespace(action="trade", reason="ok", intent=intent)
    payload = build_x3_shadow_payload(
        risk_cfg={
            "x3_resolver": {
                "enabled": True,
                "mode": "shadow",
                "policy": "piecewise_side_entry_v1",
            }
        },
        decision=decision,
        candidate_cash_required=21.5,
        raw_cash_required=21.5,
        sizing_cap_payload={"sizing_cap_applied": False},
        candidate_side="up",
    )

    assert payload["x3_action"] == "cap"
    assert payload["x3_final_cash_usd"] == 18.0
    assert payload["x3_candidate_side"] == "up"
    assert "piecewise_up_entry_gte_075_cap_usd" in payload["x3_reason_codes"]


def test_paper_x3_shadow_payload_disabled_is_empty() -> None:
    decision = SimpleNamespace(action="trade", reason="r", intent=SimpleNamespace(side="up", price=0.66, size=1.0))
    payload = build_x3_shadow_payload(
        risk_cfg={"x3_resolver": {"enabled": False, "mode": "shadow"}},
        decision=decision,
        candidate_cash_required=21.99,
        raw_cash_required=21.99,
        sizing_cap_payload={},
    )
    assert payload == {}


def _signal_decision(cash_required: float = 21.99) -> SignalDecision:
    return SignalDecision(
        action="trade",
        reason="ok",
        p_up=0.80,
        p_down=0.20,
        best_edge=0.10,
        intent=OrderIntent(
            market_slug="m",
            token_id="up-token",
            side="up",
            order_type="MAKER",
            price=0.66,
            size=33.318,
            expected_edge=0.10,
        ),
        cash_required=cash_required,
    )


def test_active_x3_cap_reduces_intent_size_and_cash() -> None:
    decision, cash, payload = apply_x3_active_to_decision(
        risk_cfg={
            "x3_resolver": {
                "enabled": True,
                "mode": "active",
                "policy": "cap_large_cash12",
                "large_cash_threshold_usd": 16.0,
                "large_cash_cap_usd": 12.0,
            }
        },
        decision=_signal_decision(),
        candidate_cash_required=21.99,
        raw_cash_required=21.99,
        sizing_cap_payload={"sizing_cap_applied": False},
    )

    assert cash == 12.0
    assert decision.intent is not None
    assert round(decision.intent.size, 6) == round(33.318 * (12.0 / 21.99), 6)
    assert decision.cash_required == 12.0
    assert payload["x3_runtime_applied"] is True
    assert payload["x3_shadow_only"] is False
    assert payload["x3_action"] == "cap"


def test_active_x3_veto_converts_trade_to_no_trade() -> None:
    decision, cash, payload = apply_x3_active_to_decision(
        risk_cfg={
            "x3_resolver": {
                "enabled": True,
                "mode": "active",
                "policy": "hard_veto",
                "global_cap_usd": 3.0,
                "min_trade_usd_after_cap": 5.0,
            }
        },
        decision=_signal_decision(),
        candidate_cash_required=21.99,
        raw_cash_required=21.99,
        sizing_cap_payload={},
    )

    assert cash == 0.0
    assert decision.action == "no_trade"
    assert decision.intent is None
    assert decision.reason.startswith("x3_veto:")
    assert payload["x3_action"] == "veto"


def test_active_x3_does_not_write_shadow_payload() -> None:
    payload = build_x3_shadow_payload(
        risk_cfg={
            "x3_resolver": {
                "enabled": True,
                "mode": "active",
                "policy": "cap_large_cash12",
                "large_cash_cap_usd": 12.0,
            }
        },
        decision=_signal_decision(),
        candidate_cash_required=21.99,
        raw_cash_required=21.99,
        sizing_cap_payload={},
    )

    assert payload == {}
