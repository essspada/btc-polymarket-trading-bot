from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.polymarket.orderbook import MarketBooks, OutcomeBook
from src.runtime.shadow_risk import build_adaptive_risk_shadow_payload, build_monitor_adaptive_risk_shadow_payload
from src.strategy.adaptive_risk import resolve_adaptive_risk_shadow

BASE_CFG = {
    "enabled": True,
    "mode": "shadow",
    "drawdown_soft_pct": 0.05,
    "drawdown_hard_pct": 0.15,
    "daily_loss_lock_frac": 0.12,
    "loss_streak_cooldown": 2,
    "loss_streak_long_cooldown": 3,
    "min_multiplier": 0.35,
    "veto_negative_economics": True,
    "min_expected_roi_cash": 0.0,
    "min_breakeven_margin": 0.0,
}


def test_adaptive_risk_disabled_is_noop_and_never_increases_cash() -> None:
    decision = resolve_adaptive_risk_shadow(cfg={"enabled": False}, candidate_cash_usd=10.0)

    assert decision.action == "allow"
    assert decision.final_cash_usd == 10.0
    assert decision.cash_multiplier == 1.0
    assert decision.runtime_applied is False
    assert decision.shadow_only is True


def test_adaptive_risk_drawdown_caps_without_runtime_authority() -> None:
    decision = resolve_adaptive_risk_shadow(
        cfg={**BASE_CFG, "max_candidate_fraction_of_equity": 1.0},
        candidate_cash_usd=20.0,
        current_equity_usd=88.0,
        peak_equity_usd=100.0,
        daily_pnl=0.0,
        expected_roi_cash=0.05,
        breakeven_margin=0.04,
    )

    assert decision.action == "cap"
    assert decision.final_cash_usd < 20.0
    assert decision.final_cash_usd <= decision.candidate_cash_usd
    assert decision.runtime_applied is False
    assert "drawdown_throttle" in decision.reason_codes


def test_adaptive_risk_negative_economics_vetoes_shadow_candidate() -> None:
    decision = resolve_adaptive_risk_shadow(
        cfg=BASE_CFG,
        candidate_cash_usd=12.0,
        current_equity_usd=100.0,
        peak_equity_usd=100.0,
        expected_roi_cash=-0.01,
        breakeven_margin=0.02,
    )

    assert decision.action == "veto"
    assert decision.final_cash_usd == 0.0
    assert decision.would_veto is True
    assert "negative_economics_expected_roi" in decision.reason_codes


def test_adaptive_risk_loss_streak_triggers_shadow_cooldown() -> None:
    decision = resolve_adaptive_risk_shadow(
        cfg=BASE_CFG,
        candidate_cash_usd=12.0,
        current_equity_usd=100.0,
        peak_equity_usd=100.0,
        loss_streak=3,
        expected_roi_cash=0.05,
        breakeven_margin=0.02,
    )

    assert decision.action == "cooldown"
    assert decision.final_cash_usd == 0.0
    assert decision.would_cooldown is True
    assert decision.suggested_cooldown_windows >= 1
    assert "loss_streak_long_cooldown" in decision.reason_codes


def test_adaptive_risk_active_mode_is_forced_to_shadow() -> None:
    decision = resolve_adaptive_risk_shadow(
        cfg={**BASE_CFG, "mode": "active"},
        candidate_cash_usd=12.0,
        current_equity_usd=100.0,
        peak_equity_usd=100.0,
        expected_roi_cash=0.05,
        breakeven_margin=0.02,
    )

    assert decision.mode == "shadow"
    assert decision.runtime_applied is False
    assert "active_mode_forced_shadow" in decision.reason_codes


def test_main_adaptive_payload_is_shadow_only_and_runtime_friendly() -> None:
    payload = build_adaptive_risk_shadow_payload(
        risk_cfg={"adaptive_risk": BASE_CFG},
        decision=SimpleNamespace(expected_roi_cash=0.05, breakeven_margin=0.02),
        candidate_cash_required=20.0,
        current_equity_usd=88.0,
        available_cash_usd=88.0,
        reserved_cash_usd=0.0,
        peak_equity_usd=100.0,
        daily_pnl=-2.0,
        loss_streak=0,
        consecutive_wins=1,
        cooldown_left=0,
        recent_accuracy=0.62,
        open_positions=1,
        max_open_positions=3,
        risk_budget_cash_usd=50.0,
        confidence=0.70,
        spread_norm=0.02,
        spot_recent_vol_5m_bps=3.0,
        seconds_to_expiry=60.0,
    )

    assert payload["adaptive_risk_enabled"] is True
    assert payload["adaptive_risk_mode"] == "shadow"
    assert payload["adaptive_risk_shadow_only"] is True
    assert payload["adaptive_risk_runtime_applied"] is False
    assert payload["adaptive_risk_final_cash_usd"] <= payload["adaptive_risk_candidate_cash_usd"]
    assert payload["adaptive_risk_reason_count"] >= 1


def test_main_adaptive_payload_disabled_is_empty() -> None:
    payload = build_adaptive_risk_shadow_payload(
        risk_cfg={"adaptive_risk": {"enabled": False, "mode": "shadow"}},
        decision=SimpleNamespace(expected_roi_cash=0.05, breakeven_margin=0.02),
        candidate_cash_required=20.0,
        current_equity_usd=100.0,
        available_cash_usd=100.0,
        reserved_cash_usd=0.0,
        peak_equity_usd=100.0,
        daily_pnl=0.0,
        loss_streak=0,
        consecutive_wins=0,
        cooldown_left=0,
        recent_accuracy=0.0,
        open_positions=0,
        max_open_positions=3,
        risk_budget_cash_usd=50.0,
        confidence=None,
        spread_norm=None,
        spot_recent_vol_5m_bps=None,
        seconds_to_expiry=None,
    )

    assert payload == {}


def test_monitor_adaptive_payload_uses_synthetic_bankroll_without_creating_trade() -> None:
    books = MarketBooks(
        up=OutcomeBook("up-token", 0.41, 0.42, 0.415, 0.01, 0.1),
        down=OutcomeBook("down-token", 0.57, 0.58, 0.575, 0.01, -0.1),
    )
    payload = build_monitor_adaptive_risk_shadow_payload(
        risk_cfg={
            "max_exposure_per_window_usd": 150.0,
            "max_balance_fraction_per_trade": 0.12,
            "max_open_positions": 3,
            "adaptive_risk": BASE_CFG,
        },
        paper_cfg={"initial_balance_usd": 100.0},
        decision=SimpleNamespace(expected_roi_cash=0.05, breakeven_margin=0.02, cash_required=20.0),
        books=books,
        candidate_side="up",
        confidence=0.70,
        spot_recent_vol_5m_bps=5.0,
        seconds_to_expiry=60.0,
    )

    assert payload["adaptive_risk_monitor_only"] is True
    assert payload["adaptive_risk_state_source"] == "monitor_only_synthetic_bankroll"
    assert payload["adaptive_risk_runtime_applied"] is False
    assert payload["adaptive_risk_final_cash_usd"] <= payload["adaptive_risk_candidate_cash_usd"]
    assert payload["adaptive_risk_risk_budget_cash_usd"] == 12.0


@pytest.mark.parametrize("candidate", [0.0, -1.0, None])
def test_adaptive_risk_no_candidate_cannot_create_trade(candidate) -> None:
    decision = resolve_adaptive_risk_shadow(cfg=BASE_CFG, candidate_cash_usd=candidate)

    assert decision.action == "no_candidate"
    assert decision.final_cash_usd == 0.0
    assert decision.cash_multiplier == 0.0
