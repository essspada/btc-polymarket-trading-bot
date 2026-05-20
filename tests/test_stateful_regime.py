from __future__ import annotations


from src.polymarket.risk import RiskManager


def _risk() -> RiskManager:
    return RiskManager(
        max_exposure_per_window_usd=150.0,
        max_daily_loss_usd=100.0,
        cooldown_after_loss_streak=99,
        cooldown_windows=3,
        max_open_positions=3,
        stateful_regime={
            "enabled": True,
            "drawdown_soft_pct": 0.05,
            "drawdown_hard_pct": 0.15,
            "loss_streak_soft": 2,
            "loss_streak_hard": 4,
            "daily_loss_soft_frac": 0.05,
            "daily_loss_hard_frac": 0.15,
            "min_multiplier": 0.4,
            "recovery_wins_required": 2,
        },
    )


def test_stateful_multiplier_monotonic_on_drawdown() -> None:
    risk = _risk()
    m1, _ = risk.stateful_trade_multiplier(available_cash=100.0, peak_cash=100.0)
    m2, _ = risk.stateful_trade_multiplier(available_cash=92.0, peak_cash=100.0)
    m3, _ = risk.stateful_trade_multiplier(available_cash=82.0, peak_cash=100.0)
    assert m1 >= m2 >= m3


def test_stateful_recovery_lock_releases_after_wins() -> None:
    risk = _risk()
    risk.on_trade_close(-5.0)
    risk.on_trade_close(-3.0)
    m_locked, reason = risk.stateful_trade_multiplier(available_cash=90.0, peak_cash=100.0)
    assert m_locked <= 0.41
    assert reason == "stateful_recovery_lock"

    risk.on_trade_close(+1.0)
    m_still_locked, _ = risk.stateful_trade_multiplier(available_cash=90.0, peak_cash=100.0)
    assert m_still_locked <= 0.41

    risk.on_trade_close(+1.0)
    m_after, reason_after = risk.stateful_trade_multiplier(available_cash=90.0, peak_cash=100.0)
    assert m_after <= 1.0
    assert reason_after in {"stateful_soft_throttle", "stateful_hard_throttle", "stateful_ok"}


def test_stateful_disabled_is_neutral() -> None:
    risk = RiskManager(
        max_exposure_per_window_usd=150.0,
        max_daily_loss_usd=100.0,
        cooldown_after_loss_streak=3,
        cooldown_windows=3,
        max_open_positions=3,
        stateful_regime={"enabled": False},
    )
    m, reason = risk.stateful_trade_multiplier(available_cash=50.0, peak_cash=70.0)
    assert m == 1.0
    assert reason == "stateful_disabled"
