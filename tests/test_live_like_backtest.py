from src.backtest import live_like as live_like_mod
from src.backtest.live_like import LiveLikeConfig, run_live_like_backtest
from src.polymarket.fees import FeeModelConfig


def _exec_cfg(**overrides):
    base = {
        "min_edge_to_trade": 0.01,
        "min_edge_for_taker": 0.02,
        "maker_preference": True,
        "allowed_order_types": ["maker"],
        "maker_fill_probability": 1.0,
        "maker_fill_floor": 1.0,
        "maker_fill_cap": 1.0,
        "maker_fill_base": 1.0,
        "maker_fill_spread_penalty": 0.0,
        "maker_fill_late_penalty_90": 0.0,
        "maker_fill_late_penalty_45": 0.0,
        "maker_ev_advantage_required": 0.0,
        "slippage_bps_taker": 0.0,
        "slippage_bps_maker": 0.0,
    }
    base.update(overrides)
    return base


def _risk_cfg(**overrides):
    base = {
        "max_exposure_per_window_usd": 150.0,
        "max_daily_loss_usd": 1000.0,
        "cooldown_after_loss_streak": 99,
        "cooldown_windows": 0,
        "max_open_positions": 3,
        "drift_enabled": False,
    }
    base.update(overrides)
    return base


def test_live_like_backtest_caps_to_balance_on_legacy_rows() -> None:
    rows = [
        {
            "market_id": "m1",
            "market_slug": "m1",
            "created_at": "2026-03-01T00:01:00+00:00",
            "actual_side": "up",
            "up_token_id": "up1",
            "down_token_id": "down1",
            "up_entry_price": 0.40,
            "down_entry_price": 0.60,
            "spot_timing_policy_p_up": 0.80,
            "seconds_to_expiry": 120.0,
        }
    ]
    out = run_live_like_backtest(
        rows,
        candidate_key="spot_timing_policy",
        fee_cfg=FeeModelConfig(maker_fee_bps=0.0, taker_fee_bps=0.0, curve_rate=0.0, min_fee=0.0),
        exec_cfg=_exec_cfg(allowed_order_types=["maker"]),
        risk_cfg=_risk_cfg(max_exposure_per_window_usd=150.0, max_balance_fraction_per_trade=0.2),
        config=LiveLikeConfig(initial_balance=10.0, min_trade_notional=1.0, include_trade_log=True),
    )
    assert out["data_mode_counts"] == {"legacy_entry_replay": 1}
    assert out["submitted_trades"] == 1
    assert out["filled_trades"] == 1
    assert out["trade_log"][0]["cash_required"] <= 2.0 + 1e-9
    assert out["final_balance"] > 10.0


def test_live_like_backtest_uses_native_runtime_path() -> None:
    rows = [
        {
            "market_id": "m1",
            "market_slug": "m1",
            "created_at": "2026-03-01T00:01:00+00:00",
            "end_time": "2026-03-01T00:05:00+00:00",
            "actual_side": "up",
            "up_token_id": "up1",
            "down_token_id": "down1",
            "up_best_bid": 0.49,
            "up_best_ask": 0.50,
            "down_best_bid": 0.49,
            "down_best_ask": 0.50,
            "up_best_bid_size": 100.0,
            "up_best_ask_size": 100.0,
            "down_best_bid_size": 100.0,
            "down_best_ask_size": 100.0,
            "spot_timing_policy_p_up": 0.75,
            "seconds_to_expiry": 120.0,
        }
    ]
    out = run_live_like_backtest(
        rows,
        candidate_key="spot_timing_policy",
        fee_cfg=FeeModelConfig(maker_fee_bps=0.0, taker_fee_bps=0.0, curve_rate=0.0, min_fee=0.0),
        exec_cfg=_exec_cfg(),
        risk_cfg=_risk_cfg(max_exposure_per_window_usd=100.0),
        config=LiveLikeConfig(initial_balance=100.0, min_trade_notional=1.0, include_trade_log=True),
    )
    assert out["data_mode_counts"] == {"native_orderbook_runtime": 1}
    assert out["order_type_counts"] == {"MAKER": 1}
    assert out["filled_trades"] == 1
    assert out["final_balance"] > 100.0


def test_live_like_backtest_can_require_orderbook() -> None:
    rows = [
        {
            "market_id": "m1",
            "market_slug": "m1",
            "created_at": "2026-03-01T00:01:00+00:00",
            "actual_side": "up",
            "up_token_id": "up1",
            "down_token_id": "down1",
            "up_entry_price": 0.40,
            "down_entry_price": 0.60,
            "spot_timing_policy_p_up": 0.80,
            "seconds_to_expiry": 120.0,
        }
    ]
    out = run_live_like_backtest(
        rows,
        candidate_key="spot_timing_policy",
        fee_cfg=FeeModelConfig(maker_fee_bps=0.0, taker_fee_bps=0.0, curve_rate=0.0, min_fee=0.0),
        exec_cfg=_exec_cfg(),
        risk_cfg=_risk_cfg(),
        config=LiveLikeConfig(initial_balance=100.0, min_trade_notional=1.0, require_orderbook=True),
    )
    assert out["submitted_trades"] == 0
    assert out["skip_reason_counts"]["orderbook_required"] == 1


def test_live_like_backtest_skips_incoherent_legacy_rows_by_default() -> None:
    rows = [
        {
            "market_id": "m1",
            "market_slug": "m1",
            "created_at": "2026-03-01T00:01:00+00:00",
            "actual_side": "up",
            "up_token_id": "up1",
            "down_token_id": "down1",
            "up_entry_price": 0.70,
            "down_entry_price": 0.55,
            "up_entry_price_norm": 0.56,
            "down_entry_price_norm": 0.44,
            "entry_price_sum_raw": 1.25,
            "spot_timing_policy_p_up": 0.80,
            "seconds_to_expiry": 120.0,
        }
    ]
    out = run_live_like_backtest(
        rows,
        candidate_key="spot_timing_policy",
        fee_cfg=FeeModelConfig(maker_fee_bps=0.0, taker_fee_bps=0.0, curve_rate=0.0, min_fee=0.0),
        exec_cfg=_exec_cfg(),
        risk_cfg=_risk_cfg(),
        config=LiveLikeConfig(initial_balance=100.0, min_trade_notional=1.0),
    )
    assert out["submitted_trades"] == 0
    assert out["skip_reason_counts"]["entry_price_sum_out_of_bounds"] == 1


def test_live_like_candidate_probability_uses_row_value_not_runtime_normalization_payload() -> None:
    row = {
        "spot_timing_policy_p_up": 0.67,
        "p_up_raw": 0.42,
        "p_up_pre_calibration": 0.44,
        "p_up_calibrated": 0.46,
        "is_normalized": False,
    }
    assert live_like_mod._candidate_prob(row, "spot_timing_policy") == 0.67


def test_live_like_backtest_applies_stateful_multiplier_after_drawdown() -> None:
    rows = [
        {
            "market_id": "m1",
            "market_slug": "m1",
            "created_at": "2026-03-01T00:01:00+00:00",
            "end_time": "2026-03-01T00:05:00+00:00",
            "actual_side": "down",
            "up_token_id": "up1",
            "down_token_id": "down1",
            "up_best_bid": 0.49,
            "up_best_ask": 0.50,
            "down_best_bid": 0.49,
            "down_best_ask": 0.50,
            "up_best_bid_size": 100.0,
            "up_best_ask_size": 100.0,
            "down_best_bid_size": 100.0,
            "down_best_ask_size": 100.0,
            "spot_timing_policy_p_up": 0.75,
            "seconds_to_expiry": 120.0,
        },
        {
            "market_id": "m2",
            "market_slug": "m2",
            "created_at": "2026-03-01T00:06:00+00:00",
            "end_time": "2026-03-01T00:10:00+00:00",
            "actual_side": "up",
            "up_token_id": "up2",
            "down_token_id": "down2",
            "up_best_bid": 0.49,
            "up_best_ask": 0.50,
            "down_best_bid": 0.49,
            "down_best_ask": 0.50,
            "up_best_bid_size": 100.0,
            "up_best_ask_size": 100.0,
            "down_best_bid_size": 100.0,
            "down_best_ask_size": 100.0,
            "spot_timing_policy_p_up": 0.75,
            "seconds_to_expiry": 120.0,
        },
    ]

    out = run_live_like_backtest(
        rows,
        candidate_key="spot_timing_policy",
        fee_cfg=FeeModelConfig(maker_fee_bps=0.0, taker_fee_bps=0.0, curve_rate=0.0, min_fee=0.0),
        exec_cfg=_exec_cfg(),
        risk_cfg=_risk_cfg(
            max_exposure_per_window_usd=100.0,
            max_balance_fraction_per_trade=0.20,
            stateful_regime={
                "enabled": True,
                "drawdown_soft_pct": 0.01,
                "drawdown_hard_pct": 0.02,
                "daily_loss_soft_frac": 0.01,
                "daily_loss_hard_frac": 0.02,
                "loss_streak_soft": 1,
                "loss_streak_hard": 1,
                "min_multiplier": 0.40,
                "recovery_wins_required": 0,
            },
        ),
        config=LiveLikeConfig(initial_balance=100.0, min_trade_notional=1.0, include_trade_log=True),
    )

    assert out["filled_trades"] == 2
    assert out["stateful_reason_counts"]["stateful_ok"] == 1
    assert out["trade_log"][0]["stateful_multiplier_applied"] is False
    assert out["trade_log"][1]["stateful_multiplier_applied"] is True
    assert out["trade_log"][1]["cash_required"] < out["trade_log"][0]["cash_required"]
