import pytest

from src.backtest.predicted_side import PredictedSideBacktestConfig, run_predicted_side_backtest
from src.polymarket.fees import FeeModelConfig


def _row(**overrides):
    base = {
        "market_id": "m1",
        "created_at": "2026-03-07T00:04:00+00:00",
        "actual_side": "up",
        "up_entry_price": 0.4,
        "down_entry_price": 0.6,
        "up_entry_price_norm": 0.4,
        "down_entry_price_norm": 0.6,
        "entry_price_sum_raw": 1.0,
        "spot_timing_policy_p_up": 0.8,
        "spot_timing_policy_stage_delay": 240,
        "spot_timing_policy_entry_mode": "fresh_due",
    }
    base.update(overrides)
    return base


def test_predicted_side_loss_loses_full_cash_required() -> None:
    result = run_predicted_side_backtest(
        [_row(actual_side="down")],
        candidate_key="spot_timing_policy",
        fee_cfg=FeeModelConfig(maker_fee_bps=0, taker_fee_bps=0, curve_rate=0.0, curve_exponent=1.0, min_fee=0.0),
        slippage_bps=0.0,
        config=PredictedSideBacktestConfig(
            initial_balance=100.0,
            min_edge_to_trade=0.01,
            max_exposure_per_window_usd=100.0,
            max_balance_fraction_per_trade=0.12,
            min_trade_notional=0.0,
            include_trade_log=True,
        ),
    )
    trade = result["trade_log"][0]
    assert trade["cash_required"] == 12.0
    assert trade["net_pnl"] == -12.0
    assert result["final_balance"] == 88.0


def test_predicted_side_win_payout_matches_binary_price() -> None:
    result = run_predicted_side_backtest(
        [_row(actual_side="up")],
        candidate_key="spot_timing_policy",
        fee_cfg=FeeModelConfig(maker_fee_bps=0, taker_fee_bps=0, curve_rate=0.0, curve_exponent=1.0, min_fee=0.0),
        slippage_bps=0.0,
        config=PredictedSideBacktestConfig(
            initial_balance=100.0,
            min_edge_to_trade=0.01,
            max_exposure_per_window_usd=100.0,
            max_balance_fraction_per_trade=0.12,
            min_trade_notional=0.0,
            include_trade_log=True,
        ),
    )
    trade = result["trade_log"][0]
    assert trade["cash_required"] == 12.0
    assert trade["net_pnl"] == pytest.approx(18.0)
    assert result["final_balance"] == pytest.approx(118.0)


def test_predicted_side_uses_probability_direction_not_edge_optimal_side() -> None:
    result = run_predicted_side_backtest(
        [
            _row(
                actual_side="up",
                spot_timing_policy_p_up=0.55,
                up_entry_price=0.8,
                down_entry_price=0.2,
                up_entry_price_norm=0.8,
                down_entry_price_norm=0.2,
            )
        ],
        candidate_key="spot_timing_policy",
        fee_cfg=FeeModelConfig(maker_fee_bps=0, taker_fee_bps=0, curve_rate=0.0, curve_exponent=1.0, min_fee=0.0),
        slippage_bps=0.0,
        config=PredictedSideBacktestConfig(
            initial_balance=100.0,
            min_edge_to_trade=0.01,
            max_exposure_per_window_usd=100.0,
            max_balance_fraction_per_trade=0.12,
            min_trade_notional=0.0,
            include_trade_log=True,
        ),
    )
    # Predicted side is still "up" because p_up >= 0.5, but this row is not traded
    # because the predicted-side contract is overpriced.
    assert result["signal_rows"] == 1
    assert result["signal_accuracy"] == 1.0
    assert result["trades_taken"] == 0
    assert result["skip_reason_counts"]["net_edge_below_threshold"] == 1


def test_predicted_side_flat_trade_notional_bypasses_balance_fraction() -> None:
    result = run_predicted_side_backtest(
        [_row(actual_side="up")],
        candidate_key="spot_timing_policy",
        fee_cfg=FeeModelConfig(maker_fee_bps=0, taker_fee_bps=0, curve_rate=0.0, curve_exponent=1.0, min_fee=0.0),
        slippage_bps=0.0,
        config=PredictedSideBacktestConfig(
            initial_balance=100.0,
            min_edge_to_trade=0.01,
            max_exposure_per_window_usd=100.0,
            max_balance_fraction_per_trade=0.12,
            flat_trade_notional=5.0,
            min_trade_notional=0.0,
            include_trade_log=True,
        ),
    )
    trade = result["trade_log"][0]
    assert trade["cash_required"] == 5.0
    assert result["final_balance"] == pytest.approx(107.5)
