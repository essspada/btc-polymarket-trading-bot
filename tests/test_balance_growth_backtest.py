from scripts.backtest_balance_growth import BalanceRiskConfig, run_balance_backtest
from src.polymarket.fees import FeeModelConfig


def test_balance_backtest_can_reach_target_on_winning_sequence() -> None:
    rows = [
        {
            "market_id": "m1",
            "created_at": "2026-03-01T00:01:00+00:00",
            "actual_side": "up",
            "up_entry_price": 0.40,
            "down_entry_price": 0.60,
            "spot_consensus_blend_p_up": 0.72,
        },
        {
            "market_id": "m2",
            "created_at": "2026-03-01T00:06:00+00:00",
            "actual_side": "up",
            "up_entry_price": 0.35,
            "down_entry_price": 0.65,
            "spot_consensus_blend_p_up": 0.78,
        },
    ]
    out = run_balance_backtest(
        rows,
        candidate_key="spot_consensus_blend",
        fee_cfg=FeeModelConfig(maker_fee_bps=0.0, taker_fee_bps=0.0, curve_rate=0.0, min_fee=0.0),
        slippage_bps=0.0,
        risk_cfg=BalanceRiskConfig(
            initial_balance=100.0,
            target_balance=120.0,
            kelly_scale=1.0,
            max_fraction=0.5,
            min_fraction=0.01,
            min_trade_notional=1.0,
            daily_loss_limit_frac=0.5,
            soft_drawdown_frac=0.5,
            hard_drawdown_frac=0.9,
        ),
    )
    assert out["final_balance"] > 120.0
    assert out["target_hit"] is True
    assert out["trades_taken"] == 2


def test_balance_backtest_respects_daily_loss_limit() -> None:
    rows = [
        {
            "market_id": "m1",
            "created_at": "2026-03-01T00:01:00+00:00",
            "actual_side": "down",
            "up_entry_price": 0.40,
            "down_entry_price": 0.60,
            "spot_consensus_blend_p_up": 0.80,
        },
        {
            "market_id": "m2",
            "created_at": "2026-03-01T00:06:00+00:00",
            "actual_side": "down",
            "up_entry_price": 0.40,
            "down_entry_price": 0.60,
            "spot_consensus_blend_p_up": 0.80,
        },
    ]
    out = run_balance_backtest(
        rows,
        candidate_key="spot_consensus_blend",
        fee_cfg=FeeModelConfig(maker_fee_bps=0.0, taker_fee_bps=0.0, curve_rate=0.0, min_fee=0.0),
        slippage_bps=0.0,
        risk_cfg=BalanceRiskConfig(
            initial_balance=100.0,
            target_balance=1000.0,
            kelly_scale=1.0,
            max_fraction=0.5,
            min_fraction=0.01,
            min_trade_notional=1.0,
            daily_loss_limit_frac=0.10,
            soft_drawdown_frac=0.5,
            hard_drawdown_frac=0.9,
        ),
    )
    assert out["trades_taken"] == 1
    assert out["reason_counts"]["daily_loss_limit"] == 1


def test_balance_backtest_hard_drawdown_throttle_does_not_freeze_forever() -> None:
    rows = [
        {
            "market_id": "m1",
            "created_at": "2026-03-01T00:01:00+00:00",
            "actual_side": "down",
            "up_entry_price": 0.40,
            "down_entry_price": 0.60,
            "spot_consensus_blend_p_up": 0.80,
        },
        {
            "market_id": "m2",
            "created_at": "2026-03-01T00:06:00+00:00",
            "actual_side": "up",
            "up_entry_price": 0.40,
            "down_entry_price": 0.60,
            "spot_consensus_blend_p_up": 0.80,
        },
    ]
    out = run_balance_backtest(
        rows,
        candidate_key="spot_consensus_blend",
        fee_cfg=FeeModelConfig(maker_fee_bps=0.0, taker_fee_bps=0.0, curve_rate=0.0, min_fee=0.0),
        slippage_bps=0.0,
        risk_cfg=BalanceRiskConfig(
            initial_balance=100.0,
            target_balance=1000.0,
            kelly_scale=1.0,
            max_fraction=0.5,
            min_fraction=0.01,
            min_trade_notional=1.0,
            daily_loss_limit_frac=0.90,
            soft_drawdown_frac=0.10,
            hard_drawdown_frac=0.10,
            hard_drawdown_mode="throttle",
            hard_drawdown_size_mult=0.2,
            hard_drawdown_cooldown_windows=0,
        ),
    )
    assert out["trades_taken"] == 2
    assert out["reason_counts"].get("hard_drawdown_stop", 0) == 0


def test_balance_backtest_can_stop_after_target() -> None:
    rows = [
        {
            "market_id": "m1",
            "created_at": "2026-03-01T00:01:00+00:00",
            "actual_side": "up",
            "up_entry_price": 0.40,
            "down_entry_price": 0.60,
            "spot_consensus_blend_p_up": 0.80,
        },
        {
            "market_id": "m2",
            "created_at": "2026-03-01T00:06:00+00:00",
            "actual_side": "up",
            "up_entry_price": 0.40,
            "down_entry_price": 0.60,
            "spot_consensus_blend_p_up": 0.80,
        },
        {
            "market_id": "m3",
            "created_at": "2026-03-01T00:11:00+00:00",
            "actual_side": "up",
            "up_entry_price": 0.40,
            "down_entry_price": 0.60,
            "spot_consensus_blend_p_up": 0.80,
        },
    ]
    out = run_balance_backtest(
        rows,
        candidate_key="spot_consensus_blend",
        fee_cfg=FeeModelConfig(maker_fee_bps=0.0, taker_fee_bps=0.0, curve_rate=0.0, min_fee=0.0),
        slippage_bps=0.0,
        risk_cfg=BalanceRiskConfig(
            initial_balance=100.0,
            target_balance=120.0,
            kelly_scale=1.0,
            max_fraction=0.5,
            min_fraction=0.01,
            min_trade_notional=1.0,
            daily_loss_limit_frac=0.90,
            soft_drawdown_frac=0.90,
            hard_drawdown_frac=0.95,
            stop_after_target=True,
        ),
    )
    assert out["target_hit"] is True
    assert out["trades_taken"] == 1
    assert out["reason_counts"]["after_target_stop"] == 2


def test_balance_backtest_sublinear_sizing_caps_growth() -> None:
    rows = [
        {
            "market_id": "m1",
            "created_at": "2026-03-01T00:01:00+00:00",
            "actual_side": "up",
            "up_entry_price": 0.20,
            "down_entry_price": 0.80,
            "spot_consensus_blend_p_up": 0.80,
        },
        {
            "market_id": "m2",
            "created_at": "2026-03-01T00:06:00+00:00",
            "actual_side": "up",
            "up_entry_price": 0.20,
            "down_entry_price": 0.80,
            "spot_consensus_blend_p_up": 0.80,
        },
    ]
    base_cfg = dict(
        initial_balance=100.0,
        target_balance=1000.0,
        kelly_scale=1.0,
        max_fraction=0.5,
        min_fraction=0.01,
        min_trade_notional=1.0,
        daily_loss_limit_frac=0.90,
        soft_drawdown_frac=0.90,
        hard_drawdown_frac=0.95,
    )
    out_linear = run_balance_backtest(
        rows,
        candidate_key="spot_consensus_blend",
        fee_cfg=FeeModelConfig(maker_fee_bps=0.0, taker_fee_bps=0.0, curve_rate=0.0, min_fee=0.0),
        slippage_bps=0.0,
        risk_cfg=BalanceRiskConfig(**base_cfg, balance_growth_exponent=1.0),
    )
    out_sublinear = run_balance_backtest(
        rows,
        candidate_key="spot_consensus_blend",
        fee_cfg=FeeModelConfig(maker_fee_bps=0.0, taker_fee_bps=0.0, curve_rate=0.0, min_fee=0.0),
        slippage_bps=0.0,
        risk_cfg=BalanceRiskConfig(**base_cfg, balance_growth_exponent=0.0),
    )
    assert out_sublinear["final_balance"] < out_linear["final_balance"]


def test_temperature_scaled_kelly_reduces_position_aggression() -> None:
    rows = [
        {
            "market_id": "m1",
            "created_at": "2026-03-01T00:01:00+00:00",
            "actual_side": "up",
            "up_entry_price": 0.40,
            "down_entry_price": 0.60,
            "spot_consensus_blend_p_up": 0.80,
        },
        {
            "market_id": "m2",
            "created_at": "2026-03-01T00:06:00+00:00",
            "actual_side": "up",
            "up_entry_price": 0.40,
            "down_entry_price": 0.60,
            "spot_consensus_blend_p_up": 0.80,
        },
    ]
    base_cfg = dict(
        initial_balance=100.0,
        target_balance=1000.0,
        kelly_scale=1.0,
        max_fraction=0.5,
        min_fraction=0.01,
        min_trade_notional=1.0,
        daily_loss_limit_frac=0.90,
        soft_drawdown_frac=0.90,
        hard_drawdown_frac=0.95,
    )
    out_raw = run_balance_backtest(
        rows,
        candidate_key="spot_consensus_blend",
        fee_cfg=FeeModelConfig(maker_fee_bps=0.0, taker_fee_bps=0.0, curve_rate=0.0, min_fee=0.0),
        slippage_bps=0.0,
        risk_cfg=BalanceRiskConfig(**base_cfg, kelly_probability_temperature=1.0),
    )
    out_cool = run_balance_backtest(
        rows,
        candidate_key="spot_consensus_blend",
        fee_cfg=FeeModelConfig(maker_fee_bps=0.0, taker_fee_bps=0.0, curve_rate=0.0, min_fee=0.0),
        slippage_bps=0.0,
        risk_cfg=BalanceRiskConfig(**base_cfg, kelly_probability_temperature=2.0),
    )
    assert out_cool["final_balance"] < out_raw["final_balance"]
