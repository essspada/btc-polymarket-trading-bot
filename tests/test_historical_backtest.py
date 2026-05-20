from scripts.backtest_historical_btc5m import _normalize_pair, _trade_policy_metrics
from src.polymarket.fees import FeeModelConfig


def test_normalize_pair_scales_to_one() -> None:
    up, down, total = _normalize_pair(0.42, 0.58)
    assert total == 1.0
    assert up == 0.42
    assert down == 0.58

    up2, down2, total2 = _normalize_pair(0.21, 0.31)
    assert round((up2 or 0.0) + (down2 or 0.0), 8) == 1.0
    assert total2 == 0.52


def test_trade_policy_can_buy_opposite_of_threshold_side_when_price_is_cheap() -> None:
    rows = [
        {
            "market_id": "m1",
            "created_at": "2026-03-01T00:00:00+00:00",
            "actual_side": "up",
            "up_entry_price": 0.35,
            "down_entry_price": 0.65,
        }
    ]

    summary = _trade_policy_metrics(
        rows,
        lambda row: 0.45,
        FeeModelConfig(maker_fee_bps=0.0, taker_fee_bps=0.0, curve_rate=0.0, min_fee=0.0),
        slippage_bps=0.0,
        min_edge_to_trade=0.0,
    )

    assert summary["rows"] == 1
    assert summary["raw_accuracy"] == 0.0
    assert summary["trades_taken"] == 1
    assert summary["selected_side_accuracy"] == 1.0
    assert summary["net_pnl_sum"] > 0.0
