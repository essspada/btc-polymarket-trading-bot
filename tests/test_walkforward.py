from src.backtest.walkforward import WalkForwardConfig, run_walkforward_backtest
from src.polymarket.fees import FeeModelConfig


def test_walkforward_runs_and_is_causal() -> None:
    rows = []
    for i in range(80):
        actual_side = "up" if i % 2 == 0 else "down"
        p_up = 0.55 if actual_side == "up" else 0.45
        rows.append(
            {
                "created_at": f"2026-01-01T00:{i:02d}:00+00:00",
                "actual_side": actual_side,
                "p_up": p_up,
                "market_p_up": 0.50,
                "proxy_p_up": p_up,
                "model_p_up": p_up,
                "up_best_ask": 0.49,
                "down_best_ask": 0.51,
                "seconds_to_expiry": 250,
            }
        )

    payload = run_walkforward_backtest(
        rows=rows,
        wf_cfg=WalkForwardConfig(lookback=40, min_train_samples=20, min_edge_to_trade=0.001, min_edge_for_taker=0.001),
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0),
        taker_slippage_bps=0.0,
    )

    assert payload["rows_valid"] == 80
    assert payload["classification"]["calibrated_accuracy"] >= 0.5
    assert payload["trading"]["trades_taken"] > 0


def test_walkforward_can_filter_dirty_books_for_trading() -> None:
    rows = []
    for i in range(20):
        rows.append(
            {
                "created_at": f"2026-01-01T00:{i:02d}:00+00:00",
                "actual_side": "up" if i % 2 == 0 else "down",
                "p_up": 0.55 if i % 2 == 0 else 0.45,
                "market_p_up": 0.50,
                "proxy_p_up": 0.50,
                "model_p_up": 0.50,
                "up_best_bid": 0.01,
                "up_best_ask": 0.99,
                "down_best_bid": 0.01,
                "down_best_ask": 0.99,
                "up_spread": 0.98,
                "down_spread": 0.98,
                "book_quality_ok": False,
                "seconds_to_expiry": 250,
            }
        )

    payload = run_walkforward_backtest(
        rows=rows,
        wf_cfg=WalkForwardConfig(
            lookback=10,
            min_train_samples=5,
            min_edge_to_trade=0.001,
            min_edge_for_taker=0.001,
            require_book_quality=True,
            max_outcome_spread=0.25,
        ),
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0),
        taker_slippage_bps=0.0,
    )

    assert payload["rows_valid"] == 20
    assert payload["trading"]["trading_rows_considered"] == 0
    assert payload["trading"]["trades_taken"] == 0
