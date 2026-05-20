from src.polymarket.fees import FeeModelConfig
from src.polymarket.orderbook import MarketBooks, OutcomeBook
from src.strategy.signals import decide_trade


def _books() -> MarketBooks:
    up = OutcomeBook(
        token_id="up",
        best_bid=0.48,
        best_ask=0.49,
        midpoint=0.485,
        spread=0.01,
        topk_imbalance=0.2,
    )
    down = OutcomeBook(
        token_id="down",
        best_bid=0.51,
        best_ask=0.52,
        midpoint=0.515,
        spread=0.01,
        topk_imbalance=-0.2,
    )
    return MarketBooks(up=up, down=down)


def test_decision_trades_when_edge_is_positive() -> None:
    d = decide_trade(
        market_slug="m",
        books=_books(),
        p_up=0.58,
        min_edge_to_trade=0.001,
        min_edge_for_taker=0.001,
        max_exposure_usd=100,
        maker_preference=True,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=50),
        taker_slippage_bps=1,
        maker_slippage_bps=1,
    )
    assert d.action == "trade"
    assert d.intent is not None


def test_decision_blocks_when_edge_is_too_low() -> None:
    d = decide_trade(
        market_slug="m",
        books=_books(),
        p_up=0.50,
        min_edge_to_trade=0.02,
        min_edge_for_taker=0.02,
        max_exposure_usd=100,
        maker_preference=True,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=1000),
        taker_slippage_bps=10,
        maker_slippage_bps=10,
    )
    assert d.action == "no_trade"
