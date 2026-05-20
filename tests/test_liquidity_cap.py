from src.polymarket.fees import FeeModelConfig
from src.polymarket.orderbook import MarketBooks, OutcomeBook
from src.strategy.signals import decide_trade


def test_liquidity_cap_applies_to_intent_size() -> None:
    # Top 3 ask size is explicitly set to 10.0 shares
    up = OutcomeBook("up", 0.49, 0.51, 0.50, 0.02, 0.0, top3_ask_size=10.0)
    down = OutcomeBook("down", 0.49, 0.51, 0.50, 0.02, 0.0, top3_ask_size=10.0)
    books = MarketBooks(up=up, down=down)
    
    # Try to execute a large trade (e.g., $100 exposure)
    # At price ~0.51, this would normally be ~196 shares
    d = decide_trade(
        market_slug="m",
        books=books,
        p_up=0.60,  # strong signal
        min_edge_to_trade=0.01,
        min_edge_for_taker=0.01,
        max_exposure_usd=100.0,
        maker_preference=False,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_rate=0.072),
        taker_slippage_bps=12,
        maker_slippage_bps=3,
        allowed_order_types=["taker"],
        sizing_mode="flat_fraction", # add this
    )
    
    assert d.action == "trade"
    assert d.intent is not None
    # 95% of 10.0 is 9.5 shares. The intent size should be strictly bounded.
    assert d.intent.size <= 9.5
