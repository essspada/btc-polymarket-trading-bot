from src.polymarket.fees import FeeModelConfig
from src.polymarket.orderbook import MarketBooks, OutcomeBook
from src.strategy.confirmation_gate import apply_confirmation_gate
from src.strategy.signals import decide_trade


def _books() -> MarketBooks:
    return MarketBooks(
        up=OutcomeBook("up", 0.50, 0.51, 0.505, 0.01, 0.0, top3_bid_size=260.0, top3_ask_size=260.0),
        down=OutcomeBook("down", 0.49, 0.50, 0.495, 0.01, 0.0, top3_bid_size=260.0, top3_ask_size=260.0),
    )


def _decision():
    return decide_trade(
        market_slug="m",
        books=_books(),
        p_up=0.56,
        min_edge_to_trade=0.03,
        min_edge_for_taker=0.20,
        max_exposure_usd=100,
        maker_preference=True,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        taker_slippage_bps=0.0,
        maker_slippage_bps=0.0,
        allowed_order_types=["maker"],
        sizing_mode="flat_fraction",
    )


def _gate(**overrides):
    cfg = {
        "enabled": True,
        "candidate_key": "proxy_logistic_market_blend",
        "min_edge": 0.03,
        "require_same_side": True,
        "max_spread": 0.02,
        "min_price": 0.05,
        "max_price": 0.95,
    }
    cfg.update(overrides)
    return cfg


def test_confirmation_gate_passes_when_proxy_agrees_and_has_edge() -> None:
    decision, payload = apply_confirmation_gate(
        decision=_decision(),
        books=_books(),
        candidate_models={"proxy_logistic_market_blend": {"p_up": 0.56}},
        gate_cfg=_gate(),
    )
    assert decision.action == "trade"
    assert decision.intent is not None
    assert payload["confirmation_gate_passed"] is True
    assert payload["confirmation_gate_edge"] == 0.06000000000000005


def test_confirmation_gate_blocks_proxy_side_disagreement() -> None:
    decision, payload = apply_confirmation_gate(
        decision=_decision(),
        books=_books(),
        candidate_models={"proxy_logistic_market_blend": {"p_up": 0.44}},
        gate_cfg=_gate(),
    )
    assert decision.action == "no_trade"
    assert decision.intent is None
    assert decision.reason == "confirmation_gate_side_disagreement"
    assert payload["confirmation_gate_reason"] == "side_disagreement"


def test_confirmation_gate_blocks_low_proxy_edge() -> None:
    decision, payload = apply_confirmation_gate(
        decision=_decision(),
        books=_books(),
        candidate_models={"proxy_logistic_market_blend": {"p_up": 0.52}},
        gate_cfg=_gate(),
    )
    assert decision.action == "no_trade"
    assert decision.intent is None
    assert decision.reason == "confirmation_gate_edge_below_min"
    assert payload["confirmation_gate_reason"] == "edge_below_min"


def test_confirmation_gate_disabled_preserves_decision() -> None:
    base = _decision()
    decision, payload = apply_confirmation_gate(
        decision=base,
        books=_books(),
        candidate_models={},
        gate_cfg={"enabled": False},
    )
    assert decision == base
    assert payload["confirmation_gate_enabled"] is False


def test_confirmation_gate_blocks_thin_top3_ask_size() -> None:
    thin_books = MarketBooks(
        up=OutcomeBook("up", 0.50, 0.51, 0.505, 0.01, 0.0, top3_bid_size=260.0, top3_ask_size=120.0),
        down=OutcomeBook("down", 0.49, 0.50, 0.495, 0.01, 0.0, top3_bid_size=260.0, top3_ask_size=260.0),
    )
    taker_decision = decide_trade(
        market_slug="m",
        books=thin_books,
        p_up=0.56,
        min_edge_to_trade=0.03,
        min_edge_for_taker=0.03,
        max_exposure_usd=100,
        maker_preference=False,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        taker_slippage_bps=0.0,
        maker_slippage_bps=0.0,
        allowed_order_types=["taker"],
        sizing_mode="flat_fraction",
    )
    decision, payload = apply_confirmation_gate(
        decision=taker_decision,
        books=thin_books,
        candidate_models={"proxy_logistic_market_blend": {"p_up": 0.56}},
        gate_cfg=_gate(min_top3_ask_size=150.0),
    )
    assert decision.action == "no_trade"
    assert decision.intent is None
    assert decision.reason == "confirmation_gate_top3_ask_size_below_min"
    assert payload["confirmation_gate_reason"] == "top3_ask_size_below_min"
    assert payload["confirmation_gate_top3_size"] == 120.0
