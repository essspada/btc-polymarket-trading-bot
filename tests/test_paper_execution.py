from datetime import UTC, datetime

from src.execution.paper_execution import PaperExecutionEngine
from src.polymarket.execution import OrderIntent


def _maker_intent(price: float = 0.42) -> OrderIntent:
    return OrderIntent(
        market_slug="m",
        token_id="up_token",
        side="BUY",
        order_type="MAKER",
        price=price,
        size=100.0,
        expected_edge=0.05,
    )


def test_maker_fill_draw_is_stable_when_only_size_changes() -> None:
    engine = PaperExecutionEngine()
    ts = datetime(2026, 3, 12, 0, 0, tzinfo=UTC)
    large = _maker_intent()
    small = OrderIntent(
        market_slug=large.market_slug,
        token_id=large.token_id,
        side=large.side,
        order_type=large.order_type,
        price=large.price,
        size=large.size * 0.25,
        expected_edge=large.expected_edge,
    )

    large_trade = engine.open_trade(
        intent=large,
        market_id="m1",
        created_at=ts,
        up_token_id="up_token",
        down_token_id="down_token",
        spread=0.01,
        seconds_to_expiry=180.0,
    )
    small_trade = engine.open_trade(
        intent=small,
        market_id="m1",
        created_at=ts,
        up_token_id="up_token",
        down_token_id="down_token",
        spread=0.01,
        seconds_to_expiry=180.0,
    )

    assert small_trade.trade_id == large_trade.trade_id
    assert small_trade.filled is large_trade.filled
    assert small_trade.fill_probability == large_trade.fill_probability
    assert small_trade.fill_size in {0.0, small.size}


def test_conservative_fill_profile_lowers_probability() -> None:
    base = PaperExecutionEngine()
    conservative = PaperExecutionEngine(
        maker_fill_floor=0.0,
        maker_fill_cap=0.35,
        maker_fill_base=0.45,
        maker_fill_spread_penalty=12.0,
        maker_fill_late_penalty_90=0.25,
        maker_fill_late_penalty_45=0.20,
    )
    assert conservative._maker_fill_probability(spread=0.01, seconds_to_expiry=180) < base._maker_fill_probability(
        spread=0.01, seconds_to_expiry=180
    )


def test_zero_cap_prevents_maker_fill() -> None:
    engine = PaperExecutionEngine(maker_fill_floor=0.0, maker_fill_cap=0.0, maker_fill_base=0.0)
    trade = engine.open_trade(
        intent=_maker_intent(),
        market_id="m1",
        created_at=datetime(2026, 3, 12, 0, 0, tzinfo=UTC),
        up_token_id="up_token",
        down_token_id="down_token",
        spread=0.01,
        seconds_to_expiry=180.0,
    )
    assert trade.fill_probability == 0.0
    assert trade.filled is False
    assert trade.fill_size == 0.0
