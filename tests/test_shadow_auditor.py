from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.execution.shadow_auditor import ShadowAuditor
from src.polymarket.orderbook import MarketBooks, OutcomeBook


def _books(
    *,
    up_bid: float = 0.49,
    up_ask: float = 0.51,
    up_bid_size: float = 100.0,
    up_ask_size: float = 100.0,
) -> MarketBooks:
    up = OutcomeBook(
        token_id="up-token",
        best_bid=up_bid,
        best_ask=up_ask,
        midpoint=(up_bid + up_ask) / 2.0,
        spread=max(0.0, up_ask - up_bid),
        topk_imbalance=0.0,
        best_bid_size=up_bid_size,
        best_ask_size=up_ask_size,
    )
    down = OutcomeBook(
        token_id="down-token",
        best_bid=1.0 - up_ask,
        best_ask=1.0 - up_bid,
        midpoint=0.5,
        spread=max(0.0, up_ask - up_bid),
        topk_imbalance=0.0,
        best_bid_size=100.0,
        best_ask_size=100.0,
    )
    return MarketBooks(up=up, down=down)


def test_shadow_auditor_normalizes_buy_and_fills_crossed_order() -> None:
    auditor = ShadowAuditor()
    t0 = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)
    order_id = auditor.place_virtual_maker_order(
        market_id="m",
        token_id="up-token",
        prediction_side="UP",
        order_side="BUY",
        price=0.49,
        size=10.0,
        books=_books(),
        taker_fee_rate=0.072,
        taker_slippage_bps=12.0,
        min_fee=0.0,
        now=t0,
    )

    auditor.on_orderbook_update(
        "m",
        _books(up_bid=0.48, up_ask=0.49, up_bid_size=50.0),
        now=t0 + timedelta(seconds=2),
    )

    assert order_id not in auditor.open_orders
    order = auditor.completed_orders[0]
    assert order.side == "buy"
    assert order.prediction_side == "up"
    assert order.status == "FILLED"
    assert order.filled_size == pytest.approx(10.0)
    assert order.latency_seconds_to_first_update == pytest.approx(2.0)


def test_shadow_auditor_tracks_partial_fill_from_queue_shrink() -> None:
    auditor = ShadowAuditor()
    t0 = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)
    order_id = auditor.place_virtual_maker_order(
        market_id="m",
        token_id="up-token",
        prediction_side="up",
        order_side="buy",
        price=0.49,
        size=10.0,
        books=_books(up_bid_size=100.0),
        taker_fee_rate=0.072,
        taker_slippage_bps=12.0,
        min_fee=0.0,
        now=t0,
    )

    auditor.on_orderbook_update(
        "m",
        _books(up_bid=0.49, up_ask=0.51, up_bid_size=94.0),
        now=t0 + timedelta(seconds=1),
    )

    order = auditor.open_orders[order_id]
    assert order.status == "PARTIAL"
    assert order.filled_size == pytest.approx(6.0)
    assert order.partial_fill_events == 1

    auditor.resolve_markets("m", "down")
    resolved = auditor.completed_orders[0]
    assert resolved.status == "PARTIAL_EXPIRED"
    assert resolved.pnl_observed == pytest.approx(6.0 * (0.0 - 0.49))


def test_shadow_auditor_resolves_correct_pnl_and_fees() -> None:
    auditor = ShadowAuditor()
    t0 = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)
    auditor.place_virtual_maker_order(
        market_id="m",
        token_id="up-token",
        prediction_side="up",
        order_side="BUY",
        price=0.49,
        size=10.0,
        books=_books(),
        taker_fee_rate=0.072,
        taker_slippage_bps=12.0,
        min_fee=0.0,
        now=t0,
    )
    auditor.on_orderbook_update(
        "m",
        _books(up_bid=0.48, up_ask=0.49, up_bid_size=50.0),
        now=t0 + timedelta(seconds=1),
    )
    auditor.resolve_markets("m", "up")

    order = auditor.completed_orders[0]
    taker_price = 0.51 * 1.0012
    taker_fee = 10.0 * 0.072 * taker_price * (1.0 - taker_price)
    assert order.pnl_observed == pytest.approx(10.0 * (1.0 - 0.49))
    assert order.taker_fee_observed == pytest.approx(taker_fee)
    assert order.taker_pnl_observed == pytest.approx(10.0 * (1.0 - taker_price) - taker_fee)

    summary = auditor.get_summary()
    assert summary["filled_orders"] == 1
    assert summary["win_fill_rate"] == 1.0
    assert summary["maker_pnl_observed"] == pytest.approx(order.pnl_observed)
