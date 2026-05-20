from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from src.polymarket.fees import FeeModelConfig, compute_trade_fee
from src.polymarket.orderbook import MarketBooks, OutcomeBook

_EPS = 1e-9


@dataclass
class ShadowOrder:
    order_id: str
    market_id: str
    token_id: str
    side: str  # normalized order side: "buy" or "sell"
    price: float
    size: float
    queue_position: float
    created_at_utc: str
    prediction_side: str
    order_type: str = "MAKER"
    status: str = "OPEN"  # OPEN, PARTIAL, FILLED, EXPIRED, PARTIAL_EXPIRED
    filled_size: float = 0.0
    pnl_observed: float = 0.0
    taker_pnl_observed: float = 0.0
    maker_fee_observed: float = 0.0
    taker_fee_observed: float = 0.0
    taker_fee_bps: float = 0.0
    taker_fee_rate: float = 0.0
    maker_fee_rate: float = 0.0
    min_fee: float = 0.0
    taker_slippage_bps: float = 12.0
    spread: float = 0.0
    entry_best_bid: float = 0.0
    entry_best_ask: float = 0.0
    entry_best_bid_size: float = 0.0
    entry_best_ask_size: float = 0.0
    taker_entry_price: float = 0.0
    update_count: int = 0
    first_update_utc: str | None = None
    last_update_utc: str | None = None
    latency_seconds_to_first_update: float | None = None
    last_best_bid: float = 0.0
    last_best_ask: float = 0.0
    last_best_bid_size: float = 0.0
    last_best_ask_size: float = 0.0
    max_observed_cross: float = 0.0
    partial_fill_events: int = 0
    actual_side: str | None = None
    actual_win: bool | None = None


class ShadowAuditor:
    def __init__(self) -> None:
        self.open_orders: dict[str, ShadowOrder] = {}
        self.completed_orders: list[ShadowOrder] = []

    def _hash_order(self, market_id: str, token_id: str, ts: str, side: str, price: float) -> str:
        key = f"{market_id}:{token_id}:{ts}:{side}:{price:.6f}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    def place_virtual_maker_order(
        self,
        market_id: str,
        token_id: str,
        prediction_side: str,
        order_side: str,
        price: float,
        size: float,
        books: MarketBooks,
        taker_fee_bps: float | None = None,
        *,
        taker_fee_rate: float | None = None,
        maker_fee_rate: float | None = None,
        taker_slippage_bps: float = 12.0,
        min_fee: float = 0.0,
        order_type: str = "MAKER",
        now: datetime | None = None,
    ) -> str:
        """Record a virtual maker order and its entry orderbook/taker snapshot."""
        now_dt = _ensure_utc(now or datetime.now(UTC))
        now_iso = now_dt.isoformat()
        side = _normalize_order_side(order_side)
        pred_side = _normalize_prediction_side(prediction_side)
        order_id = self._hash_order(market_id, token_id, now_iso, side, float(price))
        book = _book_for_order(books=books, token_id=token_id, prediction_side=pred_side)

        queue_pos = 0.0
        if side == "buy":
            if float(price) <= float(book.best_bid) + _EPS:
                queue_pos = max(0.0, float(book.best_bid_size))
            taker_ref = float(book.best_ask) if _valid_binary_price(book.best_ask) else float(price)
            taker_entry_price = taker_ref * (1.0 + float(taker_slippage_bps) / 10_000.0)
        else:
            if float(price) >= float(book.best_ask) - _EPS:
                queue_pos = max(0.0, float(book.best_ask_size))
            taker_ref = float(book.best_bid) if _valid_binary_price(book.best_bid) else float(price)
            taker_entry_price = taker_ref * (1.0 - float(taker_slippage_bps) / 10_000.0)

        taker_rate = _effective_fee_rate(fee_bps=taker_fee_bps, fee_rate=taker_fee_rate)
        maker_rate = _effective_fee_rate(fee_bps=None, fee_rate=maker_fee_rate)
        taker_bps = float(taker_fee_bps) if taker_fee_bps is not None else float(taker_rate * 10_000.0)

        order = ShadowOrder(
            order_id=order_id,
            market_id=str(market_id),
            token_id=str(token_id),
            side=side,
            price=float(price),
            size=max(0.0, float(size)),
            queue_position=float(queue_pos),
            created_at_utc=now_iso,
            prediction_side=pred_side,
            order_type=str(order_type or "MAKER").upper(),
            status="OPEN",
            taker_fee_bps=float(taker_bps),
            taker_fee_rate=float(taker_rate),
            maker_fee_rate=float(maker_rate),
            min_fee=max(0.0, float(min_fee)),
            taker_slippage_bps=float(taker_slippage_bps),
            spread=float(book.spread),
            entry_best_bid=float(book.best_bid),
            entry_best_ask=float(book.best_ask),
            entry_best_bid_size=float(book.best_bid_size),
            entry_best_ask_size=float(book.best_ask_size),
            taker_entry_price=_clamp_binary_price(taker_entry_price),
            last_best_bid=float(book.best_bid),
            last_best_ask=float(book.best_ask),
            last_best_bid_size=float(book.best_bid_size),
            last_best_ask_size=float(book.best_ask_size),
        )
        self.open_orders[order_id] = order
        return order_id

    def on_orderbook_update(self, market_id: str, books: MarketBooks, now: datetime | None = None) -> None:
        """Update virtual maker orders from the latest orderbook snapshot."""
        now_dt = _ensure_utc(now or datetime.now(UTC))
        filled_ids: list[str] = []
        for order_id, order in list(self.open_orders.items()):
            if order.market_id != str(market_id):
                continue
            book = _book_for_order(books=books, token_id=order.token_id, prediction_side=order.prediction_side)
            _record_latency_and_snapshot(order=order, book=book, now=now_dt)

            if _is_crossed(order=order, book=book):
                order.filled_size = order.size
                order.status = "FILLED"
                filled_ids.append(order_id)
                continue

            partial_fill = _partial_fill_from_visible_queue(order=order, book=book)
            if partial_fill > order.filled_size + _EPS:
                order.filled_size = min(order.size, partial_fill)
                order.partial_fill_events += 1
                order.status = "FILLED" if order.filled_size >= order.size - _EPS else "PARTIAL"
                if order.status == "FILLED":
                    filled_ids.append(order_id)

        for order_id in filled_ids:
            order = self.open_orders.pop(order_id, None)
            if order is not None:
                self.completed_orders.append(order)

    def resolve_markets(self, resolved_market_id: str, actual_side: str) -> None:
        """Resolve all shadow orders for a market and calculate maker-vs-taker PnL."""
        for order_id, order in list(self.open_orders.items()):
            if order.market_id != str(resolved_market_id):
                continue
            if order.filled_size >= order.size - _EPS and order.size > 0.0:
                order.status = "FILLED"
            elif order.filled_size > _EPS:
                order.status = "PARTIAL_EXPIRED"
            else:
                order.status = "EXPIRED"
            self.completed_orders.append(self.open_orders.pop(order_id))

        actual = _normalize_prediction_side(actual_side)
        for order in self.completed_orders:
            if order.market_id == str(resolved_market_id):
                _resolve_order_pnl(order=order, actual_side=actual)

    def get_summary(self) -> dict[str, Any]:
        orders = self.completed_orders
        total = len(orders)
        filled = sum(1 for order in orders if order.status == "FILLED")
        partial = sum(1 for order in orders if order.status == "PARTIAL_EXPIRED" or (0.0 < order.filled_size < order.size))
        expired = sum(1 for order in orders if order.status == "EXPIRED")
        any_fill = sum(1 for order in orders if order.filled_size > _EPS)
        actual_wins = sum(1 for order in orders if order.actual_win is True)
        actual_losses = sum(1 for order in orders if order.actual_win is False)
        wins_filled = sum(1 for order in orders if order.actual_win is True and order.filled_size > _EPS)
        losses_filled = sum(1 for order in orders if order.actual_win is False and order.filled_size > _EPS)
        latencies = [
            float(order.latency_seconds_to_first_update)
            for order in orders
            if order.latency_seconds_to_first_update is not None
        ]

        maker_pnl = sum(order.pnl_observed for order in orders)
        taker_pnl = sum(order.taker_pnl_observed for order in orders)
        return {
            "total_orders": total,
            "open_orders": len(self.open_orders),
            "filled_orders": filled,
            "partial_orders": partial,
            "expired_orders": expired,
            "any_fill_orders": any_fill,
            "overall_fill_rate": any_fill / total if total > 0 else 0.0,
            "full_fill_rate": filled / total if total > 0 else 0.0,
            "win_fill_rate": wins_filled / actual_wins if actual_wins > 0 else 0.0,
            "loss_fill_rate": losses_filled / actual_losses if actual_losses > 0 else 0.0,
            "maker_pnl_observed": float(maker_pnl),
            "taker_pnl_observed": float(taker_pnl),
            "maker_vs_taker_pnl_delta": float(maker_pnl - taker_pnl),
            "avg_latency_seconds_to_first_update": (sum(latencies) / len(latencies)) if latencies else None,
            "partial_fill_events": sum(order.partial_fill_events for order in orders),
        }


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_datetime(value: str) -> datetime | None:
    try:
        return _ensure_utc(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
    except Exception:
        return None


def _normalize_order_side(value: str) -> str:
    side = str(value or "").strip().lower()
    if side in {"buy", "bid", "b"}:
        return "buy"
    if side in {"sell", "ask", "s"}:
        return "sell"
    return side or "buy"


def _normalize_prediction_side(value: str) -> str:
    side = str(value or "").strip().lower()
    if side in {"up", "yes", "1", "true"}:
        return "up"
    if side in {"down", "no", "0", "false"}:
        return "down"
    return side


def _valid_binary_price(value: float) -> bool:
    try:
        price = float(value)
    except Exception:
        return False
    return math.isfinite(price) and 0.0 < price < 1.0


def _clamp_binary_price(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.5
    return float(min(0.99, max(0.01, float(value))))


def _effective_fee_rate(*, fee_bps: float | None, fee_rate: float | None) -> float:
    if fee_rate is not None:
        return max(0.0, float(fee_rate))
    if fee_bps is not None:
        return max(0.0, float(fee_bps) / 10_000.0)
    return 0.0


def _book_for_order(*, books: MarketBooks, token_id: str, prediction_side: str) -> OutcomeBook:
    if str(token_id) == str(books.up.token_id):
        return books.up
    if str(token_id) == str(books.down.token_id):
        return books.down
    return books.up if _normalize_prediction_side(prediction_side) == "up" else books.down


def _record_latency_and_snapshot(*, order: ShadowOrder, book: OutcomeBook, now: datetime) -> None:
    order.update_count += 1
    now_iso = _ensure_utc(now).isoformat()
    if order.first_update_utc is None:
        order.first_update_utc = now_iso
        created = _parse_datetime(order.created_at_utc)
        if created is not None:
            order.latency_seconds_to_first_update = max(0.0, (_ensure_utc(now) - created).total_seconds())
    order.last_update_utc = now_iso
    order.last_best_bid = float(book.best_bid)
    order.last_best_ask = float(book.best_ask)
    order.last_best_bid_size = float(book.best_bid_size)
    order.last_best_ask_size = float(book.best_ask_size)
    if order.side == "buy":
        order.max_observed_cross = max(
            order.max_observed_cross,
            max(0.0, float(order.price) - float(book.best_ask)),
            max(0.0, float(order.price) - float(book.best_bid)),
        )
    else:
        order.max_observed_cross = max(
            order.max_observed_cross,
            max(0.0, float(book.best_bid) - float(order.price)),
            max(0.0, float(book.best_ask) - float(order.price)),
        )


def _is_crossed(*, order: ShadowOrder, book: OutcomeBook) -> bool:
    if order.side == "buy":
        return float(book.best_ask) <= float(order.price) + _EPS or float(book.best_bid) < float(order.price) - _EPS
    if order.side == "sell":
        return float(book.best_bid) >= float(order.price) - _EPS or float(book.best_ask) > float(order.price) + _EPS
    return False


def _partial_fill_from_visible_queue(*, order: ShadowOrder, book: OutcomeBook) -> float:
    if order.queue_position <= _EPS:
        return order.filled_size
    if order.side == "buy" and abs(float(book.best_bid) - float(order.price)) <= _EPS:
        queue_consumed = max(0.0, float(order.queue_position) - max(0.0, float(book.best_bid_size)))
        return min(float(order.size), max(float(order.filled_size), queue_consumed))
    if order.side == "sell" and abs(float(book.best_ask) - float(order.price)) <= _EPS:
        queue_consumed = max(0.0, float(order.queue_position) - max(0.0, float(book.best_ask_size)))
        return min(float(order.size), max(float(order.filled_size), queue_consumed))
    return order.filled_size


def _resolve_order_pnl(*, order: ShadowOrder, actual_side: str) -> None:
    actual = _normalize_prediction_side(actual_side)
    order.actual_side = actual
    order.actual_win = order.prediction_side == actual
    payout = 1.0 if order.actual_win else 0.0

    maker_fee_cfg = FeeModelConfig(min_fee=0.0, maker_fee_rate=float(order.maker_fee_rate), taker_fee_rate=0.0)
    taker_fee_cfg = FeeModelConfig(min_fee=float(order.min_fee), maker_fee_rate=0.0, taker_fee_rate=float(order.taker_fee_rate))

    maker_size = max(0.0, float(order.filled_size))
    taker_size = max(0.0, float(order.size))
    order.maker_fee_observed = (
        compute_trade_fee(price=float(order.price), size=maker_size, is_taker=False, cfg=maker_fee_cfg)
        if maker_size > _EPS
        else 0.0
    )
    order.taker_fee_observed = (
        compute_trade_fee(price=float(order.taker_entry_price), size=taker_size, is_taker=True, cfg=taker_fee_cfg)
        if taker_size > _EPS
        else 0.0
    )
    order.pnl_observed = _gross_pnl(order=order, payout=payout, entry_price=float(order.price), size=maker_size) - order.maker_fee_observed
    order.taker_pnl_observed = _gross_pnl(
        order=order,
        payout=payout,
        entry_price=float(order.taker_entry_price),
        size=taker_size,
    ) - order.taker_fee_observed


def _gross_pnl(*, order: ShadowOrder, payout: float, entry_price: float, size: float) -> float:
    if size <= _EPS:
        return 0.0
    if order.side == "sell":
        return float(size) * (float(entry_price) - float(payout))
    return float(size) * (float(payout) - float(entry_price))
