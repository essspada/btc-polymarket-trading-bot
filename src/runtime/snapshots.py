"""Order-book and spot-context snapshot helpers.

Converts raw CLOB book responses into normalized `MarketBooks` plus the
auxiliary payloads (raw L2 snapshot, depth payload, spot context) that the
runtime serializes alongside every prediction or trade observation.
"""
from __future__ import annotations

from typing import Any

from src.polymarket.clients.clob_client import ClobClient
from src.polymarket.market_discovery import DiscoveredMarket
from src.polymarket.orderbook import MarketBooks, normalize_outcome_book


def _raw_l2_levels(raw_levels: Any, *, depth: int) -> list[dict[str, float]]:
    if depth <= 0:
        return []
    out: list[dict[str, float]] = []
    for lvl in raw_levels or []:
        try:
            out.append({"price": float(lvl["price"]), "size": float(lvl["size"])})
        except Exception:
            continue
    return out


def raw_book_snapshot(raw_book: dict[str, Any], *, token_id: str, depth: int) -> dict[str, Any] | None:
    if depth <= 0:
        return None
    bids = sorted(_raw_l2_levels(raw_book.get("bids"), depth=depth), key=lambda x: (x["price"], x["size"]), reverse=True)[
        :depth
    ]
    asks = sorted(_raw_l2_levels(raw_book.get("asks"), depth=depth), key=lambda x: (x["price"], x["size"]))[:depth]
    return {
        "token_id": token_id,
        "depth_limit": int(depth),
        "bid_count_logged": len(bids),
        "ask_count_logged": len(asks),
        "bids": bids,
        "asks": asks,
    }


def build_market_books_snapshot(
    clob: ClobClient,
    m: DiscoveredMarket,
    *,
    raw_l2_depth: int = 0,
) -> tuple[MarketBooks | None, dict[str, Any] | None, str | None]:
    try:
        raw_up = clob.get_book(m.up_token_id)
        raw_down = clob.get_book(m.down_token_id)
    except Exception as exc:
        return None, None, str(exc)

    up = normalize_outcome_book(m.up_token_id, raw_up)
    down = normalize_outcome_book(m.down_token_id, raw_down)
    raw_snapshot: dict[str, Any] | None = None
    if raw_l2_depth > 0:
        raw_snapshot = {
            "schema": "raw_clob_l2_topn_v1",
            "depth_limit": int(raw_l2_depth),
            "up": raw_book_snapshot(raw_up, token_id=m.up_token_id, depth=raw_l2_depth),
            "down": raw_book_snapshot(raw_down, token_id=m.down_token_id, depth=raw_l2_depth),
        }
    return MarketBooks(up=up, down=down), raw_snapshot, None


def build_market_books(clob: ClobClient, m: DiscoveredMarket) -> tuple[MarketBooks | None, str | None]:
    books, _raw_snapshot, error = build_market_books_snapshot(clob, m)
    return books, error


def book_depth_payload(books: MarketBooks) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for prefix, book in (("up", books.up), ("down", books.down)):
        payload.update(
            {
                f"{prefix}_bid_level_count": int(getattr(book, "bid_level_count", 0) or 0),
                f"{prefix}_ask_level_count": int(getattr(book, "ask_level_count", 0) or 0),
                f"{prefix}_top5_bid_size": float(getattr(book, "top5_bid_size", 0.0) or 0.0),
                f"{prefix}_top5_ask_size": float(getattr(book, "top5_ask_size", 0.0) or 0.0),
                f"{prefix}_top10_bid_size": float(getattr(book, "top10_bid_size", 0.0) or 0.0),
                f"{prefix}_top10_ask_size": float(getattr(book, "top10_ask_size", 0.0) or 0.0),
                f"{prefix}_top5_imbalance": float(getattr(book, "top5_imbalance", 0.0) or 0.0),
                f"{prefix}_top10_imbalance": float(getattr(book, "top10_imbalance", 0.0) or 0.0),
                f"{prefix}_bid_vwap_top3": float(getattr(book, "bid_vwap_top3", 0.0) or 0.0),
                f"{prefix}_ask_vwap_top3": float(getattr(book, "ask_vwap_top3", 0.0) or 0.0),
                f"{prefix}_bid_depth_1c": float(getattr(book, "bid_depth_1c", 0.0) or 0.0),
                f"{prefix}_ask_depth_1c": float(getattr(book, "ask_depth_1c", 0.0) or 0.0),
                f"{prefix}_microprice": float(getattr(book, "microprice", 0.0) or 0.0),
            }
        )
    return payload


def spot_context_payload(spot_ctx: Any | None) -> dict[str, Any] | None:
    if spot_ctx is None:
        return None
    return {
        "ts_utc": getattr(spot_ctx, "ts_utc", None),
        "symbol": getattr(spot_ctx, "symbol", None),
        "spot_price_now": getattr(spot_ctx, "spot_price_now", None),
        "spot_window_open_price": getattr(spot_ctx, "spot_window_open_price", None),
        "spot_return_bps_from_open": getattr(spot_ctx, "spot_return_bps_from_open", None),
        "spot_recent_return_1m_bps": getattr(spot_ctx, "spot_recent_return_1m_bps", None),
        "spot_recent_vol_5m_bps": getattr(spot_ctx, "spot_recent_vol_5m_bps", None),
    }
