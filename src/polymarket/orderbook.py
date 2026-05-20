from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class OutcomeBook:
    token_id: str
    best_bid: float
    best_ask: float
    midpoint: float
    spread: float
    topk_imbalance: float
    best_bid_size: float = 0.0
    best_ask_size: float = 0.0
    top3_bid_size: float = 0.0
    top3_ask_size: float = 0.0
    bid_level_count: int = 0
    ask_level_count: int = 0
    top5_bid_size: float = 0.0
    top5_ask_size: float = 0.0
    top10_bid_size: float = 0.0
    top10_ask_size: float = 0.0
    top5_imbalance: float = 0.0
    top10_imbalance: float = 0.0
    bid_vwap_top3: float = 0.0
    ask_vwap_top3: float = 0.0
    bid_depth_1c: float = 0.0
    ask_depth_1c: float = 0.0
    microprice: float = 0.0


@dataclass
class MarketBooks:
    up: OutcomeBook
    down: OutcomeBook


def _to_levels(raw_levels: Any) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for lvl in raw_levels or []:
        try:
            out.append({"price": float(lvl["price"]), "size": float(lvl["size"])})
        except Exception:
            continue
    return out


def _sorted_bids(levels: list[dict[str, float]]) -> list[dict[str, float]]:
    return sorted(levels, key=lambda x: (x["price"], x["size"]), reverse=True)


def _sorted_asks(levels: list[dict[str, float]]) -> list[dict[str, float]]:
    return sorted(levels, key=lambda x: (x["price"], x["size"]))


def _topk_imbalance(bids: list[dict[str, float]], asks: list[dict[str, float]], k: int = 3) -> float:
    bid_sz = sum(x["size"] for x in bids[:k])
    ask_sz = sum(x["size"] for x in asks[:k])
    den = bid_sz + ask_sz
    if den <= 0:
        return 0.0
    return (bid_sz - ask_sz) / den


def _sum_size(levels: list[dict[str, float]], k: int) -> float:
    return sum(x["size"] for x in levels[:k])


def _vwap(levels: list[dict[str, float]], k: int) -> float:
    selected = levels[:k]
    size = sum(x["size"] for x in selected)
    if size <= 0.0:
        return 0.0
    return sum(x["price"] * x["size"] for x in selected) / size


def _depth_within_abs(levels: list[dict[str, float]], *, anchor: float, max_abs_distance: float, side: str) -> float:
    eps = 1e-12
    if side == "bid":
        return sum(x["size"] for x in levels if anchor - x["price"] <= max_abs_distance + eps)
    return sum(x["size"] for x in levels if x["price"] - anchor <= max_abs_distance + eps)


def _microprice(best_bid: float, best_ask: float, best_bid_size: float, best_ask_size: float) -> float:
    den = best_bid_size + best_ask_size
    if den <= 0.0:
        return (best_bid + best_ask) / 2.0
    return (best_ask * best_bid_size + best_bid * best_ask_size) / den


def normalize_outcome_book(token_id: str, raw_book: dict[str, Any]) -> OutcomeBook:
    bids = _sorted_bids(_to_levels(raw_book.get("bids")))
    asks = _sorted_asks(_to_levels(raw_book.get("asks")))

    best_bid = bids[0]["price"] if bids else 0.0
    best_ask = asks[0]["price"] if asks else 1.0
    midpoint = (best_bid + best_ask) / 2.0
    spread = max(0.0, best_ask - best_bid)
    best_bid_size = bids[0]["size"] if bids else 0.0
    best_ask_size = asks[0]["size"] if asks else 0.0
    top3_bid_size = sum(x["size"] for x in bids[:3])
    top3_ask_size = sum(x["size"] for x in asks[:3])
    top5_bid_size = _sum_size(bids, 5)
    top5_ask_size = _sum_size(asks, 5)
    top10_bid_size = _sum_size(bids, 10)
    top10_ask_size = _sum_size(asks, 10)

    return OutcomeBook(
        token_id=token_id,
        best_bid=best_bid,
        best_ask=best_ask,
        midpoint=midpoint,
        spread=spread,
        topk_imbalance=_topk_imbalance(bids, asks, k=3),
        best_bid_size=best_bid_size,
        best_ask_size=best_ask_size,
        top3_bid_size=top3_bid_size,
        top3_ask_size=top3_ask_size,
        bid_level_count=len(bids),
        ask_level_count=len(asks),
        top5_bid_size=top5_bid_size,
        top5_ask_size=top5_ask_size,
        top10_bid_size=top10_bid_size,
        top10_ask_size=top10_ask_size,
        top5_imbalance=_topk_imbalance(bids, asks, k=5),
        top10_imbalance=_topk_imbalance(bids, asks, k=10),
        bid_vwap_top3=_vwap(bids, 3),
        ask_vwap_top3=_vwap(asks, 3),
        bid_depth_1c=_depth_within_abs(bids, anchor=best_bid, max_abs_distance=0.01, side="bid"),
        ask_depth_1c=_depth_within_abs(asks, anchor=best_ask, max_abs_distance=0.01, side="ask"),
        microprice=_microprice(best_bid, best_ask, best_bid_size, best_ask_size),
    )
