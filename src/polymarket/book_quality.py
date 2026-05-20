from __future__ import annotations

from dataclasses import dataclass

from .orderbook import MarketBooks


@dataclass
class BookQualityConfig:
    sentinel_bid: float = 0.01
    sentinel_ask: float = 0.99
    max_spread: float = 0.08
    max_midpoint_sum_deviation: float = 0.05
    min_midpoint: float = 0.02
    max_midpoint: float = 0.98

    @classmethod
    def from_dict(cls, raw: dict[str, float] | None) -> BookQualityConfig:
        data = raw if isinstance(raw, dict) else {}
        return cls(
            sentinel_bid=float(data.get("sentinel_bid", 0.01)),
            sentinel_ask=float(data.get("sentinel_ask", 0.99)),
            max_spread=float(data.get("max_spread", 0.08)),
            max_midpoint_sum_deviation=float(data.get("max_midpoint_sum_deviation", 0.05)),
            min_midpoint=float(data.get("min_midpoint", 0.02)),
            max_midpoint=float(data.get("max_midpoint", 0.98)),
        )


@dataclass
class BookQualityResult:
    ok: bool
    score: float
    reason: str
    flags: list[str]
    metrics: dict[str, float]


def _flag_score(flags: list[str]) -> float:
    if not flags:
        return 1.0
    penalty = 0.0
    for flag in flags:
        if flag in {"sentinel_quotes", "locked_or_crossed", "midpoint_sum_mismatch"}:
            penalty += 0.45
        elif flag in {"wide_spread", "midpoint_out_of_band"}:
            penalty += 0.2
        else:
            penalty += 0.1
    return max(0.0, 1.0 - penalty)


def assess_market_books(books: MarketBooks, cfg: BookQualityConfig) -> BookQualityResult:
    up_bid = float(books.up.best_bid)
    up_ask = float(books.up.best_ask)
    down_bid = float(books.down.best_bid)
    down_ask = float(books.down.best_ask)

    up_mid = float(books.up.midpoint)
    down_mid = float(books.down.midpoint)

    flags: list[str] = []

    if not (0.0 < up_bid < 1.0 and 0.0 < up_ask < 1.0 and 0.0 < down_bid < 1.0 and 0.0 < down_ask < 1.0):
        flags.append("non_probability_quotes")

    if up_ask <= up_bid or down_ask <= down_bid:
        flags.append("locked_or_crossed")

    if (
        up_bid <= cfg.sentinel_bid
        or down_bid <= cfg.sentinel_bid
        or up_ask >= cfg.sentinel_ask
        or down_ask >= cfg.sentinel_ask
    ):
        flags.append("sentinel_quotes")

    if books.up.spread > cfg.max_spread or books.down.spread > cfg.max_spread:
        flags.append("wide_spread")

    if not (cfg.min_midpoint <= up_mid <= cfg.max_midpoint and cfg.min_midpoint <= down_mid <= cfg.max_midpoint):
        flags.append("midpoint_out_of_band")

    midpoint_sum = up_mid + down_mid
    if abs(midpoint_sum - 1.0) > cfg.max_midpoint_sum_deviation:
        flags.append("midpoint_sum_mismatch")

    score = _flag_score(flags)
    ok = len(flags) == 0
    reason = "ok" if ok else flags[0]

    return BookQualityResult(
        ok=ok,
        score=score,
        reason=reason,
        flags=flags,
        metrics={
            "up_bid": up_bid,
            "up_ask": up_ask,
            "down_bid": down_bid,
            "down_ask": down_ask,
            "up_spread": float(books.up.spread),
            "down_spread": float(books.down.spread),
            "up_midpoint": up_mid,
            "down_midpoint": down_mid,
            "midpoint_sum": midpoint_sum,
        },
    )
