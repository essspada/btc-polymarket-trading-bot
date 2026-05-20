from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from .clients.gamma_client import GammaClient, parse_json_list_field


@dataclass
class DiscoveredMarket:
    event_title: str
    event_slug: str
    series_slug: str | None
    market_id: str
    market_slug: str
    question: str
    resolution_source: str
    start_time: datetime
    end_time: datetime
    accepting_orders: bool
    up_token_id: str
    down_token_id: str


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _extract_outcome_tokens(market: dict[str, Any]) -> dict[str, str] | None:
    outcomes = [str(x).strip().lower() for x in parse_json_list_field(market.get("outcomes"))]
    token_ids = [str(x) for x in parse_json_list_field(market.get("clobTokenIds"))]
    if len(outcomes) != 2 or len(token_ids) != 2:
        return None

    mapping = dict(zip(outcomes, token_ids, strict=True))
    up = mapping.get("up") or mapping.get("yes")
    down = mapping.get("down") or mapping.get("no")
    if not up or not down:
        return None
    return {"up": up, "down": down}


def _derive_start_time(ev: dict[str, Any], market: dict[str, Any], end_time: datetime) -> datetime:
    for key_source in (
        market.get("eventStartTime"),
        market.get("startTime"),
        ev.get("startTime"),
        ev.get("eventStartTime"),
        market.get("startDate"),
        ev.get("startDate"),
    ):
        if key_source:
            return _parse_dt(str(key_source))
    # Last fallback: infer as 5-minute window.
    return end_time - timedelta(minutes=5)


def _build_discovered_market(ev: dict[str, Any], market: dict[str, Any]) -> DiscoveredMarket | None:
    outcome_tokens = _extract_outcome_tokens(market)
    if not outcome_tokens:
        return None

    end_raw = market.get("endDate") or ev.get("endDate")
    if not end_raw:
        return None

    end_time = _parse_dt(str(end_raw))
    start_time = _derive_start_time(ev, market, end_time=end_time)

    return DiscoveredMarket(
        event_title=str(ev.get("title", "")),
        event_slug=str(ev.get("slug", "")),
        series_slug=ev.get("seriesSlug"),
        market_id=str(market.get("id")),
        market_slug=str(market.get("slug", "")),
        question=str(market.get("question", "")),
        resolution_source=str(market.get("resolutionSource", "")),
        start_time=start_time,
        end_time=end_time,
        accepting_orders=bool(market.get("acceptingOrders", False)),
        up_token_id=outcome_tokens["up"],
        down_token_id=outcome_tokens["down"],
    )


def _is_chainlink_source(source: str) -> bool:
    s = str(source).lower()
    return ("chain.link" in s) or ("chainlink" in s)


def build_btc_5m_event_slug(start_time: datetime) -> str:
    ts = int(start_time.astimezone(UTC).timestamp())
    aligned = ts - (ts % 300)
    return f"btc-updown-5m-{aligned}"


def discover_btc_5m_window_markets(
    gamma: GammaClient,
    now: datetime,
    lookback_windows: int = 1,
    lookahead_windows: int = 12,
    require_accepting_orders: bool = False,
    include_closed: bool = True,
    require_chainlink: bool = True,
    max_results: int = 50,
) -> list[DiscoveredMarket]:
    current_ts = int(now.timestamp())
    aligned_ts = current_ts - (current_ts % 300)
    base_start = datetime.fromtimestamp(aligned_ts, tz=UTC)

    out_by_id: dict[str, DiscoveredMarket] = {}
    for i in range(-lookback_windows, lookahead_windows + 1):
        start_time = base_start + timedelta(minutes=5 * i)
        slug = build_btc_5m_event_slug(start_time)
        events = gamma.get_events(slug=slug, limit=1)
        if not events:
            continue
        ev = events[0]
        if ev.get("seriesSlug") != "btc-up-or-down-5m":
            continue

        for market in ev.get("markets", []):
            item = _build_discovered_market(ev, market)
            if not item:
                continue
            is_closed = bool(market.get("closed", False))
            if require_accepting_orders and not item.accepting_orders:
                continue
            if not include_closed and is_closed:
                continue
            if require_chainlink and not _is_chainlink_source(item.resolution_source):
                continue
            out_by_id[item.market_id] = item

    out = sorted(out_by_id.values(), key=lambda x: x.start_time)
    return out[:max_results]


def extract_resolved_outcome_side(market: dict[str, Any], min_winner_price: float = 0.99) -> str | None:
    if not bool(market.get("closed", False)):
        return None

    outcomes = [str(x).strip().lower() for x in parse_json_list_field(market.get("outcomes"))]
    prices_raw = parse_json_list_field(market.get("outcomePrices"))
    if len(outcomes) != 2 or len(prices_raw) != 2:
        return None

    try:
        prices = [float(x) for x in prices_raw]
    except Exception:
        return None

    mapping = dict(zip(outcomes, prices, strict=True))
    up = mapping.get("up") if "up" in mapping else mapping.get("yes")
    down = mapping.get("down") if "down" in mapping else mapping.get("no")
    if up is None or down is None:
        return None

    # Use epsilon tolerance for float comparisons to avoid precision issues
    epsilon = 1e-6
    min_threshold = min_winner_price - epsilon
    max_threshold = (1.0 - min_winner_price) + epsilon
    
    if up >= min_threshold and down <= max_threshold:
        return "up"
    if down >= min_threshold and up <= max_threshold:
        return "down"
    return None


def discover_candidate_markets(gamma: GammaClient, cfg: dict[str, Any]) -> list[DiscoveredMarket]:
    pm_cfg = cfg.get("polymarket", {})
    disc_cfg = cfg.get("discovery", {})

    search_query = pm_cfg.get("search_query", "Bitcoin Up or Down")
    target_series = pm_cfg.get("target_series_slug")
    fallback_series = pm_cfg.get("fallback_series_slugs", [])
    allowed_series = [target_series] + [s for s in fallback_series if s]

    raw = gamma.public_search(search_query)
    events = raw.get("events", [])

    now = datetime.now(UTC)
    out: list[DiscoveredMarket] = []

    for ev in events:
        series_slug = ev.get("seriesSlug")
        if allowed_series and series_slug not in allowed_series:
            continue

        for market in ev.get("markets", []):
            accepting = bool(market.get("acceptingOrders", False))
            closed = bool(market.get("closed", False))
            item = _build_discovered_market(ev, market)
            if not item:
                continue
            end_time = item.end_time

            if disc_cfg.get("require_accepting_orders", True) and not accepting:
                continue
            if not disc_cfg.get("include_closed", False) and closed:
                continue
            if end_time <= now:
                continue

            if (
                disc_cfg.get("require_chainlink_for_5m", True)
                and series_slug == "btc-up-or-down-5m"
                and not _is_chainlink_source(str(market.get("resolutionSource", "")))
            ):
                continue

            out.append(item)

    out.sort(key=lambda x: x.end_time)
    max_results = int(disc_cfg.get("max_results", 50))
    return out[:max_results]
