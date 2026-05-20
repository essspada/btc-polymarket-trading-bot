from datetime import UTC, datetime

from src.polymarket.market_discovery import build_btc_5m_event_slug, extract_resolved_outcome_side


def test_build_btc_5m_event_slug_aligns_to_5m() -> None:
    dt = datetime(2026, 2, 27, 22, 7, 29, tzinfo=UTC)
    aligned = int(dt.timestamp())
    aligned -= aligned % 300
    assert build_btc_5m_event_slug(dt) == f"btc-updown-5m-{aligned}"


def test_extract_resolved_outcome_side_up() -> None:
    market = {
        "closed": True,
        "outcomes": '["Up", "Down"]',
        "outcomePrices": '["1", "0"]',
    }
    assert extract_resolved_outcome_side(market) == "up"


def test_extract_resolved_outcome_side_none_when_not_resolved() -> None:
    market = {
        "closed": True,
        "outcomes": '["Up", "Down"]',
        "outcomePrices": '["0.62", "0.38"]',
    }
    assert extract_resolved_outcome_side(market) is None
