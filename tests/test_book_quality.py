from src.polymarket.book_quality import BookQualityConfig, assess_market_books
from src.polymarket.orderbook import MarketBooks, OutcomeBook


def test_book_quality_flags_sentinel_quotes() -> None:
    books = MarketBooks(
        up=OutcomeBook("up", 0.01, 0.99, 0.50, 0.98, 0.0),
        down=OutcomeBook("down", 0.01, 0.99, 0.50, 0.98, 0.0),
    )
    result = assess_market_books(books, BookQualityConfig())
    assert result.ok is False
    assert "sentinel_quotes" in result.flags


def test_book_quality_accepts_clean_books() -> None:
    books = MarketBooks(
        up=OutcomeBook("up", 0.49, 0.50, 0.495, 0.01, 0.0),
        down=OutcomeBook("down", 0.50, 0.51, 0.505, 0.01, 0.0),
    )
    result = assess_market_books(books, BookQualityConfig())
    assert result.ok is True
    assert result.reason == "ok"
