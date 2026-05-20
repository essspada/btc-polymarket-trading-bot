from src.polymarket.orderbook import normalize_outcome_book


def test_normalize_book_mid_and_spread() -> None:
    raw = {
        "bids": [{"price": "0.40", "size": "100"}],
        "asks": [{"price": "0.44", "size": "120"}],
    }
    b = normalize_outcome_book("token", raw)
    assert abs(b.midpoint - 0.42) < 1e-9
    assert abs(b.spread - 0.04) < 1e-9


def test_topk_imbalance_sign() -> None:
    raw = {
        "bids": [{"price": "0.40", "size": "300"}],
        "asks": [{"price": "0.44", "size": "100"}],
    }
    b = normalize_outcome_book("token", raw)
    assert b.topk_imbalance > 0


def test_normalize_book_sorts_levels_to_best_prices() -> None:
    raw = {
        "bids": [
            {"price": "0.01", "size": "100"},
            {"price": "0.36", "size": "100"},
            {"price": "0.22", "size": "100"},
        ],
        "asks": [
            {"price": "0.99", "size": "100"},
            {"price": "0.37", "size": "100"},
            {"price": "0.61", "size": "100"},
        ],
    }
    b = normalize_outcome_book("token", raw)
    assert abs(b.best_bid - 0.36) < 1e-9
    assert abs(b.best_ask - 0.37) < 1e-9
    assert abs(b.midpoint - 0.365) < 1e-9
    assert abs(b.spread - 0.01) < 1e-9


def test_normalize_book_captures_depth_metrics() -> None:
    raw = {
        "bids": [
            {"price": "0.50", "size": "10"},
            {"price": "0.49", "size": "20"},
            {"price": "0.48", "size": "30"},
            {"price": "0.47", "size": "40"},
            {"price": "0.46", "size": "50"},
            {"price": "0.45", "size": "60"},
        ],
        "asks": [
            {"price": "0.51", "size": "15"},
            {"price": "0.52", "size": "25"},
            {"price": "0.53", "size": "35"},
            {"price": "0.54", "size": "45"},
            {"price": "0.55", "size": "55"},
            {"price": "0.56", "size": "65"},
        ],
    }

    b = normalize_outcome_book("token", raw)

    assert b.bid_level_count == 6
    assert b.ask_level_count == 6
    assert b.top5_bid_size == 150
    assert b.top5_ask_size == 175
    assert b.top10_bid_size == 210
    assert b.top10_ask_size == 240
    assert b.bid_depth_1c == 30
    assert b.ask_depth_1c == 40
    assert 0.50 <= b.microprice <= 0.51
    assert b.bid_vwap_top3 < b.best_bid
    assert b.ask_vwap_top3 > b.best_ask
