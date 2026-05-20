from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.polymarket.book_quality import BookQualityConfig, assess_market_books
from src.polymarket.market_discovery import DiscoveredMarket
from src.polymarket.orderbook import MarketBooks, OutcomeBook
from src.runtime import (
    persistence as persistence_mod,
    snapshots as snapshots_mod,
)


class _FakeClob:
    def __init__(self, books: dict[str, dict]) -> None:
        self.books = books

    def get_book(self, token_id: str) -> dict:
        return self.books[token_id]


def _market() -> DiscoveredMarket:
    start = datetime(2026, 5, 6, 12, 0, tzinfo=UTC)
    return DiscoveredMarket(
        event_title="BTC Up or Down",
        event_slug="btc-updown-5m-test",
        series_slug="btc-up-or-down-5m",
        market_id="m1",
        market_slug="btc-updown-5m-test",
        question="BTC up?",
        resolution_source="chainlink",
        start_time=start,
        end_time=start + timedelta(minutes=5),
        accepting_orders=True,
        up_token_id="up-token",
        down_token_id="down-token",
    )


def test_build_market_books_snapshot_keeps_raw_l2_topn() -> None:
    market = _market()
    clob = _FakeClob(
        {
            "up-token": {
                "bids": [{"price": "0.48", "size": "5"}, {"price": "0.47", "size": "7"}],
                "asks": [{"price": "0.51", "size": "11"}, {"price": "0.50", "size": "13"}],
            },
            "down-token": {
                "bids": [{"price": "0.49", "size": "3"}],
                "asks": [{"price": "0.52", "size": "17"}],
            },
        }
    )

    books, raw, error = snapshots_mod.build_market_books_snapshot(clob, market, raw_l2_depth=2)

    assert error is None
    assert books is not None
    assert books.up.best_ask == 0.50
    assert raw is not None
    assert raw["schema"] == "raw_clob_l2_topn_v1"
    assert raw["up"]["bids"][0] == {"price": 0.48, "size": 5.0}
    assert raw["up"]["asks"][0] == {"price": 0.50, "size": 13.0}


def test_append_monitor_observation_writes_model_book_and_raw_payload(tmp_path: Path) -> None:
    market = _market()
    now = market.start_time + timedelta(seconds=120)
    books = MarketBooks(
        up=OutcomeBook(
            "up-token",
            0.48,
            0.50,
            0.49,
            0.02,
            0.1,
            best_bid_size=5.0,
            best_ask_size=7.0,
            top3_bid_size=12.0,
            top3_ask_size=14.0,
        ),
        down=OutcomeBook(
            "down-token",
            0.50,
            0.52,
            0.51,
            0.02,
            -0.1,
            best_bid_size=6.0,
            best_ask_size=8.0,
            top3_bid_size=11.0,
            top3_ask_size=16.0,
        ),
    )
    quality = assess_market_books(books, BookQualityConfig())
    path = tmp_path / "monitor_observations_5m.jsonl"

    persistence_mod.append_monitor_observation(
        path,
        now=now,
        market=market,
        seconds_from_start=120.0,
        seconds_to_expiry=180.0,
        model_source="log240_only",
        timing_policy_active=True,
        timing_action="wait",
        timing_selection={"action": "wait", "reason": "before_first_stage"},
        current_stage_index=None,
        books=books,
        raw_orderbook={"schema": "raw_clob_l2_topn_v1", "depth_limit": 2},
        quality=quality,
        fee_bps_map={"up-token": 720.0, "down-token": 720.0},
        spot_ctx=None,
        oracle_basis=None,
        oracle_source="disabled",
        oracle_external_feed=False,
        model_weight=0.35,
        calibration_shift=0.01,
        probs={
            "p_up": 0.61,
            "p_down": 0.39,
            "predicted_side": "up",
            "confidence": 0.61,
            "model_used": "spot_consensus_blend",
            "model_p_up": 0.60,
            "market_p_up": 0.50,
            "proxy_p_up": 0.52,
            "candidate_probabilities": {"spot_logistic_online": 0.61},
        },
        candidate_models_payload={"spot_logistic_online": {"p_up": 0.61, "decision_action": "trade"}},
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["schema"] == "monitor_observation_v1"
    assert row["timing_action"] == "wait"
    assert row["p_up"] == 0.61
    assert row["up_best_ask"] == 0.50
    assert row["up_top3_ask_size"] == 14.0
    assert row["book_quality_ok"] is True
    assert row["raw_orderbook"]["schema"] == "raw_clob_l2_topn_v1"
    assert row["candidate_models"]["spot_logistic_online"]["decision_action"] == "trade"
