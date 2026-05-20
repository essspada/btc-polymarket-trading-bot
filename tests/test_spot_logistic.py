from src.strategy.spot_logistic import (
    SpotLogisticConfig,
    build_causal_spot_logistic_probability_map,
    row_key,
)


def _row(ts: str, market_id: str, actual_side: str, ret_open: float, ret_1m: float) -> dict:
    return {
        "market_id": market_id,
        "created_at": ts,
        "actual_side": actual_side,
        "spot_return_bps_from_open": ret_open,
        "spot_recent_return_1m_bps": ret_1m,
        "spot_recent_vol_5m_bps": 6.0,
        "seconds_to_expiry": 120.0,
        "market_p_up": 0.5,
        "proxy_p_up": 0.5,
    }


def test_causal_prob_map_excludes_future_rows_when_seed_overlaps_dataset() -> None:
    rows = [
        _row("2026-03-12T00:00:10+00:00", "m1", "up", 15.0, 5.0),
        _row("2026-03-12T00:01:10+00:00", "m2", "down", -15.0, -5.0),
        _row("2026-03-12T00:02:10+00:00", "m3", "up", 12.0, 4.0),
    ]
    cfg = SpotLogisticConfig(lookback=20, min_train_samples=2, min_class_samples=1)

    probs = build_causal_spot_logistic_probability_map(rows, cfg, seed_rows=rows)

    assert probs.get(row_key(rows[0])) is None
    assert probs.get(row_key(rows[1])) is None
    assert probs.get(row_key(rows[2])) is not None


def test_causal_prob_map_can_warm_start_from_older_seed_rows() -> None:
    seed_rows = [
        _row("2026-03-11T23:58:10+00:00", "s1", "up", 10.0, 3.0),
        _row("2026-03-11T23:59:10+00:00", "s2", "down", -10.0, -3.0),
    ]
    live_rows = [
        _row("2026-03-12T00:00:10+00:00", "m1", "up", 8.0, 2.0),
    ]
    cfg = SpotLogisticConfig(lookback=20, min_train_samples=2, min_class_samples=1)

    probs = build_causal_spot_logistic_probability_map(live_rows, cfg, seed_rows=seed_rows)

    assert probs.get(row_key(live_rows[0])) is not None
