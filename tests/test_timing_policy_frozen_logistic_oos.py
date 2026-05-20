from __future__ import annotations

from scripts.report_timing_policy_frozen_logistic_oos import _with_frozen_logistic
from src.strategy.spot_logistic import SpotLogisticConfig


def _row(*, created_at: str, actual_side: str | None, spot_ret: float, spot_1m: float, vol: float, market: float = 0.5, proxy: float = 0.5) -> dict:
    row = {
        "market_id": "m",
        "created_at": created_at,
        "spot_return_bps_from_open": spot_ret,
        "spot_recent_return_1m_bps": spot_1m,
        "spot_recent_vol_5m_bps": vol,
        "seconds_to_expiry": 60.0,
        "market_p_up": market,
        "proxy_p_up": proxy,
        "spot_logistic_online_p_up": None,
    }
    if actual_side is not None:
        row["actual_side"] = actual_side
    return row


def test_with_frozen_logistic_uses_train_rows_only() -> None:
    train = []
    for idx in range(6):
        train.append(_row(created_at=f"2026-03-01T00:00:0{idx}+00:00", actual_side="up", spot_ret=10 + idx, spot_1m=5, vol=4))
    for idx in range(6):
        train.append(_row(created_at=f"2026-03-01T00:01:0{idx}+00:00", actual_side="down", spot_ret=-10 - idx, spot_1m=-5, vol=4))
    test = [
        _row(created_at="2026-03-02T00:00:00+00:00", actual_side=None, spot_ret=12, spot_1m=4, vol=4),
        _row(created_at="2026-03-02T00:05:00+00:00", actual_side=None, spot_ret=-12, spot_1m=-4, vol=4),
    ]
    out = _with_frozen_logistic(train, test, SpotLogisticConfig(lookback=240, min_train_samples=10, min_class_samples=2))
    assert len(out) == 2
    assert out[0]["spot_logistic_online_p_up"] is not None
    assert out[1]["spot_logistic_online_p_up"] is not None
    assert out[0]["spot_logistic_online_p_up"] > 0.5
    assert out[1]["spot_logistic_online_p_up"] < 0.5
