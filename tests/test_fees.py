import pytest

from src.polymarket.fees import FeeModelConfig, compute_trade_fee


def test_fee_has_min_floor() -> None:
    cfg = FeeModelConfig(min_fee=0.0001)
    fee = compute_trade_fee(price=0.5, size=0.001, is_taker=True, cfg=cfg)
    assert fee >= 0.0001


def test_fee_increases_with_size() -> None:
    cfg = FeeModelConfig(min_fee=0.0)
    fee_small = compute_trade_fee(price=0.45, size=10, is_taker=True, cfg=cfg)
    fee_big = compute_trade_fee(price=0.45, size=100, is_taker=True, cfg=cfg)
    assert fee_big > fee_small


def test_explicit_taker_fee_rate_uses_polymarket_v2_formula() -> None:
    cfg = FeeModelConfig(min_fee=0.0, taker_fee_rate=0.072)
    fee = compute_trade_fee(price=0.5, size=200, is_taker=True, cfg=cfg)
    assert fee == pytest.approx(3.6)


def test_taker_fee_bps_maps_to_v2_fee_rate() -> None:
    cfg = FeeModelConfig(min_fee=0.0, taker_fee_bps=720)
    fee = compute_trade_fee(price=0.5, size=200, is_taker=True, cfg=cfg)
    assert fee == pytest.approx(3.6)


def test_zero_maker_fee_has_no_floor() -> None:
    cfg = FeeModelConfig(min_fee=0.0001, maker_fee_rate=0.0)
    assert compute_trade_fee(price=0.5, size=200, is_taker=False, cfg=cfg) == 0.0
