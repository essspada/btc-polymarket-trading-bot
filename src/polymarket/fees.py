from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class FeeModelConfig:
    maker_fee_bps: float = 0.0
    taker_fee_bps: float = 1000.0
    curve_rate: float = 0.25
    curve_exponent: float = 2.0
    min_fee: float = 0.0001
    maker_fee_rate: float | None = None
    taker_fee_rate: float | None = None


def _validate_fee_inputs(price: float, size: float) -> None:
    if not (0.0 <= price <= 1.0):
        raise ValueError(f"Price must be in [0, 1], got {price}")
    if size < 0:
        raise ValueError(f"Size must be non-negative, got {size}")
    if not math.isfinite(price) or not math.isfinite(size):
        raise ValueError("Price and size must be finite numbers")


def _v2_binary_fee(price: float, size: float, fee_rate: float) -> float:
    _validate_fee_inputs(price, size)
    if not math.isfinite(fee_rate):
        raise ValueError(f"fee_rate must be finite, got {fee_rate}")
    if fee_rate <= 0.0:
        return 0.0
    # Polymarket CLOB V2 binary fee formula: shares * feeRate * price * (1 - price).
    return float(size) * float(fee_rate) * float(price) * (1.0 - float(price))


def compute_trade_fee(price: float, size: float, is_taker: bool, cfg: FeeModelConfig) -> float:
    """
    Polymarket CLOB V2 binary fee model.

    Notes:
    - If *_fee_rate is set, it is used directly.
    - Otherwise *_fee_bps is mapped to feeRate as bps / 10_000.
    - Makers are not charged platform fees when their effective rate is zero.
    """
    if not (0.0 <= price <= 1.0):
        raise ValueError(f"Price must be in [0, 1], got {price}")
    if size <= 0:
        raise ValueError(f"Size must be positive, got {size}")

    explicit_rate = cfg.taker_fee_rate if is_taker else cfg.maker_fee_rate
    if explicit_rate is not None:
        rate = float(explicit_rate)
    else:
        bps = cfg.taker_fee_bps if is_taker else cfg.maker_fee_bps
        if not math.isfinite(float(bps)):
            raise ValueError(f"fee_bps must be finite, got {bps}")
        rate = float(bps) / 10_000.0

    fee = _v2_binary_fee(price=price, size=size, fee_rate=rate)
    if not is_taker:
        return float(fee)
    return float(max(float(cfg.min_fee), fee))
