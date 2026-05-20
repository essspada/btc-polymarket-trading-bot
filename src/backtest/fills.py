from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.polymarket.execution import OrderIntent


@dataclass
class FillResult:
    filled: bool
    fill_price: float
    fill_size: float


def simulate_fill(intent: OrderIntent, rng: np.random.Generator, spread: float) -> FillResult:
    if intent.order_type.upper() == "TAKER":
        return FillResult(True, intent.price, intent.size)

    # Maker fill probability decreases when spread is wide.
    fill_prob = float(np.clip(0.75 - 8.0 * spread, 0.05, 0.9))
    filled = bool(rng.random() < fill_prob)
    return FillResult(filled, intent.price, intent.size if filled else 0.0)
