from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.backtest.fills import simulate_fill
from src.backtest.metrics import BacktestMetrics, compute_metrics
from src.polymarket.execution import OrderIntent


@dataclass
class BacktestStep:
    intent: OrderIntent
    resolved_side: str  # up | down
    up_token_id: str
    down_token_id: str
    spread: float


def run_simple_backtest(steps: list[BacktestStep], seed: int = 42) -> BacktestMetrics:
    rng = np.random.default_rng(seed)
    pnls: list[float] = []

    for s in steps:
        fill = simulate_fill(s.intent, rng=rng, spread=s.spread)
        if not fill.filled:
            continue

        resolved = str(s.resolved_side).strip().lower()
        if resolved not in {"up", "down"}:
            raise ValueError(f"Invalid resolved_side: {resolved}")
        
        if s.intent.token_id == s.up_token_id:
            payout = 1.0 if resolved == "up" else 0.0
        elif s.intent.token_id == s.down_token_id:
            payout = 1.0 if resolved == "down" else 0.0
        else:
            raise ValueError(
                f"Token ID {s.intent.token_id} doesn't match up ({s.up_token_id}) "
                f"or down ({s.down_token_id})"
            )
        pnl = fill.fill_size * (payout - fill.fill_price)
        pnls.append(float(pnl))

    return compute_metrics(pnls)
