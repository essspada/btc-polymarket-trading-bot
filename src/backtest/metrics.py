from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BacktestMetrics:
    trades: int
    win_rate: float
    net_pnl: float
    mean_pnl: float
    sharpe_like: float
    max_drawdown: float


def compute_metrics(pnls: list[float]) -> BacktestMetrics:
    if not pnls:
        return BacktestMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)

    arr = np.asarray(pnls, dtype=float)
    eq = arr.cumsum()
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    max_dd = float(dd.min()) if len(dd) else 0.0

    wins = float((arr > 0).mean())
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    sharpe_like = float(mean / std) if std > 0 else 0.0

    return BacktestMetrics(
        trades=len(arr),
        win_rate=wins,
        net_pnl=float(arr.sum()),
        mean_pnl=mean,
        sharpe_like=sharpe_like,
        max_drawdown=max_dd,
    )
