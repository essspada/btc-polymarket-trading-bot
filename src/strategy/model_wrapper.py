from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class ModelContext:
    up_mid: float
    down_mid: float
    up_imbalance: float
    down_imbalance: float
    seconds_to_expiry: float


class MarkovProxyModel:
    """
    Minimal viable adaptation of the Markov idea for short-horizon binary markets.

    This is a lightweight regime-transition scorer over midpoint deltas + book imbalance.
    """

    def __init__(self, default_confidence: float = 0.5) -> None:
        self.default_confidence = float(default_confidence)
        self.mid_hist: deque[float] = deque(maxlen=256)
        self.state_hist: deque[int] = deque(maxlen=256)  # -1, 0, +1

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _state_from_return(ret: float, eps: float = 1e-4) -> int:
        if ret > eps:
            return 1
        if ret < -eps:
            return -1
        return 0

    def _p_up_from_markov(self) -> float:
        if len(self.state_hist) < 6:
            return self.default_confidence

        # 3x3 transition matrix over states [-1,0,1]
        idx = {-1: 0, 0: 1, 1: 2}
        trans = np.ones((3, 3), dtype=float)  # Laplace smoothing

        arr = list(self.state_hist)
        for a, b in zip(arr[:-1], arr[1:], strict=True):
            trans[idx[a], idx[b]] += 1.0

        trans /= trans.sum(axis=1, keepdims=True)
        cur = arr[-1]

        # Approximate P(up next) from current state row.
        return float(trans[idx[cur], idx[1]])

    def predict_proba(self, ctx: ModelContext) -> float:
        # synthetic "market midpoint": probability midpoint of Up token
        mid = float(np.clip(ctx.up_mid, 0.001, 0.999))
        self.mid_hist.append(mid)

        if len(self.mid_hist) >= 2:
            ret = self.mid_hist[-1] - self.mid_hist[-2]
            self.state_hist.append(self._state_from_return(ret))

        p_markov = self._p_up_from_markov()

        # microstructure features
        imbalance = float(np.clip(ctx.up_imbalance - ctx.down_imbalance, -1.0, 1.0))
        score = (
            1.6 * (p_markov - 0.5)
            + 0.9 * imbalance
        )

        p_up = self._sigmoid(score)
        return float(np.clip(p_up, 0.01, 0.99))
