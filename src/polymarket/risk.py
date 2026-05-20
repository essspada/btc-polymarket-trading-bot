from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any


@dataclass
class RiskState:
    date_utc: str
    daily_pnl: float
    loss_streak: int
    cooldown_left: int
    open_positions: int
    drift_pause_windows: int = 0
    recent_accuracy: float = 0.0
    consecutive_wins: int = 0


class RiskManager:
    def __init__(
        self,
        max_exposure_per_window_usd: float,
        max_daily_loss_usd: float,
        cooldown_after_loss_streak: int,
        cooldown_windows: int,
        max_open_positions: int,
        drift_enabled: bool = False,
        drift_lookback: int = 50,
        drift_min_samples: int = 30,
        drift_min_accuracy: float = 0.46,
        drift_cooldown_windows: int = 6,
        stateful_regime: Mapping[str, Any] | None = None,
    ) -> None:
        now = datetime.now(UTC).date().isoformat()
        self.state = RiskState(now, 0.0, 0, 0, 0)
        self.max_exposure_per_window_usd = max_exposure_per_window_usd
        self.max_daily_loss_usd = max_daily_loss_usd
        self.cooldown_after_loss_streak = cooldown_after_loss_streak
        self.cooldown_windows = cooldown_windows
        self.max_open_positions = max_open_positions
        self.drift_enabled = bool(drift_enabled)
        self.drift_lookback = max(5, int(drift_lookback))
        self.drift_min_samples = max(5, int(drift_min_samples))
        self.drift_min_accuracy = float(drift_min_accuracy)
        self.drift_cooldown_windows = max(1, int(drift_cooldown_windows))
        regime = stateful_regime if isinstance(stateful_regime, Mapping) else {}
        self.stateful_enabled = bool(regime.get("enabled", False))
        self.stateful_drawdown_soft_pct = self._bounded_float(regime.get("drawdown_soft_pct"), default=0.08, lo=0.0, hi=1.0)
        self.stateful_drawdown_hard_pct = self._bounded_float(regime.get("drawdown_hard_pct"), default=0.18, lo=0.0, hi=1.0)
        self.stateful_loss_streak_soft = max(1, int(self._bounded_float(regime.get("loss_streak_soft"), default=2.0, lo=1.0, hi=20.0)))
        self.stateful_loss_streak_hard = max(
            self.stateful_loss_streak_soft,
            int(self._bounded_float(regime.get("loss_streak_hard"), default=4.0, lo=1.0, hi=50.0)),
        )
        self.stateful_daily_loss_soft_frac = self._bounded_float(regime.get("daily_loss_soft_frac"), default=0.08, lo=0.0, hi=1.0)
        self.stateful_daily_loss_hard_frac = self._bounded_float(regime.get("daily_loss_hard_frac"), default=0.18, lo=0.0, hi=1.0)
        self.stateful_min_multiplier = self._bounded_float(regime.get("min_multiplier"), default=0.35, lo=0.01, hi=1.0)
        self.stateful_recovery_wins_required = max(
            0, int(self._bounded_float(regime.get("recovery_wins_required"), default=2.0, lo=0.0, hi=20.0))
        )
        self.window_seconds = 300
        self._last_cooldown_check_ts: datetime | None = None
        self._recent_outcomes: deque[int] = deque(maxlen=self.drift_lookback)

    @staticmethod
    def _bounded_float(value: Any, *, default: float, lo: float, hi: float) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            out = float(default)
        if out < lo:
            return float(lo)
        if out > hi:
            return float(hi)
        return float(out)

    def _roll_day(self, ts: datetime) -> None:
        day = ts.date().isoformat()
        if day != self.state.date_utc:
            self.state = RiskState(day, 0.0, 0, 0, 0, 0, 0.0, 0)
            self._last_cooldown_check_ts = None
            self._recent_outcomes.clear()

    def _advance_cooldown(self, ts: datetime) -> None:
        if self.state.cooldown_left <= 0:
            self._last_cooldown_check_ts = ts
            self.state.drift_pause_windows = 0
            return
        if self._last_cooldown_check_ts is None:
            self._last_cooldown_check_ts = ts
            return

        elapsed_seconds = (ts - self._last_cooldown_check_ts).total_seconds()
        windows_passed = int(elapsed_seconds // self.window_seconds)
        if windows_passed <= 0:
            return

        self.state.cooldown_left = max(0, self.state.cooldown_left - windows_passed)
        self.state.drift_pause_windows = self.state.cooldown_left
        self._last_cooldown_check_ts = self._last_cooldown_check_ts + timedelta(
            seconds=windows_passed * self.window_seconds
        )

    def can_trade(self, ts: datetime, exposure_usd: float) -> tuple[bool, str]:
        self._roll_day(ts)
        self._advance_cooldown(ts)

        if self.state.cooldown_left > 0:
            return False, "cooldown_or_drift"
        if self.state.daily_pnl <= -abs(self.max_daily_loss_usd):
            return False, "daily_loss_limit"
        if self.state.open_positions >= self.max_open_positions:
            return False, "max_open_positions"
        if exposure_usd > self.max_exposure_per_window_usd:
            return False, "exposure_limit"
        return True, "ok"

    def on_trade_close(self, pnl: float) -> None:
        self.state.daily_pnl += float(pnl)
        self.state.open_positions = max(0, self.state.open_positions - 1)

        if pnl < 0:
            self.state.loss_streak += 1
            self.state.consecutive_wins = 0
            if self.state.loss_streak >= self.cooldown_after_loss_streak:
                self.state.cooldown_left = self.cooldown_windows
                self.state.loss_streak = 0
                self._last_cooldown_check_ts = None
                self.state.drift_pause_windows = self.state.cooldown_left
        else:
            self.state.loss_streak = 0
            self.state.consecutive_wins += 1

    def on_trade_open(self) -> None:
        self.state.open_positions += 1

    @staticmethod
    def _linear_multiplier(*, value: float, soft: float, hard: float, min_multiplier: float) -> float:
        if hard <= soft:
            return 1.0 if value < hard else min_multiplier
        if value <= soft:
            return 1.0
        if value >= hard:
            return min_multiplier
        ratio = (value - soft) / max(1e-9, hard - soft)
        return float(1.0 - ratio * (1.0 - min_multiplier))

    def stateful_trade_multiplier(
        self,
        *,
        available_cash: float | None,
        peak_cash: float | None,
        ts: datetime | None = None,
    ) -> tuple[float, str]:
        if ts is not None:
            self._roll_day(ts)
        if not self.stateful_enabled:
            return 1.0, "stateful_disabled"

        try:
            available = float(available_cash) if available_cash is not None else 0.0
        except (TypeError, ValueError):
            available = 0.0
        if available <= 0.0:
            return self.stateful_min_multiplier, "stateful_non_positive_cash"

        peak = None
        try:
            if peak_cash is not None:
                peak = float(peak_cash)
        except (TypeError, ValueError):
            peak = None

        drawdown_pct = 0.0
        if peak is not None and peak > 0.0:
            drawdown_pct = max(0.0, min(1.0, (peak - available) / peak))
        daily_loss_frac = max(0.0, -float(self.state.daily_pnl)) / max(available, 1e-9)
        loss_streak_f = float(max(0, int(self.state.loss_streak)))

        mult_drawdown = self._linear_multiplier(
            value=drawdown_pct,
            soft=self.stateful_drawdown_soft_pct,
            hard=self.stateful_drawdown_hard_pct,
            min_multiplier=self.stateful_min_multiplier,
        )
        mult_daily = self._linear_multiplier(
            value=daily_loss_frac,
            soft=self.stateful_daily_loss_soft_frac,
            hard=self.stateful_daily_loss_hard_frac,
            min_multiplier=self.stateful_min_multiplier,
        )
        mult_streak = self._linear_multiplier(
            value=loss_streak_f,
            soft=float(self.stateful_loss_streak_soft),
            hard=float(self.stateful_loss_streak_hard),
            min_multiplier=self.stateful_min_multiplier,
        )

        multiplier = max(self.stateful_min_multiplier, min(mult_drawdown, mult_daily, mult_streak))

        if (
            self.stateful_recovery_wins_required > 0
            and multiplier < 0.999
            and self.state.consecutive_wins < self.stateful_recovery_wins_required
        ):
            reason = "stateful_recovery_lock"
            return self.stateful_min_multiplier, reason

        if multiplier >= 0.999:
            return 1.0, "stateful_ok"
        if multiplier <= self.stateful_min_multiplier + 1e-9:
            return self.stateful_min_multiplier, "stateful_hard_throttle"
        return multiplier, "stateful_soft_throttle"

    def on_outcome(self, is_correct: bool) -> None:
        if not self.drift_enabled:
            return
        self._recent_outcomes.append(1 if bool(is_correct) else 0)
        n = len(self._recent_outcomes)
        if n <= 0:
            return
        acc = float(sum(self._recent_outcomes) / n)
        self.state.recent_accuracy = acc
        if n >= self.drift_min_samples and acc < self.drift_min_accuracy:
            self.state.cooldown_left = max(self.state.cooldown_left, self.drift_cooldown_windows)
            self.state.drift_pause_windows = self.state.cooldown_left
            self._last_cooldown_check_ts = None
