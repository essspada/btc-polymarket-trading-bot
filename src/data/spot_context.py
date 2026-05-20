from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/price"


@dataclass
class SpotWindowContext:
    ts_utc: str
    symbol: str
    spot_price_now: float
    spot_window_open_price: float
    spot_return_bps_from_open: float
    spot_recent_return_1m_bps: float
    spot_recent_vol_5m_bps: float


class SpotContextClient:
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        timeout: int = 10,
        kline_limit: int = 8,
        cache_ttl_seconds: float = 5.0,
    ) -> None:
        self.symbol = symbol
        self.timeout = int(timeout)
        self.kline_limit = max(3, int(kline_limit))
        self.cache_ttl_seconds = max(0.0, float(cache_ttl_seconds))
        self._snapshot_cache: Dict[str, Any] | None = None

    def fetch_current_price(self) -> float:
        r = requests.get(BINANCE_TICKER_URL, params={"symbol": self.symbol}, timeout=self.timeout)
        r.raise_for_status()
        payload = r.json()
        return float(payload["price"])

    def fetch_recent_klines(self) -> List[List[Any]]:
        r = requests.get(
            BINANCE_KLINES_URL,
            params={"symbol": self.symbol, "interval": "1m", "limit": self.kline_limit},
            timeout=self.timeout,
        )
        r.raise_for_status()
        payload = r.json()
        return payload if isinstance(payload, list) else []

    def fetch_snapshot(self, now: datetime) -> Dict[str, Any]:
        now_ts = now.astimezone(timezone.utc).timestamp()
        if self._snapshot_cache is not None:
            age = now_ts - float(self._snapshot_cache.get("ts_epoch", 0.0))
            if age <= self.cache_ttl_seconds:
                return dict(self._snapshot_cache)

        snapshot = {
            "ts_epoch": now_ts,
            "spot_price_now": self.fetch_current_price(),
            "recent_klines": self.fetch_recent_klines(),
        }
        self._snapshot_cache = dict(snapshot)
        return snapshot

    @staticmethod
    def _window_open_reference(window_start: datetime, rows: List[List[Any]]) -> Optional[float]:
        if not rows:
            return None
        target_ms = int(window_start.replace(second=0, microsecond=0).timestamp() * 1000)
        chosen: Optional[List[Any]] = None
        for row in rows:
            try:
                open_ms = int(row[0])
            except Exception:
                continue
            if open_ms <= target_ms:
                chosen = row
            if open_ms == target_ms:
                break
        if chosen is None:
            chosen = rows[0]
        try:
            return float(chosen[1])
        except Exception:
            return None

    @staticmethod
    def _recent_minute_return_bps(rows: List[List[Any]]) -> float:
        if not rows:
            return 0.0
        row = rows[-1]
        try:
            open_px = float(row[1])
            close_px = float(row[4])
        except Exception:
            return 0.0
        if open_px <= 0:
            return 0.0
        return float(10_000.0 * (close_px - open_px) / open_px)

    @staticmethod
    def _recent_vol_bps(rows: List[List[Any]]) -> float:
        returns: List[float] = []
        for row in rows[-5:]:
            try:
                open_px = float(row[1])
                close_px = float(row[4])
            except Exception:
                continue
            if open_px <= 0:
                continue
            returns.append(10_000.0 * (close_px - open_px) / open_px)
        if len(returns) < 2:
            return 0.0
        mean = sum(returns) / len(returns)
        var = sum((value - mean) ** 2 for value in returns) / len(returns)
        return float(var ** 0.5)

    def fetch_window_context(self, now: datetime, window_start: datetime) -> Optional[SpotWindowContext]:
        try:
            snapshot = self.fetch_snapshot(now=now)
            current_price = float(snapshot["spot_price_now"])
            rows = list(snapshot["recent_klines"])
        except Exception:
            return None

        open_price = self._window_open_reference(window_start=window_start, rows=rows)
        if open_price is None or open_price <= 0:
            return None

        return SpotWindowContext(
            ts_utc=now.astimezone(timezone.utc).isoformat(),
            symbol=self.symbol,
            spot_price_now=float(current_price),
            spot_window_open_price=float(open_price),
            spot_return_bps_from_open=float(10_000.0 * (current_price - open_price) / open_price),
            spot_recent_return_1m_bps=self._recent_minute_return_bps(rows),
            spot_recent_vol_5m_bps=self._recent_vol_bps(rows),
        )
