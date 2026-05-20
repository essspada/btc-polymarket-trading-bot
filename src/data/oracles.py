from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests


BINANCE_TICKER_URL = "https://api.binance.com/api/v3/ticker/price"


@dataclass
class OracleBasisSnapshot:
    ts_utc: str
    spot_price: float
    oracle_price: float
    basis_abs: float
    basis_bps: float


class OracleClient:
    """
    Lightweight oracle adapter.

    If `oracle_price_url` is provided, it should return JSON with a numeric
    `price` field. Otherwise, oracle price falls back to spot price.
    """

    def __init__(self, symbol: str = "BTCUSDT", oracle_price_url: str | None = None, timeout: int = 10) -> None:
        self.symbol = symbol
        self.oracle_price_url = oracle_price_url
        self.timeout = int(timeout)

    def fetch_spot_price(self) -> float:
        r = requests.get(BINANCE_TICKER_URL, params={"symbol": self.symbol}, timeout=self.timeout)
        r.raise_for_status()
        payload = r.json()
        return float(payload["price"])

    def fetch_oracle_price(self, fallback_spot: float) -> float:
        if not self.oracle_price_url:
            return float(fallback_spot)
        r = requests.get(self.oracle_price_url, timeout=self.timeout)
        r.raise_for_status()
        payload = r.json()
        return float(payload["price"])

    def fetch_basis(self) -> Optional[OracleBasisSnapshot]:
        try:
            spot = self.fetch_spot_price()
            oracle = self.fetch_oracle_price(fallback_spot=spot)
        except Exception:
            return None

        basis_abs = float(spot - oracle)
        basis_bps = float(10_000.0 * basis_abs / max(1e-9, oracle))
        return OracleBasisSnapshot(
            ts_utc=datetime.now(timezone.utc).isoformat(),
            spot_price=float(spot),
            oracle_price=float(oracle),
            basis_abs=basis_abs,
            basis_bps=basis_bps,
        )
