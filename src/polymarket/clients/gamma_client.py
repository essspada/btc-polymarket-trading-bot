from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class GammaClient:
    base_url: str
    timeout: int = 20
    max_retries: int = 3
    retry_sleep_seconds: float = 1.0
    last_error: str | None = None
    last_error_type: str | None = None
    last_error_url: str | None = None

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        last_exc: Exception | None = None
        attempts = max(1, int(self.max_retries))
        for attempt in range(1, attempts + 1):
            try:
                r = requests.get(url, params=params or {}, timeout=self.timeout)
                r.raise_for_status()
                self.last_error = None
                self.last_error_type = None
                self.last_error_url = None
                return r.json()
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= attempts:
                    self.last_error = str(exc)
                    self.last_error_type = type(exc).__name__
                    self.last_error_url = url
                    raise
                time.sleep(max(0.0, float(self.retry_sleep_seconds)))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("gamma request failed without exception")

    def get_markets(self, **params: Any) -> list[dict[str, Any]]:
        try:
            data = self._get("markets", params=params)
        except requests.RequestException:
            return []
        return data if isinstance(data, list) else []

    def get_market_by_id(self, market_id: str) -> dict[str, Any] | None:
        # Prefer the single-market endpoint; it returns fresher snapshots than the list query.
        try:
            data = self._get(f"markets/{market_id}")
            if isinstance(data, dict) and str(data.get("id", "")).strip():
                return data
        except Exception:
            pass

        data = self.get_markets(id=market_id)
        if not data:
            return None
        return data[0]

    def get_events(self, **params: Any) -> list[dict[str, Any]]:
        try:
            data = self._get("events", params=params)
        except requests.RequestException:
            return []
        return data if isinstance(data, list) else []

    def get_event_by_id(self, event_id: str) -> dict[str, Any] | None:
        data = self.get_events(id=event_id)
        if not data:
            return None
        return data[0]

    def get_series(self, **params: Any) -> list[dict[str, Any]]:
        try:
            data = self._get("series", params=params)
        except requests.RequestException:
            return []
        return data if isinstance(data, list) else []

    def public_search(self, query: str) -> dict[str, Any]:
        try:
            data = self._get("public-search", params={"q": query})
        except requests.RequestException:
            return {"events": []}
        return data if isinstance(data, dict) else {"events": []}


def parse_json_list_field(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            out = json.loads(value)
            return out if isinstance(out, list) else []
        except Exception:
            return []
    return []
