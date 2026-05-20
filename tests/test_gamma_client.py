from __future__ import annotations

import requests

from src.polymarket.clients.gamma_client import GammaClient


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


def test_gamma_client_retries_after_timeout(monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_get(url, params=None, timeout=None):
        calls["count"] += 1
        if calls["count"] < 3:
            raise requests.ReadTimeout("timed out")
        return _FakeResponse([{"id": "ok"}])

    monkeypatch.setattr("src.polymarket.clients.gamma_client.requests.get", _fake_get)
    monkeypatch.setattr("src.polymarket.clients.gamma_client.time.sleep", lambda *_args, **_kwargs: None)

    client = GammaClient("https://gamma-api.polymarket.com", timeout=1, max_retries=3, retry_sleep_seconds=0.0)
    data = client.get_events(slug="btc-up-or-down-5m", limit=1)

    assert calls["count"] == 3
    assert data == [{"id": "ok"}]


def test_gamma_client_get_events_returns_empty_after_request_failure(monkeypatch) -> None:
    calls = {"count": 0}

    def _fake_get(url, params=None, timeout=None):
        calls["count"] += 1
        raise requests.ConnectionError("dns failed")

    monkeypatch.setattr("src.polymarket.clients.gamma_client.requests.get", _fake_get)
    monkeypatch.setattr("src.polymarket.clients.gamma_client.time.sleep", lambda *_args, **_kwargs: None)

    client = GammaClient("https://gamma-api.polymarket.com", timeout=1, max_retries=2, retry_sleep_seconds=0.0)

    assert client.get_events(slug="btc-up-or-down-5m", limit=1) == []
    assert calls["count"] == 2
    assert client.last_error == "dns failed"
    assert client.last_error_type == "ConnectionError"
    assert client.last_error_url == "https://gamma-api.polymarket.com/events"


def test_gamma_client_get_market_by_id_returns_none_after_request_failure(monkeypatch) -> None:
    def _fake_get(url, params=None, timeout=None):
        raise requests.ConnectionError("dns failed")

    monkeypatch.setattr("src.polymarket.clients.gamma_client.requests.get", _fake_get)
    monkeypatch.setattr("src.polymarket.clients.gamma_client.time.sleep", lambda *_args, **_kwargs: None)

    client = GammaClient("https://gamma-api.polymarket.com", timeout=1, max_retries=1, retry_sleep_seconds=0.0)

    assert client.get_market_by_id("123") is None
    assert client.last_error == "dns failed"
    assert client.last_error_type == "ConnectionError"
