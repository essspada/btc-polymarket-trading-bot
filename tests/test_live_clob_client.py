from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.polymarket.clients import clob_client as clob_mod
from src.polymarket.clients.clob_client import ClobClient
from src.polymarket.execution import (
    ExecutionEngine,
    ExecutionResult,
    OrderIntent,
    OrderReconcileResult,
    apply_reconcile_result,
    build_live_order_record,
    load_live_order_records,
    save_live_order_records,
)


class _FakeOfficialClient:
    last_instance: _FakeOfficialClient | None = None

    def __init__(
        self,
        host,
        chain_id=None,
        key=None,
        creds=None,
        signature_type=None,
        funder=None,
        builder_config=None,
        tick_size_ttl=300.0,
    ) -> None:
        self.host = host
        self.chain_id = chain_id
        self.key = key
        self.creds = creds
        self.signature_type = signature_type
        self.funder = funder
        self.builder_config = builder_config
        self.tick_size_ttl = tick_size_ttl
        self.created_order_args = None
        self.posted = None
        self.cancelled = None
        self.balance_params = None
        self.balance_refresh_params = None
        _FakeOfficialClient.last_instance = self

    def create_or_derive_api_creds(self):
        return clob_mod.ApiCreds(api_key="k", api_secret="s", api_passphrase="p")

    def set_api_creds(self, creds) -> None:
        self.creds = creds

    def create_order(self, order_args):
        self.created_order_args = order_args
        return {"signed": True, "token_id": order_args.token_id}

    def post_order(self, order, orderType=None, post_only=False):
        self.posted = {
            "order": order,
            "orderType": str(orderType),
            "post_only": bool(post_only),
        }
        return {"orderID": "oid-1", "status": "live"}

    def cancel(self, order_id):
        self.cancelled = str(order_id)
        return {"canceled": order_id}

    def get_order(self, order_id):
        return {"id": str(order_id), "status": "OPEN"}

    def get_orders(self, params=None):
        return [{"id": "oid-1", "market": getattr(params, "market", None), "asset_id": getattr(params, "asset_id", None)}]

    def get_address(self):
        return "0xabc"

    def get_collateral_address(self):
        return "0x2791"

    def update_balance_allowance(self, params=None):
        self.balance_refresh_params = params
        return {"ok": True}

    def get_balance_allowance(self, params=None):
        self.balance_params = params
        return {"balance": "125.50", "allowance": "80.25"}


def _intent(order_type: str = "MAKER") -> OrderIntent:
    return OrderIntent(
        market_slug="btc-up-down",
        token_id="token-1",
        side="BUY",
        order_type=order_type,
        price=0.47,
        size=25.0,
        expected_edge=0.03,
    )


def test_live_clob_client_maker_order_bootstraps_creds(monkeypatch) -> None:
    monkeypatch.setattr(clob_mod, "HAS_OFFICIAL_CLOB_CLIENT", True)
    monkeypatch.setattr(clob_mod, "OfficialPolymarketClobClient", _FakeOfficialClient)
    monkeypatch.setenv("LIVE_TRADING", "true")
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "0xabc123")
    monkeypatch.setenv("POLYMARKET_CHAIN_ID", "137")
    monkeypatch.delenv("POLYMARKET_API_KEY", raising=False)
    monkeypatch.delenv("POLYMARKET_API_SECRET", raising=False)
    monkeypatch.delenv("POLYMARKET_API_PASSPHRASE", raising=False)

    client = ClobClient("https://clob.polymarket.com")
    result = client.place_order(
        {
            "token_id": "token-1",
            "price": 0.47,
            "size": 25.0,
            "side": "BUY",
            "order_type": "MAKER",
        }
    )

    fake = _FakeOfficialClient.last_instance
    assert fake is not None
    assert fake.chain_id == 137
    assert fake.created_order_args is not None
    assert fake.created_order_args.side == "BUY"
    assert fake.posted == {
        "order": {"signed": True, "token_id": "token-1"},
        "orderType": "GTC",
        "post_only": True,
    }
    assert result["accepted"] is True
    assert result["order_id"] == "oid-1"
    creds = client.ensure_api_credentials()
    assert creds["api_key"] == "k"
    assert creds["address"] == "0xabc"


def test_live_clob_client_taker_maps_to_fok(monkeypatch) -> None:
    monkeypatch.setattr(clob_mod, "HAS_OFFICIAL_CLOB_CLIENT", True)
    monkeypatch.setattr(clob_mod, "OfficialPolymarketClobClient", _FakeOfficialClient)
    monkeypatch.setenv("LIVE_TRADING", "true")
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "0xabc123")
    monkeypatch.setenv("POLYMARKET_CHAIN_ID", "137")
    monkeypatch.setenv("POLYMARKET_API_KEY", "k")
    monkeypatch.setenv("POLYMARKET_API_SECRET", "s")
    monkeypatch.setenv("POLYMARKET_API_PASSPHRASE", "p")

    client = ClobClient("https://clob.polymarket.com")
    client.place_order(
        {
            "token_id": "token-1",
            "price": 0.47,
            "size": 25.0,
            "side": "BUY",
            "order_type": "TAKER",
        }
    )

    fake = _FakeOfficialClient.last_instance
    assert fake is not None
    assert fake.posted["orderType"] == "FOK"
    assert fake.posted["post_only"] is False


def test_live_clob_client_reads_collateral_balance_allowance(monkeypatch) -> None:
    monkeypatch.setattr(clob_mod, "HAS_OFFICIAL_CLOB_CLIENT", True)
    monkeypatch.setattr(clob_mod, "OfficialPolymarketClobClient", _FakeOfficialClient)
    monkeypatch.setenv("LIVE_TRADING", "true")
    monkeypatch.setenv("POLYMARKET_PRIVATE_KEY", "0xabc123")
    monkeypatch.setenv("POLYMARKET_CHAIN_ID", "137")
    monkeypatch.setenv("POLYMARKET_API_KEY", "k")
    monkeypatch.setenv("POLYMARKET_API_SECRET", "s")
    monkeypatch.setenv("POLYMARKET_API_PASSPHRASE", "p")

    client = ClobClient("https://clob.polymarket.com")
    snapshot = client.get_collateral_balance_allowance(refresh=True)

    fake = _FakeOfficialClient.last_instance
    assert fake is not None
    assert fake.balance_params is not None
    assert fake.balance_refresh_params is not None
    assert snapshot["asset_type"] == "COLLATERAL"
    assert snapshot["balance"] == 125.50
    assert snapshot["allowance"] == 80.25
    assert snapshot["available_to_trade"] == 80.25
    assert snapshot["collateral_address"] == "0x2791"


def test_execution_engine_live_submit_uses_clob_client(monkeypatch) -> None:
    class _FakeLiveClob:
        def __init__(self) -> None:
            self.payload = None

        def place_order(self, payload):
            self.payload = dict(payload)
            return {"accepted": True, "order_id": "oid-1", "status": "open"}

    monkeypatch.setenv("LIVE_TRADING", "true")
    fake = _FakeLiveClob()
    engine = ExecutionEngine(live_trading=True, require_confirm_live=True, clob_client=fake)

    result = engine.execute(_intent(order_type="MAKER"), now=datetime(2026, 3, 12, tzinfo=UTC), confirm_live=True)

    assert fake.payload is not None
    assert fake.payload["order_type"] == "MAKER"
    assert result.accepted is True
    assert result.mode == "LIVE_SUBMITTED"
    assert result.order_id == "oid-1"
    assert result.fill_size == 0.0


def test_execution_engine_reconcile_order_maps_status(monkeypatch) -> None:
    class _FakeLiveClob:
        def get_order(self, order_id):
            assert order_id == "oid-1"
            return {
                "id": "oid-1",
                "status": "MATCHED",
                "size_matched": "25",
                "avg_fill_price": "0.47",
            }

    monkeypatch.setenv("LIVE_TRADING", "true")
    engine = ExecutionEngine(live_trading=True, require_confirm_live=True, clob_client=_FakeLiveClob())

    result = engine.reconcile_order("oid-1", now=datetime(2026, 3, 12, tzinfo=UTC), confirm_live=True)

    assert result.order_id == "oid-1"
    assert result.status == "FILLED"
    assert result.terminal is True
    assert result.fill_size == 25.0
    assert result.fill_price == 0.47


def test_live_order_record_roundtrip_and_reconcile(tmp_path) -> None:
    submitted_at = datetime(2026, 3, 12, tzinfo=UTC)
    record = build_live_order_record(
        market_id="m1",
        intent=_intent(order_type="MAKER"),
        result=ExecutionResult(
            accepted=True,
            mode="LIVE_SUBMITTED",
            fill_price=0.0,
            fill_size=0.0,
            note="open",
            order_id="oid-1",
            raw_response={"status": "open"},
        ),
        now=submitted_at,
    )
    assert record is not None
    assert record.status == "OPEN"
    assert record.terminal is False

    state_path = tmp_path / "live_order_state.json"
    save_live_order_records(state_path, {"oid-1": record}, submitted_at)
    loaded = load_live_order_records(state_path)
    assert set(loaded) == {"oid-1"}
    assert loaded["oid-1"].order_id == "oid-1"

    reconciled = apply_reconcile_result(
        loaded["oid-1"],
        OrderReconcileResult(
            order_id="oid-1",
            status="FILLED",
            terminal=True,
            fill_price=0.48,
            fill_size=25.0,
            note="matched",
            raw_response={"status": "MATCHED", "size_matched": "25", "avg_fill_price": "0.48"},
        ),
        datetime(2026, 3, 12, 0, 5, tzinfo=UTC),
    )
    assert reconciled.status == "FILLED"
    assert reconciled.terminal is True
    assert reconciled.fill_size == 25.0
    assert reconciled.fill_price == 0.48


def test_live_clob_client_rejects_when_live_env_is_false(monkeypatch) -> None:
    monkeypatch.setenv("LIVE_TRADING", "false")
    client = ClobClient("https://clob.polymarket.com")
    with pytest.raises(RuntimeError):
        client.place_order({"token_id": "token-1", "price": 0.47, "size": 25.0, "side": "BUY", "order_type": "MAKER"})
