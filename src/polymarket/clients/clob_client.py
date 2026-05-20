from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import requests

try:
    from py_clob_client.client import ClobClient as OfficialPolymarketClobClient
    from py_clob_client.clob_types import (
        ApiCreds,
        AssetType,
        BalanceAllowanceParams,
        OpenOrderParams,
        OrderArgs,
        OrderType,
    )
    from py_clob_client.constants import ZERO_ADDRESS
    from py_clob_client.order_builder.constants import BUY, SELL

    HAS_OFFICIAL_CLOB_CLIENT = True
except Exception:  # pragma: no cover - import guard for environments without live deps
    OfficialPolymarketClobClient = None  # type: ignore[assignment]
    ApiCreds = None  # type: ignore[assignment]
    AssetType = None  # type: ignore[assignment]
    BalanceAllowanceParams = None  # type: ignore[assignment]
    OpenOrderParams = None  # type: ignore[assignment]
    OrderArgs = None  # type: ignore[assignment]
    OrderType = None  # type: ignore[assignment]
    BUY = "BUY"
    SELL = "SELL"
    ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
    HAS_OFFICIAL_CLOB_CLIENT = False


@dataclass
class ClobClient:
    base_url: str
    timeout: int = 20
    fee_cache_ttl_seconds: int = 120
    chain_id: int | None = None
    signature_type: int | None = None
    funder: str | None = None

    def __post_init__(self) -> None:
        self._fee_cache: dict[str, tuple[float, float]] = {}
        self._auth_client: Any = None
        self._auth_cache_key: tuple[Any, ...] | None = None

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        r = requests.get(url, params=params or {}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _live_guard(self) -> None:
        if os.getenv("LIVE_TRADING", "false").lower() != "true":
            raise RuntimeError("LIVE_TRADING env is not true. Refusing live CLOB operation.")

    def _read_auth_settings(self) -> dict[str, Any]:
        private_key = str(os.getenv("POLYMARKET_PRIVATE_KEY", "")).strip()
        chain_id_raw = str(os.getenv("POLYMARKET_CHAIN_ID", self.chain_id or "")).strip()
        funder = str(os.getenv("POLYMARKET_PROXY_ADDRESS", self.funder or "")).strip() or None
        signature_type_raw = str(os.getenv("POLYMARKET_SIGNATURE_TYPE", self.signature_type or "")).strip()

        chain_id = int(chain_id_raw) if chain_id_raw else None
        signature_type = int(signature_type_raw) if signature_type_raw else None

        api_key = str(os.getenv("POLYMARKET_API_KEY", "")).strip()
        api_secret = str(os.getenv("POLYMARKET_API_SECRET", "")).strip()
        api_passphrase = str(os.getenv("POLYMARKET_API_PASSPHRASE", "")).strip()

        return {
            "private_key": private_key,
            "chain_id": chain_id,
            "funder": funder,
            "signature_type": signature_type,
            "api_key": api_key,
            "api_secret": api_secret,
            "api_passphrase": api_passphrase,
        }

    def _api_creds_from_settings(self, settings: dict[str, Any]) -> Any:
        api_key = str(settings.get("api_key") or "").strip()
        api_secret = str(settings.get("api_secret") or "").strip()
        api_passphrase = str(settings.get("api_passphrase") or "").strip()
        if not (api_key and api_secret and api_passphrase):
            return None
        return ApiCreds(api_key=api_key, api_secret=api_secret, api_passphrase=api_passphrase)

    def _auth_cache_tuple(self, settings: dict[str, Any]) -> tuple[Any, ...]:
        return (
            str(settings.get("private_key") or ""),
            settings.get("chain_id"),
            settings.get("funder"),
            settings.get("signature_type"),
            str(settings.get("api_key") or ""),
            str(settings.get("api_secret") or ""),
            str(settings.get("api_passphrase") or ""),
        )

    def _get_auth_client(self, require_level2: bool = True) -> Any:
        self._live_guard()
        if not HAS_OFFICIAL_CLOB_CLIENT:
            raise RuntimeError("py-clob-client is not installed in the active environment.")

        settings = self._read_auth_settings()
        private_key = str(settings["private_key"] or "").strip()
        chain_id = settings["chain_id"]

        if not private_key:
            raise RuntimeError("POLYMARKET_PRIVATE_KEY is required for authenticated CLOB operations.")
        if chain_id is None:
            raise RuntimeError("POLYMARKET_CHAIN_ID is required for authenticated CLOB operations.")

        cache_key = self._auth_cache_tuple(settings)
        if self._auth_client is None or self._auth_cache_key != cache_key:
            auth_client = OfficialPolymarketClobClient(
                host=self.base_url,
                chain_id=int(chain_id),
                key=private_key,
                creds=self._api_creds_from_settings(settings),
                signature_type=settings.get("signature_type"),
                funder=settings.get("funder"),
            )
            self._auth_client = auth_client
            self._auth_cache_key = cache_key

        if require_level2 and getattr(self._auth_client, "creds", None) is None:
            creds = self._auth_client.create_or_derive_api_creds()
            if creds is None:
                raise RuntimeError("Failed to create or derive Polymarket CLOB API credentials.")
            self._auth_client.set_api_creds(creds)

        return self._auth_client

    @staticmethod
    def _normalize_order_side(side: Any) -> str:
        normalized = str(side or "").strip().upper()
        if normalized not in {BUY, SELL}:
            raise ValueError(f"Unsupported order side: {side}")
        return normalized

    @staticmethod
    def _normalize_order_type(order_payload: dict[str, Any]) -> tuple[Any, bool]:
        raw_order_type = str(order_payload.get("order_type") or "").strip().upper()
        tif_raw = str(order_payload.get("time_in_force") or "").strip().upper()
        post_only = bool(order_payload.get("post_only", False))

        if raw_order_type == "MAKER":
            return OrderType.GTC, True

        if raw_order_type == "TAKER":
            tif = tif_raw or "FOK"
            tif_map = {
                "FOK": OrderType.FOK,
                "FAK": OrderType.FAK,
            }
            if tif not in tif_map:
                raise ValueError(f"Unsupported taker time_in_force: {tif}")
            return tif_map[tif], False

        tif_map = {
            "GTC": OrderType.GTC,
            "GTD": OrderType.GTD,
            "FOK": OrderType.FOK,
            "FAK": OrderType.FAK,
        }
        if tif_raw in tif_map:
            return tif_map[tif_raw], post_only

        raise ValueError(f"Unsupported order_type payload: {raw_order_type or order_payload.get('time_in_force')}")

    def get_book(self, token_id: str) -> dict[str, Any]:
        return self._get("book", params={"token_id": token_id})

    def get_midpoint(self, token_id: str) -> dict[str, Any]:
        return self._get("midpoint", params={"token_id": token_id})

    def get_spread(self, token_id: str) -> dict[str, Any]:
        return self._get("spread", params={"token_id": token_id})

    def get_collateral_address(self) -> str | None:
        if HAS_OFFICIAL_CLOB_CLIENT:
            try:
                client = self._get_auth_client(require_level2=False)
                return str(client.get_collateral_address())
            except Exception:
                pass
        return None

    def get_prices_history(
        self,
        token_id: str,
        interval: str | None = "max",
        fidelity: int = 60,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"market": token_id, "fidelity": fidelity}
        if interval:
            params["interval"] = interval
        if start_ts is not None:
            params["startTs"] = int(start_ts)
        if end_ts is not None:
            params["endTs"] = int(end_ts)
        return self._get("prices-history", params=params)

    def get_fee_rate_bps(self, token_id: str) -> float:
        now_ts = time.time()
        cached = self._fee_cache.get(token_id)
        if cached is not None:
            cached_ts, cached_bps = cached
            if (now_ts - cached_ts) <= max(60, int(self.fee_cache_ttl_seconds)):
                return float(cached_bps)

        try:
            data = self._get("fee-rate", params={"token_id": token_id})
            if "base_fee" not in data:
                raise ValueError(f"Missing base_fee in fee-rate response for {token_id}")
            base_fee = float(data["base_fee"])
            # Validate fee is in reasonable range (0-100000 bps = 0-1000%)
            if not (0 <= base_fee <= 100000):
                raise ValueError(f"Fee out of reasonable range: {base_fee} bps")
        except Exception as e:
            # Use conservative default on any error
            print(f"[WARNING] Failed to fetch fee for {token_id}: {e}. Using default 1000 bps")
            base_fee = 1000.0
        
        self._fee_cache[token_id] = (now_ts, base_fee)
        return base_fee

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None or value == "":
                return float(default)
            return float(value)
        except Exception:
            return float(default)

    def get_balance_allowance(self, asset_type: str = "COLLATERAL", token_id: str | None = None, refresh: bool = False) -> dict[str, Any]:
        client = self._get_auth_client(require_level2=True)
        if BalanceAllowanceParams is None or AssetType is None:
            raise RuntimeError("py-clob-client balance allowance types are not available.")

        asset_type_key = str(asset_type or "COLLATERAL").strip().upper()
        asset_type_map = {
            "COLLATERAL": AssetType.COLLATERAL,
            "CONDITIONAL": AssetType.CONDITIONAL,
        }
        if asset_type_key not in asset_type_map:
            raise ValueError(f"Unsupported asset_type: {asset_type}")

        params = BalanceAllowanceParams(
            asset_type=asset_type_map[asset_type_key],
            token_id=(str(token_id).strip() if token_id else None),
        )
        if refresh:
            client.update_balance_allowance(params)
        response = client.get_balance_allowance(params)
        response_dict = dict(response) if isinstance(response, dict) else {"response": response}
        balance_raw = response_dict.get("balance")
        allowance_raw = response_dict.get("allowance")
        balance = self._safe_float(balance_raw, 0.0)
        allowance = self._safe_float(allowance_raw, 0.0)
        available_to_trade = max(0.0, min(balance, allowance))
        return {
            "asset_type": asset_type_key,
            "token_id": (str(token_id).strip() if token_id else None),
            "balance": float(balance),
            "allowance": float(allowance),
            "available_to_trade": float(available_to_trade),
            "balance_raw": balance_raw,
            "allowance_raw": allowance_raw,
            "raw_response": response_dict,
        }

    def get_collateral_balance_allowance(self, refresh: bool = False) -> dict[str, Any]:
        snapshot = self.get_balance_allowance(asset_type="COLLATERAL", token_id=None, refresh=refresh)
        snapshot["collateral_address"] = self.get_collateral_address()
        return snapshot

    def place_order(self, order_payload: dict[str, Any]) -> dict[str, Any]:
        client = self._get_auth_client(require_level2=True)
        token_id = str(order_payload.get("token_id") or "").strip()
        if not token_id:
            raise ValueError("order_payload.token_id is required")

        price = float(order_payload["price"])
        size = float(order_payload["size"])
        side = self._normalize_order_side(order_payload.get("side"))
        fee_rate_bps = int(order_payload.get("fee_rate_bps", 0))
        nonce = int(order_payload.get("nonce", 0))
        expiration = int(order_payload.get("expiration", 0))
        taker = str(order_payload.get("taker") or ZERO_ADDRESS)
        order_type, post_only = self._normalize_order_type(order_payload)

        signed_order = client.create_order(
            OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=side,
                fee_rate_bps=fee_rate_bps,
                nonce=nonce,
                expiration=expiration,
                taker=taker,
            )
        )
        response = client.post_order(signed_order, orderType=order_type, post_only=post_only)
        response_dict = dict(response) if isinstance(response, dict) else {"response": response}
        order_id = (
            response_dict.get("orderID")
            or response_dict.get("id")
            or response_dict.get("orderId")
        )
        status = str(response_dict.get("status") or response_dict.get("success") or "submitted")
        return {
            "accepted": bool(order_id or response_dict),
            "order_id": order_id,
            "status": status,
            "post_only": bool(post_only),
            "time_in_force": str(order_type),
            "response": response_dict,
        }

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        client = self._get_auth_client(require_level2=True)
        response = client.cancel(str(order_id))
        response_dict = dict(response) if isinstance(response, dict) else {"response": response}
        return {
            "accepted": True,
            "order_id": str(order_id),
            "response": response_dict,
        }

    def get_order(self, order_id: str) -> dict[str, Any]:
        client = self._get_auth_client(require_level2=True)
        response = client.get_order(str(order_id))
        return dict(response) if isinstance(response, dict) else {"response": response}

    def get_orders(self, market_id: str | None = None, token_id: str | None = None) -> list[dict[str, Any]]:
        client = self._get_auth_client(require_level2=True)
        params = OpenOrderParams(market=market_id, asset_id=token_id)
        response = client.get_orders(params=params)
        if isinstance(response, list):
            return [dict(item) if isinstance(item, dict) else {"response": item} for item in response]
        return [dict(response)] if isinstance(response, dict) else [{"response": response}]

    def ensure_api_credentials(self) -> dict[str, Any]:
        client = self._get_auth_client(require_level2=True)
        creds = getattr(client, "creds", None)
        if creds is None:
            raise RuntimeError("Polymarket CLOB API credentials are not available after auth bootstrap.")
        return {
            "api_key": getattr(creds, "api_key", None),
            "has_secret": bool(getattr(creds, "api_secret", None)),
            "has_passphrase": bool(getattr(creds, "api_passphrase", None)),
            "address": client.get_address(),
        }
