from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class WsMarketClient:
    url: str

    def subscribe_orderbook(self, token_id: str, on_message: Callable[[str], None]) -> None:
        # Placeholder for websocket streaming integration.
        # Use REST fallback by default to keep the bot deterministic and testable.
        raise NotImplementedError(
            f"WebSocket streaming is not enabled in the default build. Configure a WS loop for token_id={token_id}."
        )

    def close(self) -> None:
        return None
