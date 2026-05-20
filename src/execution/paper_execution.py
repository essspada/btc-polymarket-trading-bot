from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

from src.polymarket.execution import OrderIntent
from src.polymarket.fees import FeeModelConfig, compute_trade_fee


def estimate_maker_fill_probability(
    *,
    spread: float,
    seconds_to_expiry: float,
    maker_fill_floor: float = 0.05,
    maker_fill_cap: float = 0.9,
    maker_fill_base: float = 0.78,
    maker_fill_spread_penalty: float = 7.0,
    maker_fill_late_penalty_90: float = 0.15,
    maker_fill_late_penalty_45: float = 0.10,
) -> float:
    base = float(maker_fill_base) - float(maker_fill_spread_penalty) * max(0.0, float(spread))
    if float(seconds_to_expiry) < 90:
        base -= float(maker_fill_late_penalty_90)
    # Late penalties are cumulative: a <45s maker order is also inside the <90s bucket.
    if float(seconds_to_expiry) < 45:
        base -= float(maker_fill_late_penalty_45)
    return float(min(float(maker_fill_cap), max(float(maker_fill_floor), base)))


@dataclass
class PaperTrade:
    trade_id: str
    market_id: str
    market_slug: str
    token_id: str
    order_type: str
    side: str
    fill_price: float
    fill_size: float
    expected_edge: float
    filled: bool
    fill_probability: float
    created_at: str
    up_token_id: str
    down_token_id: str
    taker_fee_bps: float | None = None


@dataclass
class PaperSettlement:
    trade_id: str
    market_id: str
    settled_at: str
    actual_side: str
    filled: bool
    gross_pnl: float
    fee: float
    slippage: float
    net_pnl: float


class PaperExecutionEngine:
    def __init__(
        self,
        maker_fill_floor: float = 0.05,
        maker_fill_cap: float = 0.9,
        maker_fill_base: float = 0.78,
        maker_fill_spread_penalty: float = 7.0,
        maker_fill_late_penalty_90: float = 0.15,
        maker_fill_late_penalty_45: float = 0.10,
    ) -> None:
        self.maker_fill_floor = float(maker_fill_floor)
        self.maker_fill_cap = float(maker_fill_cap)
        self.maker_fill_base = float(maker_fill_base)
        self.maker_fill_spread_penalty = float(maker_fill_spread_penalty)
        self.maker_fill_late_penalty_90 = float(maker_fill_late_penalty_90)
        self.maker_fill_late_penalty_45 = float(maker_fill_late_penalty_45)

    @staticmethod
    def _hash_uniform(key: str) -> float:
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        # 12 hex chars gives enough entropy for deterministic pseudo-random in [0, 1).
        return int(h[:12], 16) / float(16**12)

    def _maker_fill_probability(self, spread: float, seconds_to_expiry: float) -> float:
        return estimate_maker_fill_probability(
            spread=spread,
            seconds_to_expiry=seconds_to_expiry,
            maker_fill_floor=self.maker_fill_floor,
            maker_fill_cap=self.maker_fill_cap,
            maker_fill_base=self.maker_fill_base,
            maker_fill_spread_penalty=self.maker_fill_spread_penalty,
            maker_fill_late_penalty_90=self.maker_fill_late_penalty_90,
            maker_fill_late_penalty_45=self.maker_fill_late_penalty_45,
        )

    def open_trade(
        self,
        intent: OrderIntent,
        market_id: str,
        created_at: datetime,
        up_token_id: str,
        down_token_id: str,
        spread: float,
        seconds_to_expiry: float,
        taker_fee_bps: float | None = None,
    ) -> PaperTrade:
        order_type = str(intent.order_type).upper()
        # Fill probability currently depends on spread/time, not order size.
        # Keep the deterministic fill draw size-stable so risk caps do not
        # accidentally change the simulated fill subset by changing only size.
        trade_key = f"{market_id}:{intent.token_id}:{created_at.isoformat()}:{order_type}:{intent.price:.6f}"
        trade_id = hashlib.sha256(trade_key.encode("utf-8")).hexdigest()[:16]

        if order_type == "TAKER":
            fill_probability = 1.0
            filled = True
        else:
            fill_probability = self._maker_fill_probability(spread=spread, seconds_to_expiry=seconds_to_expiry)
            filled = self._hash_uniform(trade_key) < fill_probability

        return PaperTrade(
            trade_id=trade_id,
            market_id=market_id,
            market_slug=intent.market_slug,
            token_id=intent.token_id,
            order_type=order_type,
            side=intent.side,
            fill_price=float(intent.price),
            fill_size=float(intent.size) if filled else 0.0,
            expected_edge=float(intent.expected_edge),
            filled=bool(filled),
            fill_probability=float(fill_probability),
            created_at=created_at.isoformat(),
            up_token_id=up_token_id,
            down_token_id=down_token_id,
            taker_fee_bps=float(taker_fee_bps) if taker_fee_bps is not None else None,
        )

    @staticmethod
    def _payout_for_token(token_id: str, up_token_id: str, down_token_id: str, actual_side: str) -> float:
        side = str(actual_side).strip().lower()
        if token_id == up_token_id:
            return 1.0 if side == "up" else 0.0
        if token_id == down_token_id:
            return 1.0 if side == "down" else 0.0
        return 0.0

    def settle_trade(
        self,
        trade: PaperTrade,
        settled_at: datetime,
        actual_side: str,
        fee_cfg: FeeModelConfig,
        taker_slippage_bps: float,
        maker_slippage_bps: float,
    ) -> PaperSettlement:
        if not trade.filled or trade.fill_size <= 0:
            return PaperSettlement(
                trade_id=trade.trade_id,
                market_id=trade.market_id,
                settled_at=settled_at.isoformat(),
                actual_side=str(actual_side).lower(),
                filled=False,
                gross_pnl=0.0,
                fee=0.0,
                slippage=0.0,
                net_pnl=0.0,
            )

        payout = self._payout_for_token(trade.token_id, trade.up_token_id, trade.down_token_id, actual_side)
        gross = trade.fill_size * (payout - trade.fill_price)

        is_taker = trade.order_type == "TAKER"
        cfg_for_trade = fee_cfg
        if is_taker and trade.taker_fee_bps is not None:
            cfg_for_trade = FeeModelConfig(
                maker_fee_bps=fee_cfg.maker_fee_bps,
                taker_fee_bps=float(trade.taker_fee_bps),
                curve_rate=fee_cfg.curve_rate,
                curve_exponent=fee_cfg.curve_exponent,
                min_fee=fee_cfg.min_fee,
                maker_fee_rate=fee_cfg.maker_fee_rate,
                taker_fee_rate=float(trade.taker_fee_bps) / 10_000.0,
            )

        fee = compute_trade_fee(
            price=trade.fill_price,
            size=trade.fill_size,
            is_taker=is_taker,
            cfg=cfg_for_trade,
        )
        slip_bps = taker_slippage_bps if is_taker else maker_slippage_bps
        slippage = trade.fill_size * trade.fill_price * (float(slip_bps) / 10_000.0)
        net = gross - fee - slippage

        return PaperSettlement(
            trade_id=trade.trade_id,
            market_id=trade.market_id,
            settled_at=settled_at.isoformat(),
            actual_side=str(actual_side).lower(),
            filled=True,
            gross_pnl=float(gross),
            fee=float(fee),
            slippage=float(slippage),
            net_pnl=float(net),
        )
