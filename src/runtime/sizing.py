"""Position-sizing, cash-required, and stateful-regime adjustments.

These helpers translate a strategy `TradeDecision` into the actual capital
commitment, applying conservative size caps, fee/slippage gross-up, and the
optional stateful risk multiplier that can shrink or veto a trade.
"""
from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from src.polymarket.execution import OrderIntent
from src.polymarket.fees import FeeModelConfig, compute_trade_fee
from src.strategy.sizing import apply_conservative_size_cap

if TYPE_CHECKING:
    from src.strategy.signals import SignalDecision as TradeDecision  # noqa: F401


def fee_cfg_with_taker_bps(fee_cfg: FeeModelConfig, taker_fee_bps: float | None) -> FeeModelConfig:
    if taker_fee_bps is None:
        return fee_cfg
    return FeeModelConfig(
        maker_fee_bps=fee_cfg.maker_fee_bps,
        taker_fee_bps=float(taker_fee_bps),
        curve_rate=fee_cfg.curve_rate,
        curve_exponent=fee_cfg.curve_exponent,
        min_fee=fee_cfg.min_fee,
        maker_fee_rate=fee_cfg.maker_fee_rate,
        taker_fee_rate=float(taker_fee_bps) / 10_000.0,
    )


def cash_required_for_order(
    *,
    price: float,
    size: float,
    order_type: str,
    fee_cfg: FeeModelConfig,
    taker_slippage_bps: float,
    maker_slippage_bps: float,
    taker_fee_bps: float | None = None,
) -> float:
    size = max(0.0, float(size))
    price = max(0.0, float(price))
    if size <= 0.0 or price <= 0.0:
        return 0.0
    is_taker = str(order_type).upper() == "TAKER"
    cfg_for_trade = fee_cfg_with_taker_bps(fee_cfg, taker_fee_bps) if is_taker else fee_cfg
    fee = compute_trade_fee(price=price, size=size, is_taker=is_taker, cfg=cfg_for_trade)
    slip_bps = float(taker_slippage_bps if is_taker else maker_slippage_bps)
    slippage = size * price * (slip_bps / 10_000.0)
    return float((size * price) + fee + slippage)


def apply_sizing_cap_to_intent(
    *,
    intent: OrderIntent,
    raw_cash_required: float,
    base_exposure_usd: float,
    risk_cfg: dict[str, Any],
) -> tuple[OrderIntent, float, dict[str, Any]]:
    capped_size, capped_cash_required, sizing_cap_payload = apply_conservative_size_cap(
        raw_size=float(intent.size),
        raw_cash_required=float(raw_cash_required),
        base_exposure_usd=float(base_exposure_usd),
        sizing_cap_cfg=risk_cfg.get("sizing_cap", {}),
    )
    updated_intent = intent
    if float(capped_size) < float(intent.size) - 1e-12:
        updated_intent = replace(intent, size=float(capped_size))
    return updated_intent, float(capped_cash_required), dict(sizing_cap_payload)


def apply_stateful_risk_multiplier_to_intent(
    *,
    decision: TradeDecision,
    raw_cash_required: float,
    multiplier: float,
) -> tuple[TradeDecision, float, dict[str, Any]]:
    payload: dict[str, Any] = {
        "stateful_regime_enabled": True,
        "stateful_multiplier_applied": False,
        "stateful_multiplier": float(max(0.0, min(1.0, multiplier))),
        "stateful_raw_cash_required": float(max(0.0, raw_cash_required)),
        "stateful_capped_cash_required": float(max(0.0, raw_cash_required)),
    }
    if decision.intent is None:
        return decision, float(raw_cash_required), payload
    try:
        multiplier_f = float(multiplier)
    except (TypeError, ValueError):
        multiplier_f = 1.0
    if multiplier_f >= 0.999:
        return decision, float(raw_cash_required), payload
    multiplier_f = max(0.0, min(1.0, multiplier_f))
    if multiplier_f <= 0.0:
        updated = replace(
            decision,
            action="no_trade",
            reason="stateful_veto_non_positive_multiplier",
            intent=None,
            cash_required=0.0,
        )
        payload.update(
            {
                "stateful_multiplier_applied": True,
                "stateful_capped_cash_required": 0.0,
            }
        )
        return updated, 0.0, payload

    updated_intent = replace(decision.intent, size=float(decision.intent.size) * multiplier_f)
    capped_cash_required = float(max(0.0, raw_cash_required)) * multiplier_f
    payload.update(
        {
            "stateful_multiplier_applied": True,
            "stateful_capped_cash_required": float(capped_cash_required),
        }
    )
    updated = replace(decision, intent=updated_intent, cash_required=float(capped_cash_required))
    return updated, float(capped_cash_required), payload
