from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def size_from_edge(max_exposure_usd: float, edge: float, min_edge: float) -> float:
    if edge <= min_edge:
        return 0.0
    # Scale linearly with edge, capped at max_exposure.
    strength = min(1.0, (edge - min_edge) / max(1e-6, 5 * min_edge))
    return max_exposure_usd * strength


def cap_exposure_by_balance(
    *,
    max_exposure_usd: float,
    balance_usd: float | None = None,
    max_balance_fraction_per_trade: float | None = None,
) -> float:
    cap = max(0.0, float(max_exposure_usd))
    if balance_usd is None:
        return cap

    balance_cap = max(0.0, float(balance_usd))
    cap = min(cap, balance_cap)

    if max_balance_fraction_per_trade is None:
        return cap

    fraction = max(0.0, min(1.0, float(max_balance_fraction_per_trade)))
    return min(cap, balance_cap * fraction)


def apply_conservative_size_cap(
    *,
    raw_size: float,
    raw_cash_required: float,
    base_exposure_usd: float,
    sizing_cap_cfg: Mapping[str, Any] | None,
) -> tuple[float, float, dict[str, Any]]:
    cfg = sizing_cap_cfg if isinstance(sizing_cap_cfg, Mapping) else {}
    enabled = bool(cfg.get("enabled", False))

    def _maybe_positive(value: Any) -> float | None:
        try:
            v = float(value)
        except Exception:
            return None
        if v <= 0.0:
            return None
        return v

    max_trade_usd = _maybe_positive(cfg.get("max_trade_usd"))
    min_trade_usd_after_cap = _maybe_positive(cfg.get("min_trade_usd_after_cap"))

    frac_raw = cfg.get("max_fraction_of_base_exposure")
    max_fraction_of_base_exposure: float | None = None
    try:
        if frac_raw is not None:
            max_fraction_of_base_exposure = max(0.0, min(1.0, float(frac_raw)))
    except Exception:
        max_fraction_of_base_exposure = None

    normalized_base_exposure_usd = max(0.0, float(base_exposure_usd))
    fraction_cap_usd: float | None = None
    if max_fraction_of_base_exposure is not None and max_fraction_of_base_exposure > 0.0:
        fraction_cap_usd = normalized_base_exposure_usd * max_fraction_of_base_exposure

    telemetry: dict[str, Any] = {
        "sizing_cap_enabled": enabled,
        "sizing_cap_applied": False,
        "sizing_cap_reason": "disabled",
        "sizing_cap_ratio": 1.0,
        "sizing_cap_below_min_trade": False,
        "sizing_cap_raw_size": float(max(0.0, raw_size)),
        "sizing_cap_capped_size": float(max(0.0, raw_size)),
        "sizing_cap_raw_cash_required": float(max(0.0, raw_cash_required)),
        "sizing_cap_capped_cash_required": float(max(0.0, raw_cash_required)),
        "sizing_cap_min_trade_usd_after_cap": min_trade_usd_after_cap,
        "sizing_cap_max_trade_usd": max_trade_usd,
        "sizing_cap_max_fraction_of_base_exposure": max_fraction_of_base_exposure,
        "sizing_cap_base_exposure_usd": normalized_base_exposure_usd,
        "sizing_cap_effective_cap_usd": None,
    }

    raw_size_f = float(raw_size)
    raw_cash_f = float(raw_cash_required)
    if raw_size_f <= 0.0 or raw_cash_f <= 0.0:
        telemetry["sizing_cap_reason"] = "invalid_raw_inputs"
        return float(max(0.0, raw_size_f)), float(max(0.0, raw_cash_f)), telemetry

    if not enabled:
        return raw_size_f, raw_cash_f, telemetry

    cap_candidates = []
    if max_trade_usd is not None:
        cap_candidates.append(("max_trade_usd", max_trade_usd))
    if fraction_cap_usd is not None:
        cap_candidates.append(("max_fraction_of_base_exposure", fraction_cap_usd))
    if not cap_candidates:
        telemetry["sizing_cap_reason"] = "no_valid_cap_config"
        return raw_size_f, raw_cash_f, telemetry

    selected_reason, effective_cap_usd = min(cap_candidates, key=lambda kv: float(kv[1]))
    telemetry["sizing_cap_effective_cap_usd"] = float(effective_cap_usd)

    if effective_cap_usd >= raw_cash_f:
        telemetry["sizing_cap_reason"] = "within_cap"
        return raw_size_f, raw_cash_f, telemetry

    if effective_cap_usd <= 0.0:
        telemetry["sizing_cap_reason"] = "non_positive_cap"
        return raw_size_f, raw_cash_f, telemetry

    cap_ratio = max(0.0, min(1.0, float(effective_cap_usd) / raw_cash_f))
    capped_size = raw_size_f * cap_ratio
    capped_cash = raw_cash_f * cap_ratio

    if capped_size <= 0.0 or capped_cash <= 0.0:
        telemetry["sizing_cap_reason"] = "invalid_capped_result"
        return raw_size_f, raw_cash_f, telemetry

    # Hardening: avoid producing a cap-induced tiny trade.
    # Keep legacy behavior by not applying this cap when it would violate the floor.
    if min_trade_usd_after_cap is not None and capped_cash < float(min_trade_usd_after_cap):
        telemetry.update(
            {
                "sizing_cap_below_min_trade": True,
                "sizing_cap_reason": "cap_below_min_trade_not_applied",
            }
        )
        return raw_size_f, raw_cash_f, telemetry

    telemetry.update(
        {
            "sizing_cap_applied": True,
            "sizing_cap_reason": str(selected_reason),
            "sizing_cap_ratio": float(cap_ratio),
            "sizing_cap_capped_size": float(capped_size),
            "sizing_cap_capped_cash_required": float(capped_cash),
        }
    )
    return float(capped_size), float(capped_cash), telemetry
