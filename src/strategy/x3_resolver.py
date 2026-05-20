from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

VALID_ACTIONS = {"allow", "cap", "veto"}


@dataclass(frozen=True)
class X3Decision:
    enabled: bool
    mode: str
    policy_name: str
    action: str
    candidate_cash_usd: float | None
    final_cash_usd: float | None
    cash_multiplier: float | None
    reason_codes: list[str]
    advisor_effective_action: str | None = None
    advisor_confidence: float | None = None
    hard_guard_applied: bool = False

    @property
    def vetoed(self) -> bool:
        return self.action == "veto"

    @property
    def capped(self) -> bool:
        return self.action == "cap"

    def to_payload(self, prefix: str = "x3_") -> dict[str, Any]:
        payload = asdict(self)
        return {f"{prefix}{key}": value for key, value in payload.items()}


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _clean_action(value: Any, default: str = "allow") -> str:
    action = str(value or default).strip().lower()
    return action if action in VALID_ACTIONS else default


def _candidate_side(side: Any) -> str:
    value = str(side or "").strip().lower()
    return value if value in {"up", "down"} else "unknown"


def _side_probability(*, side: str, p_up: Any = None, p_down: Any = None) -> float | None:
    up = _to_float(p_up)
    down = _to_float(p_down)
    if side == "up" and up is not None:
        return max(0.0, min(1.0, up))
    if side == "down":
        if down is not None:
            return max(0.0, min(1.0, down))
        if up is not None:
            return max(0.0, min(1.0, 1.0 - up))
    return None


def _payoff_win_roi(entry_price: float | None) -> float | None:
    if entry_price is None or entry_price <= 0.0:
        return None
    return (1.0 - float(entry_price)) / float(entry_price)


def _geometry_expected_roi(*, side: str, entry_price: float | None, p_up: Any = None, p_down: Any = None) -> float | None:
    p_trade = _side_probability(side=side, p_up=p_up, p_down=p_down)
    win_roi = _payoff_win_roi(entry_price)
    if p_trade is None or win_roi is None:
        return None
    return (p_trade * win_roi) - (1.0 - p_trade)


def _effective_breakeven_margin(
    *,
    side: str,
    entry_price: float | None,
    p_up: Any = None,
    p_down: Any = None,
    breakeven_margin: Any = None,
) -> float | None:
    stored = _to_float(breakeven_margin)
    if stored is not None:
        return stored
    p_trade = _side_probability(side=side, p_up=p_up, p_down=p_down)
    if p_trade is None or entry_price is None:
        return None
    return p_trade - float(entry_price)


def _append_cap(caps: list[tuple[float, str]], cap_value: Any, reason: str) -> None:
    cap = _to_float(cap_value)
    if cap is None:
        return
    if cap < 0.0:
        caps.append((0.0, reason))
        return
    caps.append((float(cap), reason))


def _advisor_fields(advisor: Mapping[str, Any] | None) -> tuple[str | None, float | None, float | None]:
    if not isinstance(advisor, Mapping):
        return None, None, None
    action = _clean_action(advisor.get("advisor_effective_action") or advisor.get("advisor_action"), default="allow")
    max_cash = _to_float(advisor.get("advisor_max_cash_usd"))
    confidence = _to_float(advisor.get("advisor_confidence"))
    if confidence is not None:
        confidence = max(0.0, min(1.0, confidence))
    return action, max_cash, confidence


def _linear_multiplier(*, value: float | None, good_at: Any, bad_at: Any, floor: Any) -> float:
    good = _to_float(good_at)
    bad = _to_float(bad_at)
    floor_f = _to_float(floor, 1.0)
    if value is None or good is None or bad is None or floor_f is None:
        return 1.0
    floor_f = max(0.0, min(1.0, floor_f))
    if good <= bad:
        return 1.0
    if value >= good:
        return 1.0
    if value <= bad:
        return floor_f
    span = good - bad
    return floor_f + ((value - bad) / span) * (1.0 - floor_f)


def _economics_composite_cap(
    *,
    cfg: Mapping[str, Any],
    cash: float,
    side: str,
    expected_roi: float | None,
    margin: float | None,
) -> float | None:
    multipliers = [
        _linear_multiplier(
            value=expected_roi,
            good_at=cfg.get("roi_good"),
            bad_at=cfg.get("roi_bad"),
            floor=cfg.get("roi_floor_multiplier"),
        ),
        _linear_multiplier(
            value=margin,
            good_at=cfg.get("margin_good"),
            bad_at=cfg.get("margin_bad"),
            floor=cfg.get("margin_floor_multiplier"),
        ),
    ]
    capital = _to_float(cfg.get("capital_usd"))
    cash_soft_limit = _to_float(cfg.get("cash_fraction_soft_limit"))
    cash_floor = _to_float(cfg.get("cash_fraction_floor_multiplier"), 1.0)
    if capital is not None and capital > 0.0 and cash_soft_limit is not None and cash_soft_limit > 0.0:
        cash_fraction = float(cash) / capital
        if cash_fraction > cash_soft_limit:
            multipliers.append(max(0.0, min(1.0, cash_floor or 1.0, cash_soft_limit / cash_fraction)))
    if side == "down":
        multipliers.append(max(0.0, min(1.0, _to_float(cfg.get("down_cash_multiplier"), 1.0) or 1.0)))
    multiplier = min(multipliers) if multipliers else 1.0
    if multiplier >= 0.999:
        return None
    return float(cash) * max(0.0, min(1.0, multiplier))


def resolve_x3_candidate(
    *,
    cfg: Mapping[str, Any] | None,
    candidate_cash_usd: Any,
    side: Any = None,
    entry_price: Any = None,
    p_up: Any = None,
    p_down: Any = None,
    expected_roi_cash: Any = None,
    breakeven_margin: Any = None,
    advisor: Mapping[str, Any] | None = None,
) -> X3Decision:
    """Resolve X1 candidate into X3 allow/cap/veto decision.

    The hard invariant is final_cash_usd <= candidate_cash_usd. This module is
    deliberately cap-only: it cannot create a trade or increase exposure.
    """

    cfg = cfg if isinstance(cfg, Mapping) else {}
    enabled = _to_bool(cfg.get("enabled"), default=False)
    mode = str(cfg.get("mode") or "off").strip().lower()
    if mode not in {"off", "shadow", "active"}:
        mode = "off"
    policy_name = str(cfg.get("policy_name") or cfg.get("policy") or "disabled")
    cash = _to_float(candidate_cash_usd)
    advisor_action, advisor_max_cash, advisor_confidence = _advisor_fields(advisor)

    if cash is None or cash <= 0.0:
        return X3Decision(
            enabled=enabled,
            mode=mode,
            policy_name=policy_name,
            action="veto",
            candidate_cash_usd=None if cash is None else max(0.0, cash),
            final_cash_usd=0.0,
            cash_multiplier=0.0,
            reason_codes=["invalid_or_missing_candidate_cash"],
            advisor_effective_action=advisor_action,
            advisor_confidence=advisor_confidence,
        )

    if not enabled or mode == "off":
        return X3Decision(
            enabled=False,
            mode=mode,
            policy_name=policy_name,
            action="allow",
            candidate_cash_usd=float(cash),
            final_cash_usd=float(cash),
            cash_multiplier=1.0,
            reason_codes=["disabled"],
            advisor_effective_action=advisor_action,
            advisor_confidence=advisor_confidence,
        )

    side_value = _candidate_side(side)
    entry = _to_float(entry_price)
    caps: list[tuple[float, str]] = []
    reason_codes: list[str] = []

    _append_cap(caps, cfg.get("global_cap_usd"), "global_cap_usd")

    if side_value == "up":
        _append_cap(caps, cfg.get("up_cap_usd"), "up_cap_usd")
    elif side_value == "down":
        _append_cap(caps, cfg.get("down_cap_usd"), "down_cap_usd")

    large_threshold = _to_float(cfg.get("large_cash_threshold_usd"), 16.0)
    if large_threshold is not None and cash > large_threshold:
        _append_cap(caps, cfg.get("large_cash_cap_usd"), "large_cash_cap_usd")

    mid_min = _to_float(cfg.get("mid_price_min"), 0.55)
    mid_max = _to_float(cfg.get("mid_price_max"), 0.75)
    if entry is not None and mid_min is not None and mid_max is not None and mid_min <= entry < mid_max:
        _append_cap(caps, cfg.get("mid_price_cap_usd"), "mid_price_cap_usd")

    if policy_name == "piecewise_side_entry_v1" and entry is not None:
        if side_value == "up":
            if entry >= 0.75:
                _append_cap(caps, cfg.get("up_entry_gte_075_cap_usd", 18.0), "piecewise_up_entry_gte_075_cap_usd")
            else:
                _append_cap(caps, cfg.get("up_entry_lt_075_cap_usd", 15.0), "piecewise_up_entry_lt_075_cap_usd")
        elif side_value == "down" and 0.55 <= entry < 0.75:
            _append_cap(caps, cfg.get("down_mid_price_cap_usd", 10.0), "piecewise_down_mid_price_cap_usd")

    if policy_name in {"economics_core_v1", "econ_composite_v1"}:
        expected_roi = _to_float(expected_roi_cash)
        if expected_roi is None:
            expected_roi = _geometry_expected_roi(side=side_value, entry_price=entry, p_up=p_up, p_down=p_down)
        margin = _effective_breakeven_margin(
            side=side_value,
            entry_price=entry,
            p_up=p_up,
            p_down=p_down,
            breakeven_margin=breakeven_margin,
        )

        roi_floor = _to_float(cfg.get("min_expected_roi"))
        if expected_roi is not None and roi_floor is not None and expected_roi < roi_floor:
            _append_cap(caps, cfg.get("thin_roi_cap_usd"), "economics_thin_roi_cap_usd")

        margin_floor = _to_float(cfg.get("min_breakeven_margin"))
        if margin is not None and margin_floor is not None and margin < margin_floor:
            _append_cap(caps, cfg.get("thin_margin_cap_usd"), "economics_thin_margin_cap_usd")

        high_entry_min = _to_float(cfg.get("high_entry_min"), 0.75)
        high_entry_roi_floor = _to_float(cfg.get("high_entry_min_expected_roi"))
        if (
            entry is not None
            and high_entry_min is not None
            and entry >= high_entry_min
            and (high_entry_roi_floor is None or expected_roi is None or expected_roi < high_entry_roi_floor)
        ):
            _append_cap(caps, cfg.get("high_entry_cap_usd"), "economics_high_entry_cap_usd")

        low_entry_max = _to_float(cfg.get("low_entry_max"))
        low_entry_roi_floor = _to_float(cfg.get("low_entry_min_expected_roi"))
        if (
            entry is not None
            and low_entry_max is not None
            and entry < low_entry_max
            and (low_entry_roi_floor is None or expected_roi is None or expected_roi < low_entry_roi_floor)
        ):
            _append_cap(caps, cfg.get("low_entry_cap_usd"), "economics_low_entry_cap_usd")

        down_margin_floor = _to_float(cfg.get("down_min_breakeven_margin"))
        if side_value == "down" and margin is not None and down_margin_floor is not None and margin < down_margin_floor:
            _append_cap(caps, cfg.get("down_weak_cap_usd"), "economics_down_weak_cap_usd")

        if _to_bool(cfg.get("composite_reducer_enabled"), default=False):
            _append_cap(
                caps,
                _economics_composite_cap(
                    cfg=cfg,
                    cash=float(cash),
                    side=side_value,
                    expected_roi=expected_roi,
                    margin=margin,
                ),
                "economics_composite_cap_usd",
            )

    if _to_bool(cfg.get("respect_advisor_veto"), default=False) and advisor_action == "veto":
        caps.append((0.0, "advisor_veto"))
    if _to_bool(cfg.get("respect_advisor_cap"), default=False) and advisor_action == "cap" and advisor_max_cash is not None:
        _append_cap(caps, min(float(cash), float(advisor_max_cash)), "advisor_cap")

    hard_guard_applied = False
    hard_max = _to_float(cfg.get("hard_max_cash_usd"))
    if hard_max is not None:
        caps.append((max(0.0, hard_max), "hard_max_cash_usd"))

    usable_caps = [(min(float(cash), cap), reason) for cap, reason in caps if cap <= cash + 1e-9]
    ignored_caps = [(cap, reason) for cap, reason in caps if cap > cash + 1e-9]
    if ignored_caps:
        reason_codes.extend(f"{reason}_ignored_above_candidate" for _, reason in ignored_caps)

    if usable_caps:
        final_cash, _ = min(usable_caps, key=lambda item: item[0])
        reason_codes.extend(reason for _, reason in usable_caps)
        hard_guard_applied = any(
            reason == "hard_max_cash_usd" and abs(cap - final_cash) < 1e-9
            for cap, reason in usable_caps
        )
    else:
        final_cash = float(cash)
        selected_reason = "within_policy"
        reason_codes.append(selected_reason)

    min_trade_after_cap = _to_float(cfg.get("min_trade_usd_after_cap"))
    if min_trade_after_cap is not None and 0.0 < final_cash < min_trade_after_cap:
        final_cash = 0.0
        reason_codes.append("below_min_trade_after_cap")

    final_cash = max(0.0, min(float(cash), float(final_cash)))
    multiplier = final_cash / float(cash) if cash > 0 else 0.0
    if final_cash <= 0.0:
        action = "veto"
    elif final_cash < float(cash) - 0.01:
        action = "cap"
    else:
        action = "allow"

    return X3Decision(
        enabled=True,
        mode=mode,
        policy_name=policy_name,
        action=action,
        candidate_cash_usd=float(cash),
        final_cash_usd=float(final_cash),
        cash_multiplier=float(multiplier),
        reason_codes=sorted(set(reason_codes)),
        advisor_effective_action=advisor_action,
        advisor_confidence=advisor_confidence,
        hard_guard_applied=hard_guard_applied,
    )
