from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

VALID_ACTIONS = {"allow", "cap", "veto", "cooldown", "no_candidate"}


@dataclass(frozen=True)
class AdaptiveRiskDecision:
    enabled: bool
    mode: str
    action: str
    candidate_cash_usd: float
    final_cash_usd: float
    cash_multiplier: float
    reason_codes: list[str]
    shadow_only: bool = True
    runtime_applied: bool = False
    would_reduce_cash: bool = False
    would_veto: bool = False
    would_cooldown: bool = False
    suggested_cooldown_windows: int = 0
    current_equity_usd: float | None = None
    available_cash_usd: float | None = None
    reserved_cash_usd: float | None = None
    peak_equity_usd: float | None = None
    drawdown_pct: float | None = None
    daily_pnl: float | None = None
    daily_loss_frac: float | None = None
    loss_streak: int | None = None
    consecutive_wins: int | None = None
    cooldown_left: int | None = None
    recent_accuracy: float | None = None
    open_positions: int | None = None
    max_open_positions: int | None = None
    risk_budget_cash_usd: float | None = None
    confidence: float | None = None
    spread_norm: float | None = None
    spot_recent_vol_5m_bps: float | None = None
    expected_roi_cash: float | None = None
    breakeven_margin: float | None = None
    seconds_to_expiry: float | None = None
    market_regime: str = "unknown"

    @property
    def capped(self) -> bool:
        return self.action == "cap"

    @property
    def vetoed(self) -> bool:
        return self.action == "veto"

    @property
    def cooldown(self) -> bool:
        return self.action == "cooldown"

    def to_payload(self, prefix: str = "adaptive_risk_") -> dict[str, Any]:
        payload = asdict(self)
        payload["reason_count"] = len(self.reason_codes)
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


def _to_int(value: Any, default: int | None = None) -> int | None:
    out = _to_float(value)
    if out is None:
        return default
    return int(out)


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _bounded_float(value: Any, *, default: float, lo: float, hi: float) -> float:
    out = _to_float(value, default)
    if out is None:
        out = float(default)
    return float(max(lo, min(hi, out)))


def _linear_risk_multiplier(*, value: float | None, soft: float, hard: float, floor: float) -> float:
    if value is None:
        return 1.0
    floor = max(0.0, min(1.0, float(floor)))
    if hard <= soft:
        return 1.0 if value < hard else floor
    if value <= soft:
        return 1.0
    if value >= hard:
        return floor
    ratio = (value - soft) / max(1e-12, hard - soft)
    return float(1.0 - ratio * (1.0 - floor))


def _action_from_final(candidate: float, final: float, reasons: list[str]) -> str:
    if candidate <= 0.0:
        return "no_candidate"
    if final <= 0.0:
        if any("cooldown" in item or "lock" in item or item == "max_open_positions" for item in reasons):
            return "cooldown"
        return "veto"
    if final < candidate - 1e-9:
        return "cap"
    return "allow"


def _market_regime(reasons: list[str]) -> str:
    if any("cooldown" in item or "lock" in item or item == "max_open_positions" for item in reasons):
        return "risk_off_cooldown"
    if any("economics" in item for item in reasons):
        return "risk_off_economics"
    if any("drawdown" in item or "daily_loss" in item or "loss_streak" in item or "risk_budget" in item for item in reasons):
        return "risk_off_bankroll"
    if any("volatility" in item or "spread" in item for item in reasons):
        return "risk_off_market_microstructure"
    if any("confidence" in item for item in reasons):
        return "risk_off_low_confidence"
    return "normal"


def resolve_adaptive_risk_shadow(
    *,
    cfg: Mapping[str, Any] | None,
    candidate_cash_usd: Any,
    available_cash_usd: Any = None,
    reserved_cash_usd: Any = None,
    current_equity_usd: Any = None,
    peak_equity_usd: Any = None,
    daily_pnl: Any = None,
    loss_streak: Any = None,
    consecutive_wins: Any = None,
    cooldown_left: Any = None,
    recent_accuracy: Any = None,
    open_positions: Any = None,
    max_open_positions: Any = None,
    risk_budget_cash_usd: Any = None,
    expected_roi_cash: Any = None,
    breakeven_margin: Any = None,
    confidence: Any = None,
    spread_norm: Any = None,
    spot_recent_vol_5m_bps: Any = None,
    seconds_to_expiry: Any = None,
) -> AdaptiveRiskDecision:
    """Evaluate reduce-only adaptive risk policy as shadow telemetry.

    This resolver is deliberately non-authoritative: it never returns
    `runtime_applied=true`, never increases cash, and cannot create a trade.
    """

    cfg = cfg if isinstance(cfg, Mapping) else {}
    enabled = _to_bool(cfg.get("enabled"), default=False)
    mode_raw = str(cfg.get("mode") or "off").strip().lower()
    mode = "shadow" if enabled and mode_raw in {"shadow", "active"} else "off"
    reasons: list[str] = []
    if enabled and mode_raw == "active":
        reasons.append("active_mode_forced_shadow")

    candidate = max(0.0, _to_float(candidate_cash_usd, 0.0) or 0.0)
    available = _to_float(available_cash_usd)
    reserved = _to_float(reserved_cash_usd)
    equity = _to_float(current_equity_usd)
    if equity is None:
        equity = max(0.0, (available or 0.0) + (reserved or 0.0)) if available is not None or reserved is not None else None
    peak = _to_float(peak_equity_usd)
    if peak is not None and equity is not None:
        peak = max(float(peak), float(equity))
    drawdown_pct = None
    if peak is not None and peak > 0.0 and equity is not None:
        drawdown_pct = max(0.0, min(1.0, (peak - equity) / peak))

    pnl_today = _to_float(daily_pnl)
    daily_loss_frac = None
    if equity is not None and equity > 0.0 and pnl_today is not None:
        daily_loss_frac = max(0.0, -float(pnl_today)) / max(float(equity), 1e-12)

    loss_streak_i = _to_int(loss_streak, 0) or 0
    consecutive_wins_i = _to_int(consecutive_wins, 0) or 0
    cooldown_left_i = _to_int(cooldown_left, 0) or 0
    open_positions_i = _to_int(open_positions)
    max_open_positions_i = _to_int(max_open_positions)
    risk_budget = _to_float(risk_budget_cash_usd)
    expected_roi = _to_float(expected_roi_cash)
    margin = _to_float(breakeven_margin)
    conf = _to_float(confidence)
    spread = _to_float(spread_norm)
    vol = _to_float(spot_recent_vol_5m_bps)
    seconds_left = _to_float(seconds_to_expiry)

    if candidate <= 0.0:
        reasons.append("x1_no_trade")
        return AdaptiveRiskDecision(
            enabled=enabled,
            mode=mode,
            action="no_candidate",
            candidate_cash_usd=0.0,
            final_cash_usd=0.0,
            cash_multiplier=0.0,
            reason_codes=reasons,
            current_equity_usd=equity,
            available_cash_usd=available,
            reserved_cash_usd=reserved,
            peak_equity_usd=peak,
            drawdown_pct=drawdown_pct,
            daily_pnl=pnl_today,
            daily_loss_frac=daily_loss_frac,
            loss_streak=loss_streak_i,
            consecutive_wins=consecutive_wins_i,
            cooldown_left=cooldown_left_i,
            recent_accuracy=_to_float(recent_accuracy),
            open_positions=open_positions_i,
            max_open_positions=max_open_positions_i,
            risk_budget_cash_usd=risk_budget,
            confidence=conf,
            spread_norm=spread,
            spot_recent_vol_5m_bps=vol,
            expected_roi_cash=expected_roi,
            breakeven_margin=margin,
            seconds_to_expiry=seconds_left,
            market_regime="no_candidate",
        )

    if not enabled or mode == "off":
        reasons.append("disabled")
        return AdaptiveRiskDecision(
            enabled=False,
            mode=mode,
            action="allow",
            candidate_cash_usd=float(candidate),
            final_cash_usd=float(candidate),
            cash_multiplier=1.0,
            reason_codes=reasons,
            current_equity_usd=equity,
            available_cash_usd=available,
            reserved_cash_usd=reserved,
            peak_equity_usd=peak,
            drawdown_pct=drawdown_pct,
            daily_pnl=pnl_today,
            daily_loss_frac=daily_loss_frac,
            loss_streak=loss_streak_i,
            consecutive_wins=consecutive_wins_i,
            cooldown_left=cooldown_left_i,
            recent_accuracy=_to_float(recent_accuracy),
            open_positions=open_positions_i,
            max_open_positions=max_open_positions_i,
            risk_budget_cash_usd=risk_budget,
            confidence=conf,
            spread_norm=spread,
            spot_recent_vol_5m_bps=vol,
            expected_roi_cash=expected_roi,
            breakeven_margin=margin,
            seconds_to_expiry=seconds_left,
            market_regime="disabled",
        )

    min_multiplier = _bounded_float(cfg.get("min_multiplier"), default=0.35, lo=0.01, hi=1.0)
    final_cash = float(candidate)
    multipliers: list[tuple[float, str]] = []
    suggested_cooldown = 0

    daily_lock_frac = _bounded_float(cfg.get("daily_loss_lock_frac"), default=0.12, lo=0.0, hi=1.0)
    loss_cooldown = max(1, int(_bounded_float(cfg.get("loss_streak_cooldown"), default=2.0, lo=1.0, hi=50.0)))
    loss_long_cooldown = max(loss_cooldown, int(_bounded_float(cfg.get("loss_streak_long_cooldown"), default=3.0, lo=1.0, hi=50.0)))
    cooldown_windows = max(1, int(_bounded_float(cfg.get("cooldown_windows"), default=1.0, lo=1.0, hi=100.0)))
    long_cooldown_windows = max(cooldown_windows, int(_bounded_float(cfg.get("long_cooldown_windows"), default=3.0, lo=1.0, hi=100.0)))

    if cooldown_left_i > 0:
        final_cash = 0.0
        suggested_cooldown = max(suggested_cooldown, cooldown_left_i)
        reasons.append("existing_cooldown")
    if daily_loss_frac is not None and daily_loss_frac >= daily_lock_frac:
        final_cash = 0.0
        suggested_cooldown = max(suggested_cooldown, long_cooldown_windows)
        reasons.append("daily_loss_lock")
    if loss_streak_i >= loss_long_cooldown:
        final_cash = 0.0
        suggested_cooldown = max(suggested_cooldown, long_cooldown_windows)
        reasons.append("loss_streak_long_cooldown")
    elif loss_streak_i >= loss_cooldown:
        final_cash = 0.0
        suggested_cooldown = max(suggested_cooldown, cooldown_windows)
        reasons.append("loss_streak_cooldown")
    if max_open_positions_i is not None and open_positions_i is not None and open_positions_i >= max_open_positions_i:
        final_cash = 0.0
        suggested_cooldown = max(suggested_cooldown, cooldown_windows)
        reasons.append("max_open_positions")

    if final_cash > 0.0 and _to_bool(cfg.get("veto_negative_economics"), default=True):
        min_roi = _to_float(cfg.get("min_expected_roi_cash"), 0.0)
        min_margin = _to_float(cfg.get("min_breakeven_margin"), 0.0)
        if expected_roi is not None and min_roi is not None and expected_roi <= min_roi:
            final_cash = 0.0
            reasons.append("negative_economics_expected_roi")
        if margin is not None and min_margin is not None and margin <= min_margin:
            final_cash = 0.0
            reasons.append("negative_economics_breakeven_margin")

    if final_cash > 0.0:
        drawdown_soft = _bounded_float(cfg.get("drawdown_soft_pct"), default=0.05, lo=0.0, hi=1.0)
        drawdown_hard = _bounded_float(cfg.get("drawdown_hard_pct"), default=0.15, lo=drawdown_soft, hi=1.0)
        drawdown_mult = _linear_risk_multiplier(value=drawdown_pct, soft=drawdown_soft, hard=drawdown_hard, floor=min_multiplier)
        if drawdown_mult < 0.999:
            multipliers.append((drawdown_mult, "drawdown_throttle"))

        daily_soft = _bounded_float(cfg.get("daily_loss_soft_frac"), default=0.06, lo=0.0, hi=1.0)
        daily_mult = _linear_risk_multiplier(value=daily_loss_frac, soft=daily_soft, hard=daily_lock_frac, floor=min_multiplier)
        if daily_mult < 0.999:
            multipliers.append((daily_mult, "daily_loss_throttle"))

        high_vol = _to_float(cfg.get("high_volatility_bps"), 20.0)
        extreme_vol = _to_float(cfg.get("extreme_volatility_bps"), 35.0)
        high_vol_mult = _bounded_float(cfg.get("high_volatility_multiplier"), default=0.65, lo=0.01, hi=1.0)
        if vol is not None and extreme_vol is not None and vol >= extreme_vol:
            multipliers.append((min_multiplier, "extreme_volatility_throttle"))
        elif vol is not None and high_vol is not None and vol >= high_vol:
            multipliers.append((high_vol_mult, "high_volatility_throttle"))

        wide_spread = _to_float(cfg.get("wide_spread_norm"), 0.06)
        extreme_spread = _to_float(cfg.get("extreme_spread_norm"), 0.10)
        wide_spread_mult = _bounded_float(cfg.get("wide_spread_multiplier"), default=0.65, lo=0.01, hi=1.0)
        if spread is not None and extreme_spread is not None and spread >= extreme_spread:
            multipliers.append((min_multiplier, "extreme_spread_throttle"))
        elif spread is not None and wide_spread is not None and spread >= wide_spread:
            multipliers.append((wide_spread_mult, "wide_spread_throttle"))

        low_conf = _to_float(cfg.get("low_confidence_threshold"), 0.58)
        low_conf_mult = _bounded_float(cfg.get("low_confidence_multiplier"), default=0.75, lo=0.01, hi=1.0)
        if conf is not None and low_conf is not None and conf < low_conf:
            multipliers.append((low_conf_mult, "low_confidence_throttle"))

        max_fraction = _to_float(cfg.get("max_candidate_fraction_of_equity"), 0.12)
        if equity is not None and equity > 0.0 and max_fraction is not None and max_fraction > 0.0:
            max_cash = float(equity) * float(max_fraction)
            if candidate > max_cash:
                multipliers.append((max(0.0, min(1.0, max_cash / candidate)), "risk_budget_fraction_cap"))
        if risk_budget is not None and risk_budget > 0.0 and candidate > risk_budget:
            multipliers.append((max(0.0, min(1.0, risk_budget / candidate)), "risk_budget_cash_cap"))

        if multipliers:
            multiplier, reason = min(multipliers, key=lambda item: item[0])
            final_cash = min(final_cash, candidate * max(0.0, min(1.0, multiplier)))
            for _, multiplier_reason in sorted(multipliers, key=lambda item: item[0]):
                if multiplier_reason not in reasons:
                    reasons.append(multiplier_reason)

        min_trade = _to_float(cfg.get("min_trade_usd_after_cap"), 0.0)
        if min_trade is not None and min_trade > 0.0 and 0.0 < final_cash < min_trade:
            final_cash = 0.0
            reasons.append("below_min_trade_after_cap")

    final_cash = max(0.0, min(float(candidate), float(final_cash)))
    multiplier = 0.0 if candidate <= 0.0 else final_cash / candidate
    if not reasons:
        reasons.append("ok")
    action = _action_from_final(candidate, final_cash, reasons)
    would_reduce = final_cash < candidate - 1e-9
    return AdaptiveRiskDecision(
        enabled=True,
        mode="shadow",
        action=action,
        candidate_cash_usd=float(candidate),
        final_cash_usd=float(final_cash),
        cash_multiplier=float(max(0.0, min(1.0, multiplier))),
        reason_codes=reasons,
        would_reduce_cash=bool(would_reduce),
        would_veto=bool(action == "veto"),
        would_cooldown=bool(action == "cooldown"),
        suggested_cooldown_windows=int(suggested_cooldown),
        current_equity_usd=equity,
        available_cash_usd=available,
        reserved_cash_usd=reserved,
        peak_equity_usd=peak,
        drawdown_pct=drawdown_pct,
        daily_pnl=pnl_today,
        daily_loss_frac=daily_loss_frac,
        loss_streak=loss_streak_i,
        consecutive_wins=consecutive_wins_i,
        cooldown_left=cooldown_left_i,
        recent_accuracy=_to_float(recent_accuracy),
        open_positions=open_positions_i,
        max_open_positions=max_open_positions_i,
        risk_budget_cash_usd=risk_budget,
        confidence=conf,
        spread_norm=spread,
        spot_recent_vol_5m_bps=vol,
        expected_roi_cash=expected_roi,
        breakeven_margin=margin,
        seconds_to_expiry=seconds_left,
        market_regime=_market_regime(reasons),
    )
