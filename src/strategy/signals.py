from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

from src.polymarket.execution import OrderIntent
from src.polymarket.fees import FeeModelConfig, compute_trade_fee
from src.polymarket.orderbook import MarketBooks
from src.strategy.actionability_calibration import evaluate_actionability_calibration
from src.strategy.sizing import size_from_edge


class TradeCandidate(NamedTuple):
    """Represents a candidate trade option with EV calculations."""
    side: str  # "up" or "down"
    order_type: str  # "taker" or "maker"
    ev_executable: float  # EV adjusted by fill probability
    ev_fill: float  # Raw EV before fill adjustment
    price: float  # Entry price
    size: float  # Order size in shares after any execution-specific caps
    fill_probability: float  # Probability this order fills
    token_id: str  # Market token ID
    cash_required: float  # Required cash at fill
    expected_roi_cash: float | None  # Executable EV per cash dollar
    breakeven_probability: float | None  # p_side required for non-negative EV (fill-level)
    breakeven_margin: float | None  # p_side - breakeven_probability


@dataclass
class SignalDecision:
    action: str  # no_trade | trade
    reason: str
    p_up: float
    p_down: float
    best_edge: float
    intent: OrderIntent | None
    score_mode: str = "expected_edge"
    score_value: float | None = None
    expected_roi_cash: float | None = None
    breakeven_probability: float | None = None
    breakeven_margin: float | None = None
    fill_probability: float | None = None
    ev_executable: float | None = None
    ev_fill: float | None = None
    cash_required: float | None = None
    calibrated_p_side: float | None = None
    calibrated_net_edge: float | None = None
    calibrated_expected_roi_cash: float | None = None
    calibrated_breakeven_margin: float | None = None
    actionability_calibration_applied: bool = False
    actionability_calibration_rejected: bool = False
    actionability_calibration_reason: str | None = None
    actionability_lower_bound_floor: float | None = None
    actionability_lower_bound_sources: list[dict[str, Any]] | None = None


def _normalized_order_types(allowed_order_types: list[str] | tuple[str, ...] | None) -> set[str]:
    if not allowed_order_types:
        return {"maker", "taker"}
    out = {
        str(value).strip().lower()
        for value in allowed_order_types
        if str(value).strip().lower() in {"maker", "taker"}
    }
    return out or {"maker", "taker"}


def _valid_price(value: float) -> bool:
    return 0.0 < float(value) < 1.0


def _net_ev_buy(
    prob_win: float,
    entry_price: float,
    size: float,
    is_taker: bool,
    fee_cfg: FeeModelConfig,
    slippage_bps: float,
) -> float:
    # Profit per share = 1-win_price if win else -entry_price -> EV/share = p - price
    gross = (prob_win - entry_price) * size
    fee, slip = _trade_costs(
        entry_price=entry_price,
        size=size,
        is_taker=is_taker,
        fee_cfg=fee_cfg,
        slippage_bps=slippage_bps,
    )
    return gross - fee - slip


def _trade_costs(
    *,
    entry_price: float,
    size: float,
    is_taker: bool,
    fee_cfg: FeeModelConfig,
    slippage_bps: float,
) -> tuple[float, float]:
    fee = compute_trade_fee(price=entry_price, size=size, is_taker=is_taker, cfg=fee_cfg)
    slip = size * entry_price * (slippage_bps / 10_000.0)
    return float(fee), float(slip)


def _candidate_economics(
    *,
    prob_win: float,
    entry_price: float,
    size: float,
    is_taker: bool,
    fee_cfg: FeeModelConfig,
    slippage_bps: float,
    fill_probability: float,
) -> dict[str, float | None]:
    fee, slip = _trade_costs(
        entry_price=entry_price,
        size=size,
        is_taker=is_taker,
        fee_cfg=fee_cfg,
        slippage_bps=slippage_bps,
    )
    cash_required = (size * entry_price) + fee + slip
    ev_fill = _net_ev_buy(
        prob_win=prob_win,
        entry_price=entry_price,
        size=size,
        is_taker=is_taker,
        fee_cfg=fee_cfg,
        slippage_bps=slippage_bps,
    )
    ev_executable = ev_fill * float(fill_probability)
    breakeven_probability: float | None = None
    breakeven_margin: float | None = None
    if size > 0:
        breakeven_probability = float(entry_price + ((fee + slip) / max(size, 1e-9)))
        breakeven_margin = float(prob_win - breakeven_probability)
    expected_roi_cash = float(ev_executable / cash_required) if cash_required > 0.0 else None
    return {
        "ev_fill": float(ev_fill),
        "ev_executable": float(ev_executable),
        "cash_required": float(cash_required),
        "expected_roi_cash": expected_roi_cash,
        "breakeven_probability": breakeven_probability,
        "breakeven_margin": breakeven_margin,
    }


def _coerce_optional_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _with_token_taker_fee(
    fee_cfg: FeeModelConfig,
    token_id: str,
    is_taker: bool,
    taker_fee_bps_by_token: dict[str, float] | None,
) -> FeeModelConfig:
    if not is_taker or not taker_fee_bps_by_token:
        return fee_cfg
    bps = taker_fee_bps_by_token.get(token_id)
    if bps is None:
        return fee_cfg
    return FeeModelConfig(
        maker_fee_bps=fee_cfg.maker_fee_bps,
        taker_fee_bps=float(bps),
        curve_rate=fee_cfg.curve_rate,
        curve_exponent=fee_cfg.curve_exponent,
        min_fee=fee_cfg.min_fee,
        maker_fee_rate=fee_cfg.maker_fee_rate,
        taker_fee_rate=float(bps) / 10_000.0,
    )


def _sized_cash_from_edge(
    *,
    max_exposure_usd: float,
    edge: float,
    min_edge: float,
    sizing_mode: str,
) -> float:
    mode = str(sizing_mode or "edge_scaled").strip().lower()
    if edge <= min_edge:
        return 0.0
    if mode in {"flat_fraction", "flat", "full_exposure"}:
        return max(0.0, float(max_exposure_usd))
    if mode in {"edge_scaled", "edge", "linear_edge"}:
        return size_from_edge(
            max_exposure_usd=max_exposure_usd,
            edge=edge,
            min_edge=min_edge,
        )
    # Preserve the historical sizing behavior if config contains an unknown value.
    return size_from_edge(
        max_exposure_usd=max_exposure_usd,
        edge=edge,
        min_edge=min_edge,
    )


def decide_trade(
    market_slug: str,
    books: MarketBooks,
    p_up: float,
    min_edge_to_trade: float,
    min_edge_for_taker: float,
    max_exposure_usd: float,
    maker_preference: bool,
    fee_cfg: FeeModelConfig,
    taker_slippage_bps: float,
    maker_slippage_bps: float,
    taker_fee_bps_by_token: dict[str, float] | None = None,
    maker_fill_probability: float = 0.65,
    maker_fill_probability_by_token: dict[str, float] | None = None,
    maker_ev_advantage_required: float = 0.0005,
    allowed_order_types: list[str] | tuple[str, ...] | None = None,
    lock_side_to_prediction: bool = True,
    min_reward_to_risk_ratio: float = 0.0,
    min_expected_roi_cash: float | None = None,
    min_breakeven_margin: float | None = None,
    sizing_mode: str = "edge_scaled",
    max_entry_price: float | None = None,
    actionability_calibration: Mapping[str, Any] | None = None,
    actionability_context: Mapping[str, Any] | None = None,
    skip_confidence_band: tuple[float, float] | None = None,
) -> SignalDecision:
    p_down = 1.0 - p_up
    enabled_order_types = _normalized_order_types(allowed_order_types)
    predicted_side = "up" if p_up >= 0.5 else "down"
    candidate_sides = {predicted_side} if lock_side_to_prediction else {"up", "down"}

    side_meta = {
        "up": {
            "prob": p_up,
            "ask": float(books.up.best_ask),
            "maker_price": float(books.up.best_bid),
            "token_id": books.up.token_id,
        },
        "down": {
            "prob": p_down,
            "ask": float(books.down.best_ask),
            "maker_price": float(books.down.best_bid),
            "token_id": books.down.token_id,
        },
    }

    # Coarse gross-edge gate for sizing only. The final decision below reuses
    # min_edge_to_trade as a stricter net EV/share gate after fees/slippage.
    edge_candidates = []
    for side in candidate_sides:
        meta = side_meta[side]
        ask = float(meta["ask"])
        maker_price = float(meta["maker_price"])
        prob = float(meta["prob"])
        if "taker" in enabled_order_types and _valid_price(ask):
            edge_candidates.append(prob - ask)
        if "maker" in enabled_order_types and _valid_price(maker_price):
            edge_candidates.append(prob - maker_price)
    best_raw_edge = max(edge_candidates) if edge_candidates else -1.0
    usd_size = _sized_cash_from_edge(
        max_exposure_usd=max_exposure_usd,
        edge=best_raw_edge,
        min_edge=min_edge_to_trade,
        sizing_mode=sizing_mode,
    )
    if usd_size <= 0:
        return SignalDecision("no_trade", "edge_below_threshold", p_up, p_down, best_raw_edge, None)

    # Convert USD exposure to shares using the actual tradable side(s), not the opposite token.
    ref_price_candidates = [
        float(side_meta[side]["ask"])
        for side in candidate_sides
        if _valid_price(side_meta[side]["ask"])
    ]
    ref_price = max([*ref_price_candidates, 0.01])
    size_shares = usd_size / ref_price

    candidates = []
    maker_fill_probability_by_token = maker_fill_probability_by_token or {}
    up_maker_fill_probability = float(
        max(0.0, min(1.0, maker_fill_probability_by_token.get(books.up.token_id, maker_fill_probability)))
    )
    down_maker_fill_probability = float(
        max(0.0, min(1.0, maker_fill_probability_by_token.get(books.down.token_id, maker_fill_probability)))
    )
    min_expected_roi_cash_value = _coerce_optional_float(min_expected_roi_cash)
    min_breakeven_margin_value = _coerce_optional_float(min_breakeven_margin)

    # Up taker / maker
    if "up" in candidate_sides and "taker" in enabled_order_types and _valid_price(books.up.best_ask):
        up_taker_fee_cfg = _with_token_taker_fee(fee_cfg, books.up.token_id, True, taker_fee_bps_by_token)
        up_max_taker_shares = float(books.up.top3_ask_size) * 0.95 if books.up.top3_ask_size > 0 else float('inf')
        up_taker_size = min(size_shares, up_max_taker_shares)
        if up_taker_size > 0:
            up_taker_econ = _candidate_economics(
                prob_win=p_up,
                entry_price=books.up.best_ask,
                size=up_taker_size,
                is_taker=True,
                fee_cfg=up_taker_fee_cfg,
                slippage_bps=taker_slippage_bps,
                fill_probability=1.0,
            )
            candidates.append(
                TradeCandidate(
                    "up",
                    "taker",
                    float(up_taker_econ["ev_executable"]),
                    float(up_taker_econ["ev_fill"]),
                    books.up.best_ask,
                    up_taker_size,
                    1.0,
                    books.up.token_id,
                    float(up_taker_econ["cash_required"]),
                    _coerce_optional_float(up_taker_econ.get("expected_roi_cash")),
                    _coerce_optional_float(up_taker_econ.get("breakeven_probability")),
                    _coerce_optional_float(up_taker_econ.get("breakeven_margin")),
                )
            )
    if "up" in candidate_sides and "maker" in enabled_order_types and _valid_price(side_meta["up"]["maker_price"]):
        up_maker_fee_cfg = _with_token_taker_fee(fee_cfg, books.up.token_id, False, taker_fee_bps_by_token)
        up_maker_econ = _candidate_economics(
            prob_win=p_up,
            entry_price=side_meta["up"]["maker_price"],
            size=size_shares,
            is_taker=False,
            fee_cfg=up_maker_fee_cfg,
            slippage_bps=maker_slippage_bps,
            fill_probability=up_maker_fill_probability,
        )
        candidates.append(
            TradeCandidate(
                "up",
                "maker",
                float(up_maker_econ["ev_executable"]),
                float(up_maker_econ["ev_fill"]),
                side_meta["up"]["maker_price"],
                size_shares,
                up_maker_fill_probability,
                books.up.token_id,
                float(up_maker_econ["cash_required"]),
                _coerce_optional_float(up_maker_econ.get("expected_roi_cash")),
                _coerce_optional_float(up_maker_econ.get("breakeven_probability")),
                _coerce_optional_float(up_maker_econ.get("breakeven_margin")),
            )
        )

    # Down taker / maker
    if "down" in candidate_sides and "taker" in enabled_order_types and _valid_price(books.down.best_ask):
        down_taker_fee_cfg = _with_token_taker_fee(fee_cfg, books.down.token_id, True, taker_fee_bps_by_token)
        down_max_taker_shares = float(books.down.top3_ask_size) * 0.95 if books.down.top3_ask_size > 0 else float('inf')
        down_taker_size = min(size_shares, down_max_taker_shares)
        if down_taker_size > 0:
            down_taker_econ = _candidate_economics(
                prob_win=p_down,
                entry_price=books.down.best_ask,
                size=down_taker_size,
                is_taker=True,
                fee_cfg=down_taker_fee_cfg,
                slippage_bps=taker_slippage_bps,
                fill_probability=1.0,
            )
            candidates.append(
                TradeCandidate(
                    "down",
                    "taker",
                    float(down_taker_econ["ev_executable"]),
                    float(down_taker_econ["ev_fill"]),
                    books.down.best_ask,
                    down_taker_size,
                    1.0,
                    books.down.token_id,
                    float(down_taker_econ["cash_required"]),
                    _coerce_optional_float(down_taker_econ.get("expected_roi_cash")),
                    _coerce_optional_float(down_taker_econ.get("breakeven_probability")),
                    _coerce_optional_float(down_taker_econ.get("breakeven_margin")),
                )
            )
    if "down" in candidate_sides and "maker" in enabled_order_types and _valid_price(side_meta["down"]["maker_price"]):
        down_maker_fee_cfg = _with_token_taker_fee(fee_cfg, books.down.token_id, False, taker_fee_bps_by_token)
        down_maker_econ = _candidate_economics(
            prob_win=p_down,
            entry_price=side_meta["down"]["maker_price"],
            size=size_shares,
            is_taker=False,
            fee_cfg=down_maker_fee_cfg,
            slippage_bps=maker_slippage_bps,
            fill_probability=down_maker_fill_probability,
        )
        candidates.append(
            TradeCandidate(
                "down",
                "maker",
                float(down_maker_econ["ev_executable"]),
                float(down_maker_econ["ev_fill"]),
                side_meta["down"]["maker_price"],
                size_shares,
                down_maker_fill_probability,
                books.down.token_id,
                float(down_maker_econ["cash_required"]),
                _coerce_optional_float(down_maker_econ.get("expected_roi_cash")),
                _coerce_optional_float(down_maker_econ.get("breakeven_probability")),
                _coerce_optional_float(down_maker_econ.get("breakeven_margin")),
            )
        )

    if not candidates:
        return SignalDecision("no_trade", "no_allowed_candidates", p_up, p_down, best_raw_edge, None)

    # Select by expected executable EV, not raw fill EV.
    best_overall = max(candidates, key=lambda x: x.ev_executable)
    (
        side,
        order_type,
        best_ev_exec,
        best_ev_fill,
        price,
        selected_size,
        fill_prob,
        token_id,
        selected_cash_required,
        selected_expected_roi_cash,
        selected_breakeven_probability,
        selected_breakeven_margin,
    ) = (
        best_overall.side,
        best_overall.order_type,
        best_overall.ev_executable,
        best_overall.ev_fill,
        best_overall.price,
        best_overall.size,
        best_overall.fill_probability,
        best_overall.token_id,
        best_overall.cash_required,
        best_overall.expected_roi_cash,
        best_overall.breakeven_probability,
        best_overall.breakeven_margin,
    )
    if maker_preference and order_type == "taker":
        maker_candidates = [x for x in candidates if x.order_type == "maker"]
        if maker_candidates:
            best_maker = max(maker_candidates, key=lambda x: x.ev_executable)
            maker_ev_exec = best_maker.ev_executable
            best_ev_exec_per_share = best_ev_exec / max(selected_size, 1e-9)
            maker_ev_exec_per_share = maker_ev_exec / max(best_maker.size, 1e-9)
            ev_gap_per_share = best_ev_exec_per_share - maker_ev_exec_per_share
            if maker_ev_exec > 0 and ev_gap_per_share <= max(0.0, maker_ev_advantage_required):
                (
                    side,
                    order_type,
                    best_ev_exec,
                    best_ev_fill,
                    price,
                    selected_size,
                    fill_prob,
                    token_id,
                    selected_cash_required,
                    selected_expected_roi_cash,
                    selected_breakeven_probability,
                    selected_breakeven_margin,
                ) = (
                    best_maker.side,
                    best_maker.order_type,
                    best_maker.ev_executable,
                    best_maker.ev_fill,
                    best_maker.price,
                    best_maker.size,
                    best_maker.fill_probability,
                    best_maker.token_id,
                    best_maker.cash_required,
                    best_maker.expected_roi_cash,
                    best_maker.breakeven_probability,
                    best_maker.breakeven_margin,
                )

    edge_per_share = best_ev_exec / max(selected_size, 1e-9)

    selected_prob = p_up if side == "up" else p_down
    calibration_context: dict[str, Any] = dict(actionability_context or {})
    calibration_context.update(
        {
            "selected_side": side,
            "selected_price": float(price),
            "selected_spread": float(books.up.spread if side == "up" else books.down.spread),
            "selected_top3_ask_size": float(books.up.top3_ask_size if side == "up" else books.down.top3_ask_size),
            "selected_top3_bid_size": float(books.up.top3_bid_size if side == "up" else books.down.top3_bid_size),
            "confidence": float(max(selected_prob, 1.0 - selected_prob)),
            "decision_best_edge": float(edge_per_share),
            "decision_expected_roi_cash": selected_expected_roi_cash,
            "decision_breakeven_margin": selected_breakeven_margin,
        }
    )

    calibration_result = evaluate_actionability_calibration(
        raw_p_side=float(selected_prob),
        cash_per_share=_coerce_optional_float(
            selected_breakeven_probability if selected_breakeven_probability is not None else price
        ),
        breakeven_probability=selected_breakeven_probability,
        size_shares=selected_size,
        fill_probability=fill_prob,
        cash_required=selected_cash_required,
        context=calibration_context,
        calibration_cfg=actionability_calibration,
    )
    if calibration_result is not None:
        if calibration_result.rejected:
            return SignalDecision(
                "no_trade",
                calibration_result.reason,
                p_up,
                p_down,
                edge_per_share,
                None,
                score_mode="expected_edge",
                score_value=float(edge_per_share),
                expected_roi_cash=selected_expected_roi_cash,
                breakeven_probability=selected_breakeven_probability,
                breakeven_margin=selected_breakeven_margin,
                fill_probability=float(fill_prob),
                ev_executable=float(best_ev_exec),
                ev_fill=float(best_ev_fill),
                cash_required=float(selected_cash_required),
                calibrated_p_side=calibration_result.calibrated_p_side,
                calibrated_net_edge=calibration_result.calibrated_net_edge,
                calibrated_expected_roi_cash=calibration_result.calibrated_expected_roi_cash,
                calibrated_breakeven_margin=calibration_result.calibrated_breakeven_margin,
                actionability_calibration_applied=True,
                actionability_calibration_rejected=True,
                actionability_calibration_reason=calibration_result.reason,
                actionability_lower_bound_floor=calibration_result.lower_bound_floor,
                actionability_lower_bound_sources=list(calibration_result.lower_bound_sources),
            )
        if calibration_result.calibrated_net_edge is not None:
            edge_per_share = float(calibration_result.calibrated_net_edge)

    # Confidence skip band — research finding from Phase Y 775-fill dataset (2026-05-18):
    # log180_then_log240 trades with confidence in [0.65, 0.85) lose net money on a
    # walk-forward split (TRAIN -$163, HOLDOUT +$36 baseline → skip yields +$58/+$137).
    # This is the adverse-selection zone where model edge exists directionally but the
    # market price already reflects most of it, leaving asymmetric payoff working against us.
    # Default disabled; gated explicitly by config.
    if skip_confidence_band is not None:
        lo, hi = float(skip_confidence_band[0]), float(skip_confidence_band[1])
        if lo < hi and lo <= float(selected_prob) < hi:
            return SignalDecision(
                "no_trade",
                "confidence_in_skip_band",
                p_up,
                p_down,
                edge_per_share,
                None,
                score_mode="expected_edge",
                score_value=float(edge_per_share),
                expected_roi_cash=selected_expected_roi_cash,
                breakeven_probability=selected_breakeven_probability,
                breakeven_margin=selected_breakeven_margin,
                fill_probability=float(fill_prob),
                ev_executable=float(best_ev_exec),
                ev_fill=float(best_ev_fill),
                cash_required=float(selected_cash_required),
            )

    if edge_per_share < min_edge_to_trade:
        return SignalDecision("no_trade", "net_ev_below_threshold", p_up, p_down, edge_per_share, None)

    if order_type == "taker" and edge_per_share < min_edge_for_taker:
        return SignalDecision("no_trade", "edge_not_enough_for_taker", p_up, p_down, edge_per_share, None)

    if max_entry_price is not None and float(price) > float(max_entry_price):
        return SignalDecision("no_trade", "entry_price_above_max", p_up, p_down, edge_per_share, None)

    reward_to_risk_ratio = (1.0 - float(price)) / max(float(price), 1e-9)
    if reward_to_risk_ratio < max(0.0, float(min_reward_to_risk_ratio)):
        return SignalDecision(
            "no_trade",
            "reward_too_small",
            p_up,
            p_down,
            edge_per_share,
            None,
            score_mode="expected_edge",
            score_value=float(edge_per_share),
            expected_roi_cash=selected_expected_roi_cash,
            breakeven_probability=selected_breakeven_probability,
            breakeven_margin=selected_breakeven_margin,
            fill_probability=float(fill_prob),
            ev_executable=float(best_ev_exec),
            ev_fill=float(best_ev_fill),
            cash_required=float(selected_cash_required),
            calibrated_p_side=(calibration_result.calibrated_p_side if calibration_result is not None else None),
            calibrated_net_edge=(calibration_result.calibrated_net_edge if calibration_result is not None else None),
            calibrated_expected_roi_cash=(
                calibration_result.calibrated_expected_roi_cash if calibration_result is not None else None
            ),
            calibrated_breakeven_margin=(
                calibration_result.calibrated_breakeven_margin if calibration_result is not None else None
            ),
            actionability_calibration_applied=(calibration_result is not None),
            actionability_calibration_rejected=False,
            actionability_calibration_reason=(calibration_result.reason if calibration_result is not None else None),
            actionability_lower_bound_floor=(
                calibration_result.lower_bound_floor if calibration_result is not None else None
            ),
            actionability_lower_bound_sources=(
                list(calibration_result.lower_bound_sources) if calibration_result is not None else None
            ),
        )

    if min_expected_roi_cash_value is not None:
        roi_val = selected_expected_roi_cash
        if roi_val is None or float(roi_val) < float(min_expected_roi_cash_value):
            return SignalDecision(
                "no_trade",
                "expected_roi_cash_below_threshold",
                p_up,
                p_down,
                edge_per_share,
                None,
                score_mode="expected_edge",
                score_value=float(edge_per_share),
                expected_roi_cash=roi_val,
                breakeven_probability=selected_breakeven_probability,
                breakeven_margin=selected_breakeven_margin,
                fill_probability=float(fill_prob),
                ev_executable=float(best_ev_exec),
                ev_fill=float(best_ev_fill),
                cash_required=float(selected_cash_required),
                calibrated_p_side=(calibration_result.calibrated_p_side if calibration_result is not None else None),
                calibrated_net_edge=(calibration_result.calibrated_net_edge if calibration_result is not None else None),
                calibrated_expected_roi_cash=(
                    calibration_result.calibrated_expected_roi_cash if calibration_result is not None else None
                ),
                calibrated_breakeven_margin=(
                    calibration_result.calibrated_breakeven_margin if calibration_result is not None else None
                ),
                actionability_calibration_applied=(calibration_result is not None),
                actionability_calibration_rejected=False,
                actionability_calibration_reason=(calibration_result.reason if calibration_result is not None else None),
                actionability_lower_bound_floor=(
                    calibration_result.lower_bound_floor if calibration_result is not None else None
                ),
                actionability_lower_bound_sources=(
                    list(calibration_result.lower_bound_sources) if calibration_result is not None else None
                ),
            )

    if min_breakeven_margin_value is not None:
        margin_val = selected_breakeven_margin
        if margin_val is None or float(margin_val) < float(min_breakeven_margin_value):
            return SignalDecision(
                "no_trade",
                "breakeven_margin_below_threshold",
                p_up,
                p_down,
                edge_per_share,
                None,
                score_mode="expected_edge",
                score_value=float(edge_per_share),
                expected_roi_cash=selected_expected_roi_cash,
                breakeven_probability=selected_breakeven_probability,
                breakeven_margin=margin_val,
                fill_probability=float(fill_prob),
                ev_executable=float(best_ev_exec),
                ev_fill=float(best_ev_fill),
                cash_required=float(selected_cash_required),
                calibrated_p_side=(calibration_result.calibrated_p_side if calibration_result is not None else None),
                calibrated_net_edge=(calibration_result.calibrated_net_edge if calibration_result is not None else None),
                calibrated_expected_roi_cash=(
                    calibration_result.calibrated_expected_roi_cash if calibration_result is not None else None
                ),
                calibrated_breakeven_margin=(
                    calibration_result.calibrated_breakeven_margin if calibration_result is not None else None
                ),
                actionability_calibration_applied=(calibration_result is not None),
                actionability_calibration_rejected=False,
                actionability_calibration_reason=(calibration_result.reason if calibration_result is not None else None),
                actionability_lower_bound_floor=(
                    calibration_result.lower_bound_floor if calibration_result is not None else None
                ),
                actionability_lower_bound_sources=(
                    list(calibration_result.lower_bound_sources) if calibration_result is not None else None
                ),
            )

    intent = OrderIntent(
        market_slug=market_slug,
        token_id=token_id,
        side="BUY",
        order_type=order_type.upper(),
        price=float(price),
        size=float(selected_size),
        expected_edge=float(edge_per_share),
    )
    return SignalDecision(
        "trade",
        "ok",
        p_up,
        p_down,
        edge_per_share,
        intent,
        score_mode="expected_edge",
        score_value=float(edge_per_share),
        expected_roi_cash=selected_expected_roi_cash,
        breakeven_probability=selected_breakeven_probability,
        breakeven_margin=selected_breakeven_margin,
        fill_probability=float(fill_prob),
        ev_executable=float(best_ev_exec),
        ev_fill=float(best_ev_fill),
        cash_required=float(selected_cash_required),
        calibrated_p_side=(calibration_result.calibrated_p_side if calibration_result is not None else None),
        calibrated_net_edge=(calibration_result.calibrated_net_edge if calibration_result is not None else None),
        calibrated_expected_roi_cash=(
            calibration_result.calibrated_expected_roi_cash if calibration_result is not None else None
        ),
        calibrated_breakeven_margin=(
            calibration_result.calibrated_breakeven_margin if calibration_result is not None else None
        ),
        actionability_calibration_applied=(calibration_result is not None),
        actionability_calibration_rejected=False,
        actionability_calibration_reason=(calibration_result.reason if calibration_result is not None else None),
        actionability_lower_bound_floor=(calibration_result.lower_bound_floor if calibration_result is not None else None),
        actionability_lower_bound_sources=(
            list(calibration_result.lower_bound_sources) if calibration_result is not None else None
        ),
    )
