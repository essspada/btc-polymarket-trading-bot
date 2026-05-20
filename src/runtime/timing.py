"""Runtime timing-policy application and ancillary helpers.

The bot can defer a prediction to a later stage of the 5-minute window if the
early-stage candidate doesn't clear the configured extra-edge gate. This
module implements:
  * pruning the deferred-decisions registry once windows resolve,
  * applying the chosen timing policy to the runtime probability payload,
  * a local-time risk-reduction window helper,
  * the TransferredMarkov predictor factory (with sensible empty-default
    handling so the open-source distribution stays usable without the
    external PyTorch project).
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from src.runtime.utilities import parse_iso_utc
from src.strategy.calibration import clip_prob
from src.strategy.timing_policy import (
    DeferredTimingDecision,
    TimingPolicyStage,
    select_timing_policy_stage,
    timing_policy_entry_mode,
)
from src.strategy.transferred_markov import TransferredMarkovConfig, TransferredMarkovPredictor


def parse_local_hhmm_to_minutes(value: Any, *, default_minutes: int) -> int:
    text = str(value or "").strip()
    if not text:
        return int(default_minutes)
    try:
        hh_raw, mm_raw = text.split(":", 1)
        hh = max(0, min(23, int(hh_raw)))
        mm = max(0, min(59, int(mm_raw)))
        return (hh * 60) + mm
    except Exception:
        return int(default_minutes)


def local_time_in_window(*, now: datetime, timezone_name: str, start_local: str, end_local: str) -> bool:
    local_now = now.astimezone(ZoneInfo(str(timezone_name or "Europe/Warsaw")))
    minute_of_day = (local_now.hour * 60) + local_now.minute
    start_minute = parse_local_hhmm_to_minutes(start_local, default_minutes=0)
    end_minute = parse_local_hhmm_to_minutes(end_local, default_minutes=0)
    if start_minute == end_minute:
        return True
    if start_minute < end_minute:
        return start_minute <= minute_of_day < end_minute
    return minute_of_day >= start_minute or minute_of_day < end_minute


def risk_window_exposure_multiplier(*, now: datetime, risk_cfg: dict[str, Any]) -> float:
    multiplier = float(risk_cfg.get("reduced_risk_exposure_multiplier", 1.0))
    if multiplier >= 0.999999:
        return 1.0
    timezone_name = str(risk_cfg.get("reduced_risk_timezone", "Europe/Warsaw")).strip() or "Europe/Warsaw"
    start_local = str(risk_cfg.get("reduced_risk_start_local", "00:00")).strip()
    end_local = str(risk_cfg.get("reduced_risk_end_local", "05:30")).strip()
    try:
        in_window = local_time_in_window(
            now=now,
            timezone_name=timezone_name,
            start_local=start_local,
            end_local=end_local,
        )
    except Exception:
        return 1.0
    return max(0.0, min(1.0, multiplier)) if in_window else 1.0


def prune_deferred_timing(
    *,
    deferred: dict[str, DeferredTimingDecision],
    resolved_ids: set[str],
    now: datetime,
    settlement_grace_seconds: int,
) -> None:
    cutoff = now - timedelta(seconds=max(0, int(settlement_grace_seconds)))
    for market_id in list(deferred.keys()):
        item = deferred[market_id]
        if market_id in resolved_ids:
            deferred.pop(market_id, None)
            continue
        try:
            end_time = parse_iso_utc(item.end_time)
        except Exception:
            deferred.pop(market_id, None)
            continue
        if end_time < cutoff:
            deferred.pop(market_id, None)


def apply_runtime_timing_policy(
    *,
    model_source: str,
    timing_policy_stages: tuple[TimingPolicyStage, ...] | None,
    seconds_from_start: float,
    min_edge_to_trade: float,
    candidate_models_payload: dict[str, Any] | None,
    probs: dict[str, Any],
    current_stage_index: int | None,
    late_start_grace_seconds: float = 0.0,
) -> tuple[str, dict[str, Any] | None, dict[str, Any], dict[str, Any] | None]:
    if not timing_policy_stages:
        return "finalize", None, probs, candidate_models_payload

    selection = select_timing_policy_stage(
        stages=timing_policy_stages,
        seconds_from_start=seconds_from_start,
        candidate_models_payload=candidate_models_payload,
        min_edge_to_trade=min_edge_to_trade,
        current_stage_index=current_stage_index,
    )
    action = str(selection.get("action") or "skip")
    if action != "finalize":
        return action, selection, probs, candidate_models_payload

    stage = selection.get("stage")
    item = selection.get("candidate_item")
    if not isinstance(stage, TimingPolicyStage) or not isinstance(item, dict):
        return "skip", selection, probs, candidate_models_payload

    p_up = clip_prob(float(item.get("p_up", probs.get("p_up", 0.5))))
    p_up_raw = float(item.get("p_up_raw", p_up))
    p_up_pre_calibration = float(item.get("p_up_pre_calibration", p_up))
    predicted_side = "up" if p_up >= 0.5 else "down"
    updated = dict(probs)
    updated["p_up"] = p_up
    updated["p_down"] = 1.0 - p_up
    updated["predicted_side"] = predicted_side
    updated["confidence"] = p_up if predicted_side == "up" else (1.0 - p_up)
    updated["p_pre_calibration"] = p_up_pre_calibration
    updated["model_used"] = model_source
    updated["model_p_up"] = p_up_raw
    updated["timing_policy_name"] = model_source
    updated["timing_policy_source"] = stage.candidate_key
    updated["timing_policy_stage_delay"] = int(stage.delay_seconds)
    updated["timing_policy_stage_extra_edge"] = float(stage.extra_edge)
    updated["timing_policy_reason"] = str(selection.get("reason") or "stage_fallback")
    updated["timing_policy_entry_mode"] = timing_policy_entry_mode(
        stage=stage,
        current_stage_index=current_stage_index,
        seconds_from_start=seconds_from_start,
        late_start_grace_seconds=late_start_grace_seconds,
    )

    payload = dict(candidate_models_payload or {})
    meta_item = dict(item)
    meta_item["timing_policy_name"] = model_source
    meta_item["timing_policy_source"] = stage.candidate_key
    meta_item["timing_policy_stage_delay"] = int(stage.delay_seconds)
    meta_item["timing_policy_stage_extra_edge"] = float(stage.extra_edge)
    meta_item["timing_policy_reason"] = str(selection.get("reason") or "stage_fallback")
    meta_item["timing_policy_entry_mode"] = updated["timing_policy_entry_mode"]
    payload[model_source] = meta_item

    candidate_probabilities = dict(updated.get("candidate_probabilities") or {})
    candidate_probabilities[model_source] = {
        "p_up_raw": p_up_raw,
        "p_up_pre_calibration": p_up_pre_calibration,
        "p_up_calibrated": float(item.get("p_up_calibrated", p_up)),
        "p_up": p_up,
        "is_normalized": True,
    }
    updated["candidate_probabilities"] = candidate_probabilities
    return "finalize", selection, updated, payload


def build_transferred_markov_predictor(
    *,
    transfer_cfg_raw: dict[str, Any],
    model_source: str,
) -> TransferredMarkovPredictor | None:
    """Construct the optional TransferredMarkov predictor.

    The predictor wraps an external PyTorch sequence model that lives in a
    separate research project. Defaults are empty strings: if the operator
    leaves the relevant paths unset, the predictor is silently skipped instead
    of failing the run, which keeps the open-source default config usable.
    """
    if not bool(transfer_cfg_raw.get("enabled", False)):
        return None
    if model_source not in {"transferred_markov", "adaptive_blend"}:
        return None

    python_bin = str(transfer_cfg_raw.get("python_bin", "") or "").strip()
    project_root = str(transfer_cfg_raw.get("project_root", "") or "").strip()
    model_config_path = str(transfer_cfg_raw.get("model_config_path", "") or "").strip()
    model_weights_path = str(transfer_cfg_raw.get("model_weights_path", "") or "").strip()
    if not all([python_bin, project_root, model_config_path, model_weights_path]):
        return None

    cfg = TransferredMarkovConfig(
        python_bin=python_bin,
        project_root=project_root,
        predictor_module=str(transfer_cfg_raw.get("predictor_module", "src.models.predict_sequence")),
        model_config_path=model_config_path,
        model_weights_path=model_weights_path,
        output_dir=str(transfer_cfg_raw.get("output_dir", "outputs/markov_bridge")),
        symbol=str(transfer_cfg_raw.get("symbol", "BTCUSDT")),
        interval=str(transfer_cfg_raw.get("interval", "5m")),
        candles_limit=int(transfer_cfg_raw.get("candles_limit", 14000)),
        request_limit=int(transfer_cfg_raw.get("request_limit", 1000)),
        request_timeout_seconds=int(transfer_cfg_raw.get("request_timeout_seconds", 20)),
        request_pause_seconds=float(transfer_cfg_raw.get("request_pause_seconds", 0.1)),
        predict_timeout_seconds=int(transfer_cfg_raw.get("predict_timeout_seconds", 120)),
        hold_weight=float(transfer_cfg_raw.get("hold_weight", 0.0)),
        max_prediction_age_minutes=int(transfer_cfg_raw.get("max_prediction_age_minutes", 45)),
        min_refresh_seconds=int(transfer_cfg_raw.get("min_refresh_seconds", 30)),
    )
    return TransferredMarkovPredictor(cfg)
