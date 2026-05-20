from __future__ import annotations

from scripts.report_timing_policy_family_search import filter_families_by_available_delays
from scripts.report_timing_policy_oos import build_timing_policy_rows
from src.polymarket.fees import FeeModelConfig


def _row(
    *,
    market_id: str,
    delay: int,
    consensus: float | None = None,
    logistic: float | None = None,
    up: float = 0.45,
    down: float = 0.55,
    entry_delay_seconds: float | None = None,
) -> dict:
    return {
        "market_id": market_id,
        "created_at": f"2026-03-01T00:0{delay // 60}:00+00:00",
        "entry_delay_seconds": (entry_delay_seconds if entry_delay_seconds is not None else delay),
        "spot_consensus_blend_p_up": consensus,
        "spot_logistic_online_p_up": logistic,
        "up_entry_price": up,
        "down_entry_price": down,
        "actual_side": "up",
    }


def test_timing_policy_takes_early_stage_when_edge_hits() -> None:
    rows_by_delay = {
        120: [_row(market_id="m1", delay=120, consensus=0.70, up=0.40, down=0.60)],
        180: [_row(market_id="m1", delay=180, consensus=0.55, up=0.48, down=0.52)],
        240: [_row(market_id="m1", delay=240, logistic=0.52, up=0.49, down=0.51)],
    }
    rows, diag = build_timing_policy_rows(
        rows_by_delay,
        stages=[(120, "spot_consensus_blend", 0.0), (180, "spot_consensus_blend", 0.0), (240, "spot_logistic_online", 0.0)],
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        slippage_bps=0.0,
        min_edge_to_trade=0.01,
    )
    assert len(rows) == 1
    assert rows[0]["spot_timing_policy_stage_delay"] == 120
    assert rows[0]["spot_timing_policy_source"] == "spot_consensus_blend"
    assert rows[0]["spot_timing_policy_reason"] == "edge_hit"
    assert diag["selected_delay_counts"]["120"] == 1


def test_timing_policy_waits_for_later_stage_if_early_edge_is_too_small() -> None:
    rows_by_delay = {
        120: [_row(market_id="m1", delay=120, consensus=0.53, up=0.50, down=0.50)],
        180: [_row(market_id="m1", delay=180, consensus=0.52, up=0.50, down=0.50)],
        240: [_row(market_id="m1", delay=240, logistic=0.75, up=0.45, down=0.55)],
    }
    rows, diag = build_timing_policy_rows(
        rows_by_delay,
        stages=[(120, "spot_consensus_blend", 0.0), (180, "spot_consensus_blend", 0.0), (240, "spot_logistic_online", 0.0)],
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        slippage_bps=0.0,
        min_edge_to_trade=0.05,
    )
    assert len(rows) == 1
    assert rows[0]["spot_timing_policy_stage_delay"] == 240
    assert rows[0]["spot_timing_policy_source"] == "spot_logistic_online"
    assert diag["selected_delay_counts"]["240"] == 1


def test_timing_policy_falls_back_to_last_available_candidate_row() -> None:
    rows_by_delay = {
        120: [_row(market_id="m1", delay=120, consensus=0.51, up=0.50, down=0.50)],
        240: [_row(market_id="m1", delay=240, logistic=0.49, up=0.50, down=0.50)],
    }
    rows, diag = build_timing_policy_rows(
        rows_by_delay,
        stages=[(120, "spot_consensus_blend", 0.0), (180, "spot_consensus_blend", 0.0), (240, "spot_logistic_online", 0.0)],
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        slippage_bps=0.0,
        min_edge_to_trade=0.10,
    )
    assert len(rows) == 1
    assert rows[0]["spot_timing_policy_stage_delay"] == 240
    assert rows[0]["spot_timing_policy_reason"] == "stage_fallback"
    assert diag["missing_stage_counts"]["180:spot_consensus_blend"] == 1


def test_timing_policy_handles_missing_stage_by_using_any_available_row() -> None:
    rows_by_delay = {
        240: [_row(market_id="m1", delay=240, logistic=0.60, up=0.45, down=0.55)],
    }
    rows, diag = build_timing_policy_rows(
        rows_by_delay,
        stages=[(180, "spot_consensus_blend", 0.0)],
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        slippage_bps=0.0,
        min_edge_to_trade=0.01,
    )
    assert len(rows) == 1
    assert rows[0]["created_at"].endswith("+00:00")
    assert rows[0]["spot_timing_policy_p_up"] is None
    assert rows[0]["spot_timing_policy_reason"] == "no_candidate_probability"
    assert diag["missing_stage_counts"]["180:spot_consensus_blend"] == 1


def test_timing_policy_extra_edge_margin_can_force_wait() -> None:
    rows_by_delay = {
        180: [_row(market_id="m1", delay=180, consensus=0.70, up=0.64, down=0.36)],
        240: [_row(market_id="m1", delay=240, logistic=0.80, up=0.45, down=0.55)],
    }
    rows, _ = build_timing_policy_rows(
        rows_by_delay,
        stages=[(180, "spot_consensus_blend", 0.05), (240, "spot_logistic_online", 0.0)],
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        slippage_bps=0.0,
        min_edge_to_trade=0.01,
    )
    assert len(rows) == 1
    assert rows[0]["spot_timing_policy_stage_delay"] == 240
    assert rows[0]["spot_timing_policy_source"] == "spot_logistic_online"


def test_family_search_skips_families_with_unavailable_delays() -> None:
    valid, skipped = filter_families_by_available_delays(
        {
            "cons120_then_log240": [("train", 120, "spot_consensus_blend"), ("fixed", 240, "spot_logistic_online")],
            "path180_then_path240": [("train", 180, "spot_window_path"), ("fixed", 240, "spot_window_path")],
        },
        available_delays={180, 240},
    )
    assert list(valid) == ["path180_then_path240"]
    assert skipped == {"cons120_then_log240": [120]}


def test_timing_policy_grace_can_wait_for_future_stage_instead_of_late_fresh_start() -> None:
    rows_by_delay = {
        180: [_row(market_id="m1", delay=180, consensus=0.75, up=0.45, down=0.55, entry_delay_seconds=191.0)],
        240: [_row(market_id="m1", delay=240, logistic=0.80, up=0.45, down=0.55, entry_delay_seconds=240.0)],
    }
    rows, diag = build_timing_policy_rows(
        rows_by_delay,
        stages=[(180, "spot_consensus_blend", 0.0), (240, "spot_logistic_online", 0.0)],
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        slippage_bps=0.0,
        min_edge_to_trade=0.01,
        allow_late_fresh_start=False,
        late_start_grace_seconds=10.0,
    )
    assert len(rows) == 1
    assert rows[0]["spot_timing_policy_stage_delay"] == 240
    assert rows[0]["spot_timing_policy_source"] == "spot_logistic_online"
    assert diag["skipped_market_reason_counts"] == {}


def test_timing_policy_grace_skips_market_when_final_stage_is_too_late() -> None:
    rows_by_delay = {
        240: [_row(market_id="m1", delay=240, logistic=0.80, up=0.45, down=0.55, entry_delay_seconds=251.0)],
    }
    rows, diag = build_timing_policy_rows(
        rows_by_delay,
        stages=[(240, "spot_logistic_online", 0.0)],
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        slippage_bps=0.0,
        min_edge_to_trade=0.01,
        allow_late_fresh_start=False,
        late_start_grace_seconds=10.0,
    )
    assert rows == []
    assert diag["rows_selected"] == 0
    assert diag["skipped_market_reason_counts"] == {"late_fresh_start_disabled": 1}
