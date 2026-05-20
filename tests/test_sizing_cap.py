from dataclasses import asdict

from src.polymarket.fees import FeeModelConfig
from src.polymarket.orderbook import MarketBooks, OutcomeBook
from src.runtime.sizing import apply_sizing_cap_to_intent, cash_required_for_order
from src.runtime.state import PendingPaperTrade
from src.strategy.signals import decide_trade
from src.strategy.sizing import apply_conservative_size_cap


def _books() -> MarketBooks:
    up = OutcomeBook("up", 0.39, 0.40, 0.395, 0.01, 0.2)
    down = OutcomeBook("down", 0.59, 0.60, 0.595, 0.01, -0.2)
    return MarketBooks(up=up, down=down)


def test_sizing_cap_disabled_keeps_old_behavior() -> None:
    size, cash, telemetry = apply_conservative_size_cap(
        raw_size=12.0,
        raw_cash_required=6.0,
        base_exposure_usd=100.0,
        sizing_cap_cfg={"enabled": False, "max_trade_usd": 3.0},
    )
    assert size == 12.0
    assert cash == 6.0
    assert telemetry["sizing_cap_applied"] is False
    assert telemetry["sizing_cap_reason"] == "disabled"
    assert telemetry["sizing_cap_ratio"] == 1.0


def test_sizing_cap_enabled_clamps_size() -> None:
    size, cash, telemetry = apply_conservative_size_cap(
        raw_size=20.0,
        raw_cash_required=10.0,
        base_exposure_usd=150.0,
        sizing_cap_cfg={"enabled": True, "max_trade_usd": 4.0},
    )
    assert size == 8.0
    assert cash == 4.0
    assert telemetry["sizing_cap_applied"] is True
    assert telemetry["sizing_cap_reason"] == "max_trade_usd"
    assert telemetry["sizing_cap_ratio"] == 0.4


def test_sizing_cap_below_min_trade_not_applied_with_telemetry() -> None:
    size, cash, telemetry = apply_conservative_size_cap(
        raw_size=20.0,
        raw_cash_required=10.0,
        base_exposure_usd=150.0,
        sizing_cap_cfg={"enabled": True, "max_trade_usd": 4.0, "min_trade_usd_after_cap": 5.0},
    )
    assert size == 20.0
    assert cash == 10.0
    assert telemetry["sizing_cap_applied"] is False
    assert telemetry["sizing_cap_below_min_trade"] is True
    assert telemetry["sizing_cap_reason"] == "cap_below_min_trade_not_applied"
    assert telemetry["sizing_cap_min_trade_usd_after_cap"] == 5.0


def test_sizing_cap_works_with_small_balance_fraction_cap() -> None:
    size, cash, telemetry = apply_conservative_size_cap(
        raw_size=10.0,
        raw_cash_required=10.0,
        base_exposure_usd=6.0,
        sizing_cap_cfg={"enabled": True, "max_fraction_of_base_exposure": 0.5},
    )
    assert size == 3.0
    assert cash == 3.0
    assert telemetry["sizing_cap_applied"] is True
    assert telemetry["sizing_cap_reason"] == "max_fraction_of_base_exposure"


def test_sizing_cap_does_not_generate_invalid_zero_or_negative_size() -> None:
    size, cash, telemetry = apply_conservative_size_cap(
        raw_size=3.0,
        raw_cash_required=1.5,
        base_exposure_usd=100.0,
        sizing_cap_cfg={"enabled": True, "max_trade_usd": 0.0, "max_fraction_of_base_exposure": 0.0},
    )
    assert size == 3.0
    assert cash == 1.5
    assert telemetry["sizing_cap_applied"] is False
    assert telemetry["sizing_cap_reason"] == "no_valid_cap_config"


def test_apply_sizing_cap_to_intent_does_not_change_prediction_or_expected_edge() -> None:
    decision = decide_trade(
        market_slug="m",
        books=_books(),
        p_up=0.80,
        min_edge_to_trade=0.001,
        min_edge_for_taker=0.005,
        max_exposure_usd=100.0,
        maker_preference=True,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        taker_slippage_bps=0.0,
        maker_slippage_bps=0.0,
        allowed_order_types=["maker"],
    )
    assert decision.intent is not None

    predicted_side_before = "up" if decision.p_up >= 0.5 else "down"
    p_up_before = decision.p_up
    expected_edge_before = decision.intent.expected_edge
    intent_before_payload = asdict(decision.intent)
    raw_size_before = decision.intent.size
    raw_cash_required = cash_required_for_order(
        price=decision.intent.price,
        size=decision.intent.size,
        order_type=decision.intent.order_type,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        taker_slippage_bps=0.0,
        maker_slippage_bps=0.0,
        taker_fee_bps=None,
    )

    capped_intent, capped_cash_required, telemetry = apply_sizing_cap_to_intent(
        intent=decision.intent,
        raw_cash_required=raw_cash_required,
        base_exposure_usd=100.0,
        risk_cfg={"sizing_cap": {"enabled": True, "max_trade_usd": raw_cash_required * 0.5}},
    )

    predicted_side_after = "up" if decision.p_up >= 0.5 else "down"
    assert predicted_side_after == predicted_side_before
    assert decision.p_up == p_up_before
    assert capped_intent.expected_edge == expected_edge_before
    intent_after_payload = asdict(capped_intent)
    assert intent_after_payload["size"] < intent_before_payload["size"]
    for key, value in intent_before_payload.items():
        if key == "size":
            continue
        assert intent_after_payload[key] == value
    assert capped_intent.size < raw_size_before
    assert capped_cash_required < raw_cash_required
    assert telemetry["sizing_cap_applied"] is True


def test_sizing_cap_telemetry_is_serialized_in_pending_trade_payload() -> None:
    trade = PendingPaperTrade(
        trade_id="t1",
        market_id="m1",
        market_slug="m1",
        event_slug="e1",
        series_slug="s1",
        start_time="2026-04-01T00:00:00+00:00",
        end_time="2026-04-01T00:05:00+00:00",
        created_at="2026-04-01T00:03:00+00:00",
        predicted_side="up",
        p_up=0.8,
        p_down=0.2,
        model_source="log240_only",
        token_id="up1",
        order_type="MAKER",
        fill_price=0.4,
        fill_size=10.0,
        filled=True,
        fill_probability=0.65,
        expected_edge=0.04,
        up_token_id="up1",
        down_token_id="down1",
        up_best_ask=0.4,
        down_best_ask=0.6,
        cash_required=4.0,
        sizing_cap_enabled=True,
        sizing_cap_applied=True,
        sizing_cap_reason="max_trade_usd",
        sizing_cap_ratio=0.5,
        sizing_cap_below_min_trade=False,
        sizing_cap_raw_size=20.0,
        sizing_cap_capped_size=10.0,
        sizing_cap_raw_cash_required=8.0,
        sizing_cap_capped_cash_required=4.0,
        sizing_cap_min_trade_usd_after_cap=1.0,
        sizing_cap_max_trade_usd=4.0,
        sizing_cap_max_fraction_of_base_exposure=0.5,
        sizing_cap_base_exposure_usd=12.0,
        sizing_cap_effective_cap_usd=4.0,
    )
    payload = asdict(trade)
    assert payload["sizing_cap_applied"] is True
    assert payload["sizing_cap_reason"] == "max_trade_usd"
    assert payload["sizing_cap_below_min_trade"] is False
    assert payload["sizing_cap_raw_cash_required"] == 8.0
    assert payload["sizing_cap_capped_cash_required"] == 4.0
    assert payload["sizing_cap_min_trade_usd_after_cap"] == 1.0
