from src.polymarket.fees import FeeModelConfig
from src.polymarket.orderbook import MarketBooks, OutcomeBook
from src.strategy.signals import decide_trade


def _books(spread: float = 0.01) -> MarketBooks:
    up = OutcomeBook("up", 0.49 - spread / 2, 0.49 + spread / 2, 0.49, spread, 0.3)
    down = OutcomeBook("down", 0.51 - spread / 2, 0.51 + spread / 2, 0.51, spread, -0.3)
    return MarketBooks(up=up, down=down)


def test_taker_requires_higher_edge() -> None:
    d = decide_trade(
        market_slug="m",
        books=_books(),
        p_up=0.53,
        min_edge_to_trade=0.001,
        min_edge_for_taker=0.02,
        max_exposure_usd=100,
        maker_preference=False,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0),
        taker_slippage_bps=0,
        maker_slippage_bps=0,
    )
    assert d.action in {"trade", "no_trade"}
    if d.intent is not None:
        assert d.intent.order_type in {"MAKER", "TAKER"}


def test_maker_preference_keeps_taker_when_ev_gap_is_large() -> None:
    d = decide_trade(
        market_slug="m",
        books=_books(spread=0.02),
        p_up=0.70,
        min_edge_to_trade=0.001,
        min_edge_for_taker=0.005,
        max_exposure_usd=100,
        maker_preference=True,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=1000),
        taker_slippage_bps=20,
        maker_slippage_bps=1,
    )
    assert d.action == "trade"
    assert d.intent is not None
    assert d.intent.order_type == "TAKER"


def test_positive_maker_edge_is_not_blocked_by_negative_taker_edge() -> None:
    books = MarketBooks(
        up=OutcomeBook("up", 0.51, 0.55, 0.53, 0.04, 0.0),
        down=OutcomeBook("down", 0.44, 0.48, 0.46, 0.04, 0.0),
    )
    d = decide_trade(
        market_slug="m",
        books=books,
        p_up=0.53,
        min_edge_to_trade=0.005,
        min_edge_for_taker=0.02,
        max_exposure_usd=100,
        maker_preference=True,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        taker_slippage_bps=0.0,
        maker_slippage_bps=0.0,
    )
    assert d.action == "trade"
    assert d.intent is not None
    assert d.intent.order_type == "MAKER"
    assert d.intent.token_id == "up"


def test_allowed_order_types_can_force_maker_only() -> None:
    d = decide_trade(
        market_slug="m",
        books=_books(spread=0.02),
        p_up=0.53,
        min_edge_to_trade=0.001,
        min_edge_for_taker=0.02,
        max_exposure_usd=100,
        maker_preference=True,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        taker_slippage_bps=0.0,
        maker_slippage_bps=0.0,
        allowed_order_types=["maker"],
    )
    assert d.action == "trade"
    assert d.intent is not None
    assert d.intent.order_type == "MAKER"


def test_flat_fraction_sizing_uses_full_balance_capped_exposure() -> None:
    books = MarketBooks(
        up=OutcomeBook("up", 0.50, 0.51, 0.505, 0.01, 0.0),
        down=OutcomeBook("down", 0.49, 0.50, 0.495, 0.01, 0.0),
    )
    edge_scaled = decide_trade(
        market_slug="m",
        books=books,
        p_up=0.56,
        min_edge_to_trade=0.03,
        min_edge_for_taker=0.20,
        max_exposure_usd=100,
        maker_preference=True,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        taker_slippage_bps=0.0,
        maker_slippage_bps=0.0,
        allowed_order_types=["maker"],
    )
    flat = decide_trade(
        market_slug="m",
        books=books,
        p_up=0.56,
        min_edge_to_trade=0.03,
        min_edge_for_taker=0.20,
        max_exposure_usd=100,
        maker_preference=True,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        taker_slippage_bps=0.0,
        maker_slippage_bps=0.0,
        allowed_order_types=["maker"],
        sizing_mode="flat_fraction",
    )
    assert edge_scaled.intent is not None
    assert flat.intent is not None
    assert edge_scaled.cash_required is not None
    assert flat.cash_required is not None
    assert abs(flat.cash_required - (100 * 0.50 / 0.51)) < 1e-9
    assert flat.cash_required > edge_scaled.cash_required


def test_maker_preference_advantage_threshold_is_per_share() -> None:
    books = MarketBooks(
        up=OutcomeBook("up", 0.51, 0.52, 0.515, 0.01, 0.0),
        down=OutcomeBook("down", 0.48, 0.49, 0.485, 0.01, 0.0),
    )
    common = {
        "market_slug": "m",
        "books": books,
        "p_up": 0.55,
        "min_edge_to_trade": 0.0001,
        "min_edge_for_taker": 0.0001,
        "max_exposure_usd": 100,
        "maker_preference": True,
        "fee_cfg": FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        "taker_slippage_bps": 0.0,
        "maker_slippage_bps": 0.0,
        "maker_fill_probability": 0.70,
    }
    strict = decide_trade(**common, maker_ev_advantage_required=0.001)
    assert strict.action == "trade"
    assert strict.intent is not None
    assert strict.intent.order_type == "TAKER"

    tolerant = decide_trade(**common, maker_ev_advantage_required=0.005)
    assert tolerant.action == "trade"
    assert tolerant.intent is not None
    assert tolerant.intent.order_type == "MAKER"


def test_maker_preference_does_not_change_maker_only_path() -> None:
    books = MarketBooks(
        up=OutcomeBook("up", 0.51, 0.52, 0.515, 0.01, 0.0),
        down=OutcomeBook("down", 0.48, 0.49, 0.485, 0.01, 0.0),
    )
    common = {
        "market_slug": "m",
        "books": books,
        "p_up": 0.55,
        "min_edge_to_trade": 0.0001,
        "min_edge_for_taker": 0.0001,
        "max_exposure_usd": 100,
        "fee_cfg": FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        "taker_slippage_bps": 0.0,
        "maker_slippage_bps": 0.0,
        "maker_fill_probability": 0.70,
        "maker_ev_advantage_required": 0.005,
        "allowed_order_types": ["maker"],
    }
    without_preference = decide_trade(**common, maker_preference=False)
    with_preference = decide_trade(**common, maker_preference=True)

    assert without_preference.action == with_preference.action == "trade"
    assert without_preference.intent is not None
    assert with_preference.intent is not None
    assert without_preference.intent.order_type == with_preference.intent.order_type == "MAKER"
    assert without_preference.intent.price == with_preference.intent.price
    assert without_preference.intent.size == with_preference.intent.size
    assert without_preference.best_edge == with_preference.best_edge


def test_missing_bid_does_not_create_fake_maker_trade() -> None:
    books = MarketBooks(
        up=OutcomeBook("up", 0.0, 0.55, 0.275, 0.55, 0.0),
        down=OutcomeBook("down", 0.0, 0.45, 0.225, 0.45, 0.0),
    )
    d = decide_trade(
        market_slug="m",
        books=books,
        p_up=0.70,
        min_edge_to_trade=0.001,
        min_edge_for_taker=0.001,
        max_exposure_usd=100,
        maker_preference=True,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        taker_slippage_bps=0.0,
        maker_slippage_bps=0.0,
        allowed_order_types=["maker"],
    )
    assert d.action == "no_trade"
    assert d.reason == "edge_below_threshold"


def test_maker_decision_can_use_per_token_fill_probability() -> None:
    books = MarketBooks(
        up=OutcomeBook("up", 0.49, 0.50, 0.495, 0.01, 0.0),
        down=OutcomeBook("down", 0.47, 0.48, 0.475, 0.01, 0.0),
    )
    d = decide_trade(
        market_slug="m",
        books=books,
        p_up=0.52,
        min_edge_to_trade=0.001,
        min_edge_for_taker=0.02,
        max_exposure_usd=100,
        maker_preference=True,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        taker_slippage_bps=0.0,
        maker_slippage_bps=0.0,
        allowed_order_types=["maker"],
        maker_fill_probability=0.65,
        maker_fill_probability_by_token={"up": 0.10, "down": 0.90},
    )
    assert d.action == "trade"
    assert d.intent is not None
    assert d.intent.order_type == "MAKER"
    assert d.intent.token_id == "up"


def test_decision_does_not_flip_to_opposite_token_when_opposite_is_cheaper() -> None:
    books = MarketBooks(
        up=OutcomeBook("up", 0.04, 0.05, 0.045, 0.01, 0.0),
        down=OutcomeBook("down", 0.95, 0.96, 0.955, 0.01, 0.0),
    )
    d = decide_trade(
        market_slug="m",
        books=books,
        p_up=0.07,
        min_edge_to_trade=0.001,
        min_edge_for_taker=0.02,
        max_exposure_usd=100,
        maker_preference=True,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        taker_slippage_bps=0.0,
        maker_slippage_bps=0.0,
        allowed_order_types=["maker"],
    )
    assert d.action == "no_trade"
    assert d.reason == "edge_below_threshold"


def test_decision_blocks_trade_when_reward_is_too_small() -> None:
    books = MarketBooks(
        up=OutcomeBook("up", 0.94, 0.95, 0.945, 0.01, 0.0),
        down=OutcomeBook("down", 0.05, 0.06, 0.055, 0.01, 0.0),
    )
    d = decide_trade(
        market_slug="m",
        books=books,
        p_up=0.99,
        min_edge_to_trade=0.001,
        min_edge_for_taker=0.02,
        max_exposure_usd=100,
        maker_preference=True,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        taker_slippage_bps=0.0,
        maker_slippage_bps=0.0,
        allowed_order_types=["maker"],
        min_reward_to_risk_ratio=0.10,
    )
    assert d.action == "no_trade"
    assert d.reason == "reward_too_small"


def test_decision_exposes_economics_telemetry_on_trade() -> None:
    books = MarketBooks(
        up=OutcomeBook("up", 0.94, 0.95, 0.945, 0.01, 0.0),
        down=OutcomeBook("down", 0.05, 0.06, 0.055, 0.01, 0.0),
    )
    d = decide_trade(
        market_slug="m",
        books=books,
        p_up=0.98,
        min_edge_to_trade=0.001,
        min_edge_for_taker=0.02,
        max_exposure_usd=100,
        maker_preference=True,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        taker_slippage_bps=0.0,
        maker_slippage_bps=0.0,
        allowed_order_types=["maker"],
    )
    assert d.action == "trade"
    assert d.intent is not None
    assert d.score_mode == "expected_edge"
    assert d.score_value == d.best_edge
    assert d.expected_roi_cash is not None
    assert d.breakeven_probability is not None
    assert d.breakeven_margin is not None
    assert d.fill_probability is not None
    assert d.ev_executable is not None
    assert d.ev_fill is not None
    assert d.cash_required is not None and d.cash_required > 0.0


def test_decision_blocks_trade_when_expected_roi_cash_below_threshold() -> None:
    books = MarketBooks(
        up=OutcomeBook("up", 0.94, 0.95, 0.945, 0.01, 0.0),
        down=OutcomeBook("down", 0.05, 0.06, 0.055, 0.01, 0.0),
    )
    base = decide_trade(
        market_slug="m",
        books=books,
        p_up=0.96,
        min_edge_to_trade=0.001,
        min_edge_for_taker=0.02,
        max_exposure_usd=100,
        maker_preference=True,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        taker_slippage_bps=0.0,
        maker_slippage_bps=0.0,
        allowed_order_types=["maker"],
    )
    assert base.action == "trade"
    blocked = decide_trade(
        market_slug="m",
        books=books,
        p_up=0.96,
        min_edge_to_trade=0.001,
        min_edge_for_taker=0.02,
        max_exposure_usd=100,
        maker_preference=True,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        taker_slippage_bps=0.0,
        maker_slippage_bps=0.0,
        allowed_order_types=["maker"],
        min_expected_roi_cash=0.03,
    )
    assert blocked.action == "no_trade"
    assert blocked.reason == "expected_roi_cash_below_threshold"
    assert blocked.p_up == base.p_up
    assert blocked.p_down == base.p_down


def test_decision_blocks_trade_when_breakeven_margin_below_threshold() -> None:
    books = MarketBooks(
        up=OutcomeBook("up", 0.94, 0.95, 0.945, 0.01, 0.0),
        down=OutcomeBook("down", 0.05, 0.06, 0.055, 0.01, 0.0),
    )
    d = decide_trade(
        market_slug="m",
        books=books,
        p_up=0.96,
        min_edge_to_trade=0.001,
        min_edge_for_taker=0.02,
        max_exposure_usd=100,
        maker_preference=True,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        taker_slippage_bps=0.0,
        maker_slippage_bps=0.0,
        allowed_order_types=["maker"],
        min_breakeven_margin=0.03,
    )
    assert d.action == "no_trade"
    assert d.reason == "breakeven_margin_below_threshold"


def test_decision_can_block_trade_via_actionability_calibration_reject_policy() -> None:
    books = MarketBooks(
        up=OutcomeBook("up", 0.49, 0.50, 0.495, 0.01, 0.0),
        down=OutcomeBook("down", 0.49, 0.50, 0.495, 0.01, 0.0),
    )
    d = decide_trade(
        market_slug="m",
        books=books,
        p_up=0.62,
        min_edge_to_trade=0.01,
        min_edge_for_taker=0.01,
        max_exposure_usd=100,
        maker_preference=False,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        taker_slippage_bps=0.0,
        maker_slippage_bps=0.0,
        allowed_order_types=["taker"],
        actionability_calibration={
            "enabled": True,
            "fallback_policy": "reject",
            "min_bucket_samples": 5,
            "artifact": {
                "schema": "actionability_calibration_v1",
                "global": {"n": 20, "wins": 12, "lb_wr": 0.52},
                "features": {
                    "confidence": {
                        "buckets": [
                            {"lower": 0.50, "upper": 0.60, "n": 10, "wins": 5, "lb_wr": 0.40},
                        ]
                    }
                },
            },
        },
        actionability_context={"confidence": 0.80},
    )
    assert d.action == "no_trade"
    assert d.reason == "actionability_insufficient_bucket_support"
    assert d.actionability_calibration_applied is True
    assert d.actionability_calibration_rejected is True


def test_decision_uses_calibrated_edge_when_actionability_calibration_is_enabled() -> None:
    books = MarketBooks(
        up=OutcomeBook("up", 0.49, 0.50, 0.495, 0.01, 0.0),
        down=OutcomeBook("down", 0.49, 0.50, 0.495, 0.01, 0.0),
    )
    d = decide_trade(
        market_slug="m",
        books=books,
        p_up=0.66,
        min_edge_to_trade=0.05,
        min_edge_for_taker=0.05,
        max_exposure_usd=100,
        maker_preference=False,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        taker_slippage_bps=0.0,
        maker_slippage_bps=0.0,
        allowed_order_types=["taker"],
        actionability_calibration={
            "enabled": True,
            "fallback_policy": "conservative_shrink",
            "min_bucket_samples": 5,
            "artifact": {
                "schema": "actionability_calibration_v1",
                "global": {"n": 20, "wins": 12, "lb_wr": 0.52},
                "features": {
                    "confidence": {
                        "buckets": [
                            {"lower": 0.60, "upper": 0.70, "n": 20, "wins": 14, "lb_wr": 0.61},
                        ]
                    }
                },
            },
        },
        actionability_context={"confidence": 0.66},
    )
    assert d.action == "trade"
    assert d.actionability_calibration_applied is True
    assert d.actionability_calibration_rejected is False
    assert d.calibrated_p_side is not None
    assert d.calibrated_net_edge is not None
    assert d.best_edge == d.calibrated_net_edge


def test_skip_confidence_band_blocks_mid_conviction() -> None:
    # Confidence on the up side will be max(p_up, p_down) = 0.75 — inside the band.
    d = decide_trade(
        market_slug="m",
        books=_books(spread=0.02),
        p_up=0.75,
        min_edge_to_trade=0.001,
        min_edge_for_taker=0.005,
        max_exposure_usd=100,
        maker_preference=True,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        taker_slippage_bps=0.0,
        maker_slippage_bps=0.0,
        skip_confidence_band=(0.65, 0.85),
    )
    assert d.action == "no_trade"
    assert d.reason == "confidence_in_skip_band"
    assert d.intent is None


def test_skip_confidence_band_allows_low_conviction() -> None:
    # Confidence 0.55 sits below the band, should be allowed through (subject to edge).
    d = decide_trade(
        market_slug="m",
        books=_books(spread=0.02),
        p_up=0.55,
        min_edge_to_trade=0.001,
        min_edge_for_taker=0.005,
        max_exposure_usd=100,
        maker_preference=True,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        taker_slippage_bps=0.0,
        maker_slippage_bps=0.0,
        skip_confidence_band=(0.65, 0.85),
    )
    # 0.55 is outside the band — must not be rejected for that reason.
    assert d.reason != "confidence_in_skip_band"


def test_skip_confidence_band_allows_high_conviction() -> None:
    d = decide_trade(
        market_slug="m",
        books=_books(spread=0.02),
        p_up=0.92,
        min_edge_to_trade=0.001,
        min_edge_for_taker=0.005,
        max_exposure_usd=100,
        maker_preference=True,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        taker_slippage_bps=0.0,
        maker_slippage_bps=0.0,
        skip_confidence_band=(0.65, 0.85),
    )
    assert d.reason != "confidence_in_skip_band"


def test_skip_confidence_band_disabled_when_none() -> None:
    d = decide_trade(
        market_slug="m",
        books=_books(spread=0.02),
        p_up=0.75,
        min_edge_to_trade=0.001,
        min_edge_for_taker=0.005,
        max_exposure_usd=100,
        maker_preference=True,
        fee_cfg=FeeModelConfig(min_fee=0.0, taker_fee_bps=0.0, maker_fee_bps=0.0),
        taker_slippage_bps=0.0,
        maker_slippage_bps=0.0,
        skip_confidence_band=None,
    )
    assert d.reason != "confidence_in_skip_band"
