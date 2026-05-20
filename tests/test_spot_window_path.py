from src.strategy.spot_window_path import (
    SpotWindowPathConfig,
    blend_spot_with_market_probability,
    predict_spot_window_path_probability,
)


def test_spot_window_path_probability_tracks_direction() -> None:
    cfg = SpotWindowPathConfig(min_sigma_bps=4.0, drift_weight=0.25, prob_shrink=0.9)

    p_up = predict_spot_window_path_probability(
        spot_return_bps_from_open=12.0,
        spot_recent_return_1m_bps=3.0,
        spot_recent_vol_5m_bps=6.0,
        seconds_to_expiry=120.0,
        cfg=cfg,
    )
    p_down = predict_spot_window_path_probability(
        spot_return_bps_from_open=-12.0,
        spot_recent_return_1m_bps=-3.0,
        spot_recent_vol_5m_bps=6.0,
        seconds_to_expiry=120.0,
        cfg=cfg,
    )

    assert p_up is not None and p_up > 0.5
    assert p_down is not None and p_down < 0.5


def test_spot_market_blend_stays_between_inputs() -> None:
    cfg = SpotWindowPathConfig(spot_model_weight=0.4)
    blended = blend_spot_with_market_probability(market_p_up=0.35, spot_p_up=0.75, cfg=cfg)

    assert 0.35 < blended < 0.75
    assert round(blended, 6) == round(0.35 * 0.6 + 0.75 * 0.4, 6)
