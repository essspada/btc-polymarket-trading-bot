from __future__ import annotations

from pathlib import Path

import yaml


def test_monitor_taker_config_is_aligned() -> None:
    cfg_path = Path("configs/monitor_taker.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    assert cfg["app"]["mode"] == "monitor-5m"
    assert cfg["execution"]["live_trading"] is False
    assert cfg["execution"]["allowed_order_types"] == ["taker"]
    assert cfg["execution"]["maker_preference"] is False
    assert cfg["execution"]["min_edge_to_trade"] == 0.01
    assert cfg["execution"]["min_edge_for_taker"] == 0.01
    assert cfg["execution"]["max_entry_price"] == 0.70
    assert cfg["execution"]["confirmation_gate"]["enabled"] is True
    assert cfg["execution"]["confirmation_gate"]["min_edge"] == 0.0
    assert cfg["execution"]["confirmation_gate"]["max_spread"] == 0.02
