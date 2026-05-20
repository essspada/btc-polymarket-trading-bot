from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config import AppConfig
from src.utils.config import load_app_config, load_config


CONFIG_DIR = Path("configs")


def test_all_repository_configs_validate() -> None:
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        cfg = load_app_config(str(path))
        assert cfg.app.name == "btc_polymarket_bot"
        assert cfg.execution.allowed_order_types
        assert cfg.to_runtime_dict()["app"]["seed"] == cfg.app.seed


def test_config_schema_adds_defaults_for_sparse_yaml(tmp_path: Path) -> None:
    path = tmp_path / "sparse.yaml"
    path.write_text("app:\n  mode: sim\n", encoding="utf-8")

    cfg = load_config(str(path))

    assert cfg["app"]["mode"] == "sim"
    assert cfg["execution"]["live_trading"] is False
    assert cfg["monitor"]["predictions_file"] == "predictions_5m.jsonl"
    assert cfg["risk"]["adaptive_risk"]["mode"] == "shadow"


def test_config_schema_rejects_invalid_price_window() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {
                "execution": {
                    "confirmation_gate": {
                        "min_price": 0.8,
                        "max_price": 0.2,
                    }
                }
            }
        )


def test_config_schema_rejects_unknown_top_level_section() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"app": {}, "exection": {}})


def test_live_trading_env_override_is_applied_after_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"execution": {"live_trading": False}}), encoding="utf-8")

    monkeypatch.setenv("LIVE_TRADING", "true")
    cfg = load_app_config(str(path))

    assert cfg.execution.live_trading is True
