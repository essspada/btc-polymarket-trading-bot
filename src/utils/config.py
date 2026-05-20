from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.config import AppConfig

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


ENV_BOOL_TRUE = {"1", "true", "yes", "on"}


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in ENV_BOOL_TRUE


def load_app_config(path: str = "configs/default.yaml") -> AppConfig:
    """Load and validate a YAML config as a pydantic model."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = AppConfig.model_validate(raw)
    if "LIVE_TRADING" in os.environ:
        cfg.execution.live_trading = _as_bool(os.environ["LIVE_TRADING"])
    return cfg


def load_config(path: str = "configs/default.yaml") -> dict[str, Any]:
    """Return the validated config in the legacy dict shape."""
    return load_app_config(path).to_runtime_dict()


def ensure_dirs(cfg: dict[str, Any]) -> None:
    paths = cfg.get("paths", {})
    for key in ("logs_dir", "outputs_dir", "reports_dir"):
        p = paths.get(key)
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
