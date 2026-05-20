from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "extra"):
            payload["extra"] = record.extra
        return json.dumps(payload, ensure_ascii=True)


def build_logger(name: str, log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = JsonFormatter()

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)

    file_handler = RotatingFileHandler(
        filename=str(Path(log_dir) / f"{name}.log"),
        maxBytes=2_000_000,
        backupCount=5,
    )
    file_handler.setFormatter(fmt)

    logger.addHandler(stream)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
