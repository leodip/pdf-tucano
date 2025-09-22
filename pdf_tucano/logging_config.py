from __future__ import annotations

import json
import logging
import logging.config
from typing import Any, Tuple


_RESERVED_LOG_RECORD_ATTRS = set(logging.makeLogRecord({}).__dict__.keys())


def _normalize_context(value: Any) -> Any:
    """Ensure values in the structured context are JSON serializable."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class ContextualFormatter(logging.Formatter):
    """Formatter that appends structured context stored in LogRecord extras."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - formatting helper
        message = super().format(record)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_LOG_RECORD_ATTRS and not key.startswith("_")
        }
        if not extras:
            return message

        serialized = json.dumps({k: _normalize_context(v) for k, v in extras.items()}, default=str, ensure_ascii=False)
        return f"{message} | context={serialized}"


def configure_logging(log_level_name: str) -> Tuple[int, bool]:
    """Configure root logging with contextual formatting and return resolved level."""

    level_mapping = getattr(logging, "getLevelNamesMapping", lambda: logging._nameToLevel)()
    normalized_name = (log_level_name or "").upper()

    resolved_level = level_mapping.get(normalized_name)
    fallback = False
    if resolved_level is None:
        resolved_level = logging.INFO
        fallback = True

    logging.captureWarnings(True)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "context": {
                    "()": "pdf_tucano.logging_config.ContextualFormatter",
                    "fmt": "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": resolved_level,
                    "formatter": "context",
                }
            },
            "root": {
                "handlers": ["console"],
                "level": resolved_level,
            },
            "loggers": {
                "uvicorn": {
                    "handlers": ["console"],
                    "level": resolved_level,
                    "propagate": False,
                },
                "uvicorn.error": {
                    "handlers": ["console"],
                    "level": resolved_level,
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["console"],
                    "level": resolved_level,
                    "propagate": False,
                },
            },
        }
    )

    return resolved_level, fallback
