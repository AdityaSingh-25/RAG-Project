"""JSON-structured logging shared across the engine.

Each log record carries a ``trace_id`` (per query) and a free-form payload
of ``extra`` fields, so downstream tools can group events by query and
filter by node/stage. ``configure_logging`` is idempotent.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record.

    Standard fields are always present; everything passed via ``extra``
    lands as top-level keys. ``exc_info`` is rendered into ``exception``.
    """

    _RESERVED = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 6),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in self._RESERVED or key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _ContextFilter(logging.Filter):
    """Default-fills ``trace_id`` so JSON output is always shaped uniformly."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "trace_id"):
            record.trace_id = "-"
        return True


_OWNED_MARKER = "_rag_engine_handler"


def configure_logging(log_format: str = "json", level: str = "INFO") -> None:
    """Install our handler on the root logger. Safe to call repeatedly.

    Only handlers this module installed are removed; pytest's caplog and any
    other test-time handlers are left in place.
    """
    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, _OWNED_MARKER, False):
            root.removeHandler(existing)

    handler = logging.StreamHandler(sys.stderr)
    setattr(handler, _OWNED_MARKER, True)
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s trace=%(trace_id)s %(message)s")
        )
    handler.addFilter(_ContextFilter())
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    trace_id: str,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit one structured event."""
    extra = {"event": event, "trace_id": trace_id, **fields}
    logger.log(level, event, extra=extra)


def now_ms() -> float:
    """Monotonic milliseconds for duration timing."""
    return time.perf_counter() * 1000.0
