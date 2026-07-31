"""
===============================================================================
WATER INTELLIGENCE PLATFORM - SHARED LOGGING UTILITY
Module: src/common/logging_utils.py
===============================================================================

LAYER: Common / Infrastructure
PURPOSE:
    Provides the logging philosophy shared across every Stage 11 module and future
    platform stages: a configured logger factory, an ENTRY/EXIT call-tracing
    decorator, a block-level timing context manager, and a bounded value-
    summarization helper so DEBUG-level logs stay informative without flooding the
    log stream with full feature vectors or fitted model objects.

ARCHITECTURAL NOTES:
    - Root Namespace: Uses 'water_intelligence' as the single root hierarchy. Child
      loggers resolve to 'water_intelligence.stage11.<module_name>'.
    - Thread-Safe Configuration: Uses a reentrant lock to make configure_logging()
      fully thread-safe during concurrent initialization.
    - Zero Exception Interference: @log_call logs errors with full traceback via
      logger.exception() and re-raises the exact exception unchanged.

IMPORTS:
    - functools, inspect, logging, sys, threading, time, typing
"""

from __future__ import annotations

import functools
import logging
import sys
import threading
import time
from typing import Any, Callable, TypeVar

__all__ = [
    "WATER_INTELLIGENCE_LOGGER",
    "LogTimer",
    "configure_logging",
    "get_logger",
    "log_call",
    "summarize_value",
]

# ---------------------------------------------------------------------------
# Logger Configuration & Hierarchy
# ---------------------------------------------------------------------------

#: Root logger namespace for the entire Water Intelligence Platform hierarchy.
WATER_INTELLIGENCE_LOGGER = "water_intelligence"

_LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | "
    "%(funcName)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False
_config_lock = threading.Lock()


def configure_logging(
    level: int = logging.DEBUG,
    propagate: bool = False,
) -> None:
    """Configure the "water_intelligence" logger hierarchy exactly once per process.

    Thread-safe and idempotent: safe to call from multiple entry points (a script,
    FastAPI app factory, or test runner) without attaching duplicate handlers or
    emitting duplicate log lines.

    Parameters
    ----------
    level : int, default=logging.DEBUG
        Log level for the root logger hierarchy.
    propagate : bool, default=False
        Whether to propagate log records to the standard Python root logger.
        Set to True when integrating with centralized server logging frameworks.
    """
    global _configured
    if _configured:
        return

    with _config_lock:
        if _configured:
            return

        logger = logging.getLogger(WATER_INTELLIGENCE_LOGGER)
        logger.setLevel(level)

        # Attach standard output stream handler if none present
        if not logger.handlers:
            handler = logging.StreamHandler(stream=sys.stdout)
            handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
            logger.addHandler(handler)

        logger.propagate = propagate
        _configured = True


def get_logger(module_name: str) -> logging.Logger:
    """Return a child logger under 'water_intelligence.stage11' for *module_name*.

    Calls configure_logging() idempotently so modules can obtain loggers at import time.

    Parameters
    ----------
    module_name : str
        Identifies the calling module in log lines (e.g. "inference.model_loader").

    Returns
    -------
    logging.Logger
        Configured child logger instance.
    """
    configure_logging()
    return logging.getLogger(f"{WATER_INTELLIGENCE_LOGGER}.stage11.{module_name}")


# ---------------------------------------------------------------------------
# Bounded Value Summarization
# ---------------------------------------------------------------------------

_MAX_REPR_LENGTH = 200
_MAX_DICT_KEY_PREVIEW = 3


def summarize_value(value: Any) -> str:
    """Produce a short, bounded, log-safe description of *value*.

    Never raises: summarization failure falls back to a type-name-only string.

    Behavior by type:
    - Objects with `.shape` (NumPy arrays, DataFrames) -> "TypeName[shape=(...)]"
    - Dicts -> "dict[keys=N:first_keys=['a', 'b']]"
    - Collections (list, tuple, set) -> "TypeName[len=N]"
    - Scalars & primitives -> repr(value), truncated if exceeding _MAX_REPR_LENGTH.
    """
    try:
        if hasattr(value, "shape"):
            return f"{type(value).__name__}[shape={tuple(value.shape)}]"

        if isinstance(value, dict):
            key_count = len(value)
            if key_count > 0:
                first_keys = [
                    repr(k) for k in list(value.keys())[:_MAX_DICT_KEY_PREVIEW]
                ]
                keys_preview = f":first_keys=[{', '.join(first_keys)}]"
            else:
                keys_preview = ""
            return f"dict[keys={key_count}{keys_preview}]"

        if isinstance(value, (list, tuple, set)):
            return f"{type(value).__name__}[len={len(value)}]"

        text = repr(value)
        if len(text) > _MAX_REPR_LENGTH:
            return text[:_MAX_REPR_LENGTH] + "...(truncated)"
        return text

    except Exception:  # noqa: BLE001
        return f"<{type(value).__name__}: unrepresentable>"


def _summarize_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Build a bounded, single-line input summary for log_call's ENTRY log."""
    parts = [summarize_value(a) for a in args]
    parts += [f"{k}={summarize_value(v)}" for k, v in kwargs.items()]
    return ", ".join(parts) if parts else "()"


# ---------------------------------------------------------------------------
# Execution Tracing Decorator
# ---------------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])


def log_call(func: F) -> F:
    """Decorator that logs ENTRY, EXIT, and ERROR around function execution."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = get_logger(func.__module__)
        qualified_name = func.__qualname__
        input_summary = _summarize_args(args, kwargs)

        logger.debug("ENTRY %s | args=%s", qualified_name, input_summary)

        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            logger.exception(
                "ERROR %s | elapsed_ms=%.3f | exception_type=%s | args=%s",
                qualified_name,
                elapsed_ms,
                type(exc).__name__,
                input_summary,
            )
            raise

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        logger.debug(
            "EXIT %s | elapsed_ms=%.3f | result=%s",
            qualified_name,
            elapsed_ms,
            summarize_value(result),
        )
        return result

    return wrapper  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Block-Level Timing Context Manager
# ---------------------------------------------------------------------------


class LogTimer:
    """Context manager that logs execution duration of a code block."""

    def __init__(self, logger: logging.Logger, operation_name: str) -> None:
        self._logger = logger
        self._operation_name = operation_name
        self._start_time: float = 0.0

    def __enter__(self) -> LogTimer:
        self._start_time = time.perf_counter()
        self._logger.debug("TIMING START | %s", self._operation_name)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0
        if exc_type is None:
            self._logger.debug(
                "TIMING END | %s | elapsed_ms=%.3f",
                self._operation_name,
                elapsed_ms,
            )
        else:
            self._logger.debug(
                "TIMING END (failed) | %s | elapsed_ms=%.3f | exception_type=%s",
                self._operation_name,
                elapsed_ms,
                exc_type.__name__,
            )
