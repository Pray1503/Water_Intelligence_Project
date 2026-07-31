"""Shared logging utilities for Stage 10.

Provides a decorator that logs START/END/execution-time/inputs/outputs
for every public function, per the Stage 10 debug logging requirement.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger("stage10")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)

F = TypeVar("F", bound=Callable[..., Any])


def _summarize(value: Any, max_len: int = 200) -> str:
    """Return a short, log-safe string representation of *value*."""
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001
        text = f"<unrepr-able {type(value).__name__}>"
    if len(text) > max_len:
        text = text[:max_len] + "...(truncated)"
    return text


def log_call(func: F) -> F:
    """Decorator: logs START, important inputs, END, execution time, and
    important outputs for a public function."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        qualified_name = func.__qualname__
        arg_summary = ", ".join(
            [_summarize(a) for a in args]
            + [f"{k}={_summarize(v)}" for k, v in kwargs.items()]
        )
        logger.debug("START %s(%s)", qualified_name, arg_summary)
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - start_time
            logger.debug(
                "FAILED %s after %.4fs: %s: %s",
                qualified_name,
                elapsed,
                type(exc).__name__,
                exc,
            )
            raise
        elapsed = time.perf_counter() - start_time
        logger.debug(
            "END %s | elapsed=%.4fs | output=%s",
            qualified_name,
            elapsed,
            _summarize(result),
        )
        return result

    return wrapper  # type: ignore[return-value]
