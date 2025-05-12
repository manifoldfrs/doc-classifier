from __future__ import annotations

import logging
import sys
import time
import uuid
from typing import Any, Awaitable, Callable

# structlog must be imported before its typing helpers
import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from structlog.types import Processor

__all__: list[str] = [
    "configure_logging",
    "RequestLoggingMiddleware",
]


def _ensure_request_context(
    logger: Any,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """Guarantee *request_id*, *user* and *path* keys exist in *event_dict*."""

    event_dict.setdefault("request_id", None)
    event_dict.setdefault("user", None)
    event_dict.setdefault("path", None)
    return event_dict


# Re-assemble the JSON processor chain with the new helper placed *after*
# ``merge_contextvars`` so it only fills in missing keys.
_JSON_PROCESSORS: list[Processor] = [
    structlog.contextvars.merge_contextvars,
    _ensure_request_context,
    structlog.processors.add_log_level,
    structlog.processors.TimeStamper(fmt="iso"),
    structlog.processors.JSONRenderer(),
]


def _configure_stdlib_logging(level: int) -> None:
    """Configure the built-in *logging* module to route records to structlog.

    The FastAPI ecosystem (Uvicorn, Starlette) still uses stdlib logging.  We
    therefore configure a **StreamHandler** pointing to *stderr* and set a
    simple formatter.  Structlog will pick up the record and apply its own
    processors, ensuring consistent output.
    """

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(message)s"))  # structlog formats

    # Remove default handlers to avoid duplicate logs in some runtimes.
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


_LOGGING_CONFIGURED: bool = False


def configure_logging(debug: bool = False) -> None:
    """Initialise `structlog` for the entire process.

    Call **exactly once** and *before* any loggers are created.  `uvicorn`
    imports this module *after* its own setup so we cannot rely on its default
    formatter.  The function is idempotent – multiple calls are safe but no-op
    after the first.

    Parameters
    ----------
    debug:
        When *True* lowers the log level to ``DEBUG``; otherwise ``INFO``.
    """

    global _LOGGING_CONFIGURED

    if _LOGGING_CONFIGURED:
        return

    level: int = logging.DEBUG if debug else logging.INFO

    _configure_stdlib_logging(level)

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(level),
        processors=_JSON_PROCESSORS,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _LOGGING_CONFIGURED = True


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log each HTTP request with structured details and latency.

    The middleware is inserted **early** in the FastAPI stack so that all
    downstream handlers inherit the bound `request_id`, `path`, and `method`
    context variables – this ensures consistency across log entries.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        start: float = time.perf_counter()
        request_id: str = (
            request.headers.get("x-request-id") or uuid.uuid4().hex  # 32-char hex
        )

        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            path=request.url.path,
            method=request.method,
        )

        try:
            response: Response = await call_next(request)
        finally:
            duration_ms: float = (time.perf_counter() - start) * 1000
            logger = structlog.get_logger("http")
            logger.info(
                "request_completed",
                status_code=response.status_code if "response" in locals() else 500,
                duration_ms=round(duration_ms, 2),
                user=getattr(request.state, "user", None),
            )
            structlog.contextvars.clear_contextvars()

        response.headers["X-Request-ID"] = request_id
        return response
