from __future__ import annotations

from http import HTTPStatus
from typing import Any, Dict

import structlog
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

__all__: list[str] = ["add_exception_handlers"]

logger = structlog.get_logger("errors")


def _build_error_payload(
    code: str | int,
    message: str,
    request_id: str | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return a JSON-serialisable error envelope.

    Parameters
    ----------
    code:
        A machine-readable error code (snake_case) or HTTP status integer.
    message:
        Human-readable description (English, sentence-cased).
    request_id:
        Optional correlation ID injected by `RequestLoggingMiddleware`.
    extra:
        Optional additional payload for debugging (e.g. validation details).
    """

    payload: Dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id,
        }
    }
    if extra:
        payload["error"].update(extra)
    return payload


async def _http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    """Handle exceptions explicitly raised by the application/routers."""

    logger.warning(
        "http_exception",
        path=request.url.path,
        status_code=exc.status_code,
        detail=str(exc.detail),
    )

    payload = _build_error_payload(
        code=exc.status_code,
        message=str(exc.detail),
        request_id=request.headers.get("x-request-id"),
    )
    # Maintain compatibility with FastAPI default error schema expected by
    # some client tests that assert on ``response.json()['detail']``.
    payload["detail"] = str(exc.detail)

    return JSONResponse(status_code=exc.status_code, content=payload)


async def _validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle body/query/path parameter validation failures (422)."""

    logger.warning(
        "validation_error",
        path=request.url.path,
        errors=exc.errors(),
    )

    payload = _build_error_payload(
        code="validation_error",
        message="Invalid request parameters.",
        request_id=request.headers.get("x-request-id"),
        extra={"details": exc.errors()},
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=payload
    )


async def _unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:  # noqa: D401 – FastAPI handler sig
    """Catch-all for unexpected errors – returns HTTP 500."""

    logger.exception(
        "unhandled_exception",
        path=request.url.path,
        error=str(exc),
    )

    payload = _build_error_payload(
        code="internal_server_error",
        message="An unexpected error occurred.",
        request_id=request.headers.get("x-request-id"),
    )
    return JSONResponse(
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,  # 500
        content=payload,
    )


def add_exception_handlers(app: FastAPI) -> None:  # noqa: D401 – imperative
    """Register all global exception handlers on **app**."""

    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)
