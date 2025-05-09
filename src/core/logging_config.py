"""
Logging configuration for the application.

This module sets up structured JSON logging for the application,
with appropriate handlers and formatters for development and production.
"""

import logging
import sys
from typing import Dict, Any

from pythonjsonlogger import jsonlogger


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure application-wide logging with JSON formatting.

    Args:
        log_level: The minimum log level to capture ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
                  Defaults to 'INFO'.

    Returns:
        None
    """
    # Get the root logger
    root_logger = logging.getLogger()

    # Clear any existing handlers
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    # Set the log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # JSON formatter with standard fields
    formatter = jsonlogger.JsonFormatter(
        fmt="%(timestamp)s %(level)s %(name)s %(message)s",
        rename_fields={"levelname": "level", "asctime": "timestamp"},
        json_ensure_ascii=False,
    )
    console_handler.setFormatter(formatter)

    # Add handler to the root logger
    root_logger.addHandler(console_handler)

    # Disable propagation for uvicorn access logs to avoid duplicate logs
    logging.getLogger("uvicorn.access").propagate = False

    # Log that logging has been set up
    root_logger.info("Logging configured", extra={"log_level": log_level})


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger with the given name.

    Args:
        name: The name for the logger, typically the module name.

    Returns:
        logging.Logger: A configured logger instance.
    """
    return logging.getLogger(name)


class LoggingMiddleware:
    """
    Middleware to add request/response logging for FastAPI.

    This middleware logs information about incoming requests and outgoing responses,
    including method, path, status code, and timing information.
    """

    async def __call__(self, request: Any, call_next: Any) -> Any:
        """Process an incoming request and log request/response details."""
        import time
        import uuid
        from fastapi import Request

        request_id = str(uuid.uuid4())
        logger = get_logger("api")

        # Start timer and log request
        start_time = time.time()

        # Parse request as FastAPI Request if it isn't already
        if not isinstance(request, Request):
            request = Request(scope=request.scope)

        # Log the incoming request
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client_ip": request.client.host if request.client else None,
            },
        )

        # Process the request
        try:
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Log the response
            logger.info(
                f"Request completed: {request.method} {request.url.path} - {response.status_code}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "processing_time_ms": round(process_time * 1000, 2),
                },
            )

            return response

        except Exception as e:
            # Log any unhandled exceptions
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} - {str(e)}",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "processing_time_ms": round(process_time * 1000, 2),
                },
                exc_info=True,
            )
            raise
