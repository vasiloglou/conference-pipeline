"""Logging configuration for the NeurIPS pipeline."""

import logging
import sys
from pathlib import Path
from typing import Any

# Flag to track if structlog is available and configured
_structlog_configured = False
_use_structlog = False


def _try_setup_structlog(
    level: str = "INFO",
    log_file: Path | None = None,
    json_format: bool = False,
) -> bool:
    """Try to configure structlog if available.

    Args:
        level: Logging level.
        log_file: Optional file path for logging output.
        json_format: Whether to use JSON formatting.

    Returns:
        True if structlog was configured successfully.
    """
    global _structlog_configured, _use_structlog

    try:
        import structlog
        from structlog.stdlib import (
            BoundLogger,
            ProcessorFormatter,
            add_log_level,
            add_logger_name,
            filter_by_level,
        )

        # Configure structlog processors
        shared_processors: list[Any] = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
        ]

        if json_format:
            # JSON format for production/structured logging
            shared_processors.append(structlog.processors.format_exc_info)
            renderer = structlog.processors.JSONRenderer()
        else:
            # Console format for development
            renderer = structlog.dev.ConsoleRenderer(
                colors=sys.stdout.isatty(),
                exception_formatter=structlog.dev.plain_traceback,
            )

        structlog.configure(
            processors=shared_processors
            + [
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        # Configure stdlib logging to work with structlog
        formatter = ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )

        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper()))

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File handler (if specified)
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, level.upper()))
            # Use JSON for file output
            json_formatter = ProcessorFormatter(
                foreign_pre_chain=shared_processors,
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),
                ],
            )
            file_handler.setFormatter(json_formatter)
            root_logger.addHandler(file_handler)

        _structlog_configured = True
        _use_structlog = True
        return True

    except ImportError:
        return False


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    name: str = "conference_pipeline",
    use_structlog: bool = True,
    json_format: bool = False,
) -> logging.Logger:
    """Configure and return a logger for the pipeline.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path for logging output.
        name: Logger name.
        use_structlog: Whether to use structlog if available.
        json_format: Whether to use JSON formatting (structlog only).

    Returns:
        Configured logger instance.
    """
    global _use_structlog, _structlog_configured

    # Clear root logger handlers to prevent duplicates
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Try structlog first if requested
    if use_structlog and not _structlog_configured:
        if _try_setup_structlog(level, log_file, json_format):
            return get_logger(name)

    # Fall back to stdlib logging
    _use_structlog = False
    _structlog_configured = False
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Prevent propagation to root to avoid duplicate logs
    logger.propagate = False

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "conference_pipeline") -> Any:
    """Get an existing logger or create a basic one.

    If structlog is configured, returns a structlog BoundLogger.
    Otherwise, returns a stdlib Logger.

    Args:
        name: Logger name.

    Returns:
        Logger instance (structlog.BoundLogger or logging.Logger).
    """
    global _use_structlog

    if _use_structlog:
        try:
            import structlog

            return structlog.get_logger(name)
        except ImportError:
            pass

    logger = logging.getLogger(name)
    if not logger.handlers:
        # Set up basic logging if not configured
        setup_logging(name=name, use_structlog=False)
    return logger


def bind_context(**kwargs: Any) -> None:
    """Bind context variables for structured logging.

    These will be included in all subsequent log messages in the current context.

    Args:
        **kwargs: Key-value pairs to bind to the logging context.
    """
    if _use_structlog:
        try:
            import structlog

            structlog.contextvars.bind_contextvars(**kwargs)
        except ImportError:
            pass


def unbind_context(*keys: str) -> None:
    """Unbind context variables from structured logging.

    Args:
        *keys: Keys to remove from the logging context.
    """
    if _use_structlog:
        try:
            import structlog

            structlog.contextvars.unbind_contextvars(*keys)
        except ImportError:
            pass


def clear_context() -> None:
    """Clear all bound context variables."""
    if _use_structlog:
        try:
            import structlog

            structlog.contextvars.clear_contextvars()
        except ImportError:
            pass
