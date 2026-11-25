# utils/logging.py - Structured Logging

Logging configuration supporting both stdlib logging and structlog.

## Location

`src/utils/logging.py`

## Functions

### setup_logging

Configure and return a logger for the pipeline.

```python
def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    name: str = "conference_pipeline",
    use_structlog: bool = True,
    json_format: bool = False,
) -> logging.Logger:
    """
    Configure logging for the pipeline.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging output
        name: Logger name
        use_structlog: Whether to use structlog if available
        json_format: Whether to use JSON formatting (structlog only)

    Returns:
        Configured logger instance
    """
```

### get_logger

Get an existing logger or create a basic one.

```python
def get_logger(name: str = "conference_pipeline") -> Any:
    """
    Get logger instance.

    Returns structlog.BoundLogger if configured, otherwise logging.Logger.
    """
```

### bind_context

Bind context variables for structured logging.

```python
def bind_context(**kwargs: Any) -> None:
    """
    Bind context variables for all subsequent log messages.

    Example:
        bind_context(presentation_id="paper-123", slide_index=5)
    """
```

### unbind_context

Remove context variables from structured logging.

```python
def unbind_context(*keys: str) -> None:
    """
    Remove keys from logging context.

    Example:
        unbind_context("presentation_id", "slide_index")
    """
```

### clear_context

Clear all bound context variables.

```python
def clear_context() -> None:
    """Clear all context variables."""
```

## Features

### Automatic Fallback

- Uses structlog if installed and requested
- Falls back to stdlib logging otherwise
- No code changes needed

### Output Formats

**Console (Development):**
```
2024-01-15 10:30:45 [info     ] Processing presentation      presentation_id=paper-123
```

**JSON (Production/File):**
```json
{"timestamp": "2024-01-15T10:30:45", "level": "info", "event": "Processing presentation", "presentation_id": "paper-123"}
```

### Context Variables

Context variables are automatically included in all log messages:

```python
bind_context(presentation_id="paper-123")
logger.info("Downloading assets")     # Includes presentation_id
logger.info("OCR complete")           # Includes presentation_id
unbind_context("presentation_id")
```

## Usage Examples

### Basic Setup

```python
from utils.logging import setup_logging, get_logger

# Setup at application start
setup_logging(level="INFO")

# Get logger in any module
logger = get_logger(__name__)
logger.info("Pipeline started")
```

### With File Output

```python
from pathlib import Path

setup_logging(
    level="DEBUG",
    log_file=Path("logs/pipeline.log"),
    json_format=True,  # JSON in file
)
```

### With Context

```python
from utils.logging import get_logger, bind_context, clear_context

logger = get_logger(__name__)

def process_presentation(pres_id: str):
    bind_context(presentation_id=pres_id)
    try:
        logger.info("Starting processing")
        # ... processing ...
        logger.info("Processing complete")
    finally:
        clear_context()
```

### Log Levels

```python
logger.debug("Detailed debug info")
logger.info("Normal operation")
logger.warning("Something unexpected")
logger.error("An error occurred")
logger.exception("Error with traceback")  # Includes stack trace
```

## Configuration

### Environment Variables

```env
LOG_LEVEL=INFO
LOG_FILE=./logs/pipeline.log
USE_STRUCTURED_LOGGING=true
```

### In Settings

```python
@dataclass
class PipelineSettings:
    log_level: str = "INFO"
    log_file: Path | None = None
    use_structured_logging: bool = True
```

## Dependencies

**Required:** None (uses stdlib logging)

**Optional:**
```bash
pip install structlog  # For structured logging
```

## Output Examples

### Development (Console)

```
2024-01-15 10:30:45 [info     ] Discovering presentations
2024-01-15 10:30:46 [info     ] Found 1500 presentations
2024-01-15 10:30:47 [info     ] Downloading assets           presentation_id=paper-001
2024-01-15 10:30:48 [warning  ] Rate limited, waiting        domain=slideslive.com wait_seconds=2.5
2024-01-15 10:30:50 [info     ] Download complete            presentation_id=paper-001 size_mb=15.2
```

### Production (JSON File)

```json
{"timestamp": "2024-01-15T10:30:45.123Z", "level": "info", "logger": "conference_pipeline.cli", "event": "Discovering presentations"}
{"timestamp": "2024-01-15T10:30:46.456Z", "level": "info", "logger": "conference_pipeline.cli", "event": "Found 1500 presentations"}
{"timestamp": "2024-01-15T10:30:47.789Z", "level": "info", "logger": "conference_pipeline.downloader", "event": "Downloading assets", "presentation_id": "paper-001"}
```

## Source References

- `src/utils/logging.py`
