from __future__ import annotations

import logging
from rich.console import Console
from rich.logging import RichHandler

# Shared console instance for consistent display
_console: Console | None = None


def get_console() -> Console:
    """Get or create the shared console instance."""
    global _console
    if _console is None:
        _console = Console(stderr=True, force_terminal=True)
    return _console


def setup_logging(level: int = logging.INFO) -> None:
    """Configure Rich logging for the application.

    Ensures idempotency if called multiple times.
    """
    root = logging.getLogger()
    if any(isinstance(h, RichHandler) for h in root.handlers):
        return
    console = get_console()
    handler = RichHandler(console=console, rich_tracebacks=True, show_time=True, show_path=True)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[handler],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

