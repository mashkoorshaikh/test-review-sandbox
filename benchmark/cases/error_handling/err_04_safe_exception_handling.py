"""Clean error handling with specific exception types and proper chaining."""

import logging

logger = logging.getLogger(__name__)


def parse_and_validate_port(raw_port: str) -> int:
    """Safely converts string to port number, guarding against ValueError with proper chaining."""
    try:
        port = int(raw_port)
    except ValueError as exc:
        logger.warning("Invalid port number format: %s", raw_port)
        raise ValueError(f"Port '{raw_port}' is not a valid integer.") from exc
    if not (1 <= port <= 65535):
        raise ValueError(f"Port {port} is out of valid TCP/UDP range 1-65535.")
    return port

