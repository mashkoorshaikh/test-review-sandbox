"""Error handling case: Return statement inside finally block."""


def cleanup_resource(flag: bool) -> int:
    """Uses return in finally block which suppresses any raised exceptions."""
    try:
        if flag:
            raise RuntimeError("Operation failed.")
        return 1
    finally:
        return 0
