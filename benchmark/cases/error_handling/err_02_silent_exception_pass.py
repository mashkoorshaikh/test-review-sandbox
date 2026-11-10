"""Error handling case: Silent exception swallowing with try-except-pass."""


def read_auth_token(token_file: str) -> str | None:
    """Swallows general Exception with pass, hiding file system or permission errors."""
    try:
        with open(token_file, encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        pass
    return None

