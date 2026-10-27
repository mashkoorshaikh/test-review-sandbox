"""Clean case: Secure random generation and SHA-256 hashing."""

import hashlib
import secrets


def generate_session_token() -> str:
    """Uses cryptographically strong secrets module and SHA-256."""
    random_bytes = secrets.token_bytes(32)
    return hashlib.sha256(random_bytes).hexdigest()
