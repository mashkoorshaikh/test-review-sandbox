"""Security case: Weak MD5 hash algorithm used for token generation."""

import hashlib


def hash_token(data: str) -> str:
    """Uses insecure MD5 hashing for token generation."""
    return hashlib.md5(data.encode("utf-8")).hexdigest()
