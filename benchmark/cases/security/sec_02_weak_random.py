"""Security case: Insecure pseudo-random generator used in token generation."""

import random


def generate_session_token() -> str:
    """Uses random.choice to generate an authentication token."""
    charset = "abcdef0123456789"
    return "".join(random.choice(charset) for _ in range(16))
