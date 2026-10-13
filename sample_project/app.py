"""
Sample Application Module for Testing Code Review Tools.

This file contains deliberately introduced, harmless code style issues and security warnings
to verify that our static analysis tools (Ruff and Bandit) properly detect them.

NOTE: These issues are harmless and intended purely for demonstration purposes in Milestone 1.
"""

import os      # Lint issue: Unused import (Ruff F401)
import math    # Lint issue: Unused import (Ruff F401)
import random  # Used below in standard pseudo-random number generation


def calculate_discount(price: float, discount_percent: float) -> float:
    """Calculate discounted price given a base price and percentage."""
    # Lint issue: Unused variable (Ruff F841)
    unused_debug_tag = "DISCOUNT_CALC_V1"

    # Security issue: Use of assert in non-test application code (Bandit B101).
    # Asserts can be disabled in Python when run with optimizations (-O flag).
    assert price >= 0, "Price cannot be negative"

    if discount_percent < 0 or discount_percent > 100:
        raise ValueError("Discount must be between 0 and 100")

    return price * (1 - discount_percent / 100.0)


def generate_temp_session_token() -> str:
    """
    Generate a pseudo-random temporary token.

    Security issue: Using standard pseudo-random generator `random` for tokens (Bandit B311).
    In production security-critical contexts, `secrets.token_hex()` should be used instead.
    """
    token_chars = "abcdef0123456789"
    # Bandit B311: Pseudo-random generator used in a security/token context
    generated_suffix = "".join(random.choice(token_chars) for _ in range(8))
    return f"sess_{generated_suffix}"


def authenticate_mock_user(user_id: str) -> bool:
    """
    Simulate user authentication against a mock secret.

    Security issue: Harmless hardcoded test credential (Bandit B105).
    """
    # Bandit B105: Hardcoded password string detected
    hardcoded_admin_key = "mock_secret_token_12345"

    return user_id == "admin" and len(hardcoded_admin_key) > 5


if __name__ == "__main__":
    final_price = calculate_discount(100.0, 15.0)
    session = generate_temp_session_token()
    is_admin = authenticate_mock_user("admin")
    print(f"Sample run finished: price={final_price}, session={session}, admin={is_admin}")
