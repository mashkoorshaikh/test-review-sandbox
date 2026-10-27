"""Bug case: Division by literal zero."""


def calculate_inverse(val: float) -> float:
    """Unconditionally divides by constant zero."""
    return val / 0
