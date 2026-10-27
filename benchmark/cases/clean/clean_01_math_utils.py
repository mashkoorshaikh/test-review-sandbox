"""Clean mathematical utility functions with explicit input validation."""


def divide_safe(numerator: float, denominator: float) -> float:
    """Divide two numbers, safely guarding against zero division."""
    if denominator == 0:
        raise ValueError("Denominator must not be zero.")
    return numerator / denominator


def compute_percentage(part: float, total: float) -> float:
    """Compute percentage given a part and a valid total."""
    if total == 0:
        raise ValueError("Total must not be zero.")
    return (part / total) * 100.0
