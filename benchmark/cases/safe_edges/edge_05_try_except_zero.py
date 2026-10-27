"""Safe edge case: Division guarded by try-except ZeroDivisionError."""


def try_divide(numerator: float, denominator: float) -> float:
    """Handles potential zero divisor via explicit ZeroDivisionError block."""
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return 0.0
