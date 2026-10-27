"""Bug case: Division by zero in else branch of non-zero guard."""


def compute_inverse(n: int) -> float:
    """Divides by zero in the inverted else branch."""
    if n != 0:
        return 1.0 / n
    else:
        return 100.0 / n
