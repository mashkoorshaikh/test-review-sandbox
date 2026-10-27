"""Bug case: Use of undefined variable name."""


def calculate_tax(subtotal: float) -> float:
    """References undefined surcharge variable."""
    rate = 0.05
    return subtotal * rate + surcharge
