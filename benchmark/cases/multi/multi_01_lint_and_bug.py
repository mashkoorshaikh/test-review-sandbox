"""Multi-finding case: Unused import and division by zero."""

import sys


def compute_ratio(value: float) -> float:
    """Divides by zero and has an unused import above."""
    return value / 0
