"""Safe edge case: Comprehension target shadowing an outer zero."""


def filter_valid_divisors(items: list[int]) -> list[float]:
    """Outer variable x is zero, but comprehension target x shadows it safely."""
    x = 0
    results = [10 / x for x in items if x != 0]
    if x != 0:
        results.append(float(x))
    return results
