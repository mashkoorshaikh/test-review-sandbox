"""Safe edge case: Short-circuit guard protecting None and key access."""


def safe_short_circuit(data: dict[str, str] | None) -> str:
    """Protects dictionary access using short-circuit logical and."""
    if data is not None and "key" in data:
        return data["key"]
    return ""
