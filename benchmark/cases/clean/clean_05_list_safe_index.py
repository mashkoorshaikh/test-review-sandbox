"""Clean case: Safe list index protected by len bounds guard."""


def get_second_item(items: list[int]) -> int:
    """Safely retrieves second element after length check."""
    if len(items) >= 2:
        return items[1]
    return 0
