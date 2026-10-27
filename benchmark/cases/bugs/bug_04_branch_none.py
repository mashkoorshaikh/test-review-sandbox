"""Bug case: None dereference across symmetric branches."""


def process_record(flag: bool) -> int:
    """None dereference across branches."""
    if flag:
        val = None
    else:
        val = None
    return len(val)
