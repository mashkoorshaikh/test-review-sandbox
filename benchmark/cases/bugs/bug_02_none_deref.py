"""Bug case: Definite None dereference."""


def get_header_value() -> str:
    """Attempts attribute access on a variable assigned to None."""
    header = None
    return header.strip()
