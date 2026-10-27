"""Bug case: List index out of bounds."""


def get_color() -> str:
    """Subscripts list index 5 on length 2 list."""
    palette = ["red", "green"]
    return palette[5]
