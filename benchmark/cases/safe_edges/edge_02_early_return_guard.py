"""Safe edge case: None-defaulted parameter with early-return guard."""


def format_header(title: str | None = None) -> str:
    """Guarded parameter returning early when None."""
    if title is None:
        return ""
    return title.strip().upper()
