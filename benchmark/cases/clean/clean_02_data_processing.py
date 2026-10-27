"""Clean data processing functions with proper None checking."""


def normalize_tags(raw_tags: list[str] | None) -> list[str]:
    """Normalize a list of string tags, handling None safely."""
    if raw_tags is None:
        return []
    return [tag.strip().lower() for tag in raw_tags if tag]


def get_display_name(first_name: str | None, last_name: str | None) -> str:
    """Format full display name safely with None guards."""
    parts: list[str] = []
    if first_name is not None:
        parts.append(first_name.strip())
    if last_name is not None:
        parts.append(last_name.strip())
    return " ".join(parts)
