"""Clean case: Safe dictionary lookup using membership guard."""


def get_setting(settings: dict[str, str], key: str) -> str:
    """Safely retrieves value with 'in' membership guard."""
    if key in settings:
        return settings[key]
    return "default"
