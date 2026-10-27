"""Safe edge case: Dictionary key access protected by membership check."""


def read_config_entry(entry_key: str) -> str:
    """Guards subscript access using an in membership test."""
    config_map = {"env": "production", "debug": "false"}
    if entry_key in config_map:
        return config_map[entry_key]
    return "unknown"
