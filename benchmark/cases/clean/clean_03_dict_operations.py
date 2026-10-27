"""Clean dictionary operations using membership tests and safe defaults."""


def get_user_role(roles: dict[str, str], username: str) -> str:
    """Retrieve user role using safe membership guard."""
    if username in roles:
        return roles[username]
    return "guest"


def merge_configs(base: dict[str, str], overrides: dict[str, str]) -> dict[str, str]:
    """Safely combine two configuration dictionaries."""
    result = dict(base)
    for key, val in overrides.items():
        result[key] = val
    return result
