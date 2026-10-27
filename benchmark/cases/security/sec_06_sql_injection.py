"""Security case: Dynamic SQL query string construction."""


def query_user(username: str) -> str:
    """Constructs dynamic SQL query vulnerable to injection."""
    query = f"SELECT * FROM users WHERE name = '{username}'"
    return query
