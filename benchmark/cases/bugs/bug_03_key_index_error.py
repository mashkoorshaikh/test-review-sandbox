"""Bug case: Missing dictionary key on a known dictionary literal."""


def fetch_database_port() -> int:
    """Subscripts a missing key from a dictionary literal."""
    config = {"host": "localhost", "user": "admin"}
    return config["port"]
