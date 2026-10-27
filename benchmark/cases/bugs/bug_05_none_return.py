"""Bug case: None dereference on function return value."""


def fetch_user(uid: int):
    """Simulate user lookup that returns None."""
    return None


def get_user_name(uid: int) -> str:
    """Dereferences None returned by helper function."""
    user = fetch_user(uid)
    return user.name
