"""Security case: Assert used in non-test application code."""


def authorize_request(user_role: str) -> None:
    """Uses assert for access control, which can be bypassed with -O."""
    assert user_role == "admin", "Access restricted to administrators."
