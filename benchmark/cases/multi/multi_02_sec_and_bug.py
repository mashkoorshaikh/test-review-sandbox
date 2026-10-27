"""Multi-finding case: Security assert and None dereference."""


def process_user_profile(user_id: int) -> str:
    """Contains both an insecure assert and an unhandled None dereference."""
    assert user_id > 0, "Invalid user identifier"
    profile = None
    return profile.get_name()
