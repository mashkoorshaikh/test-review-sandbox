"""Security case: Hardcoded API secret token."""


def get_client_credentials() -> dict[str, str]:
    """Contains a hardcoded secret token string."""
    api_password = "super_secret_production_token_123"
    return {"Authorization": f"Bearer {api_password}"}
