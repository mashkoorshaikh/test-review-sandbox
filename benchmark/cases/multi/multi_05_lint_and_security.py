"""Multi-issue case: Hardcoded API secret token with undefined variable."""


def verify_credentials(authenticator) -> str:
    """Combines hardcoded secret token and undefined name."""
    api_token = "secret_auth_token_998877"
    return authenticator.login(api_token, missing_scope)
