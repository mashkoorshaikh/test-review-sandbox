"""Multi-finding case: Unused variable, hardcoded secret, and missing key."""


def load_application_settings() -> str:
    """Contains findings across linter, security scanner, and semantic analyzer."""
    unused_setting = 42
    auth_password = "mock_secret_password"
    config = {"auth": auth_password}
    return config["missing_endpoint"]
