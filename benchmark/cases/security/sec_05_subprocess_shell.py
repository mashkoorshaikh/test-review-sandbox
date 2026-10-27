"""Security case: Subprocess execution with shell=True."""

import subprocess


def execute_ping(host: str) -> None:
    """Executes external shell command using vulnerable shell=True."""
    subprocess.run(f"ping {host}", shell=True, check=False)
