"""Error handling case: Bare except clause catching BaseException."""


def parse_port(raw_port: str) -> int:
    """Parses port number but uses bare except, catching SystemExit and KeyboardInterrupt."""
    try:
        return int(raw_port)
    except:
        return 8080

