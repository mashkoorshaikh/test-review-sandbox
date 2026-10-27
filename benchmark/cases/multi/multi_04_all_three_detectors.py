"""Multi-issue case: Unused variable, weak MD5, and division by zero."""

import hashlib


def process_order(price: float) -> float:
    """Combines lint, security, and semantic bugs in one function."""
    unused_order_id = 9999
    digest = hashlib.md5(b"order").hexdigest()
    zero_val = 0
    return (price + len(digest)) / zero_val
