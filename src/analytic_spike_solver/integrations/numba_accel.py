from __future__ import annotations


def numba_available() -> bool:
    try:
        import numba  # noqa: F401
    except Exception:
        return False
    return True
