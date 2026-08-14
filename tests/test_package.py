"""Smoke tests: the package imports and reports a sane version."""

from __future__ import annotations

import fluoroflow


def test_version_is_a_nonempty_string() -> None:
    assert isinstance(fluoroflow.__version__, str)
    assert fluoroflow.__version__


def test_public_api_is_explicit() -> None:
    # Every name in __all__ must actually exist. This guards against the classic
    # "renamed the function, forgot the export" breakage.
    for name in fluoroflow.__all__:
        assert hasattr(fluoroflow, name), name
