"""Deprecated location. Canonical home: src.gui.config_validation.

Validation exists to drive the GUI's error surface; it has no consumer in the
simulation path. Note src.gui's root module is deliberately Qt-free, so this
stays importable on headless nodes.
"""
from ..gui.config_validation import *          # noqa: F401,F403
from ..gui.config_validation import __all__    # noqa: F401
from ..gui.config_validation import (          # public, but absent from __all__
    PATH_ISSUE_KINDS,
    PATH_KEYS,
)
