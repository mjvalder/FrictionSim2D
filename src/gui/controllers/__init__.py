"""Application state and the operations on it.

Qt-free: these hold and manipulate config state, and the widgets observe them.
Keeping them free of Qt is what lets the interesting behaviour -- import,
export, validation, sweep counting -- be tested without a display.
"""

from .document import ConfigDocument, DocumentError

__all__ = ['ConfigDocument', 'DocumentError']
