"""Keeping the terminal readable.

Rendering the simulation cell makes ovito fail to bind a vertex buffer for its
thick-line shader during the viewport's *picking* pass. Measured: with the cell
shown, one interaction produced 286 messages; with it hidden, none. Picking runs
whenever the pointer moves over the viewport, so the output arrives two lines at
a time and buries everything else.

Setting the cell's ``line_width`` to zero does not avoid it -- the picking pass
uses the thick-line shader whatever the width -- and the alternative, not drawing
the cell, would remove something worth seeing. The failure is in the renderer,
is cosmetic (the box still draws), and is intermittent, which makes it noise
rather than a signal.

So the messages are collapsed and reported once with a count on exit, rather
than either flooding the terminal or being hidden outright. Everything not
matching :data:`SUPPRESSED_PATTERNS` is passed through untouched.
"""

from __future__ import annotations

import sys
from collections import Counter
from typing import Optional

from PySide6.QtCore import QtMsgType, qInstallMessageHandler

__all__ = ['install_message_filter', 'suppressed_summary', 'SUPPRESSED_PATTERNS']

#: Substrings identifying renderer noise worth collapsing. Deliberately narrow:
#: anything not matching here is passed through untouched.
SUPPRESSED_PATTERNS = (
    'OpenGLShaderHelper::bindBuffer',
    'Failed to bind OpenGL vertex buffer',
    'line_thick_picking',
)

_counts: Counter = Counter()

_LEVELS = {
    QtMsgType.QtDebugMsg: 'debug',
    QtMsgType.QtInfoMsg: 'info',
    QtMsgType.QtWarningMsg: 'warning',
    QtMsgType.QtCriticalMsg: 'critical',
    QtMsgType.QtFatalMsg: 'fatal',
}


def _handler(mode, _context, message: str) -> None:
    for pattern in SUPPRESSED_PATTERNS:
        if pattern in message:
            _counts[pattern] += 1
            return
    level = _LEVELS.get(mode, 'info')
    sys.stderr.write(f"[qt {level}] {message}\n")


def install_message_filter() -> None:
    """Route Qt messages through the filter."""
    qInstallMessageHandler(_handler)


def suppressed_summary() -> Optional[str]:
    """One line describing what was collapsed, or None if nothing was."""
    if not _counts:
        return None
    total = sum(_counts.values())
    return (
        f"Suppressed {total} renderer message(s) from the viewport's picking "
        f"pass (a driver-side shader limitation; the geometry still draws)."
    )
