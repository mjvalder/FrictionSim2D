"""GUI test guards: offscreen Qt, no real modal dialog, and a hang backstop.

Offscreen, a modal nobody patched waits for a click that never comes, so the
run hangs. Every ``QMessageBox`` / ``QFileDialog`` / ``QInputDialog`` static
call and every ``QDialog.exec`` is stubbed here: reaching one raises, and the
test fails at teardown even when Qt swallowed the exception inside a slot. A
test that expects a dialog patches it itself (its ``monkeypatch`` wins).

The backstop: an item (setup, call and teardown) that runs longer than
``FS2D_GUI_TEST_TIMEOUT`` seconds (default 300) dumps every thread's traceback
and exits the run (``faulthandler``; pytest-timeout is not installed).
"""

from __future__ import annotations

import faulthandler
import os
import sys

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

TIMEOUT_S = float(os.environ.get('FS2D_GUI_TEST_TIMEOUT', '300'))

_MODALS = {
    'QMessageBox': ('question', 'warning', 'critical', 'information', 'about', 'exec'),
    'QFileDialog': ('getOpenFileName', 'getOpenFileNames', 'getSaveFileName',
                    'getExistingDirectory'),
    'QInputDialog': ('getText', 'getInt', 'getDouble', 'getItem', 'getMultiLineText'),
    'QDialog': ('exec',),
}


class UnexpectedModal(AssertionError):
    """A modal dialog no test patched was opened."""


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):  # pylint: disable=unused-argument
    """Arm the hang backstop around one item, teardown included."""
    faulthandler.dump_traceback_later(TIMEOUT_S, exit=True, file=sys.stderr)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


@pytest.fixture(autouse=True)
def _no_real_modals(monkeypatch):
    try:
        from PySide6 import QtWidgets
    except ImportError:
        yield
        return

    reached = []

    def _stub(owner, name):
        def _modal(*args, **_kwargs):
            title = next((a for a in args if isinstance(a, str)), '')
            call = f'{owner}.{name}({title!r})'
            reached.append(call)
            raise UnexpectedModal(f'unpatched modal opened: {call}')
        return _modal

    for owner, names in _MODALS.items():
        cls = getattr(QtWidgets, owner)
        for name in names:
            monkeypatch.setattr(cls, name, _stub(owner, name))
    yield
    if reached:
        pytest.fail('unpatched modal dialog(s) reached: ' + '; '.join(reached))
