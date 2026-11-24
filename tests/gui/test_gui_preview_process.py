"""A preview build must never leave the Render button stuck.

``PreviewController`` runs a build in a ``QProcess`` and wires exactly one of
``finished``/``failed``/``cancelled`` to follow ``started`` -- ``MainWindow``
re-enables Render only on one of those three. Qt's own docs carve out one
exception: if the process fails to start at all, ``finished`` is *never*
emitted, only ``errorOccurred(FailedToStart)``. Left unwired, that path never
told the controller the build was over, so ``running`` stayed True and the
Render button stayed disabled -- indistinguishable, from the user's seat,
from a build that runs forever.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from frictionsim2d.gui.controllers.preview import PreviewController, PreviewRequest  # noqa: E402
from frictionsim2d.gui.workspace import Workspace  # noqa: E402


def _run_until(signal, timeout_ms: int = 5000) -> list:
    """Pump a local event loop until ``signal`` fires or ``timeout_ms`` elapses.

    No pytest-qt in this environment (declared in pyproject.toml's ``dev``
    extra but not installed here), so this drives the real Qt event loop
    with plain QtCore -- the same primitives the app itself uses -- rather
    than pulling in a new test dependency.

    Returns the arguments the signal was emitted with, or ``[]`` on timeout
    (distinguishable from a signal genuinely emitted with no arguments,
    which none of the controller's signals are).
    """
    loop = QEventLoop()
    received: list = []

    def _capture(*args):
        received.append(args)
        loop.quit()

    signal.connect(_capture)
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(timeout_ms)

    loop.exec()

    timer.stop()
    signal.disconnect(_capture)
    return received[0] if received else []


class TestPreviewControllerNeverStaysStuck:
    """A build that never reaches Qt's ``finished`` signal must still end."""

    @pytest.fixture(name='qt_app')
    def _qt_app(self):
        return QApplication.instance() or QApplication([])

    @pytest.fixture(name='workspace')
    def _workspace(self, tmp_path):
        return Workspace.open(tmp_path / 'ws')

    def test_failed_to_start_process_ends_the_build(self, qt_app, workspace, monkeypatch):
        """No python interpreter at all: QProcess emits ``errorOccurred
        (FailedToStart)`` and, per Qt, never emits ``finished``. Before
        wiring ``errorOccurred``, nothing else in PreviewController would
        ever run again -- ``running`` stayed True and the caller (the
        Render button) stayed disabled forever."""
        del qt_app  # only needed to ensure a QApplication exists
        import frictionsim2d.gui.controllers.preview as preview_module
        monkeypatch.setattr(
            preview_module.sys, 'executable',
            '/nonexistent/not-a-real-interpreter-xyz')

        controller = PreviewController(workspace)
        config = {'2D': {'mat': 'h_MoS2', 'x': 60, 'y': 60}}
        request = PreviewRequest(config=config, model='indent')

        controller.start(request)
        result = _run_until(controller.failed, timeout_ms=5000)

        assert result, 'the "failed" signal never arrived -- the build is stuck'
        assert not controller.running, \
            'a failed-to-start build must not leave the controller "running" forever'


class TestPartialBuildWarnings:
    """A preview built while some members failed must say so (GC-10)."""

    def test_build_warnings_reach_the_system(self, tmp_path, monkeypatch):
        import json
        from types import SimpleNamespace

        import frictionsim2d.gui.controllers.preview as preview_module

        QApplication.instance() or QApplication([])
        system = SimpleNamespace(warnings=['skipped x'])
        monkeypatch.setattr(preview_module, 'parse_assembly', lambda *a: object())
        monkeypatch.setattr(preview_module, 'assemble', lambda *a: system)
        result_file = tmp_path / 'result.json'
        result_file.write_text(json.dumps({
            'ok': True, 'init_script': 'in.x', 'simulation_root': str(tmp_path),
            'settings_file': None, 'warnings': ['afm/b: build failed'],
        }), encoding='utf-8')

        controller = PreviewController(Workspace.open(tmp_path / 'ws'))
        delivered = []
        controller.finished.connect(delivered.append)
        controller._deliver(result_file)  # pylint: disable=protected-access

        assert delivered == [system]
        assert system.warnings == ['skipped x', 'afm/b: build failed']
