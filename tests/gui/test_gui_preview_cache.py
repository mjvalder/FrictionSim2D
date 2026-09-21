"""A failed preview must not be cached.

`PreviewController.start` reused any `preview.json` under the request's cache
key, including `ok: false`. A failure caused by something outside the key --
a path the staging got wrong, a potential file added since -- then replayed
forever for that config, even after the cause was fixed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QProcess  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.core.config import parse_config  # noqa: E402
from src.gui.controllers.preview import PreviewController, PreviewRequest  # noqa: E402
from src.gui.workspace import Workspace  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[2] / 'examples'


@pytest.fixture(scope='module', name='qt_app')
def _qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture(name='request_and_controller')
def _request_and_controller(qt_app, tmp_path, monkeypatch):
    monkeypatch.chdir(EXAMPLES.parent)
    started = []
    monkeypatch.setattr(QProcess, 'start', lambda self, *a: started.append(True))
    controller = PreviewController(Workspace.open(tmp_path / 'ws'))
    request = PreviewRequest(
        config=parse_config(EXAMPLES / 'afm_config.ini'), model='afm', layer=None,
        settings_file=None, referenced_files=[], base_dir=EXAMPLES)
    result_file = controller.workspace.preview_path(request.key()) / 'preview.json'
    result_file.parent.mkdir(parents=True)
    return controller, request, result_file, started


def test_a_cached_failure_is_rebuilt(request_and_controller):
    controller, request, result_file, started = request_and_controller
    result_file.write_text(json.dumps({'ok': False, 'error': 'stale'}), encoding='utf-8')
    failures = []
    controller.failed.connect(failures.append)

    controller.start(request)

    assert failures == []
    assert started


def test_a_cached_success_is_still_reused(request_and_controller):
    controller, request, result_file, started = request_and_controller
    result_file.write_text(json.dumps({'ok': True}), encoding='utf-8')

    controller.start(request)

    assert not started
