"""The database controller's process lifecycle.

It copies `PreviewController` deliberately, including the one case Qt carves
out: a process that fails to start never emits `finished`, only
`errorOccurred(FailedToStart)`. Unwired, that leaves `running` True forever
and the pane's Search button disabled with nothing to re-enable it.
"""

from __future__ import annotations

import json
import os

import pytest

pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.core.config import GlobalSettings  # noqa: E402
from src.gui.controllers.database import (  # noqa: E402
    DatabaseController, DatabaseRequest, request_key)
from src.gui.workspace import Workspace  # noqa: E402


def _run_until(signal, timeout_ms: int = 5000) -> list:
    """Pump a local event loop until `signal` fires or `timeout_ms` elapses.

    No pytest-qt in this environment (declared in pyproject.toml's `dev`
    extra but not installed here), so this drives the real Qt event loop
    with plain QtCore. Returns the signal's arguments, or [] on timeout.
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


@pytest.fixture(name='qt_app', scope='module')
def _qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(name='controller')
def _controller(qt_app, tmp_path):
    del qt_app
    workspace = Workspace.open(tmp_path / 'ws')
    settings = GlobalSettings()
    settings.aiida.profile = 'friction2d'
    settings.database.api_url = 'http://localhost:8000'
    settings.database.api_key = 'k'
    return DatabaseController(workspace, settings)


def test_the_request_key_is_stable_and_input_sensitive():
    first = request_key('list', 'local', {'filters': {'family': 'afm'}})
    same = request_key('list', 'local', {'filters': {'family': 'afm'}})
    other = request_key('list', 'local', {'filters': {'family': 'hetero'}})

    assert first == same
    assert first != other
    assert first != request_key('list', 'central', {'filters': {'family': 'afm'}})
    assert len(first) == 24


def test_start_writes_a_request_carrying_the_local_profile(controller):
    request = DatabaseRequest(kind='list', source='local',
                              payload={'filters': {'family': 'afm'}})

    controller.start(request)
    controller.cancel()

    written = json.loads(
        (controller.request_dir(request.key()) / 'request.json').read_text(
            encoding='utf-8'))
    assert written['kind'] == 'list'
    assert written['source'] == 'local'
    assert written['profile'] == 'friction2d'
    assert written['filters'] == {'family': 'afm'}
    assert written['result_file'].endswith('result.json')
    assert 'api_key' not in written


def test_a_central_request_carries_the_url_but_never_the_key(controller):
    """`request.json` is written into the workspace and kept as the D13 cache.
    A key written there outlives the session in a directory the user may sync
    or share, so it goes through the child's environment instead."""
    request = DatabaseRequest(kind='list', source='central', payload={'filters': {}})

    controller.start(request)
    environment = controller._process.processEnvironment()
    controller.cancel()

    written = json.loads(
        (controller.request_dir(request.key()) / 'request.json').read_text(
            encoding='utf-8'))
    assert written['api_url'] == 'http://localhost:8000'
    assert 'api_key' not in written
    assert 'profile' not in written
    # Not `'k' not in json.dumps(written)`: the payload necessarily contains
    # the substring "kind" (asserted above), which already contains 'k' and
    # makes that check unsatisfiable regardless of whether the secret leaked.
    # Checking for the quoted JSON string value instead catches a leaked 'k'
    # without tripping on 'kind'.
    assert '"k"' not in json.dumps(written)
    assert environment.value(DatabaseController.API_KEY_ENV) == 'k'


def test_a_local_request_does_not_leak_the_key_into_the_environment(controller):
    controller.start(DatabaseRequest(kind='list', source='local',
                                     payload={'filters': {}}))
    environment = controller._process.processEnvironment()
    controller.cancel()

    assert environment.value(DatabaseController.API_KEY_ENV) == ''


def test_a_write_in_flight_names_itself(controller):
    """An export writes into a destination directory; the window refuses to
    start anything else while one is running rather than SIGKILLing it
    half-way through a copy."""
    controller.start(DatabaseRequest(kind='export', source='local',
                                     payload={'what': 'run_tree'}))

    assert controller.current_kind == 'export'
    assert controller.current_kind in DatabaseController.WRITE_KINDS

    controller.cancel()
    assert controller.current_kind == ''


def test_a_failed_to_start_process_ends_the_request(controller, monkeypatch):
    import src.gui.controllers.database as database_module
    monkeypatch.setattr(database_module.sys, 'executable',
                        '/nonexistent/not-a-real-interpreter-xyz')

    controller.start(DatabaseRequest(kind='list', source='local',
                                     payload={'filters': {}}))
    result = _run_until(controller.failed, timeout_ms=5000)

    assert result, 'the "failed" signal never arrived -- the request is stuck'
    assert result[0] == 'list'
    assert 'failed to start' in result[1]
    assert not controller.running


def test_cancel_reports_the_kind_it_cancelled(controller):
    controller.start(DatabaseRequest(kind='list', source='local',
                                     payload={'filters': {}}))
    seen = []
    controller.cancelled.connect(seen.append)

    controller.cancel()

    assert seen == ['list']
    assert not controller.running
