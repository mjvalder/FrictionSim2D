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
    CACHE_KEEP, DatabaseController, DatabaseRequest, request_key)
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
        (controller.directory_for(request) / 'request.json').read_text(
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
        (controller.directory_for(request) / 'request.json').read_text(
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


class TestTheWorkerReallyRuns:
    """One test that spawns the actual `python -m src.gui.db_worker`.

    Everything else in this suite calls `dispatch` in-process or stubs the
    controller, which leaves the parts that only exist between two processes
    untested: the `-m` entry point resolving, the working directory making
    `src` importable, the newline-delimited JSON on stdout, the result file
    landing where the request said, and the exit code. Those are exactly the
    failures that make a worker look hung rather than broken, so one end-to-end
    seam earns its runtime. `check` against an unconfigured central source is
    the one request kind that needs no aiida, no httpx call and no network.
    """

    def test_a_check_round_trips_through_the_real_entry_point(self, controller):
        controller.settings.database.api_url = ''
        progress = []
        controller.progress.connect(progress.append)

        controller.start(DatabaseRequest(kind='check', source='central',
                                         payload={}))
        result = _run_until(controller.finished, timeout_ms=60000)

        assert result, 'the worker never finished -- the -m entry point is broken'
        kind, payload = result
        assert kind == 'check'
        assert payload['ok'] is True
        assert payload['ready'] is False
        assert any('not configured' in c['detail'] for c in payload['checks'])
        assert not controller.running


_EMPTY_LIST_RESULT = {'ok': True, 'kind': 'list', 'source': 'local', 'records': [],
                      'facets': {}, 'total_count': 0, 'returned': 0,
                      'elapsed_s': 0.0, 'warnings': []}


def _seed(directory, result) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'result.json').write_text(
        result if isinstance(result, str) else json.dumps(result), encoding='utf-8')


class TestResultCache:
    """A local query over 13k nodes is seconds, not milliseconds."""

    def test_a_cached_list_result_is_delivered_without_a_process(self, controller):
        request = DatabaseRequest(kind='list', source='local',
                                  payload={'filters': {'family': 'afm'}})
        _seed(controller.directory_for(request), _EMPTY_LIST_RESULT)
        seen = []
        controller.finished.connect(lambda kind, result: seen.append(kind))

        controller.start(request)

        assert seen == ['list']
        assert not controller.running, 'no subprocess should have been started'

    def test_refresh_bypasses_and_clears_the_cache(self, controller):
        request = DatabaseRequest(kind='list', source='local',
                                  payload={'filters': {'family': 'afm'}})
        directory = controller.directory_for(request)
        _seed(directory, '{"ok": true, "kind": "list", "records": []}')
        stale = directory / 'result.json'
        seen = []
        controller.finished.connect(lambda kind, result: seen.append(kind))

        controller.start(request, use_cache=False)
        controller.cancel()

        assert seen == [], 'the stale result must not be delivered'
        assert not stale.exists()

    def test_only_list_results_are_cached(self, controller):
        request = DatabaseRequest(kind='export', source='local',
                                  payload={'what': 'summary_csv'})
        _seed(controller.directory_for(request),
              '{"ok": true, "kind": "export", "files": []}')
        seen = []
        controller.finished.connect(lambda kind, result: seen.append(kind))

        controller.start(request)
        controller.cancel()

        assert seen == [], 'an export must always run, never be served from disk'
        assert controller.CACHED_KINDS == ('list',)

    def test_a_cached_failure_is_never_replayed(self, controller):
        """PreviewController's rule: a failure's cause can lie outside the
        key (a profile fixed, a server back up), so replaying it outlives
        the fix."""
        request = DatabaseRequest(kind='list', source='local',
                                  payload={'filters': {}})
        directory = controller.directory_for(request)
        _seed(directory, {'ok': False, 'error': 'RuntimeError: no profile'})
        failed = []
        controller.failed.connect(lambda kind, message: failed.append(message))

        controller.start(request)
        assert controller.running, 'a cached failure must re-run the request'
        controller.cancel()

        assert failed == []
        assert not (directory / 'result.json').exists()

    def test_the_cache_key_includes_the_connection(self, controller):
        """A different profile, URL or API key is a different database: the
        same filters must not be answered from another store's result."""
        local = DatabaseRequest(kind='list', source='local', payload={'filters': {}})
        central = DatabaseRequest(kind='list', source='central', payload={'filters': {}})
        before_local = controller.directory_for(local)
        before_central = controller.directory_for(central)

        controller.settings.aiida.profile = 'other'
        controller.settings.database.api_url = 'https://db.example.org'
        assert controller.directory_for(local) != before_local
        changed_url = controller.directory_for(central)
        assert changed_url != before_central

        controller.settings.database.api_key = 'another-key'
        assert controller.directory_for(central) != changed_url

    def test_the_api_key_is_hashed_into_the_key_never_written(self, controller):
        controller.settings.database.api_key = 'sekrit-token-123'
        request = DatabaseRequest(kind='list', source='central', payload={'filters': {}})

        controller.start(request)
        controller.cancel()

        directory = controller.directory_for(request)
        for path in directory.rglob('*'):
            assert 'sekrit-token-123' not in path.read_text(encoding='utf-8')
        assert 'sekrit-token-123' not in str(directory)

    def test_non_cached_kinds_share_one_scratch_directory_per_kind(self, controller):
        """A `detail` fires on every row click; one directory per click would
        grow `.preview/_db` without bound."""
        first = DatabaseRequest(kind='detail', source='local',
                                payload={'run_uuid': 'a', 'result_uuid': 'b'})
        second = DatabaseRequest(kind='detail', source='local',
                                 payload={'run_uuid': 'c', 'result_uuid': 'd'})
        export = DatabaseRequest(kind='export', source='local',
                                 payload={'what': 'summary_csv'})

        assert controller.directory_for(first) == controller.directory_for(second)
        assert controller.directory_for(first) != controller.directory_for(export)
        assert controller.directory_for(first).parent.name == '_scratch'

    def test_a_stale_scratch_result_is_never_delivered(self, controller):
        """The scratch directory is reused, so a previous request's result
        file must be gone before the next request runs -- otherwise a worker
        that dies without writing one would deliver the previous answer."""
        request = DatabaseRequest(kind='detail', source='local',
                                  payload={'run_uuid': 'a', 'result_uuid': 'b'})
        directory = controller.directory_for(request)
        _seed(directory, {'ok': True, 'kind': 'detail'})

        controller.start(request)
        controller.cancel()

        assert not (directory / 'result.json').exists()

    def test_a_successful_import_invalidates_cached_lists(self, controller):
        listing = DatabaseRequest(kind='list', source='local', payload={'filters': {}})
        cached = controller.directory_for(listing) / 'result.json'
        _seed(cached.parent, _EMPTY_LIST_RESULT)
        importing = DatabaseRequest(kind='import', source='local',
                                    payload={'path': '/tmp/tree'})
        result_file = controller.directory_for(importing) / 'result.json'
        _seed(result_file.parent, {'ok': True, 'kind': 'import'})
        controller._kind = 'import'

        controller._deliver(result_file)

        assert not cached.exists()

    def test_a_failed_import_keeps_cached_lists(self, controller):
        listing = DatabaseRequest(kind='list', source='local', payload={'filters': {}})
        cached = controller.directory_for(listing) / 'result.json'
        _seed(cached.parent, _EMPTY_LIST_RESULT)
        importing = DatabaseRequest(kind='import', source='local',
                                    payload={'path': '/tmp/tree'})
        result_file = controller.directory_for(importing) / 'result.json'
        _seed(result_file.parent, {'ok': False, 'error': 'nope'})
        controller._kind = 'import'

        controller._deliver(result_file)

        assert cached.exists()


class TestCacheMarking:
    """A replayed result must not read as a fresh one.

    `elapsed_s` belongs to the query that first fetched it, so the pane needs
    to know which branch delivered the result -- `_deliver` is the only place
    that does.
    """

    def test_a_cached_result_is_marked_cached(self, controller):
        request = DatabaseRequest(kind='list', source='local',
                                  payload={'filters': {'family': 'afm'}})
        _seed(controller.directory_for(request), _EMPTY_LIST_RESULT)
        seen = []
        controller.finished.connect(lambda kind, result: seen.append(result))

        controller.start(request)

        assert seen[0]['cached'] is True

    def test_a_result_read_after_a_run_is_not(self, controller):
        request = DatabaseRequest(kind='list', source='local',
                                  payload={'filters': {'family': 'afm'}})
        result_file = controller.directory_for(request) / 'result.json'
        _seed(result_file.parent, _EMPTY_LIST_RESULT)
        controller._kind = 'list'
        controller._result_file = result_file
        seen = []
        controller.finished.connect(lambda kind, result: seen.append(result))

        controller._deliver(result_file)

        assert seen[0]['cached'] is False


class TestCachePruning:
    """`.preview/_db` is the one scratch area nothing else cleans up.

    `Workspace.prune_previews` skips `_`-prefixed entries on purpose (pruning
    to the newest few builds would throw the query cache away after eight
    renders) and `_invalidate_cached_lists` only unlinks the result files,
    leaving the directories behind. Every distinct query makes another one.
    """

    def _keyed(self, controller):
        return sorted(p.name for p in controller._db_dir.iterdir()
                      if p.is_dir() and p.name != '_scratch')

    def test_a_new_query_directory_prunes_to_the_newest_keep(self, controller):
        made = []
        for index in range(CACHE_KEEP + 3):
            directory = controller.directory_for(
                DatabaseRequest(kind='list', source='local',
                                payload={'filters': {'family': f"f{index}"}}))
            made.append(directory)
            # mtimes in a fixed, strictly increasing order: several
            # directories made in the same millisecond would otherwise tie,
            # and "the newest N" would not be a testable claim.
            os.utime(directory, (1_600_000_000 + index, 1_600_000_000 + index))

        assert self._keyed(controller) == sorted(p.name for p in made[3:])
        assert all(p.exists() for p in made[3:])
        assert not any(p.exists() for p in made[:3])

    def test_the_scratch_directory_survives_the_prune(self, controller):
        scratch = controller.directory_for(
            DatabaseRequest(kind='detail', source='local', payload={'run_uuid': 'a'}))
        scratch.mkdir(parents=True, exist_ok=True)
        os.utime(scratch.parent, (1, 1))

        for index in range(CACHE_KEEP + 2):
            controller.directory_for(
                DatabaseRequest(kind='list', source='local',
                                payload={'filters': {'family': f"g{index}"}}))

        assert scratch.is_dir()

    def test_an_existing_directory_is_reused_without_pruning(self, controller,
                                                             monkeypatch):
        """A cache hit must not count as a new directory: re-querying the
        same filters would otherwise age out a neighbour on every Search."""
        request = DatabaseRequest(kind='list', source='local',
                                  payload={'filters': {'family': 'afm'}})
        directory = controller.directory_for(request)
        (directory / 'result.json').write_text('{}', encoding='utf-8')
        removed = []
        monkeypatch.setattr(controller, '_prune_cached_results',
                            lambda *args, **kwargs: removed.append(1))

        assert controller.directory_for(request) == directory
        assert removed == []
