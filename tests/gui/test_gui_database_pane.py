"""The Database pane, driven offscreen with the controller stubbed out.

Nothing here starts a real query: `MainWindow.database.start` is replaced by a
list so the assertion is on the request that *would* have run. That is the
pattern the smoke file uses, and it is what keeps this file free of aiida,
subprocesses and modals.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication  # noqa: E402

from src.gui.widgets.database.filters import EMPTY_FILTERS  # noqa: E402
from src.gui.widgets.database.pane import SOURCES, DatabasePane  # noqa: E402
from src.gui.workspace import Workspace  # noqa: E402

RECORDS = [
    {'set_label': 's1', 'family': 'afm', 'materials': ['h-MoS2'], 'layers': 2,
     'load_kind': 'force', 'load': 30.0, 'mean_cof': 0.0731,
     'run_uuid': '1a2b', 'result_uuid': '9f8e'},
    {'set_label': 's1', 'family': 'hetero', 'materials': ['h-WS2'], 'layers': 1,
     'load_kind': 'force', 'load': 10.0, 'mean_cof': 0.0125,
     'run_uuid': '3c4d', 'result_uuid': '7b6a'},
]


@pytest.fixture(name='qt_app', scope='module')
def _qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(name='pane')
def _pane(qt_app):
    del qt_app
    pane = DatabasePane()
    pane.set_records(RECORDS)
    return pane


@pytest.fixture(name='window')
def _window(qt_app, tmp_path):
    del qt_app
    from src.gui.app import MainWindow
    window = MainWindow(Workspace.open(tmp_path / 'ws'))
    yield window
    window.document.dirty = False
    window.settings_document.dirty = False
    window.close()


class TestPaneLayout:

    def test_the_sources_are_local_and_central(self, pane):
        assert SOURCES == ('local', 'central')
        assert pane.source == 'local'

    def test_switching_source_announces_it(self, pane):
        seen = []
        pane.source_changed.connect(seen.append)

        pane.set_source('central')

        assert pane.source == 'central'
        assert seen == ['central']

    def test_search_asks_for_a_query_with_the_current_filters(self, pane):
        seen = []
        pane.query_requested.connect(lambda source, filters: seen.append((source, filters)))

        pane.search_button.click()

        assert seen == [('local', EMPTY_FILTERS)]

    def test_refresh_is_its_own_signal(self, pane):
        seen = []
        pane.refresh_requested.connect(lambda source, filters: seen.append(source))

        pane.refresh_button.click()

        assert seen == ['local']

    def test_free_text_filters_the_table_without_a_query(self, pane):
        queries = []
        pane.query_requested.connect(lambda *args: queries.append(args))

        pane.filters.search.setText('h-WS2')

        assert pane.proxy.rowCount() == 1
        assert queries == []

    def test_selecting_a_row_asks_for_its_detail(self, pane):
        seen = []
        pane.detail_requested.connect(lambda source, payload: seen.append(payload))

        pane.table.selectRow(0)

        assert seen == [{'run_uuid': '1a2b', 'result_uuid': '9f8e'}]

    def test_selected_and_visible_records(self, pane):
        pane.table.selectRow(1)

        assert pane.selected_records() == [RECORDS[1]]
        assert pane.visible_records() == RECORDS

    def test_all_columns_widens_the_table(self, pane):
        before = pane.model.columnCount()

        pane.all_columns_box.setChecked(True)

        assert pane.model.columnCount() != before
        assert 'run_uuid' in [c.key for c in pane.model.columns]

    def test_the_message_panel_is_the_only_error_surface(self, pane):
        pane.message('nothing matched', 'warn')

        assert 'nothing matched' in pane.messages.toPlainText()

    def test_busy_disables_the_query_buttons_and_offers_cancel(self, pane):
        assert not pane.cancel_button.isEnabled()

        pane.set_busy(True)
        assert not pane.search_button.isEnabled()
        assert pane.cancel_button.isEnabled()

        pane.set_busy(False)
        assert pane.search_button.isEnabled()
        assert not pane.cancel_button.isEnabled()

    def test_cancel_asks_for_a_stop(self, pane):
        seen = []
        pane.cancel_requested.connect(lambda: seen.append(1))
        pane.set_busy(True)

        pane.cancel_button.click()

        assert seen == [1]


class TestModalSeams:
    """Every modal reachable from this package must be a named seam on the
    pane. Today there is one, `_ask_directory`; Task 10 adds `_confirm` for
    the whole-set download and rewrites this class.

    An unpatched modal in a teardown path hangs the whole suite -- that is why
    a bare `pytest` cannot be run in this repo. Few seams means few things for
    a test to override, and this test is what keeps the count from growing.
    """

    def test_no_other_dialog_is_constructed_in_the_package(self):
        package = Path('src/gui/widgets/database')
        sources = {path.name: path.read_text(encoding='utf-8')
                   for path in package.glob('*.py')}

        assert 'QMessageBox' not in ''.join(sources.values())
        offenders = [name for name, text in sources.items()
                     if 'QFileDialog' in text and name != 'pane.py']
        assert offenders == []
        assert sources['pane.py'].count('QFileDialog.') == 1


class TestWindowWiring:

    def test_the_database_tab_holds_the_pane(self, window):
        assert window.top_tabs.widget(1) is window.database_page
        assert isinstance(window.database_pane, DatabasePane)

    def test_a_query_reaches_the_controller_as_a_list_request(self, window, monkeypatch):
        calls = []
        monkeypatch.setattr(window.database, 'start', calls.append)

        window.database_pane.search_button.click()

        assert len(calls) == 1
        assert calls[0].kind == 'list'
        assert calls[0].source == 'local'
        assert calls[0].payload == {'filters': EMPTY_FILTERS}

    def test_a_finished_list_fills_the_table_and_the_combos(self, window):
        window._on_db_finished('list', {
            'ok': True, 'kind': 'list', 'source': 'local', 'records': RECORDS,
            'facets': {'materials': ['h-MoS2'], 'families': ['afm']},
            'total_count': 2, 'returned': 2, 'elapsed_s': 0.4, 'warnings': []})

        assert window.database_pane.model.rowCount() == 2
        assert window.database_pane.filters.material.count() == 2
        assert '2 of 2' in window.database_pane.messages.toPlainText()

    def test_a_truncated_central_list_reports_the_limit_not_a_bogus_total(self, window):
        """`total_count` is None whenever the central client's own `limit`
        cut the rows short (see `_list` in db_worker.py) -- the message must
        not read "N of None results", which looks like a broken query rather
        than a truncated one."""
        window._on_db_finished('list', {
            'ok': True, 'kind': 'list', 'source': 'central', 'records': RECORDS,
            'facets': {}, 'total_count': None, 'returned': 2, 'limit': 2,
            'elapsed_s': 0.4, 'warnings': []})

        text = window.database_pane.messages.toPlainText()
        assert '2 results (limit 2) in 0.4 s' in text
        assert 'None' not in text

    def test_a_failure_lands_in_the_pane_not_a_dialog(self, window):
        window._on_db_failed('list', 'RuntimeError: no profile')

        assert 'no profile' in window.database_pane.messages.toPlainText()

    def test_a_list_warning_is_shown_not_swallowed(self, window):
        window._on_db_finished('list', {
            'ok': True, 'kind': 'list', 'source': 'local', 'records': RECORDS,
            'facets': {}, 'set_sizes': {'s1': 37}, 'total_count': 2,
            'returned': 2, 'elapsed_s': 0.4,
            'warnings': ['the set choices come from the 2 rows fetched']})

        assert 'rows fetched' in window.database_pane.messages.toPlainText()
        assert window.database_pane.set_sizes == {'s1': 37}

    def test_nothing_starts_while_a_download_is_running(self, window, monkeypatch):
        """`start` cancels what is running, and cancelling is SIGKILL. A
        `detail` request fires on every row selection, so an export could be
        killed mid-copy by a stray click."""
        started = []
        monkeypatch.setattr(window.database, 'start', started.append)
        monkeypatch.setattr(type(window.database), 'current_kind',
                            property(lambda self: 'export'))

        window._on_db_detail('local', {'run_uuid': '1a2b', 'result_uuid': '9f8e'})

        assert started == []
        assert 'still running' in window.database_pane.messages.toPlainText()

    def test_cancel_says_what_it_left_behind(self, window, monkeypatch):
        monkeypatch.setattr(type(window.database), 'running',
                            property(lambda self: True))
        monkeypatch.setattr(type(window.database), 'current_kind',
                            property(lambda self: 'export'))
        cancelled = []
        monkeypatch.setattr(window.database, 'cancel',
                            lambda: cancelled.append(True))

        window._on_db_cancel()

        assert cancelled == [True]
        text = window.database_pane.messages.toPlainText()
        assert 'Cancelled the export' in text
        assert 'still there' in text

    def test_cancel_of_a_read_only_kind_does_not_claim_leftover_files(self, window, monkeypatch):
        """C13: a list/detail/check cancellation writes nothing outside the
        workspace, so the message must not claim files are "still there"."""
        monkeypatch.setattr(type(window.database), 'running',
                            property(lambda self: True))
        monkeypatch.setattr(type(window.database), 'current_kind',
                            property(lambda self: 'list'))
        monkeypatch.setattr(window.database, 'cancel', lambda: None)

        window._on_db_cancel()

        text = window.database_pane.messages.toPlainText()
        assert 'Cancelled the list' in text
        assert 'still there' not in text

    def test_closing_cancels_a_running_request(self, window, monkeypatch):
        cancelled = []
        monkeypatch.setattr(window.database, 'cancel',
                            lambda: cancelled.append(True))
        window.document.dirty = False
        window.settings_document.dirty = False

        window.close()

        assert cancelled == [True]
