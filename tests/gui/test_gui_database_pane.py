"""The Database pane, driven offscreen with the controller stubbed out.

Nothing here starts a real query: `MainWindow.database.start` is replaced by a
list so the assertion is on the request that *would* have run. That is the
pattern the smoke file uses, and it is what keeps this file free of aiida,
subprocesses and modals.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication, QFileDialog  # noqa: E402

from frictionsim2d.gui.widgets.database.filters import EMPTY_FILTERS  # noqa: E402
from frictionsim2d.gui.widgets.database.pane import SOURCES, DatabasePane  # noqa: E402
from frictionsim2d.gui.workspace import RecentWorkspaces, Workspace  # noqa: E402

RECORDS = [
    {'set_label': 's1', 'family': 'afm', 'materials': ['h-MoS2'], 'layers': 2,
     'load_kind': 'force', 'load': 30.0, 'mean_cof': 0.0731,
     'run_uuid': 'r1', 'node_uuid': '1a2b', 'point_uuid': '9f8e'},
    {'set_label': 's1', 'family': 'hetero', 'materials': ['h-WS2'], 'layers': 1,
     'load_kind': 'force', 'load': 10.0, 'mean_cof': 0.0125,
     'run_uuid': 'r2', 'node_uuid': '3c4d', 'point_uuid': '7b6a'},
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
    from frictionsim2d.gui.app import MainWindow
    window = MainWindow(Workspace.open(tmp_path / 'ws'))
    yield window
    window.document.dirty = False
    window.settings_document.dirty = False
    window.close()


class TestPaneLayout:

    def test_local_is_the_only_source(self, pane):
        assert SOURCES == ('local',)
        assert pane.source == 'local'

    def test_there_is_no_source_switch(self, pane):
        from PySide6.QtWidgets import QRadioButton

        assert pane.findChildren(QRadioButton) == []
        assert pane.source_label.text() == 'Local store'
        assert not hasattr(pane, 'set_source')

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

        assert seen == [{'node_uuid': '1a2b', 'point_uuid': '9f8e'}]

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


class TestDownloads:
    """Four downloads, one dialog. The dialog is overridden here -- an
    unpatched QFileDialog would block the run forever."""

    @pytest.fixture(name='ready')
    def _ready(self, pane, tmp_path, monkeypatch):
        monkeypatch.setattr(pane, '_ask_directory', lambda: tmp_path / 'dl')
        pane.table.selectRow(0)
        return pane

    def test_there_is_a_button_for_each_export_kind(self, pane):
        from frictionsim2d.gui.db_worker import EXPORT_KINDS
        assert tuple(pane.download_buttons) == EXPORT_KINDS

    def test_results_files_export_carries_the_selection(self, ready):
        seen = []
        ready.export_requested.connect(lambda source, payload: seen.append((source, payload)))

        ready.download_buttons['results'].click()

        source, payload = seen[0]
        assert source == 'local'
        assert payload['what'] == 'results'
        assert payload['dest'].endswith('dl')
        assert payload['selection'] == [
            {'node_uuid': '1a2b', 'point_uuid': '9f8e', 'set_label': 's1'}]
        assert payload['rebuild'] is False

    def test_provenance_and_run_tree_carry_the_same_selection(self, ready):
        seen = []
        ready.export_requested.connect(lambda source, payload: seen.append(payload))

        ready.download_buttons['provenance'].click()
        ready.rebuild_box.setChecked(True)
        ready.download_buttons['run_tree'].click()

        assert seen[0]['what'] == 'provenance'
        assert seen[1]['what'] == 'run_tree'
        assert seen[1]['rebuild'] is True

    def test_the_warning_appears_before_a_destination_is_even_asked(
            self, ready, monkeypatch):
        """One selected row exports every run in its set. That is the
        feature, but it must never be a surprise (N2): a run_tree download
        writes the warning to the message panel before the directory chooser
        even opens, rather than gating on a second modal."""
        order = []
        monkeypatch.setattr(ready, 'message', lambda text, level='info':
                            order.append(('message', level, text)))
        monkeypatch.setattr(ready, '_ask_directory',
                            lambda: order.append(('asked',)) or (
                                Path('/tmp/dl')))
        seen = []
        ready.export_requested.connect(lambda *args: seen.append(args))

        ready.download_buttons['run_tree'].click()

        assert [step[0] for step in order] == ['message', 'asked']
        assert order[0][1] == 'warn'
        assert 'whole sets' in order[0][2]
        assert seen  # the chooser answered, so the request still goes out

    def test_cancelling_the_chooser_after_the_warning_asks_for_nothing(
            self, ready, monkeypatch):
        monkeypatch.setattr(ready, '_ask_directory', lambda: None)
        seen = []
        ready.export_requested.connect(lambda *args: seen.append(args))

        ready.download_buttons['run_tree'].click()

        assert seen == []

    def test_the_confirmation_quotes_the_set_and_its_run_count(self, ready):
        ready.set_set_sizes({'s1': 37})

        question = ready._run_tree_warning(ready.selected_records())

        assert 'whole sets' in question
        assert 's1: 37 runs' in question
        assert '1 selected row' in question

    def test_the_confirmation_extrapolates_a_size_when_it_has_one(self, ready):
        ready.set_set_sizes({'s1': 10})
        ready.show_detail({'run': {'uuid': '', 'node_type': '', 'attributes': {}},
                           'result': {'uuid': '', 'node_type': '', 'attributes': {}},
                           'files': [{'node': 'run', 'path': 'a', 'bytes': 1024}],
                           'total_bytes': 1024})

        question = ready._run_tree_warning(ready.selected_records())

        assert '10 runs, roughly 10.0 kB' in question
        assert 'extrapolated' in question

    def test_the_confirmation_says_so_when_it_cannot_size_the_set(self, ready):
        ready.set_set_sizes({})

        assert 'run count unknown' in ready._run_tree_warning(
            ready.selected_records())

    def test_the_other_three_downloads_write_no_such_warning(self, ready):
        seen = []
        ready.export_requested.connect(lambda source, payload: seen.append(payload))

        for what in ('results', 'provenance', 'summary_csv'):
            ready.download_buttons[what].click()

        assert [p['what'] for p in seen] == ['results', 'provenance', 'summary_csv']
        assert 'whole sets' not in ready.messages.toPlainText()

    def test_summary_csv_carries_the_visible_rows_and_column_order(self, ready):
        seen = []
        ready.export_requested.connect(lambda source, payload: seen.append(payload))

        ready.download_buttons['summary_csv'].click()

        payload = seen[0]
        assert payload['columns'][:3] == ['set_label', 'family', 'materials']
        # One row per visible record, carrying exactly the columns shown.
        assert [r['family'] for r in payload['records']] == ['afm', 'hetero']
        assert [r['load'] for r in payload['records']] == [30.0, 10.0]
        assert set(payload['records'][0]) == set(payload['columns'])

    def test_summary_csv_round_trips_the_shown_columns(
            self, pane, tmp_path, monkeypatch):
        """The Summary CSV promises the visible rows "with the columns shown"
        (D6): the worker writes exactly the payload's columns."""
        from frictionsim2d.gui import db_worker
        monkeypatch.setattr(pane, '_ask_directory', lambda: tmp_path / 'dl')
        pane.set_records([{'set_label': 's1', 'family': 'afm',
                           'materials': ['h-MoS2'], 'load_kind': 'force',
                           'load': 30.0, 'angle': 15.0, 'speed': 5.0,
                           'status': 'complete', 'node_uuid': 'n1'}])
        seen = []
        pane.export_requested.connect(lambda source, payload: seen.append(payload))

        pane.download_buttons['summary_csv'].click()
        result = db_worker.dispatch({
            'kind': 'export', 'source': 'local', 'what': 'summary_csv',
            'dest': str(tmp_path / 'dl'), 'records': seen[0]['records'],
            'columns': seen[0]['columns']})

        assert result['ok'] is True
        with open(result['files'][0], encoding='utf-8', newline='') as handle:
            row = next(csv.DictReader(handle))
        assert row['load'] == '30.0'
        assert row['angle'] == '15.0'
        assert row['speed'] == '5.0'
        assert row['materials'] == 'h-MoS2'

    def test_summary_csv_does_not_need_a_selection(self, pane, tmp_path, monkeypatch):
        monkeypatch.setattr(pane, '_ask_directory', lambda: tmp_path / 'dl')
        seen = []
        pane.export_requested.connect(lambda source, payload: seen.append(payload))

        pane.download_buttons['summary_csv'].click()

        assert seen and seen[0]['what'] == 'summary_csv'

    def test_the_other_three_refuse_an_empty_selection(self, pane, tmp_path, monkeypatch):
        monkeypatch.setattr(pane, '_ask_directory', lambda: tmp_path / 'dl')
        seen = []
        pane.export_requested.connect(lambda *args: seen.append(args))

        pane.download_buttons['results'].click()

        assert seen == []
        assert 'Select at least one row' in pane.messages.toPlainText()

    def test_a_cancelled_dialog_asks_for_nothing(self, pane, monkeypatch):
        monkeypatch.setattr(pane, '_ask_directory', lambda: None)
        pane.table.selectRow(0)
        seen = []
        pane.export_requested.connect(lambda *args: seen.append(args))

        pane.download_buttons['results'].click()

        assert seen == []

class TestModalSeams:
    """The only modal reachable from this package is `_ask_directory`.

    Task 10 considered a second modal (`_confirm`) for the whole-set
    download and rejected it (N2): the run_tree warning is written to the
    message panel instead, before the destination is even asked, so there
    is still exactly one modal to override.

    An unpatched modal in a teardown path hangs the whole suite -- that is why
    a bare `pytest` cannot be run in this repo. One seam means one thing for
    a test to override, and this test is what keeps the count from growing.
    """

    def test_no_other_dialog_is_constructed_in_the_package(self):
        package = (Path(__file__).resolve().parents[2] / 'frictionsim2d' / 'gui'
                   / 'widgets' / 'database')
        sources = {path.name: path.read_text(encoding='utf-8')
                   for path in package.glob('*.py')}

        offenders = [name for name, text in sources.items()
                     if name != 'pane.py'
                     and ('QFileDialog' in text or 'QMessageBox' in text)]
        assert offenders == []
        assert 'QMessageBox' not in sources['pane.py']
        assert sources['pane.py'].count('QFileDialog.') == 1

    def test_the_file_dialog_lives_only_in_ask_directory(self):
        package = (Path(__file__).resolve().parents[2] / 'frictionsim2d' / 'gui'
                   / 'widgets' / 'database')
        text = (package / 'pane.py').read_text(encoding='utf-8')
        start = text.index('def _ask_directory')
        end = text.find('\n    def ', start + 1)
        body = text[start:len(text) if end == -1 else end]
        assert body.count('QFileDialog.') == 1


class TestExportReachesTheController:

    def test_a_download_becomes_an_export_request(self, window, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(window.database, 'start',
                            lambda request, **kwargs: calls.append(request))
        pane = window.database_pane
        pane.set_records(RECORDS)
        monkeypatch.setattr(pane, '_ask_directory', lambda: tmp_path / 'dl')
        pane.table.selectRow(0)

        pane.download_buttons['results'].click()

        # selectRow(0) itself fires a 'detail' request (wired since Task 8);
        # the download is the last call to reach the controller.
        assert calls[-1].kind == 'export'
        assert calls[-1].payload['what'] == 'results'

    def test_a_finished_export_reports_what_was_written(self, window):
        window._on_db_finished('export', {
            'ok': True, 'kind': 'export', 'what': 'results',
            'dest': '/tmp/dl', 'files': ['/tmp/dl/a.csv', '/tmp/dl/b.csv'],
            'bytes': 2048, 'warnings': []})

        text = window.database_pane.messages.toPlainText()
        assert '2 files' in text
        assert '2.0 kB' in text
        assert '/tmp/dl' in text

    def test_an_export_warning_is_shown(self, window):
        window._on_db_finished('export', {
            'ok': True, 'kind': 'export', 'what': 'run_tree',
            'dest': '/tmp/dl', 'files': ['/tmp/dl/a.csv'], 'bytes': 10,
            'warnings': ["set 's1' was exported in full, not only the "
                         "selected rows"]})

        assert 'exported in full' in window.database_pane.messages.toPlainText()


class TestWindowWiring:

    def test_the_database_tab_holds_the_pane(self, window):
        assert window.top_tabs.widget(1) is window.database_page
        assert isinstance(window.database_pane, DatabasePane)

    def test_a_query_reaches_the_controller_as_a_list_request(self, window, monkeypatch):
        calls = []
        monkeypatch.setattr(window.database, 'start',
                            lambda request, **kwargs: calls.append((request, kwargs)))

        window.database_pane.search_button.click()

        assert len(calls) == 1
        request, kwargs = calls[0]
        assert request.kind == 'list'
        assert request.source == 'local'
        assert request.payload == {'filters': EMPTY_FILTERS}
        assert kwargs.get('use_cache', True) is True

    def test_refresh_reaches_the_controller_bypassing_the_cache(self, window, monkeypatch):
        calls = []
        monkeypatch.setattr(window.database, 'start',
                            lambda request, **kwargs: calls.append((request, kwargs)))

        window.database_pane.refresh_button.click()

        assert len(calls) == 1
        request, kwargs = calls[0]
        assert request.kind == 'list'
        assert kwargs == {'use_cache': False}

    def test_refresh_does_not_cancel_a_running_export(self, window, monkeypatch):
        """Refresh is still a `start`, and `start` SIGKILLs what is running:
        it must go through the same write guard as every other request."""
        started = []
        monkeypatch.setattr(window.database, 'start',
                            lambda request, **kwargs: started.append(request))
        monkeypatch.setattr(type(window.database), 'current_kind',
                            property(lambda self: 'export'))

        window.database_pane.refresh_button.click()

        assert started == []
        assert 'still running' in window.database_pane.messages.toPlainText()

    def test_a_finished_list_fills_the_table_and_the_combos(self, window):
        window._on_db_finished('list', {
            'ok': True, 'kind': 'list', 'source': 'local', 'records': RECORDS,
            'facets': {'materials': ['h-MoS2'], 'families': ['afm']},
            'total_count': 2, 'returned': 2, 'elapsed_s': 0.4, 'warnings': []})

        assert window.database_pane.model.rowCount() == 2
        assert window.database_pane.filters.material.count() == 2
        assert '2 of 2' in window.database_pane.messages.toPlainText()

    def test_a_cached_list_says_so_rather_than_quoting_a_stale_timing(self, window):
        """`elapsed_s` comes off the result file with the rest of it: a
        replayed answer would otherwise report the original query's timing as
        if the store had just been read."""
        window._on_db_finished('list', {
            'ok': True, 'kind': 'list', 'source': 'local', 'records': RECORDS,
            'facets': {}, 'total_count': 2, 'returned': 2, 'elapsed_s': 0.4,
            'warnings': [], 'cached': True})

        text = window.database_pane.messages.toPlainText()
        assert '2 of 2 results in 0.4 s (cached — Refresh to re-query)' in text

    def test_a_fresh_list_carries_no_cache_marker(self, window):
        window._on_db_finished('list', {
            'ok': True, 'kind': 'list', 'source': 'local', 'records': RECORDS,
            'facets': {}, 'total_count': 2, 'returned': 2, 'elapsed_s': 0.4,
            'warnings': [], 'cached': False})

        assert 'cached' not in window.database_pane.messages.toPlainText()

    def test_a_truncated_list_reports_the_limit_not_a_bogus_total(self, window):
        """`total_count` is None whenever a `limit` cut the rows short --
        the message must
        not read "N of None results", which looks like a broken query rather
        than a truncated one."""
        window._on_db_finished('list', {
            'ok': True, 'kind': 'list', 'source': 'local', 'records': RECORDS,
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
        monkeypatch.setattr(window.database, 'start',
                            lambda request, **kwargs: started.append(request))
        monkeypatch.setattr(type(window.database), 'current_kind',
                            property(lambda self: 'export'))

        window._on_db_detail('local', {'node_uuid': '1a2b', 'point_uuid': '9f8e'})

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


class TestImport:
    """D7: the one write action the pane offers besides downloads."""

    def test_import_asks_for_a_directory_and_names_the_set_after_it(
            self, pane, tmp_path, monkeypatch):
        chosen = tmp_path / 'simulation_20260901_120000'
        chosen.mkdir()
        monkeypatch.setattr(pane, '_ask_directory', lambda remember=True: chosen)
        seen = []
        pane.import_requested.connect(lambda source, payload: seen.append((source, payload)))

        pane.import_button.click()

        assert seen == [('local', {'root': str(chosen),
                                   'label': 'simulation_20260901_120000',
                                   'dry_run': False})]

    def test_import_does_not_remember_the_chosen_directory_as_last_dest(
            self, pane, tmp_path, monkeypatch):
        """C15: a run directory picked to ingest must not become the next
        download destination."""
        chosen = tmp_path / 'simulation_20260901_120000'
        chosen.mkdir()
        seen_remember = []

        def fake_ask_directory(remember=True):
            seen_remember.append(remember)
            return chosen

        monkeypatch.setattr(pane, '_ask_directory', fake_ask_directory)
        pane.last_dest = None

        pane.import_button.click()

        assert seen_remember == [False]
        assert pane.last_dest is None

    def test_a_cancelled_dialog_imports_nothing(self, pane, monkeypatch):
        monkeypatch.setattr(pane, '_ask_directory', lambda remember=True: None)
        seen = []
        pane.import_requested.connect(lambda *args: seen.append(args))

        pane.import_button.click()

        assert seen == []

    @staticmethod
    def _import_result(**fields):
        """A worker `import` response: the real `ImportReport` field names."""
        result = {
            'ok': True, 'kind': 'import', 'ingest_ok': True, 'label': 'x',
            'group_uuid': '77aa', 'created_decks': ['afm/a:slide'],
            'skipped_decks': [], 'superseded_decks': [], 'already_imported': [],
            'n_points': 1, 'verdict_counts': {'complete': 1}, 'log_missing': [],
            'no_version': [], 'unparseable': [], 'failures': [], 'warnings': [],
            'dry_run': False, 'report': "set 'x' (77aa)\n  decks created:       1"}
        result.update(fields)
        return result

    def test_a_finished_import_shows_the_report(self, window):
        window._on_db_finished('import', self._import_result())

        assert 'decks created' in window.database_pane.messages.toPlainText()

    def test_missing_logs_and_versions_are_reported_not_buried(self, window):
        window._on_db_finished('import', self._import_result(
            log_missing=['afm/a:slide'], no_version=['afm/a:system', 'afm/b:system']))

        text = window.database_pane.messages.toPlainText()
        assert '1 deck(s) with no log yet: afm/a:slide' in text
        assert '2 deck(s) with no LAMMPS version' in text

    def test_failures_are_reported_at_error(self, window, monkeypatch):
        seen = []
        monkeypatch.setattr(window.database_pane, 'message',
                            lambda text, level='info': seen.append((text, level)))

        window._on_db_finished('import', self._import_result(
            ingest_ok=False, failures=[['afm/b', 'no results file']]))

        (text, level), = seen
        assert '1 failure(s): afm/b: no results file' in text
        assert level == 'error'

    def test_unparseable_paths_are_reported_not_buried(self, window):
        """N8: `aiida import` exits non-zero on `unparseable`; the pane must
        not bury it either."""
        window._on_db_finished('import', self._import_result(unparseable=['afm/broken']))

        text = window.database_pane.messages.toPlainText()
        assert '1 path(s) unparseable' in text
        assert 'afm/broken' in text

    def test_unparseable_makes_a_clean_ingest_not_quiet(self, window, monkeypatch):
        """N8: the quiet/level logic must treat `unparseable` as not-quiet,
        even when `ingest_ok` is True and nothing else is amiss."""
        levels = []
        monkeypatch.setattr(window.database_pane, 'message',
                            lambda text, level='info': levels.append(level))

        window._on_db_finished('import', self._import_result(unparseable=['afm/broken']))

        assert levels == ['warn']


class TestConnectionCheck:

    def test_the_button_asks_for_a_check_of_the_current_source(self, pane):
        seen = []
        pane.check_requested.connect(seen.append)

        pane.check_button.click()

        assert seen == ['local']

    def test_the_report_renders_in_the_message_panel(self, window):
        window._on_db_finished('check', {
            'ok': True, 'kind': 'check', 'source': 'local', 'ready': False,
            'checks': [{'name': 'aiida installed', 'ok': True, 'detail': '2.7.3'},
                       {'name': 'profile loaded', 'ok': False, 'detail': 'boom'}],
            'notes': ['Check settings.aiida.profile']})

        text = window.database_pane.messages.toPlainText()
        assert '[OK  ] aiida installed: 2.7.3' in text
        assert '[FAIL] profile loaded: boom' in text
        assert 'Check settings.aiida.profile' in text
        assert 'not ready' in text


class TestWorkspaceMemory:
    """Nothing new on disk: the workspace's existing key/value state."""

    def test_a_query_records_the_filters_not_the_source(self, window, monkeypatch):
        monkeypatch.setattr(window.database, 'start', lambda request, **kwargs: None)
        window.database_pane.filters.family.addItem('afm')
        window.database_pane.filters.family.setCurrentText('afm')

        window.database_pane.search_button.click()

        assert 'db_source' not in window.workspace.state  # one source: nothing to remember
        assert window.workspace.state['db_filters']['family'] == 'afm'

    def test_a_download_records_its_destination(self, window, tmp_path, monkeypatch):
        monkeypatch.setattr(window.database, 'start', lambda request, **kwargs: None)
        pane = window.database_pane
        pane.set_records(RECORDS)
        monkeypatch.setattr(pane, '_ask_directory', lambda: tmp_path / 'dl')
        pane.table.selectRow(0)

        pane.download_buttons['results'].click()

        assert window.workspace.state['db_dest'] == str(tmp_path / 'dl')

    def test_a_new_window_restores_what_the_last_one_left(
            self, qt_app, tmp_path, monkeypatch):
        del qt_app
        from frictionsim2d.gui.app import MainWindow
        workspace = Workspace.open(tmp_path / 'ws')
        workspace.remember('db_source', 'local')
        workspace.remember('db_filters', dict(EMPTY_FILTERS, family='hetero'))
        workspace.remember('db_dest', str(tmp_path / 'dl'))

        window = MainWindow(workspace)
        try:
            assert window.database_pane.source == 'local'
            assert window.database_pane.filters.value()['family'] == 'hetero'
            assert window.database_pane.last_dest == str(tmp_path / 'dl')
        finally:
            window.document.dirty = False
            window.settings_document.dirty = False
            window.close()

    def test_changing_workspace_re_points_the_controller_and_restores(
            self, window, tmp_path, monkeypatch):
        other = Workspace.open(tmp_path / 'other')
        other.remember('db_source', 'local')
        monkeypatch.setattr(RecentWorkspaces, 'add', lambda self, root: [])
        monkeypatch.setattr(QFileDialog, 'getExistingDirectory',
                            lambda *args, **kwargs: str(other.root))

        window._choose_workspace()

        assert window.database.workspace.root == other.root
        assert window.database_pane.source == 'local'

    def test_switching_workspace_does_not_clobber_the_new_workspaces_filters(
            self, window, tmp_path, monkeypatch):
        """Regression: a write to the shared state dict while a workspace's
        saved state is being applied once wrote the OLD workspace's
        still-on-screen filters before db_filters was even read back out of
        it -- so the new workspace's saved filters were clobbered both on
        screen and on disk before the user touched anything."""
        other = Workspace.open(tmp_path / 'other')
        other.remember('db_source', 'central')
        other.remember('db_filters', dict(EMPTY_FILTERS, family='hetero'))
        monkeypatch.setattr(RecentWorkspaces, 'add', lambda self, root: [])
        monkeypatch.setattr(QFileDialog, 'getExistingDirectory',
                            lambda *args, **kwargs: str(other.root))

        window._choose_workspace()

        assert window.database_pane.filters.value()['family'] == 'hetero'
        reloaded = Workspace.open(other.root)
        assert reloaded.state['db_filters']['family'] == 'hetero'


class TestDegradation:
    """A machine without aiida still gets a window.

    The pane is reached from `frictionsim2d.gui.app`, which is reached from `frictionsim2d.cli`;
    an unconditional `import aiida` here would break the CLI on every compute
    node that has no store.
    """

    def test_missing_backend_probes_without_importing(self, monkeypatch):
        import frictionsim2d.gui.widgets.database.pane as pane_module
        monkeypatch.setattr(pane_module, 'find_spec',
                            lambda name: None if name == 'aiida' else object())

        assert pane_module.missing_backend('local') == ['aiida']
        assert pane_module.missing_backend('central') == [], 'no such source'

    def test_a_missing_backend_says_what_to_install(
            self, qt_app, monkeypatch):
        del qt_app
        import frictionsim2d.gui.widgets.database.pane as pane_module
        monkeypatch.setattr(pane_module, 'missing_backend',
                            lambda source: ['aiida'])

        pane = pane_module.DatabasePane()

        assert not pane.source_label.isEnabled()
        text = pane.messages.toPlainText()
        assert 'aiida' in text
        assert "pip install -e '.[aiida]'" in text

    def test_both_backends_missing_leaves_a_usable_but_inert_pane(
            self, qt_app, monkeypatch):
        del qt_app
        import frictionsim2d.gui.widgets.database.pane as pane_module
        monkeypatch.setattr(pane_module, 'missing_backend', lambda source: ['x'])

        pane = pane_module.DatabasePane()

        assert not pane.search_button.isEnabled()
        assert not pane.import_button.isEnabled()
        assert 'x' in pane.messages.toPlainText()

    def test_a_present_backend_is_left_alone(self, pane):
        assert pane.source_label.isEnabled()
        assert pane.search_button.isEnabled()


class TestRemovedCentralSourceIsNeverReachable:
    """The central database is gone. A workspace that still remembers
    `db_source == 'central'` must fall back to the local store, and no
    request may target the removed source."""

    def test_a_remembered_central_source_is_ignored_on_switch(
            self, window, tmp_path, monkeypatch):
        other = Workspace.open(tmp_path / 'other')
        other.remember('db_source', 'central')
        monkeypatch.setattr(RecentWorkspaces, 'add', lambda self, root: [])
        monkeypatch.setattr(QFileDialog, 'getExistingDirectory',
                            lambda *args, **kwargs: str(other.root))

        window._choose_workspace()

        assert window.database_pane.source == 'local'
        assert window.workspace.state['db_source'] == 'central'

        seen = []
        monkeypatch.setattr(window.database, 'start',
                            lambda request, **kwargs: seen.append(request))
        window.database_pane.search_button.click()

        assert seen and seen[0].source == 'local'

class TestMessageLevels:

    def test_an_error_message_uses_the_error_token(self, pane):
        from frictionsim2d.gui.style import STATUS_ERROR

        pane.message('boom', 'error')

        assert pane.messages.styleSheet() == STATUS_ERROR

    def test_a_warning_message_uses_the_warn_token(self, pane):
        from frictionsim2d.gui.style import STATUS_WARN

        pane.message('careful', 'warn')

        assert pane.messages.styleSheet() == STATUS_WARN

    def test_an_info_message_clears_the_colour(self, pane):
        pane.message('boom', 'error')
        pane.message('fine')

        assert pane.messages.styleSheet() == ''
