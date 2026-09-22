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

from PySide6.QtWidgets import QApplication, QFileDialog  # noqa: E402

from src.gui.widgets.database.filters import EMPTY_FILTERS  # noqa: E402
from src.gui.widgets.database.pane import SOURCES, DatabasePane  # noqa: E402
from src.gui.workspace import RecentWorkspaces, Workspace  # noqa: E402

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


class TestDownloads:
    """Four downloads, one dialog. The dialog is overridden here -- an
    unpatched QFileDialog would block the run forever."""

    @pytest.fixture(name='ready')
    def _ready(self, pane, tmp_path, monkeypatch):
        monkeypatch.setattr(pane, '_ask_directory', lambda: tmp_path / 'dl')
        pane.table.selectRow(0)
        return pane

    def test_there_is_a_button_for_each_export_kind(self, pane):
        from src.gui.db_worker import EXPORT_KINDS
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
            {'run_uuid': '1a2b', 'result_uuid': '9f8e', 'set_label': 's1'}]
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
        assert payload['records'] == RECORDS
        assert payload['columns'][:3] == ['set_label', 'family', 'materials']

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

    def test_the_central_source_offers_only_the_summary_csv(self, pane):
        pane.set_source('central')

        assert pane.download_buttons['summary_csv'].isEnabled()
        assert not pane.download_buttons['results'].isEnabled()
        assert not pane.download_buttons['provenance'].isEnabled()
        assert not pane.download_buttons['run_tree'].isEnabled()


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
        package = (Path(__file__).resolve().parents[2] / 'src' / 'gui'
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
        package = (Path(__file__).resolve().parents[2] / 'src' / 'gui'
                   / 'widgets' / 'database')
        text = (package / 'pane.py').read_text(encoding='utf-8')
        start = text.index('def _ask_directory')
        end = text.find('\n    def ', start + 1)
        body = text[start:len(text) if end == -1 else end]
        assert body.count('QFileDialog.') == 1


class TestOwnerContact:
    """The central database holds summaries, not files. Reaching the person
    who ran it is the only way to get the rest, so it is one click (D6)."""

    @pytest.fixture(name='central')
    def _central(self, pane, monkeypatch):
        copied = []
        monkeypatch.setattr(pane, '_set_clipboard', copied.append)
        pane.set_source('central')
        pane.set_records([dict(RECORDS[0], contact='someone@example.org'),
                          dict(RECORDS[1], contact=None)])
        pane._copied = copied
        return pane

    def test_the_button_is_central_only(self, pane):
        assert not pane.copy_contact_button.isEnabled()
        pane.set_source('central')
        assert pane.copy_contact_button.isEnabled()

    def test_it_copies_the_selected_row_contact(self, central):
        central.table.selectRow(0)

        central.copy_contact_button.click()

        assert central._copied == ['someone@example.org']
        assert 'someone@example.org' in central.messages.toPlainText()

    def test_a_row_with_no_contact_says_so_rather_than_copying_nothing(self, central):
        central.table.selectRow(1)

        central.copy_contact_button.click()

        assert central._copied == []
        assert 'no owner contact' in central.messages.toPlainText()

    def test_it_refuses_an_empty_selection(self, central):
        central.copy_contact_button.click()

        assert central._copied == []
        assert 'Select a row' in central.messages.toPlainText()


class TestExportReachesTheController:

    def test_a_download_becomes_an_export_request(self, window, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(window.database, 'start', calls.append)
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

    def test_import_is_disabled_for_the_central_source(self, pane):
        pane.set_source('central')

        assert not pane.import_button.isEnabled()

    def test_a_finished_import_shows_the_report(self, window):
        window._on_db_finished('import', {
            'ok': True, 'kind': 'import', 'ingest_ok': True, 'label': 'x',
            'set_uuid': '77aa', 'created_runs': ['afm/a'], 'created_results': [],
            'skipped_results': [], 'incomplete': [], 'empty_runs': [],
            'unparseable': [], 'failures': [], 'dry_run': False,
            'report': "set 'x' (77aa)\n  runs created:    1"})

        assert 'runs created' in window.database_pane.messages.toPlainText()

    def test_incomplete_and_empty_runs_are_reported_not_buried(self, window):
        window._on_db_finished('import', {
            'ok': True, 'kind': 'import', 'ingest_ok': True, 'label': 'x',
            'set_uuid': '77aa', 'created_runs': ['afm/a'],
            'created_results': ['afm/a#force30_a0_s2'], 'skipped_results': [],
            'incomplete': ['afm/a#force30_a0_s2'], 'empty_runs': ['afm/c'],
            'unparseable': [], 'failures': [], 'dry_run': False,
            'report': "set 'x' (77aa)"})

        text = window.database_pane.messages.toPlainText()
        assert '1 run(s) incomplete' in text
        assert '1 run(s) with no results at all' in text

    def test_unparseable_paths_are_reported_not_buried(self, window):
        """N8: `aiida import` exits non-zero on `unparseable`; the pane must
        not bury it either."""
        window._on_db_finished('import', {
            'ok': True, 'kind': 'import', 'ingest_ok': True, 'label': 'x',
            'set_uuid': '77aa', 'created_runs': [], 'created_results': [],
            'skipped_results': [], 'incomplete': [], 'empty_runs': [],
            'unparseable': ['afm/broken'], 'failures': [], 'dry_run': False,
            'report': "set 'x' (77aa)"})

        text = window.database_pane.messages.toPlainText()
        assert '1 path(s) unparseable' in text
        assert 'afm/broken' in text

    def test_unparseable_makes_a_clean_ingest_not_quiet(self, window, monkeypatch):
        """N8: the quiet/level logic must treat `unparseable` as not-quiet,
        even when `ingest_ok` is True and nothing else is amiss."""
        levels = []
        monkeypatch.setattr(window.database_pane, 'message',
                            lambda text, level='info': levels.append(level))

        window._on_db_finished('import', {
            'ok': True, 'kind': 'import', 'ingest_ok': True, 'label': 'x',
            'set_uuid': '77aa', 'created_runs': [], 'created_results': [],
            'skipped_results': [], 'incomplete': [], 'empty_runs': [],
            'unparseable': ['afm/broken'], 'failures': [], 'dry_run': False,
            'report': "set 'x' (77aa)"})

        assert levels == ['warn']


class TestConnectionCheck:

    def test_the_button_asks_for_a_check_of_the_current_source(self, pane):
        seen = []
        pane.check_requested.connect(seen.append)

        pane.set_source('central')
        pane.check_button.click()

        assert seen == ['central']

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

    def test_a_query_records_the_source_and_the_filters(self, window, monkeypatch):
        monkeypatch.setattr(window.database, 'start', lambda request: None)
        window.database_pane.filters.family.addItem('afm')
        window.database_pane.filters.family.setCurrentText('afm')

        window.database_pane.search_button.click()

        assert window.workspace.state['db_source'] == 'local'
        assert window.workspace.state['db_filters']['family'] == 'afm'

    def test_switching_source_is_recorded_immediately(self, window):
        # D11 disables central by default; configure it first so the switch
        # this test is actually about (not D11) has somewhere to land.
        window.settings.database.api_url = 'https://db.example.org'
        window._apply_database_availability()

        window.database_pane.set_source('central')

        assert window.workspace.state['db_source'] == 'central'

    def test_a_download_records_its_destination(self, window, tmp_path, monkeypatch):
        monkeypatch.setattr(window.database, 'start', lambda request: None)
        pane = window.database_pane
        pane.set_records(RECORDS)
        monkeypatch.setattr(pane, '_ask_directory', lambda: tmp_path / 'dl')
        pane.table.selectRow(0)

        pane.download_buttons['results'].click()

        assert window.workspace.state['db_dest'] == str(tmp_path / 'dl')

    def test_a_new_window_restores_what_the_last_one_left(
            self, qt_app, tmp_path, monkeypatch):
        del qt_app
        import src.gui.app as app_module
        from src.gui.app import MainWindow
        # D11 disables central by default (no central database is configured);
        # this test is about restoring the remembered source, not about D11,
        # so pretend it is configured.
        monkeypatch.setattr(app_module, 'central_configured', lambda s: True)
        workspace = Workspace.open(tmp_path / 'ws')
        workspace.remember('db_source', 'central')
        workspace.remember('db_filters', dict(EMPTY_FILTERS, family='hetero'))
        workspace.remember('db_dest', str(tmp_path / 'dl'))

        window = MainWindow(workspace)
        try:
            assert window.database_pane.source == 'central'
            assert window.database_pane.filters.value()['family'] == 'hetero'
            assert window.database_pane.last_dest == str(tmp_path / 'dl')
        finally:
            window.document.dirty = False
            window.settings_document.dirty = False
            window.close()

    def test_changing_workspace_re_points_the_controller_and_restores(
            self, window, tmp_path, monkeypatch):
        import src.gui.app as app_module
        # D11 disables central by default; this test is about restoring the
        # remembered source on a workspace switch, not about D11, so pretend
        # central is configured.
        monkeypatch.setattr(app_module, 'central_configured', lambda s: True)
        other = Workspace.open(tmp_path / 'other')
        other.remember('db_source', 'central')
        monkeypatch.setattr(RecentWorkspaces, 'add', lambda self, root: [])
        monkeypatch.setattr(QFileDialog, 'getExistingDirectory',
                            lambda *args, **kwargs: str(other.root))

        window._choose_workspace()

        assert window.database.workspace.root == other.root
        assert window.database_pane.source == 'central'

    def test_switching_workspace_does_not_clobber_the_new_workspaces_filters(
            self, window, tmp_path, monkeypatch):
        """Regression: restoring a source different from the pane's current
        one used to fire source_changed -> _remember_database_state, which
        wrote the OLD workspace's still-on-screen filters into the shared
        state dict before db_filters was even read back out of it -- so the
        new workspace's saved filters were clobbered both on screen and on
        disk before the user touched anything."""
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

    The pane is reached from `src.gui.app`, which is reached from `src.cli`;
    an unconditional `import aiida` here would break the CLI on every compute
    node that has no store.
    """

    def test_missing_backend_probes_without_importing(self, monkeypatch):
        import src.gui.widgets.database.pane as pane_module
        monkeypatch.setattr(pane_module, 'find_spec',
                            lambda name: None if name == 'aiida' else object())

        assert pane_module.missing_backend('local') == ['aiida']
        assert pane_module.missing_backend('central') == []

    def test_a_missing_backend_disables_its_source_and_says_why(
            self, qt_app, monkeypatch):
        del qt_app
        import src.gui.widgets.database.pane as pane_module
        monkeypatch.setattr(pane_module, 'missing_backend',
                            lambda source: ['aiida'] if source == 'local' else [])

        pane = pane_module.DatabasePane()

        assert pane.source == 'central', 'a disabled source must not stay selected'
        assert not pane._source_buttons.button(0).isEnabled()
        text = pane.messages.toPlainText()
        assert 'aiida' in text
        assert "pip install -e '.[aiida]'" in text

    def test_both_backends_missing_leaves_a_usable_but_inert_pane(
            self, qt_app, monkeypatch):
        del qt_app
        import src.gui.widgets.database.pane as pane_module
        monkeypatch.setattr(pane_module, 'missing_backend', lambda source: ['x'])

        pane = pane_module.DatabasePane()

        assert not pane.search_button.isEnabled()
        assert not pane.import_button.isEnabled()
        assert 'x' in pane.messages.toPlainText()

    def test_a_present_backend_is_left_alone(self, pane):
        assert pane._source_buttons.button(0).isEnabled()
        assert pane.search_button.isEnabled()


class TestNotConfigured:
    """Installed is not the same as configured, and today there is no central
    database at all."""

    def test_the_default_url_alone_is_not_a_configured_database(self):
        from types import SimpleNamespace
        from src.gui.controllers.database import DEFAULT_API_URL, central_configured

        bare = SimpleNamespace(database=SimpleNamespace(
            api_url=DEFAULT_API_URL, api_key='', host=''))
        keyed = SimpleNamespace(database=SimpleNamespace(
            api_url=DEFAULT_API_URL, api_key='k', host=''))
        elsewhere = SimpleNamespace(database=SimpleNamespace(
            api_url='https://db.example.org', api_key='', host=''))
        empty = SimpleNamespace(database=SimpleNamespace(
            api_url='', api_key='', host=''))

        assert central_configured(bare) is False
        assert central_configured(keyed) is True
        assert central_configured(elsewhere) is True
        assert central_configured(empty) is False

    def test_an_unconfigured_central_source_is_disabled_with_a_reason(self, window):
        window.settings.database.api_url = ''
        window._apply_database_availability()

        text = window.database_pane.messages.toPlainText()
        assert not window.database_pane._source_buttons.button(1).isEnabled()
        assert 'not configured' in text
        assert 'Local store' in text

    def test_configuring_it_enables_the_source(self, window):
        window.settings.database.api_url = 'https://db.example.org'
        window._apply_database_availability()

        assert window.database_pane._source_buttons.button(1).isEnabled()


class TestAvailabilityChangeDoesNotClobberMemory:
    """Regression: `_apply_settings` calls `_apply_database_availability()`
    live, unguarded, any time Settings is applied -- not only at startup or
    on a workspace switch, which already ran it under `_restoring_db_state`
    (see `_restore_database_state`). Clearing `database.api_url` while
    Central was selected disabled the source, `set_source_enabled` fell back
    to Local, and the resulting `source_changed` -> `_remember_database_state`
    wrote 'local' over the workspace's remembered 'central' -- exactly the
    write the D11 fallback promises never to make ("only the on-screen
    selection falls back")."""

    def test_a_live_availability_change_does_not_overwrite_the_remembered_source(
            self, window):
        window.settings.database.api_url = 'https://db.example.org'
        window._apply_database_availability()
        window.database_pane.set_source('central')
        assert window.workspace.state['db_source'] == 'central'

        window.settings.database.api_url = ''
        window._apply_database_availability()

        assert window.database_pane.source == 'local'
        assert window.workspace.state['db_source'] == 'central'

        window.settings.database.api_url = 'https://db.example.org'
        window._apply_database_availability()

        assert window.database_pane.source == 'central'
        assert window.workspace.state['db_source'] == 'central'


class TestDisabledSourceIsNeverReachable:
    """Regression: a disabled-but-selected 'central' fired real central
    requests, because `set_source` did not consult the button's enabled
    state. `_restore_database_state` could put the pane on a source that
    D11 had just disabled (not installed, or not configured), and every
    action gated on `self._source` -- Search, Refresh, Check connection --
    would then target it anyway with no button ever having been clickable
    to get there. Fixed in `DatabasePane.set_source`; these drive it through
    `MainWindow`, not a bare pane, because the bug was in how a *window*
    restores a remembered source, not in the pane in isolation."""

    def test_a_missing_backend_beats_a_remembered_central_source(
            self, qt_app, tmp_path, monkeypatch):
        del qt_app
        import src.gui.widgets.database.pane as pane_module
        from src.gui.app import MainWindow
        # The probe, not `missing_backend` itself: `missing_backend` is one
        # function object shared by the pane's own construction-time check
        # and the window's `_apply_database_availability` -- patching the
        # probe it calls affects both call sites the way a real missing
        # package would, rather than only whichever one imported the name.
        monkeypatch.setattr(pane_module, 'find_spec',
                            lambda name: None if name == 'httpx' else object())
        workspace = Workspace.open(tmp_path / 'ws')
        workspace.remember('db_source', 'central')

        window = MainWindow(workspace)
        try:
            assert window.database_pane.source == 'local'
            # The fallback is on-screen only: the remembered choice on disk
            # survives it untouched.
            assert window.workspace.state['db_source'] == 'central'

            seen = []
            monkeypatch.setattr(window.database, 'start', seen.append)
            window.database_pane.search_button.click()

            assert seen and seen[0].source == 'local'
        finally:
            window.document.dirty = False
            window.settings_document.dirty = False
            window.close()

    def test_an_unconfigured_central_beats_a_remembered_central_source_on_switch(
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
        monkeypatch.setattr(window.database, 'start', seen.append)
        window.database_pane.search_button.click()

        assert seen and seen[0].source == 'local'

    def test_set_source_refuses_a_disabled_button(self, window):
        # Simulate a stray call (a future regression, a test) reaching
        # `set_source('central')` directly while D11 has it disabled: it
        # must not select it, and must not be able to write it to disk
        # either, since no signal fires for a selection that never happened.
        window.workspace.remember('db_source', 'central')
        assert not window.database_pane._source_buttons.button(1).isEnabled()

        window.database_pane.set_source('central')

        assert window.database_pane.source == 'local'
        assert window.workspace.state['db_source'] == 'central'


class TestMessageLevels:

    def test_an_error_message_uses_the_error_token(self, pane):
        from src.gui.style import STATUS_ERROR

        pane.message('boom', 'error')

        assert pane.messages.styleSheet() == STATUS_ERROR

    def test_a_warning_message_uses_the_warn_token(self, pane):
        from src.gui.style import STATUS_WARN

        pane.message('careful', 'warn')

        assert pane.messages.styleSheet() == STATUS_WARN

    def test_an_info_message_clears_the_colour(self, pane):
        pane.message('boom', 'error')
        pane.message('fine')

        assert pane.messages.styleSheet() == ''
