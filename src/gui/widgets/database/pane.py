"""The Database pane: source toggle, filters, table, detail, actions, messages.

One non-modal message panel is the pane's only error surface, the way `issues`
is for the builder, and exactly one modal is reachable from here --
`_ask_directory`. Both rules exist because an unpatched dialog in a teardown
path hangs the test suite.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ...diagnostics import Check
from ...style import STATUS_ERROR, STATUS_WARN
from .detail import DetailPanel, format_bytes
from .filters import FilterBar
from .model import ResultSortProxyModel, ResultTableModel, record_value

__all__ = ['DatabasePane', 'EXPORT_LABELS', 'SOURCES', 'SOURCE_LABELS',
          'BACKEND_MODULES', 'INSTALL_HINTS', 'missing_backend',
          'render_checks']

#: The pane's third axis, beside the AiiDA profile.
SOURCES: Tuple[str, str] = ('local', 'central')
SOURCE_LABELS = {'local': 'Local store', 'central': 'Central'}

#: (kind, button text, tooltip). Buttons rather than a menu: a QMenu popup is
#: one more thing that can be left open in a teardown path.
EXPORT_LABELS: Tuple[Tuple[str, str, str], ...] = (
    ('results', 'Results files',
     'The stored tier files and logs of the selected runs, verbatim'),
    ('provenance', 'Provenance bundle',
     'provenance.json, config.json, settings.yaml, cif and potentials'),
    ('summary_csv', 'Summary CSV',
     'The rows currently visible in the table, with the columns shown'),
    ('run_tree', 'Recreate run tree',
     'A directory tree `postprocess read` can read'),
)

#: Downloads the central summary DB cannot serve -- it holds no files.
_LOCAL_ONLY_EXPORTS = ('results', 'provenance', 'run_tree')

#: What each source needs installed. Probed with `find_spec`, never imported:
#: this module is reached from `src.cli`, and a compute node has neither.
BACKEND_MODULES: Dict[str, Tuple[str, ...]] = {
    'local': ('aiida',),
    'central': ('httpx',),
}

INSTALL_HINTS: Dict[str, str] = {
    'local': "pip install -e '.[aiida]'",
    'central': "pip install -e '.[api]'",
}

#: `message`'s level -> stylesheet. 'info' clears back to the palette.
_MESSAGE_STYLES = {'warn': STATUS_WARN, 'error': STATUS_ERROR, 'info': ''}


def missing_backend(source: str) -> List[str]:
    """Modules this source needs that are not installed."""
    missing = []
    for name in BACKEND_MODULES.get(source, ()):
        try:
            if find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    return missing


def render_checks(result: Dict[str, Any]) -> str:
    """A check response as `diagnostics.Report` renders one.

    Same `Check` lines as `FrictionSim2D gui --check`, with this pane's own
    heading -- the environment check's heading would be a lie here.
    """
    source = SOURCE_LABELS.get(result.get('source', ''), result.get('source', ''))
    lines = [f"{source} connection check", '']
    lines += [Check(c['name'], bool(c['ok']), c.get('detail', '')).render()
              for c in result.get('checks') or []]
    notes = result.get('notes') or []
    if notes:
        lines += ['', 'Notes:'] + [f"  - {note}" for note in notes]
    lines += ['', 'Result: ' + ('ready' if result.get('ready') else 'not ready')]
    return '\n'.join(lines)


class DatabasePane(QWidget):
    """Browse and download the two stores."""

    query_requested = Signal(str, dict)     # source, filters
    refresh_requested = Signal(str, dict)   # source, filters
    detail_requested = Signal(str, dict)    # source, row identity
    export_requested = Signal(str, dict)    # source, export payload
    import_requested = Signal(str, dict)    # source, {'root', 'label', 'dry_run'}
    check_requested = Signal(str)           # source
    cancel_requested = Signal()             # stop whatever is running
    source_changed = Signal(str)            # source

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._source = 'local'
        #: Where the last download went; the next dialog opens there (D10).
        self.last_dest: Optional[str] = None
        #: {set_label: n_runs}, from the last `list` response.
        self.set_sizes: Dict[str, int] = {}
        #: Stored bytes of the last inspected run, as a per-run yardstick.
        self.last_detail_bytes: Optional[int] = None
        #: One standing explanation per disabled source (Task 14).
        self._notes: Dict[str, str] = {}

        layout = QVBoxLayout(self)
        layout.addLayout(self._build_source_row())

        self.filters = FilterBar(self)
        self.filters.submitted.connect(self._emit_query)
        self.filters.search.textChanged.connect(self._on_text_changed)
        layout.addWidget(self.filters)

        splitter = QSplitter(Qt.Vertical, self)
        splitter.addWidget(self._build_table())
        self.detail = DetailPanel(splitter)
        splitter.addWidget(self.detail)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        layout.addLayout(self._build_action_row())

        self.messages = QPlainTextEdit(self)
        self.messages.setReadOnly(True)
        self.messages.setMaximumHeight(120)
        self.messages.setPlaceholderText(
            'Query results, download reports and connection checks appear here.')
        layout.addWidget(self.messages)

        self._apply_backend_availability()

    # ------------------------------------------------------------------- ui

    def _build_source_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._source_buttons = QButtonGroup(self)
        for index, name in enumerate(SOURCES):
            button = QRadioButton(SOURCE_LABELS[name], self)
            button.setChecked(name == self._source)
            button.toggled.connect(
                lambda checked, key=name: self._on_source_toggled(key, checked))
            self._source_buttons.addButton(button, index)
            row.addWidget(button)

        row.addSpacing(16)
        self.search_button = QPushButton('Search', self)
        self.search_button.clicked.connect(self._emit_query)
        row.addWidget(self.search_button)

        self.refresh_button = QPushButton('Refresh', self)
        self.refresh_button.setToolTip('Re-run the query, ignoring the cache')
        self.refresh_button.clicked.connect(
            lambda: self.refresh_requested.emit(self._source, self.filters.value()))
        row.addWidget(self.refresh_button)

        self.check_button = QPushButton('Check connection', self)
        self.check_button.clicked.connect(
            lambda: self.check_requested.emit(self._source))
        row.addWidget(self.check_button)

        # A query over a large store takes seconds and a download takes
        # minutes; without this the only way out of either is closing the
        # window, which is what "greyed out and I couldn't press it" was for
        # the Render button.
        self.cancel_button = QPushButton('Cancel', self)
        self.cancel_button.setToolTip(
            'Stop the running request. Files already written to a download '
            'destination are left where they are.')
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        row.addWidget(self.cancel_button)

        row.addStretch(1)
        return row

    def _build_table(self) -> QWidget:
        self.model = ResultTableModel(self)
        self.proxy = ResultSortProxyModel(self)
        self.proxy.setSourceModel(self.model)

        self.table = QTableView(self)
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.table.selectionModel().selectionChanged.connect(
            self._on_selection_changed)
        return self.table

    def _build_action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.all_columns_box = QCheckBox('All columns', self)
        self.all_columns_box.setToolTip('Show every stored field, not the curated subset')
        self.all_columns_box.toggled.connect(self.model.set_all_columns)
        row.addWidget(self.all_columns_box)

        row.addStretch(1)

        self.download_buttons: Dict[str, QPushButton] = {}
        for what, label, tooltip in EXPORT_LABELS:
            button = QPushButton(label, self)
            button.setToolTip(tooltip)
            button.clicked.connect(lambda _checked=False, kind=what:
                                   self._request_export(kind))
            self.download_buttons[what] = button
            row.addWidget(button)

        self.rebuild_box = QCheckBox('rebuild lammps/', self)
        self.rebuild_box.setToolTip(
            'After exporting the tree, regenerate its lammps/ inputs from the '
            'stored config')
        row.addWidget(self.rebuild_box)

        self.copy_contact_button = QPushButton('Copy owner contact', self)
        self.copy_contact_button.setToolTip(
            'The e-mail of whoever uploaded the selected central row -- the '
            'central database holds no files, so this is how you ask for them')
        self.copy_contact_button.clicked.connect(self._copy_contact)
        row.addWidget(self.copy_contact_button)

        self.import_button = QPushButton('Import run directory…', self)
        self.import_button.setToolTip(
            'Ingest a simulation directory into the local store')
        self.import_button.clicked.connect(self._request_import)
        row.addWidget(self.import_button)

        self._apply_source_actions()
        return row

    def _apply_source_actions(self) -> None:
        """The central DB holds summaries only: three downloads are local,
        and the owner contact is the one action that is central-only."""
        local = self._source == 'local'
        for what in _LOCAL_ONLY_EXPORTS:
            button = self.download_buttons.get(what)
            if button is not None:
                button.setEnabled(local)
        self.rebuild_box.setEnabled(local)
        self.copy_contact_button.setEnabled(not local)
        self.import_button.setEnabled(local)

    # ---------------------------------------------------------------- state

    @property
    def source(self) -> str:
        return self._source

    def set_source(self, source: str) -> None:
        """Select `source` -- unless its button is disabled (D11).

        `QRadioButton.setChecked` succeeds on a disabled button; without this
        check, a caller (`_restore_database_state`, a stray test) could put
        the pane on a source with no working backend or no central database
        configured, and every action gated on `self._source` -- Search,
        Refresh, Check connection -- would fire against it anyway with no
        button ever having been clickable to get there.
        """
        if source not in SOURCES or source == self._source:
            return
        button = self._source_buttons.button(SOURCES.index(source))
        if button is None or not button.isEnabled():
            return
        button.setChecked(True)

    def set_source_enabled(self, source: str, enabled: bool,
                           reason: str = '') -> None:
        """Enable or disable one source, saying why in the message panel.

        One note per source, replaced rather than appended, so re-enabling a
        source clears its message instead of leaving a stale explanation on
        screen. A source that is disabled while selected hands over to the
        other one if that is usable.
        """
        if source not in SOURCES:
            return
        button = self._source_buttons.button(SOURCES.index(source))
        if button is not None:
            button.setEnabled(enabled)
            button.setToolTip(reason)

        cleared = self._notes.pop(source, None) if enabled else None
        if not enabled and reason:
            self._notes[source] = reason
        if self._notes:
            self.message('\n'.join(self._notes[s] for s in SOURCES
                                   if s in self._notes), 'warn')
        elif cleared:
            # The last standing note just went away: leaving it on screen
            # would keep telling the user to configure something they have
            # already configured. Only when this call is what removed it --
            # re-enabling an already-enabled source must not wipe a query
            # result out of the panel.
            self.message('')

        if not enabled and self._source == source:
            other = next((s for s in SOURCES if s != source), None)
            other_button = (self._source_buttons.button(SOURCES.index(other))
                            if other else None)
            if other_button is not None and other_button.isEnabled():
                self.set_source(other)

    def _apply_backend_availability(self) -> None:
        """Build even with no backend installed (D11): disable what cannot
        work and say what to install, rather than raising at construction.

        "Not installed" only. Whether a source is *configured* is the window's
        question -- it is the only one holding the settings.
        """
        usable = []
        for source in SOURCES:
            missing = missing_backend(source)
            if missing:
                self.set_source_enabled(
                    source, False,
                    f"{SOURCE_LABELS[source]}: {', '.join(missing)} is not "
                    f"installed. {INSTALL_HINTS[source]}")
            else:
                usable.append(source)

        if not usable:
            self.set_busy(True)
            self.cancel_button.setEnabled(False)
            self.import_button.setEnabled(False)
            for button in self.download_buttons.values():
                button.setEnabled(False)
            return
        if self._source not in usable:
            self.set_source(usable[0])

    def _on_source_toggled(self, source: str, checked: bool) -> None:
        if not checked or source == self._source:
            return
        self._source = source
        # The rows on screen belong to the store we just left, and every
        # action reads `self._source`: clicking one would ask the new store
        # for the old store's row -- a central `result_id` of None (the worker
        # does `int(...)`) or a central uuid handed to `load_node`. The
        # Summary CSV would export the old store's rows under the new one's
        # name. Nothing survives a source change.
        self.clear_results()
        self.source_changed.emit(source)
        self._apply_source_actions()

    def clear_results(self) -> None:
        """Drop the rows, the detail and everything measured from them."""
        self.model.set_records([])
        self.detail.clear()
        self.set_sizes = {}
        self.last_detail_bytes = None

    def set_records(self, records: Sequence[Dict[str, Any]]) -> None:
        self.model.set_records(records)
        self.detail.clear()

    def set_facets(self, facets: Dict[str, List[Any]]) -> None:
        self.filters.set_choices(facets)

    def set_set_sizes(self, sizes: Dict[str, int]) -> None:
        """How many runs each set holds -- what the run_tree confirmation
        quotes, so the user learns the scale before the copy starts."""
        self.set_sizes = dict(sizes or {})

    def show_detail(self, detail: Dict[str, Any]) -> None:
        self.detail.show_detail(detail)
        self.last_detail_bytes = detail.get('total_bytes') or None

    def message(self, text: str, level: str = 'info') -> None:
        """The pane's one error surface. `level` is 'info', 'warn' or 'error'."""
        self.messages.setStyleSheet(_MESSAGE_STYLES.get(level, ''))
        self.messages.setPlainText(text)

    def set_busy(self, busy: bool) -> None:
        for button in (self.search_button, self.refresh_button, self.check_button):
            button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    # ------------------------------------------------------------- requests

    def _emit_query(self) -> None:
        self.query_requested.emit(self._source, self.filters.value())

    def _on_text_changed(self, text: str) -> None:
        self.proxy.set_text_filter(text)

    def _on_selection_changed(self, *_args) -> None:
        records = self.selected_records()
        if not records:
            return
        record = records[0]
        if self._source == 'central':
            payload = {'result_id': record.get('id')}
        else:
            payload = {'run_uuid': record.get('run_uuid'),
                       'result_uuid': record.get('result_uuid')}
        self.detail_requested.emit(self._source, payload)

    def selected_records(self) -> List[Dict[str, Any]]:
        rows = {self.proxy.mapToSource(index).row()
                for index in self.table.selectionModel().selectedRows()}
        return [self.model.record_at(row) for row in sorted(rows)]

    def visible_records(self) -> List[Dict[str, Any]]:
        return [self.model.record_at(self.proxy.mapToSource(
            self.proxy.index(row, 0)).row()) for row in range(self.proxy.rowCount())]

    def exported_rows(self) -> List[Dict[str, Any]]:
        """The visible rows reduced to the columns shown, values resolved.

        The Summary CSV promises "the rows currently visible, with the columns
        shown" (D6), and what a column shows is `record_value` -- key first,
        then the column's fallbacks. The worker only ever sees `record[key]`,
        so a central row, whose load/angle/speed live under `force_nN`,
        `scan_angle` and `scan_speed`, exported three blank columns that the
        table on screen had filled. Resolving here rather than in the worker
        keeps the fallback vocabulary in the one module that defines it, and
        leaves the worker's list/dict cell formatting where it is.
        """
        columns = self.model.columns
        return [{c.key: record_value(record, c) for c in columns}
                for record in self.visible_records()]

    # ---------------------------------------------------------------- modal

    def _ask_directory(self, remember: bool = True) -> Optional[Path]:
        """Where a download -- or an import source -- comes from. The only
        modal this package opens: tests override this method rather than
        trying to click a real dialog.

        `remember=False` (the import path) leaves `last_dest` alone: a run
        directory picked to ingest is not where the next download should
        land (C15)."""
        chosen = QFileDialog.getExistingDirectory(
            self, 'Choose a destination', self.last_dest or str(Path.home()))
        if not chosen:
            return None
        if remember:
            self.last_dest = chosen
        return Path(chosen)

    # ---------------------------------------------------------------- export

    def _request_export(self, what: str) -> None:
        """Ask the controller for one of the four downloads (D6).

        `run_tree` copies every run in the selected rows' sets, not just the
        selected rows. That is not gated behind a second modal (N2): the
        warning is written to the message panel, at 'warn', before the
        destination is even asked -- the user backs out by cancelling the
        directory chooser that follows, same as any other download.
        """
        selection = self.selected_records()
        if what != 'summary_csv' and not selection:
            self.message('Select at least one row first.', 'warn')
            return
        if what == 'run_tree':
            self.message(self._run_tree_warning(selection), 'warn')

        dest = self._ask_directory()
        if dest is None:
            return

        payload: Dict[str, Any] = {
            'what': what,
            'dest': str(dest),
            'selection': [{'run_uuid': r.get('run_uuid') or '',
                           'result_uuid': r.get('result_uuid') or '',
                           'set_label': r.get('set_label') or ''}
                          for r in selection],
            'rebuild': bool(self.rebuild_box.isChecked()),
        }
        if what == 'summary_csv':
            payload['records'] = self.exported_rows()
            payload['columns'] = [c.key for c in self.model.columns]
        self.export_requested.emit(self._source, payload)

    def _run_tree_warning(self, selection: List[Dict[str, Any]]) -> str:
        """What "Recreate run tree" is actually about to copy.

        The unit of this download is the SET, not the row: one selected row
        drags every run beside it. That is the same hazard the provenance
        bundle was written to avoid, and here it is the point of the feature
        -- so it is warned about rather than prevented, with the scale quoted.
        """
        labels = sorted({str(r.get('set_label') or '') for r in selection} - {''})
        lines = []
        for label in labels:
            runs = self.set_sizes.get(label)
            if runs is None:
                lines.append(f"  {label}: run count unknown")
            elif self.last_detail_bytes:
                lines.append(f"  {label}: {runs} runs, roughly "
                             f"{format_bytes(runs * self.last_detail_bytes)}")
            else:
                lines.append(f"  {label}: {runs} runs")
        if not lines:
            lines = ['  (the selected rows name no set)']
        rows = f"{len(selection)} selected row" + ('' if len(selection) == 1 else 's')
        return ('This exports whole sets, not the ' + rows + ':\n'
                + '\n'.join(lines)
                + '\n\nAny size shown is extrapolated from the last run you '
                  'inspected and is a rough figure.\n\nA destination chooser '
                  'opens next -- cancel it to back out.')

    def _request_import(self) -> None:
        """The one write action (D7). Delete and submit stay in the CLI."""
        root = self._ask_directory(remember=False)
        if root is None:
            return
        self.import_requested.emit(
            self._source, {'root': str(root), 'label': root.name, 'dry_run': False})

    def _copy_contact(self) -> None:
        """Put the selected rows' owner contacts on the clipboard (D6).

        The central database holds summaries, not files -- reaching the
        person who ran it is the only way to get the rest.
        """
        records = self.selected_records()
        if not records:
            self.message('Select a row first.', 'warn')
            return
        contacts = sorted({str(r.get('contact') or '').strip() for r in records}
                          - {''})
        if not contacts:
            self.message('That row carries no owner contact.', 'warn')
            return
        self._set_clipboard('\n'.join(contacts))
        self.message('Copied to the clipboard: ' + ', '.join(contacts))

    def _set_clipboard(self, text: str) -> None:
        """The clipboard seam. A method rather than an inline call for the
        same reason `_ask_directory` is one: a test can override it instead of
        depending on what the offscreen platform plugin implements."""
        QGuiApplication.clipboard().setText(text)
