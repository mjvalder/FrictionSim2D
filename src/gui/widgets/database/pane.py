"""The Database pane: source toggle, filters, table, detail, actions, messages.

One non-modal message panel is the pane's only error surface, the way `issues`
is for the builder, and exactly one modal is reachable from here --
`_ask_directory`. Both rules exist because an unpatched dialog in a teardown
path hangs the test suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, Signal
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

from .detail import DetailPanel
from .filters import FilterBar
from .model import ResultSortProxyModel, ResultTableModel

__all__ = ['DatabasePane', 'SOURCES', 'SOURCE_LABELS']

#: The pane's third axis, beside the AiiDA profile.
SOURCES: Tuple[str, str] = ('local', 'central')
SOURCE_LABELS = {'local': 'Local store', 'central': 'Central'}


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
        return row

    # ---------------------------------------------------------------- state

    @property
    def source(self) -> str:
        return self._source

    def set_source(self, source: str) -> None:
        if source not in SOURCES or source == self._source:
            return
        button = self._source_buttons.button(SOURCES.index(source))
        if button is not None:
            button.setChecked(True)

    def _on_source_toggled(self, source: str, checked: bool) -> None:
        if not checked or source == self._source:
            return
        self._source = source
        self.source_changed.emit(source)

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
        del level  # Task 15 colours the panel by level
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

    # ---------------------------------------------------------------- modal

    def _ask_directory(self) -> Optional[Path]:
        """Where a download goes. The only modal this package opens: tests
        override this method rather than trying to click a real dialog."""
        chosen = QFileDialog.getExistingDirectory(
            self, 'Choose a destination', self.last_dest or str(Path.home()))
        if not chosen:
            return None
        self.last_dest = chosen
        return Path(chosen)
