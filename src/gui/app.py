"""The application window.

Split view: configuration on the left, the rendered system on the right, which
is the shape the whole thing exists for -- see what you configured before
committing it to a cluster.

Imports Qt, so it must only be reached through :func:`src.gui.main`.
"""

from __future__ import annotations

import copy
import itertools
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .controllers.database import (DatabaseController, DatabaseRequest,
                                   central_configured)
from .controllers.document import ConfigDocument, DocumentError
from .controllers.generate import GenerateController, GenerateRequest
from .controllers.preview import PreviewController, PreviewRequest
from ..run.hpc_dispatch import hpc_generation_blocker
from ..run.naming import AFM_LIKE_MODELS
from .controllers.settings_document import SettingsDocument
from .qt_logging import install_message_filter, suppressed_summary
from .style import CAPTION
from .preview.viewport import FINITE_COMPONENTS, ViewportPanel
from .schema.simulation_types import FAMILIES, MODES, family_for_model
from .widgets.database.detail import format_bytes
from .widgets.database.pane import (SOURCES, DatabasePane, missing_backend,
                                    render_checks)
from .widgets.form import ConfigForm
from .workspace import RecentWorkspaces, Workspace

logger = logging.getLogger(__name__)

APP_NAME = 'FrictionSim2D'

#: What the Database pane remembers between sessions (D10). Workspace state
#: is a plain key/value store, so this is the whole schema.
DB_STATE_KEYS = ('db_source', 'db_filters', 'db_dest')

#: Workspace state key for the Assistant dock: {'open': bool, 'width': int}.
ASSISTANT_STATE_KEY = 'assistant'
ASSISTANT_DEFAULT_WIDTH = 420


@dataclass
class AssistantUndo:
    """Everything one applied assistant change can be undone back to.

    Taken immediately before the apply: deep copies of both documents (source
    path and workspace dirs included) and, when the change writes a material
    list, that file's prior state (``material_bytes`` None: it did not exist).
    """

    id: int
    #: None for a plot change, which touches only the Plots tab.
    document: Optional[ConfigDocument] = None
    settings_document: Optional[SettingsDocument] = None
    config_keys: List[Tuple[str, str]] = field(default_factory=list)
    settings_keys: List[Tuple[str, str]] = field(default_factory=list)
    #: Highlights this change replaced (the key was already marked by an
    #: earlier change), put back by its Undo.
    replaced_config: Dict[Tuple[str, str], str] = field(default_factory=dict)
    replaced_settings: Dict[Tuple[str, str], str] = field(default_factory=dict)
    touched_settings: bool = False
    material_path: Optional[Path] = None
    material_bytes: Optional[bytes] = None
    #: The Builder right after the apply -- (model, config values, settings
    #: values) and their fingerprint -- to tell whether Undo would also
    #: revert edits the user made since.
    after: Optional[Tuple[str, dict, dict]] = None
    after_fingerprint: str = ''
    #: A plot change: the Plots tab's document before it (a `PlotDocument`),
    #: and as a dict right after it, to tell whether Undo would also revert
    #: plot edits the user made since.
    plots_before: Any = None
    plots_after: Optional[dict] = None


def _merge_highlights(form: ConfigForm, new: Dict[Tuple[str, str], str]):
    """``new`` with a key an earlier change already marked reading from that
    change's *before* ("a → c", not "b → c"); plus the marks it replaces."""
    current = form.highlights()
    replaced = {key: current[key] for key in new if key in current}
    merged = {key: (current[key].split(' → ', 1)[0] + ' → ' + text.split(' → ', 1)[1]
                    if key in replaced else text) for key, text in new.items()}
    return merged, replaced


def _highlight_map(change, scope: str) -> Dict[Tuple[str, str], str]:
    """``(section, key) -> "before → after"`` for one scope of a change's diff."""
    return {(d.section, d.key): f'{d.before!r} → {d.after!r}'
            for d in change.diff if d.scope == scope and d.key and d.section != '(model)'}


def plural(count: int, word: str) -> str:
    """'1 config', '2 configs'. Every word here pluralises with a bare s."""
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _min_numeric(value: object) -> Optional[float]:
    """The smallest numeric value in `value`, which may be a scalar, a sweep
    list, absent (None), or a non-numeric placeholder.

    Returns None -- "nothing usable" -- rather than raising or silently
    dropping bad entries, so a precheck built on this can skip a field
    entirely instead of half-trusting a list with one unresolved member.
    """
    items = value if isinstance(value, list) else [value]
    if not items:
        return None
    numbers = []
    for item in items:
        try:
            numbers.append(float(item))
        except (TypeError, ValueError):
            return None
    return min(numbers)


def _is_true(value: object) -> bool:
    """Truthiness of a config flag that may still be raw `.ini` text.

    A document read from a file holds `'false'` for an unticked box until the
    pydantic model coerces it, and `bool('false')` is True -- which would make
    every non-finite_sheet config look finite.
    """
    if isinstance(value, str):
        return value.strip().lower() in ('true', 'yes', 'on', '1')
    return bool(value)


def _fold_controls(parent: QWidget, form: ConfigForm) -> QHBoxLayout:
    """Collapse/expand-all buttons for a form.

    Both tabs run to nine sections, which is more than fits on screen; folding
    them one at a time to find the one you want is its own chore.
    """
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)

    collapse = QPushButton('Collapse all', parent)
    collapse.setToolTip('Fold every section')
    collapse.clicked.connect(lambda: form.set_all_collapsed(True))
    row.addWidget(collapse)

    expand = QPushButton('Expand all', parent)
    expand.setToolTip('Unfold every section')
    expand.clicked.connect(lambda: form.set_all_collapsed(False))
    row.addWidget(expand)

    row.addStretch(1)
    return row


class MainWindow(QMainWindow):
    """Configuration on the left, rendered system on the right."""

    def __init__(self, workspace: Workspace, *,
                 agent_settings_loader: Optional[Callable[[], object]] = None,
                 confirm_undo: Optional[Callable[[str], bool]] = None,
                 user_busy: Optional[Callable[[], bool]] = None) -> None:
        super().__init__()
        self.workspace = workspace
        # Injectable so tests never meet a modal: asked before an Undo that
        # would also revert later edits (default: a Yes/No box, No default).
        self._confirm_undo = confirm_undo or self._default_confirm_undo
        # Whether the user is mid-edit, so a direct change waits as a card.
        self._user_busy = user_busy or self._default_user_busy
        # Injectable so a test can make the Assistant's build fail; None
        # means src.agent.settings.load_agent_settings, imported lazily.
        self._agent_settings_loader = agent_settings_loader
        self.document = ConfigDocument.blank('afm')
        # An unsaved config still needs somewhere to write a material list --
        # see ConfigDocument.write_material_names' docstring for why it
        # refuses to fall back to the process's cwd.
        self.document.workspace_dir = workspace.configs_dir
        self.document.workspace_root = workspace.root
        # A workspace settings.yaml is used when present, but only because it
        # is passed explicitly; nothing picks it up from the directory itself.
        workspace_settings = workspace.settings_path
        self.settings_document = SettingsDocument.load(
            workspace_settings if workspace_settings.exists() else None)
        self.settings = self.settings_document.as_settings()

        self.preview = PreviewController(workspace, self)
        self.generator = GenerateController(workspace, self)
        self.database = DatabaseController(workspace, self.settings, self)
        # What was last rendered, so an edit afterwards can be flagged as
        # stale rather than leaving the viewport showing a superseded system
        # with no indication anything changed.
        self._render_snapshot: Optional[dict] = None
        # Set around _restore_database_state so the pane's own signals
        # (source_changed) don't loop back into _remember_database_state
        # while a workspace's saved state is still being applied.
        self._restoring_db_state = False

        self.setWindowTitle(f"{APP_NAME} — Simulation Builder")
        self.resize(1500, 950)

        self._build_ui()
        self._connect()
        self._refresh_counts()

    # ------------------------------------------------------------------- ui

    def _build_ui(self) -> None:
        self.top_tabs = QTabWidget(self)
        # In the tab bar's corner, so the toggle is there on every tab.
        self.assistant_button = QPushButton('Assistant', self.top_tabs)
        self.assistant_button.setCheckable(True)
        self.assistant_button.setToolTip('Show or hide the Assistant pane (Ctrl+Shift+A)')
        self.assistant_button.toggled.connect(self.set_assistant_open)
        self.top_tabs.setCornerWidget(self.assistant_button, Qt.TopRightCorner)
        self.top_tabs.addTab(self._build_builder_tab(), 'Builder')
        self.database_page = self._build_database_tab()
        self.top_tabs.addTab(self.database_page, 'Database')
        self.plots_page = self._build_plots_tab()
        self.top_tabs.addTab(self.plots_page, 'Plots')
        self.top_tabs.currentChanged.connect(self._on_top_tab_changed)

        # The Assistant docks beside the tabs, so one conversation stays in
        # view whichever tab is open. Its pane is built on first expand, so
        # pydantic-ai is never imported at startup.
        self.assistant_dock = QWidget(self)
        dock_layout = QVBoxLayout(self.assistant_dock)
        dock_layout.setContentsMargins(0, 0, 0, 0)
        self.assistant_dock.hide()
        self._assistant_pane = None
        self._assistant_width = ASSISTANT_DEFAULT_WIDTH
        self._assistant_applied_width: Optional[int] = None
        self._assistant_width_pending = False
        #: Applied assistant changes, oldest first; only the last can be undone.
        self._assistant_undo: List[AssistantUndo] = []
        self._assistant_ids = itertools.count(1)

        self.main_splitter = QSplitter(Qt.Horizontal, self)
        self.main_splitter.addWidget(self.top_tabs)
        self.main_splitter.addWidget(self.assistant_dock)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 0)
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, False)
        # A drag emits splitterMoved per step; save the width once it settles.
        self._assistant_width_timer = QTimer(self)
        self._assistant_width_timer.setSingleShot(True)
        self._assistant_width_timer.setInterval(400)
        self._assistant_width_timer.timeout.connect(self._remember_assistant_state)
        self.main_splitter.splitterMoved.connect(
            lambda *_: self._assistant_width_timer.start())

        self.assistant_shortcut = QShortcut(QKeySequence('Ctrl+Shift+A'), self)
        self.assistant_shortcut.activated.connect(self.toggle_assistant)

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.addWidget(self.main_splitter, 1)
        self.setCentralWidget(container)
        self._restore_assistant_state()

        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage(f"Workspace: {self.workspace.root}")

    def _build_builder_tab(self) -> QWidget:
        """Everything that existed before D1, unchanged, inside one tab."""
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 5)

        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.addWidget(self._build_header())
        layout.addWidget(splitter, 1)
        layout.addLayout(self._build_actions())
        self.builder_page = page
        return page

    def _build_database_tab(self) -> QWidget:
        """The database pane (D1/D2). It never queries on construction: a
        query happens only when the user presses Search or Refresh."""
        self.database_pane = DatabasePane(self)
        self._restore_database_state()
        return self.database_pane

    def _build_plots_tab(self) -> QWidget:
        """The Plots tab: a plot document editor over the same form machinery.

        Imported here rather than at module level so `import src.gui.app`
        does not pull the plot widgets -- and, through them, one refactor
        away, matplotlib -- into the process ovito's Qt backend owns.
        """
        from .widgets.plots import PlotsPane  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        self.plots_pane = PlotsPane(self.workspace, self)
        return self.plots_pane

    def _render_plot(self) -> None:
        """Render the Plots tab's preview (replaced in tests, which must never
        start the plot worker)."""
        self.plots_pane.render()

    def _on_top_tab_changed(self, index: int) -> None:
        if hasattr(self._assistant_pane, 'set_context_label'):
            self._assistant_pane.set_context_label(self.top_tabs.tabText(index))

    def toggle_assistant(self) -> None:
        """Expand or collapse the Assistant dock (button and Ctrl+Shift+A)."""
        self.set_assistant_open(self.assistant_dock.isHidden())

    def set_assistant_open(self, open_: bool, *, remember: bool = True) -> None:
        """Show the dock at its remembered width, or hide it; build on first show.

        `remember=False` is a restore: it neither captures the on-screen width
        (that belongs to the previous workspace) nor writes state.
        """
        if not open_ and remember:
            self._capture_assistant_width()
        if open_ and self._assistant_pane is None:
            self._build_assistant()
        was_open = self.assistant_dock.isVisibleTo(self)
        self.assistant_dock.setVisible(open_)
        if open_ and (not was_open or not remember):
            self._apply_assistant_width()
        self.assistant_button.blockSignals(True)
        try:
            self.assistant_button.setChecked(open_)
        finally:
            self.assistant_button.blockSignals(False)
        if remember:
            self._remember_assistant_state()

    def _apply_assistant_width(self) -> None:
        """Give the dock its remembered width. Before the window is shown the
        splitter has no real geometry, so the next `showEvent` applies it --
        once per expand, so re-showing the window never undoes a drag."""
        if not self.isVisible():
            self._assistant_width_pending = True
            return
        self._assistant_width_pending = False
        total = sum(self.main_splitter.sizes()) or self.main_splitter.width()
        width = self._assistant_width
        self.main_splitter.setSizes([max(total - width, 1), width])
        # What the splitter actually gave; if it had to clip, this differs
        # from the request and must not replace the remembered width.
        self._assistant_applied_width = self.main_splitter.sizes()[1]

    def showEvent(self, event) -> None:  # pylint: disable=invalid-name
        """Lay the dock out at its remembered width once there is a geometry."""
        super().showEvent(event)
        if self._assistant_width_pending and self.assistant_dock.isVisibleTo(self):
            self._apply_assistant_width()

    def _capture_assistant_width(self) -> None:
        # Only a laid-out window has a splitter size worth keeping, and a size
        # unchanged since `_apply_assistant_width` is the splitter's clip of
        # the remembered width, not a choice the user made.
        if self.isVisible() and self.assistant_dock.isVisibleTo(self):
            width = self.main_splitter.sizes()[1]
            if width > 0 and width != self._assistant_applied_width:
                self._assistant_width = width

    def _remember_assistant_state(self) -> None:
        self._assistant_width_timer.stop()  # this write covers any pending drag
        self._capture_assistant_width()
        self.workspace.remember(ASSISTANT_STATE_KEY, {
            'open': self.assistant_dock.isVisibleTo(self),
            'width': self._assistant_width})

    def _restore_assistant_state(self) -> None:
        """Put the dock back the way this workspace left it (collapsed, at the
        default width, if it saved nothing)."""
        state = self.workspace.state.get(ASSISTANT_STATE_KEY)
        state = state if isinstance(state, dict) else {}
        width = state.get('width')
        valid = isinstance(width, int) and not isinstance(width, bool) and width > 0
        self._assistant_width = width if valid else ASSISTANT_DEFAULT_WIDTH
        self._assistant_applied_width = None
        self.set_assistant_open(state.get('open') is True, remember=False)

    def _build_assistant(self) -> None:
        """Build the pane into the dock; any failure becomes a message there."""
        try:
            self._assistant_pane = self._make_assistant_pane()
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception('the Assistant pane could not be built')
            self._assistant_pane = self._dock_message(f'The Assistant could not start: {exc}')
            return
        self.assistant_dock.layout().addWidget(self._assistant_pane)

    def _dock_message(self, text: str) -> QLabel:
        label = QLabel(text, self.assistant_dock)
        label.setWordWrap(True)
        self.assistant_dock.layout().addWidget(label)
        return label

    def _make_assistant_pane(self) -> QWidget:
        from ..agent import _AGENT_AVAILABLE  # pylint: disable=import-outside-toplevel
        if not _AGENT_AVAILABLE:
            return self._dock_message(
                'The Assistant needs the optional "agent" extra (Python 3.11+):\n'
                '  pip install -e ".[agent]"')
        # pylint: disable=import-outside-toplevel
        from ..agent.ollama_server import OllamaServer
        from ..agent.runtime import AgentSession
        from ..agent.settings import load_agent_settings
        from .assistant.pane import AssistantPane
        settings = (self._agent_settings_loader or load_agent_settings)()
        pane = AssistantPane(AgentSession(settings), self.assistant_deps, settings,
                             self.assistant_dock, ollama_server=OllamaServer())
        pane.apply_requested.connect(self._apply_assistant_change)
        pane.fs_apply_requested.connect(self._apply_assistant_fs_change)
        pane.plot_apply_requested.connect(
            lambda change: self._apply_assistant_plot_change(change, from_card=True))
        pane.show_tab_requested.connect(self._show_tab)
        pane.change_proposed.connect(self._apply_assistant_change_now)
        pane.undo_requested.connect(self._undo_assistant_change)
        pane.accept_requested.connect(self._accept_assistant_changes)
        pane.chat_cleared.connect(lambda: self._end_assistant_undo(lock=False))
        pane.set_context_label(self.top_tabs.tabText(self.top_tabs.currentIndex()))
        return pane

    def assistant_deps(self, document=None):
        """Snapshot of the Builder for one assistant turn (GUI thread only)."""
        # pylint: disable=import-outside-toplevel
        from ..agent.context import snapshot_deps
        from ..agent.materials_db import default_materials_db
        from ..agent.sandbox import SearchRoots, find_examples_dir
        self.form.commit()
        roots = SearchRoots(self.workspace.root, find_examples_dir())
        return snapshot_deps(document or self.document, self.settings_document,
                             self.settings, roots, plot_document=self._plots_state(),
                             materials_db=default_materials_db)

    def _plots_state(self) -> Optional[dict]:
        """The Plots tab's document as a dict (None before the tab exists)."""
        if getattr(self, 'plots_pane', None) is None:
            return None
        try:
            return self.plots_pane.document().model_dump(mode='json', exclude_unset=True)
        except ValueError:  # a plot being edited does not validate yet
            logger.info('the Plots document does not validate; not shown to the assistant')
            return None

    def _show_tab(self, title: str) -> None:
        for index in range(self.top_tabs.count()):
            if self.top_tabs.tabText(index) == title:
                self.top_tabs.setCurrentIndex(index)
                return

    def _apply_assistant_plot_change(self, change, from_card: bool = False) -> None:
        """Add a proposed plot (and the datasets it needs) to the Plots tab,
        select it and render its preview; offer Undo. The tab is not switched."""
        # pylint: disable=import-outside-toplevel
        from ..agent.plot_changes import apply_plot_change
        from ..postprocessing.plot_config import PlotDocument
        pane = self._assistant_pane
        if not from_card and self._user_busy():
            pane.queue_card(change, 'Not added yet: kept as a card because you were busy. '
                                    'Add it when ready.')
            return
        try:
            before = self.plots_pane.document()
            new, _added = apply_plot_change(
                change, before.model_dump(mode='json', exclude_unset=True), self.workspace.root)
            document = PlotDocument.model_validate(new)
        except ValueError as exc:  # a refused dataset, or a plot that no longer validates
            message = f'Not added: {change.title}: {exc}'
            if from_card:
                pane.finish_card(change, message, error=True)
            else:
                pane.note(message, error=True)
            return
        if from_card:
            pane.finish_card(change, '')
        self.plots_pane.set_document(document)
        self.plots_pane.plot_list.setCurrentRow(len(document.plots) - 1)
        self._render_plot()
        undo = AssistantUndo(id=next(self._assistant_ids), plots_before=before,
                             plots_after=self._plots_state())
        self._assistant_undo.append(undo)
        hidden = self.top_tabs.currentWidget() is not self.plots_page
        pane.add_applied_line(undo.id, change.summary(), show_tab='Plots' if hidden else '')
        self._refresh_undo_states()
        self.statusBar().showMessage(change.summary())

    def _undo_plot_change(self, entry: AssistantUndo) -> None:
        """Put the Plots tab's document back as it was before ``entry``."""
        if self._plots_state() != entry.plots_after and not self._confirm_undo(
                'Undo also reverts your later edits in the Plots tab. Undo anyway?'):
            return
        self._assistant_undo.pop()
        self.plots_pane.set_document(entry.plots_before.model_copy(deep=True))
        self._assistant_pane.set_undo_state(entry.id, 'undone')
        self._refresh_undo_states()
        self.statusBar().showMessage('Undid the assistant\'s last plot.')

    def _resync_family_and_mode(self) -> None:
        """Point both selectors at the document's model without re-entering handlers."""
        family = MODES[self.document.model][0]
        index = self.family_box.findData(family.key)
        if index >= 0 and index != self.family_box.currentIndex():
            self.family_box.blockSignals(True)
            try:
                self.family_box.setCurrentIndex(index)
            finally:
                self.family_box.blockSignals(False)
            self._populate_modes()
        self._resync_mode_box()

    def _apply_assistant_change(self, change, as_new: bool) -> None:
        """Apply a change set from a card (ask-before-applying, or Apply as new)."""
        # pylint: disable=import-outside-toplevel
        from ..agent.changes import ToolInputError, apply_changeset, build_changeset
        pane = self._assistant_pane
        proposed = change
        self.form.commit()  # a value still in an editor counts as a Builder edit
        # Undo covers an in-place apply; a new config throws the open one away.
        if as_new and self.document.dirty:
            answer = QMessageBox.question(
                self, 'Unsaved changes',
                'The Builder has unsaved changes that this will replace. Continue?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return
        target = self.document
        try:
            if as_new:
                target = ConfigDocument.blank(self.document.model)
                target.workspace_dir = self.workspace.configs_dir
                target.workspace_root = self.workspace.root
                change = build_changeset(self.assistant_deps(document=target), change.title,
                                         change.ops, change.notes)
                undo = self._assistant_snapshot(change, target)
                result = apply_changeset(change, target, self.settings_document, force=True)
            else:
                undo = self._assistant_snapshot(change, target)
                result = apply_changeset(change, target, self.settings_document)
                if result.stale:
                    fresh = build_changeset(self.assistant_deps(), change.title,
                                            change.ops, change.notes)
                    pane.replace_change(proposed, fresh,
                                        'The Builder changed since that was proposed; '
                                        're-checked against the current config. Apply again '
                                        'to use it.')
                    return
        except ToolInputError as exc:
            pane.finish_card(proposed, f'Not applied: {exc}', error=True)
            return
        pane.finish_card(proposed, '' if result.applied else result.message,
                         error=not result.applied)
        if result.applied or result.partial:
            self._land_assistant_change(change, target, undo, result.message,
                                        partial=result.partial)

    def _apply_assistant_change_now(self, change) -> None:
        """Apply a config/settings change set as it arrives (no card).

        If the user edited the Builder during the turn, the change is rebuilt
        against the current documents and that is applied instead.
        """
        # pylint: disable=import-outside-toplevel
        from ..agent.changes import ToolInputError, apply_changeset, build_changeset
        from ..agent.plot_changes import PlotChange
        if isinstance(change, PlotChange):
            self._apply_assistant_plot_change(change)
            return
        pane = self._assistant_pane
        self.form.commit()  # a value still in an editor counts as a Builder edit
        if self._user_busy():
            pane.queue_card(change, 'Not applied yet: kept as a card because you were '
                                    'editing the Builder. Apply it when ready.')
            return
        try:
            undo = self._assistant_snapshot(change, self.document)
            result = apply_changeset(change, self.document, self.settings_document)
            if result.stale:
                change = build_changeset(self.assistant_deps(), change.title,
                                         change.ops, change.notes)
                pane.note('The Builder changed during the turn; re-checked against '
                          'the current config.')
                undo = self._assistant_snapshot(change, self.document)
                result = apply_changeset(change, self.document, self.settings_document)
        except ToolInputError as exc:
            pane.note(f'Not applied: {change.title}: {exc}', error=True)
            return
        if not result.applied:
            pane.note(result.message, error=True)
            if not result.partial:
                return
        self._land_assistant_change(change, self.document, undo, result.message,
                                    partial=result.partial)

    def _default_user_busy(self) -> bool:
        """The user is in the middle of something a direct apply would trample:
        a refused commit, a modal dialog, or the cursor in a form field."""
        if self.form.refusals() or self.settings_form.refusals():
            return True
        if QApplication.activeModalWidget() is not None:
            return True
        focus = QApplication.focusWidget()
        return focus is not None and (self.form.isAncestorOf(focus)
                                      or self.settings_form.isAncestorOf(focus))

    def _assistant_snapshot(self, change, target) -> AssistantUndo:
        """The undo record for ``change``, taken before it touches anything.

        The material-list path is where `write_material_names` writes for
        ``target`` as it stands now -- an op earlier in the same change that
        re-points `materials_list` is not followed (no tool emits one).
        """
        undo = AssistantUndo(id=next(self._assistant_ids),
                             document=copy.deepcopy(self.document),
                             settings_document=copy.deepcopy(self.settings_document))
        if any(op.kind == 'materials' and op.value for op in change.ops):
            path = target.material_list_target()
            if path is not None:
                undo.material_path = Path(path)
                undo.material_bytes = path.read_bytes() if path.is_file() else None
        return undo

    def _land_assistant_change(self, change, target, undo: AssistantUndo,
                               message: str, partial: bool = False) -> None:
        """Refresh the Builder after an apply, mark what changed, offer Undo.

        ``partial``: a file error stopped the apply part-way; nothing is
        highlighted (what landed is unknown) but Undo is offered.
        """
        # pylint: disable=import-outside-toplevel
        from ..agent.changes import applied_summary, fingerprint
        pane = self._assistant_pane
        if target is not self.document:
            self.document = target
            self.form.document = target
        self._resync_family_and_mode()
        self.form.rebuild()
        if change.touches_settings():
            self.settings_form.rebuild()
            self._on_settings_changed()
            try:
                self.settings = self.settings_document.as_settings()
            except Exception as exc:  # pylint: disable=broad-except
                # apply_changeset refuses invalid settings; this is a backstop.
                pane.note(f'Applied, but the settings are invalid: {exc}', error=True)
            else:
                self.form.set_settings(self.settings)
        self._refresh_counts()
        self._validate()
        self._on_config_changed()
        undo.touched_settings = change.touches_settings()
        if not partial:
            config, undo.replaced_config = _merge_highlights(
                self.form, _highlight_map(change, 'config'))
            settings, undo.replaced_settings = _merge_highlights(
                self.settings_form, _highlight_map(change, 'settings'))
            undo.config_keys, undo.settings_keys = list(config), list(settings)
            self.form.highlight(config)
            self.settings_form.highlight(settings)
        undo.after = (self.document.model, copy.deepcopy(self.document.values),
                      copy.deepcopy(self.settings_document.values))
        undo.after_fingerprint = fingerprint(self.document, self.settings_document)
        self._assistant_undo.append(undo)
        summary = applied_summary(change)
        pane.add_applied_line(undo.id, f'Partly applied, then stopped: {summary}'
                              if partial else summary)
        self._refresh_undo_states()
        self.statusBar().showMessage(message)

    def _refresh_undo_states(self) -> None:
        pane = self._assistant_pane
        for index, entry in enumerate(self._assistant_undo):
            last = index == len(self._assistant_undo) - 1
            pane.set_undo_state(entry.id, 'enabled' if last else 'blocked')
        pane.set_accept_enabled(bool(self.form.highlights() or self.settings_form.highlights()))

    def _undo_assistant_change(self, entry_id: int) -> None:
        """Put the Builder back as it was before the most recent change."""
        if not self._assistant_undo or self._assistant_undo[-1].id != entry_id:
            return
        if self._assistant_undo[-1].plots_before is not None:
            self._undo_plot_change(self._assistant_undo[-1])
            return
        later = self._edits_since(self._assistant_undo[-1])
        if later and not self._confirm_undo(
                f'Undo also reverts your later edits to {later}. Undo anyway?'):
            return
        entry = self._assistant_undo.pop()
        self.document = entry.document
        self.form.document = entry.document
        self.settings_document = entry.settings_document
        self.settings_form.document = entry.settings_document
        if entry.material_path is not None:
            try:
                if entry.material_bytes is None:
                    entry.material_path.unlink(missing_ok=True)
                else:
                    entry.material_path.write_bytes(entry.material_bytes)
            except OSError as exc:
                self._assistant_pane.note(f'Could not restore {entry.material_path}: {exc}',
                                          error=True)
        self._resync_family_and_mode()
        self.form.rebuild()
        self.settings_form.rebuild()
        self._refresh_settings_origin()
        try:
            self.settings = self.settings_document.as_settings()
        except Exception:  # pylint: disable=broad-except
            logger.exception('settings restored by Undo do not validate')
        else:
            self.form.set_settings(self.settings)
        if entry.touched_settings:
            # The restored values must reach settings.yaml at the next build,
            # whatever the snapshot's own flag said.
            self.settings_document.dirty = True
        self.form.clear_highlights(entry.config_keys)
        self.settings_form.clear_highlights(entry.settings_keys)
        self.form.highlight(entry.replaced_config)
        self.settings_form.highlight(entry.replaced_settings)
        self._refresh_counts()
        self._validate()
        self._on_config_changed()
        self._assistant_pane.set_undo_state(entry_id, 'undone')
        self._refresh_undo_states()
        self.statusBar().showMessage('Undid the assistant\'s last change.')

    def _edits_since(self, entry: AssistantUndo) -> str:
        """The fields the user changed after ``entry`` was applied, as text
        ('' when none: Undo then reverts only the assistant's change)."""
        # pylint: disable=import-outside-toplevel
        from ..agent.changes import field_diff, fingerprint
        self.form.commit()
        if entry.after is None or (fingerprint(self.document, self.settings_document)
                                   == entry.after_fingerprint):
            return ''
        model, config, settings = entry.after
        names = (['the simulation model'] if model != self.document.model else [])
        names += [f'[{d.section}] {d.key}'.strip()
                  for d in field_diff('config', config, self.document.values)]
        names += [f'settings [{d.section}] {d.key}'.strip()
                  for d in field_diff('settings', settings, self.settings_document.values)]
        if not names:
            return 'the Builder'
        return ', '.join(names[:6]) + (', …' if len(names) > 6 else '')

    def _default_confirm_undo(self, text: str) -> bool:
        answer = QMessageBox.question(self, 'Undo the assistant\'s change?', text,
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return answer == QMessageBox.Yes

    def _accept_assistant_changes(self) -> None:
        """Accept: the changes stay, their yellow marks go."""
        self.form.clear_highlights()
        self.settings_form.clear_highlights()
        if hasattr(self._assistant_pane, 'set_accept_enabled'):
            self._assistant_pane.set_accept_enabled(False)

    def _end_assistant_undo(self, lock: bool = True) -> None:
        """Forget the undo records: the config was saved or replaced, or the
        chat holding their Undo links was cleared. ``lock`` greys those links.

        A config or settings write (``lock``) leaves plot changes undoable:
        their Undo restores only the Plots tab, which no such write touches."""
        kept = [e for e in self._assistant_undo if lock and e.plots_before is not None]
        if lock and hasattr(self._assistant_pane, 'set_undo_state'):
            for entry in self._assistant_undo:
                if entry.plots_before is None:
                    self._assistant_pane.set_undo_state(entry.id, 'locked')
        self._assistant_undo = kept
        if kept and hasattr(self._assistant_pane, 'set_undo_state'):
            self._refresh_undo_states()

    def _apply_assistant_fs_change(self, change) -> None:
        from ..agent.changes import apply_fs_changeset  # pylint: disable=import-outside-toplevel
        result = apply_fs_changeset(change, self.workspace.root)
        self._assistant_pane.finish_card(change, result.message, error=not result.applied)
        self.statusBar().showMessage(result.message)

    def _shutdown_assistant(self) -> None:
        if hasattr(self._assistant_pane, 'shutdown'):
            self._assistant_pane.shutdown()

    def _build_header(self) -> QWidget:
        header = QWidget(self)
        row = QHBoxLayout(header)
        row.setContentsMargins(0, 0, 0, 0)

        row.addWidget(QLabel('Simulation:', header))
        self.family_box = QComboBox(header)
        for family in FAMILIES:
            self.family_box.addItem(family.label, family.key)
            self.family_box.setItemData(
                self.family_box.count() - 1, family.description, Qt.ToolTipRole)
        row.addWidget(self.family_box)

        self.mode_box = QComboBox(header)
        row.addWidget(self.mode_box)

        row.addSpacing(16)
        self.counts = QLabel('', header)
        self.counts.setToolTip(
            'Configs get their own output directory; builds are the L{n} trees; '
            'jobs are the run scripts an HPC array would submit.')
        row.addWidget(self.counts)

        row.addStretch(1)

        self.workspace_label = QLabel('', header)
        self.workspace_label.setStyleSheet(CAPTION)
        row.addWidget(self.workspace_label)

        change = QPushButton('Change workspace…', header)
        change.clicked.connect(self._choose_workspace)
        row.addWidget(change)

        self._populate_modes()
        self._update_workspace_label()
        return header

    def _build_left(self) -> QWidget:
        self.tabs = QTabWidget(self)

        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        self.form = ConfigForm(self.document, page, settings=self.settings)
        layout.addLayout(_fold_controls(page, self.form))

        self.config_scroll = QScrollArea(page)
        self.config_scroll.setWidget(self.form)
        self.config_scroll.setWidgetResizable(True)
        layout.addWidget(self.config_scroll, 1)

        self.tabs.addTab(page, 'Config')
        self.tabs.addTab(self._build_settings_tab(), 'Settings')
        return self.tabs

    def _build_settings_tab(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)

        self.settings_origin = QLabel('', page)
        self.settings_origin.setWordWrap(True)
        self.settings_origin.setStyleSheet(CAPTION)
        layout.addWidget(self.settings_origin)

        self.settings_form = ConfigForm(self.settings_document, page,
                                        start_collapsed=True)
        self.settings_form.changed.connect(self._on_settings_changed)
        layout.addLayout(_fold_controls(page, self.settings_form))

        scroll = QScrollArea(page)
        scroll.setWidget(self.settings_form)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll, 1)

        row = QHBoxLayout()
        save = QPushButton('Save to workspace', page)
        save.setToolTip(
            'Write settings.yaml into the workspace. It is passed explicitly to '
            'every build from here; it is not picked up automatically by the CLI.')
        save.clicked.connect(self._save_settings)
        row.addWidget(save)

        revert = QPushButton('Revert to defaults', page)
        revert.clicked.connect(self._revert_settings)
        row.addWidget(revert)
        row.addStretch(1)
        layout.addLayout(row)

        self._refresh_settings_origin()
        return page

    def _build_right(self) -> QWidget:
        right = QWidget(self)
        layout = QVBoxLayout(right)
        layout.setContentsMargins(0, 0, 0, 0)

        self.viewport = ViewportPanel(right)
        layout.addWidget(self.viewport, 1)

        self.issues = QPlainTextEdit(right)
        self.issues.setReadOnly(True)
        self.issues.setMaximumHeight(130)
        self.issues.setPlaceholderText('Validation messages appear here.')
        layout.addWidget(self.issues)

        return right

    def _build_actions(self) -> QHBoxLayout:
        row = QHBoxLayout()

        for label, slot in (('Open config…', self._open_config),
                            ('Save config…', self._save_config)):
            button = QPushButton(label, self)
            button.clicked.connect(slot)
            row.addWidget(button)

        row.addStretch(1)

        self.hpc_box = QCheckBox('Also generate HPC scripts', self)
        row.addWidget(self.hpc_box)

        self.render_button = QPushButton('Render', self)
        self.render_button.setShortcut('Ctrl+R')
        self.render_button.setToolTip(
            'Build the configured components and show the assembled system. '
            'Never runs a melt-quench or a minimisation. (Ctrl+R)')
        self.render_button.clicked.connect(self._on_render_clicked)
        row.addWidget(self.render_button)

        self.generate_button = QPushButton('Generate', self)
        self.generate_button.setShortcut('Ctrl+G')
        self.generate_button.setToolTip(
            'Write the full simulation package into the workspace. (Ctrl+G)')
        self.generate_button.clicked.connect(self._generate)
        row.addWidget(self.generate_button)

        return row

    def _connect(self) -> None:
        self.family_box.currentIndexChanged.connect(self._on_family_changed)
        self.mode_box.currentIndexChanged.connect(self._on_mode_changed)
        self.form.changed.connect(self._on_config_changed)
        # A refusal from a per-keystroke commit has no validation pass behind
        # it to write it into the issues panel, so it is said in the status
        # bar. The ones raised by `_validate`'s own commit are folded into the
        # panel there instead, where they stay readable.
        self.form.refused.connect(self._on_commit_refused)
        self.settings_form.refused.connect(self._on_commit_refused)

        self.preview.started.connect(self._on_preview_started)
        self.preview.progress.connect(self.statusBar().showMessage)
        self.preview.finished.connect(self._on_preview_ready)
        self.preview.failed.connect(self._on_preview_failed)
        self.preview.cancelled.connect(self._on_preview_cancelled)

        self.generator.started.connect(lambda: self._busy('Generating…', True))
        self.generator.progress.connect(self.statusBar().showMessage)
        self.generator.finished.connect(self._on_generated)
        self.generator.failed.connect(self._on_generate_failed)

        self._connect_database()

    def _connect_database(self) -> None:
        pane = self.database_pane
        pane.query_requested.connect(self._on_db_query)
        pane.refresh_requested.connect(self._on_db_refresh)
        pane.detail_requested.connect(self._on_db_detail)
        pane.export_requested.connect(self._on_db_export)
        pane.import_requested.connect(self._on_db_import)
        pane.check_requested.connect(self._on_db_check)
        pane.cancel_requested.connect(self._on_db_cancel)
        pane.source_changed.connect(lambda _source: self._remember_database_state())

        self.database.started.connect(self._on_db_started)
        self.database.progress.connect(self.statusBar().showMessage)
        self.database.finished.connect(self._on_db_finished)
        self.database.failed.connect(self._on_db_failed)
        self.database.cancelled.connect(self._on_db_cancelled)

    # ------------------------------------------------------------- database

    def _start_db(self, kind: str, source: str, payload: dict, *,
                  use_cache: bool = True) -> None:
        # A download or an import writes outside the workspace. `start` cancels
        # whatever is running, and cancelling means SIGKILL -- mid-copy that
        # leaves a half-written destination directory nobody cleans up. Since
        # selecting a table row fires a `detail` request on its own, that was
        # one stray click away at all times.
        running = self.database.current_kind
        if running in self.database.WRITE_KINDS:
            self.database_pane.message(
                f"A {running} is still running. Wait for it to finish, or press "
                f"Cancel, before starting a {kind}.", 'warn')
            return
        # The controller reads the profile and API credentials off `settings`
        # at start time, so a settings edit takes effect without a restart.
        self.database.settings = self.settings
        request = DatabaseRequest(kind=kind, source=source, payload=payload)
        self.database.start(request, use_cache=use_cache)

    def _on_db_cancel(self) -> None:
        if not self.database.running:
            return
        kind = self.database.current_kind
        self.database.cancel()
        if kind in self.database.WRITE_KINDS:
            self.database_pane.message(
                f"Cancelled the {kind} request. Anything already written to "
                f"the destination is still there.", 'warn')
        else:
            self.database_pane.message(f"Cancelled the {kind} request.", 'warn')

    def _restore_database_state(self) -> None:
        """Put back the source, filters and destination this workspace last used.

        Snapshot all three values before touching the pane: `set_source` below
        can emit `source_changed`, which is wired to `_remember_database_state`
        (so a live source toggle is recorded even without a query). Since
        `self.workspace.state` is the same dict object read here and written
        there, applying `set_source` first and only then reading back
        `db_filters` would read whatever `_remember_database_state` had just
        written from the pane's still-stale, pre-switch filters -- silently
        clobbering the new workspace's saved filters with the old workspace's.
        The `_restoring_db_state` guard blocks that write outright; the
        snapshot means correctness never depends on the two agreeing.

        Availability (D11) is re-applied *before* `set_source`, under the same
        guard, for both directions: a remembered source that is not usable
        here must not get selected (`set_source` refuses a disabled button on
        its own, but only if the button was already disabled -- checking here
        first is what disables it in time), and a remembered source that has
        become usable again (a workspace switch onto settings where central
        is configured) must be re-enabled before `set_source` gets a chance to
        select it. Guarded so a resulting handover -- disabling the source
        this workspace remembered hands the pane to the other one -- writes
        nothing back to disk; only the on-screen selection falls back. This
        is also what re-checks availability on a workspace switch
        (`_choose_workspace` only calls this method) without a second,
        unguarded call that could write the fallback to the new workspace's
        state.
        """
        state = self.workspace.state
        source = state.get('db_source')
        filters = state.get('db_filters')
        dest = state.get('db_dest')
        self._restoring_db_state = True
        try:
            self._apply_database_availability()
            if source in SOURCES:
                self.database_pane.set_source(source)
            if isinstance(filters, dict):
                self.database_pane.filters.set_value(filters)
            if isinstance(dest, str):
                self.database_pane.last_dest = dest
        finally:
            self._restoring_db_state = False

    def _remember_database_state(self) -> None:
        if self._restoring_db_state:
            return
        self.workspace.remember('db_source', self.database_pane.source)
        self.workspace.remember('db_filters', self.database_pane.filters.value())

    def _apply_database_availability(self) -> None:
        """Disable the central source until this installation is pointed at a
        central database.

        `missing_backend` answers "is httpx installed"; it cannot answer "does
        a central database exist", and the answer today is no. Left enabled,
        every central action ends as a raw ConnectionError against a localhost
        URL nobody ever configured -- a stack-trace string in place of "you
        have not set this up".

        Called from `_restore_database_state` (already under the
        `_restoring_db_state` guard) but also from `_apply_settings`, live,
        any time the Settings tab is applied -- clearing `database.api_url`
        while Central is selected must not let the resulting fallback
        (`set_source_enabled` hands the pane to Local) look like the user
        chose Local: `source_changed` -> `_remember_database_state` would
        otherwise overwrite the workspace's remembered 'central' with 'local'
        outside of any restore. Wrapping the body in the same guard (saving
        and restoring its previous value, since a call from
        `_restore_database_state` already has it set) makes an
        availability-driven fallback silent the same way a restore-driven one
        already is. The other half of "comes back once the missing piece is
        fixed": re-enabling Central here, while the workspace still remembers
        it and the pane is showing the Local fallback, switches the pane back
        rather than leaving it stranded on Local until the next restart.
        """
        if missing_backend('central'):
            return          # the pane already said what to install
        previous = self._restoring_db_state
        self._restoring_db_state = True
        try:
            if not central_configured(self.settings):
                self.database_pane.set_source_enabled(
                    'central', False,
                    'Central: not configured. Set database.api_url (and '
                    'database.api_key) in the Settings tab. The Local store '
                    'needs none of this.')
            else:
                self.database_pane.set_source_enabled('central', True)
                if (self.workspace.state.get('db_source') == 'central'
                        and self.database_pane.source != 'central'):
                    self.database_pane.set_source('central')
        finally:
            self._restoring_db_state = previous

    def _on_db_query(self, source: str, filters: dict) -> None:
        self._remember_database_state()
        self._start_db('list', source, {'filters': filters})

    def _on_db_refresh(self, source: str, filters: dict) -> None:
        # Through `_start_db`, not `database.start`: a Refresh is still a
        # `start`, which SIGKILLs a running export or import.
        self._remember_database_state()
        self._start_db('list', source, {'filters': filters}, use_cache=False)

    def _on_db_detail(self, source: str, payload: dict) -> None:
        self._start_db('detail', source, payload)

    def _on_db_export(self, source: str, payload: dict) -> None:
        self.workspace.remember('db_dest', payload.get('dest', ''))
        self._start_db('export', source, payload)

    def _on_db_import(self, source: str, payload: dict) -> None:
        self._start_db('import', source, payload)

    def _on_db_check(self, source: str) -> None:
        self._start_db('check', source, {})

    def _on_db_started(self, kind: str) -> None:
        self.database_pane.set_busy(True)
        self.statusBar().showMessage(f"Database: {kind}…")

    def _on_db_cancelled(self, kind: str) -> None:
        del kind
        self.database_pane.set_busy(False)

    def _on_db_failed(self, kind: str, message: str) -> None:
        self.database_pane.set_busy(False)
        self.database_pane.message(f"{kind}: {message}", 'error')

    def _on_db_finished(self, kind: str, result: object) -> None:
        self.database_pane.set_busy(False)
        if not isinstance(result, dict):
            return
        if kind == 'list':
            self.database_pane.set_records(result.get('records') or [])
            self.database_pane.set_facets(result.get('facets') or {})
            self.database_pane.set_set_sizes(result.get('set_sizes') or {})
            returned = result.get('returned', 0)
            total = result.get('total_count')
            elapsed = result.get('elapsed_s', 0)
            limit = result.get('limit')
            if total is None and limit:
                # The central client has no true total once its own `limit`
                # cut the rows short (see `_list` in db_worker.py) -- "N of
                # None results" would read like a broken query rather than a
                # truncated one.
                lines = [f"{returned} results (limit {limit}) in {elapsed} s"]
            else:
                lines = [f"{returned} of {total} results in {elapsed} s"]
            if result.get('cached'):
                # `elapsed_s` belongs to the query this result was first
                # fetched by, not to what just happened -- without saying so,
                # a replay of a week-old answer reads as a fresh one.
                lines[0] += ' (cached — Refresh to re-query)'
            warnings = result.get('warnings') or []
            lines += [f"note: {w}" for w in warnings]
            self.database_pane.message('\n'.join(lines),
                                       'warn' if warnings else 'info')
        elif kind == 'detail':
            self.database_pane.show_detail(result)
        elif kind == 'export':
            written = result.get('files') or []
            warnings = result.get('warnings') or []
            lines = [f"Wrote {plural(len(written), 'file')} "
                     f"({format_bytes(int(result.get('bytes') or 0))}) "
                     f"to {result.get('dest')}"]
            lines += [f"note: {w}" for w in warnings]
            self.database_pane.message('\n'.join(lines),
                                       'warn' if warnings else 'info')
        elif kind == 'import':
            lines = [result.get('report', '')]
            for label, key in (('incomplete', 'incomplete'),
                               ('with no results at all', 'empty_runs')):
                entries = result.get(key) or []
                if entries:
                    lines.append(f"{len(entries)} run(s) {label}: "
                                 + ', '.join(entries[:5])
                                 + (' …' if len(entries) > 5 else ''))
            unparseable = result.get('unparseable') or []
            if unparseable:
                lines.append(f"{len(unparseable)} path(s) unparseable: "
                             + ', '.join(unparseable[:5])
                             + (' …' if len(unparseable) > 5 else ''))
            quiet = not (result.get('incomplete') or result.get('empty_runs')
                        or unparseable)
            self.database_pane.message(
                '\n'.join(line for line in lines if line),
                'info' if result.get('ingest_ok') and quiet else
                ('warn' if result.get('ingest_ok') else 'error'))
        elif kind == 'check':
            self.database_pane.message(
                render_checks(result), 'info' if result.get('ready') else 'error')

    # -------------------------------------------------------------- helpers

    def _populate_modes(self) -> None:
        family_key = self.family_box.currentData()
        family = next((f for f in FAMILIES if f.key == family_key), FAMILIES[0])

        self.mode_box.blockSignals(True)
        self.mode_box.clear()
        for mode in family.modes:
            self.mode_box.addItem(mode.label, mode.model)
            self.mode_box.setItemData(
                self.mode_box.count() - 1, mode.description, Qt.ToolTipRole)
        self.mode_box.blockSignals(False)

    def _busy(self, message: str, busy: bool) -> None:
        self.statusBar().showMessage(message)
        self.render_button.setEnabled(not busy)
        self.generate_button.setEnabled(not busy)

    def _set_render_building(self, building: bool) -> None:
        """Keep Render pressable, as a Cancel action, while a preview builds.

        `_busy` disables Render along with Generate; this overrides that for
        Render specifically, right after. A build that is slow (one heavy
        member) or hung must always leave the user a way out, rather than a
        button that looks the same whether it will re-enable in a second or
        never -- which is what "greyed out and I couldn't press it" was.
        """
        self.render_button.setText('Cancel' if building else 'Render')
        self.render_button.setEnabled(True)

    def _on_render_clicked(self) -> None:
        if self.preview.running:
            self.preview.cancel()
        else:
            self._render()

    def _on_preview_started(self) -> None:
        self._busy('Building preview…', True)
        self._set_render_building(True)

    def _on_preview_cancelled(self) -> None:
        self._set_render_building(False)
        self._busy('Preview cancelled.', False)

    def _update_workspace_label(self) -> None:
        self.workspace_label.setText(str(self.workspace.root))

    def _refresh_settings_origin(self) -> None:
        origin = self.settings_document.source_path
        changed = self.settings_document.differs_from_defaults()
        text = f"Settings from: {origin or 'built-in defaults'}"
        if changed:
            groups = ', '.join(sorted(changed))
            text += f"  —  changed from defaults in: {groups}"
        self.settings_origin.setText(text)

    def _on_settings_changed(self) -> None:
        self._refresh_settings_origin()

    def _save_settings(self) -> None:
        """Write the complete settings file into the workspace."""
        self.settings_form.commit()
        if self.settings_form.refusals():
            QMessageBox.warning(
                self, 'Settings were not saved',
                self._record_refusals('Save settings',
                                      self.settings_form.refusals()))
            return
        result = self.settings_document.validate()
        self.settings_form.show_errors(result.issues)
        if not result.ok:
            QMessageBox.warning(self, 'Settings are not valid', result.render())
            return

        try:
            written = self.settings_document.save(self.workspace.settings_path)
        except Exception as exc:  # pylint: disable=broad-except
            QMessageBox.critical(self, 'Could not save settings', str(exc))
            return

        self.settings = self.settings_document.as_settings()
        self.form.set_settings(self.settings)
        self._refresh_settings_origin()
        self.settings_form.clear_highlights()
        self._end_assistant_undo()  # settings.yaml now holds these values
        self.statusBar().showMessage(f"Saved {written}")

    def _revert_settings(self) -> None:
        self.settings_document = SettingsDocument.defaults()
        self.settings_form.document = self.settings_document
        self.settings_form.clear_highlights()
        self.settings_form.rebuild()
        self._refresh_settings_origin()
        # _apply_settings and _save_settings both refresh self.settings and
        # fan it out to the config form after changing the settings document
        # (Task 7); this path did not, so a star -- and _validate, which
        # hands self.settings straight to document.validate -- kept reading
        # the pre-revert object even though the Settings tab itself already
        # showed defaults. A fresh `SettingsDocument.defaults()` is never
        # dirty, so `_apply_settings`'s early return could not repair this
        # later either.
        self.settings = self.settings_document.as_settings()
        self.form.set_settings(self.settings)

    def _apply_settings(self) -> Optional[str]:
        """Make what is on the Settings tab the settings the build will use.

        The tab edits a document, but a build reads a file. Without this,
        typing a value and pressing Generate uses the previous one -- the tab
        looks live while quietly having no effect.

        Returns:
            None on success, otherwise why the settings could not be applied.
        """
        self.settings_form.commit()
        # Checked before the dirty gate: a section that refused its commit
        # may be exactly the reason the document looks unchanged.
        if self.settings_form.refusals():
            return self._record_refusals('Applying settings',
                                         self.settings_form.refusals())
        if not self.settings_document.dirty:
            return None

        result = self.settings_document.validate()
        if not result.ok:
            self.settings_form.show_errors(result.issues)
            return result.render()

        self.settings_document.save(self.workspace.settings_path)
        # Written to disk: an Undo now would leave settings.yaml (what builds
        # read) on values the Builder no longer shows.
        self._end_assistant_undo()
        self.settings = self.settings_document.as_settings()
        self.form.set_settings(self.settings)
        self._refresh_settings_origin()
        self.statusBar().showMessage('Applied settings changes.')
        self._apply_database_availability()
        return None

    def _sync_settings(self) -> bool:
        """Apply the settings, reporting a failure. False stops the build."""
        error = self._apply_settings()
        if error is None:
            return True

        self.tabs.setCurrentIndex(1)
        QMessageBox.warning(
            self, 'Settings are not valid',
            f"The build was not started because the settings cannot be "
            f"applied:\n\n{error}")
        return False

    @property
    def settings_file(self) -> Optional[Path]:
        """The settings file builds should use, if the workspace has one.

        Passed explicitly because nothing loads a settings.yaml from the
        working directory on its own -- only an explicit argument or the file
        in ~/.config/FrictionSim2D.
        """
        path = self.workspace.settings_path
        return path if path.exists() else None

    def _refresh_counts(self) -> None:
        try:
            summary = self.document.expansion()
            text = (
                f"{plural(summary.configs, 'config')} → "
                f"{plural(summary.builds, 'build')} → "
                f"{plural(summary.jobs, 'job')}")
        except Exception as exc:  # pylint: disable=broad-except
            text = '—'
            logger.debug("count failed: %s", exc)
        if text != self.counts.text():
            self.counts.setText(text)

    def _tip_footprint_issues(self) -> List[str]:
        """The tip-vs-periodic-image precheck, as issue lines.

        Non-blocking: a footprint under 2x the tip diameter lets the tip
        interact with its own periodic image, but plenty of valid setups
        (small scans, deliberately, or values still being sized) fall in this
        range, so it is reported and not gated on.

        Silent on a two-material config, where the two dimensions it reads
        are not the footprint that gets built. `hetero_afm` replicates the
        coincidence cell until the periodic footprint is >= `4 * tip_r`
        (`compute_replication_factors`) whatever `[2D] x`/`y` say -- neither
        `hetero_afm.py` nor `hetero.py` reads them -- so the warning was
        advice about numbers the build never consults, on a geometry that
        satisfies the rule by construction. The homogeneous AFM path does
        build the footprint out of those two fields, and keeps warning.

        Its own method rather than a block inside `_validate`: it needs three
        locals of its own to read a value that may be a sweep list, and
        `_validate` had grown past the point where three more could be told
        apart from the ten it already carries.
        """
        if self.document.model not in AFM_LIKE_MODELS:
            return []
        if self.document.has_second_material():
            return []

        sheet_x = _min_numeric(self.document.get('2D', 'x'))
        sheet_y = _min_numeric(self.document.get('2D', 'y'))
        tip_r = _min_numeric(self.document.get('tip', 'r'))
        if sheet_x is None or sheet_y is None or tip_r is None:
            return []
        if min(sheet_x, sheet_y) >= 4 * tip_r:
            return []
        return ["[tip] sheet footprint < 2× tip diameter; the tip will "
                "interact with its own periodic image"]

    def _validate(self, enforce_required: bool = True,
                  refusals_block: bool = True) -> bool:
        """Validate and surface the result. Returns True when valid.

        `enforce_required` gates only the starred-but-empty-and-blocking
        term: `_render` passes False so a cleared `force`/`pressure` (marked
        `required_blocks` for Generate, but not read by the preview builder)
        does not stop a user from previewing the geometry while trying
        another load value. `_generate` leaves it True -- an actually
        unbuildable LAMMPS script must still be refused. Every other
        validation failure (the document's own `validate()`, unsupported
        lists) blocks both regardless, and the issue lines are reported the
        same either way.

        `refusals_block` decides whether a section the form could not commit
        stops the action. It does by default: after a refusal the document no
        longer matches the form, so Generate would build from a config the
        user never approved. `_render` passes False -- a preview persists
        nothing and launches nothing, and being able to look at the geometry
        is often how the disagreement gets diagnosed -- so there it is
        reported and not gated on.
        """
        self.form.commit()
        result = self.document.validate(self.settings)
        self.form.show_errors(result.issues)

        unsupported = self.document.unsupported_lists()
        # A section the form could not commit is the first thing wrong with
        # this config: what is on screen is not what would be built. Reported
        # here, alongside every other reason the config will not build, rather
        # than as the modal-per-section this used to raise.
        refusals = self.form.refusals()
        lines = list(refusals)
        lines += [issue.render() for issue in result.issues]
        lines += [
            f"[{section}] {key}: a list here is not supported -- it would not "
            f"change the output directory, so the runs would overwrite each other"
            for section, key in unsupported
        ]

        missing = self.form.missing_required()
        lines += [
            f"[{sec}] {label}: required, currently empty"
            for sec, _key, label, _blocks in missing
        ]
        blocking = [blocks for _sec, _key, _label, blocks in missing]

        expansion = None
        try:
            expansion = self.document.expansion()
        except Exception as exc:  # pylint: disable=broad-except
            # The sweep cannot be counted -- a materials_list whose
            # per-material files do not resolve, or whose lines name a
            # different number of materials than there are sheet sections.
            #
            # Reported, not swallowed. `expansion_output_dirs()` runs the same
            # expansion and hands back its message verbatim, which is the
            # sentence saying what to fix ("...names a single material, but
            # this config has a [2D-2] section and expects ordered pairs...").
            # Generate has shown it since it started asking that method first;
            # here it was caught into a debug log, so Render and this panel
            # showed a config that read as fine.
            logger.debug("expansion count failed: %s", exc)
            lines.append(self.document.expansion_output_dirs().error or str(exc))

        if expansion is not None:
            # Only the material-sweep notes: a materials_list that could not
            # be read is reported here rather than silently falling back to
            # "nothing configured", which is how a typo in the path used to
            # become a full sweep. ExpansionSummary.notes also carries
            # routine, non-error information (outer_loop splitting the run
            # phase, a PES refinement script) that does not belong beside a
            # "config is valid" message.
            lines += [n for n in expansion.notes
                     if n.startswith('[2D] materials_list')]

        lines += self._tip_footprint_issues()

        # Non-blocking by construction: each of these builds and runs, it just
        # does not do what it looks like it does (see `ConfigDocument.conflicts`),
        # so it is said out loud and `ok` is left alone.
        lines += self.document.conflicts()

        if self.document.pending_template_issues:
            # Still worth saying the placeholders are pending -- but only the
            # working case may promise they get filled in. When the expansion
            # was refused above, "they will be substituted across each
            # material in the sweep when you generate" contradicts the
            # refusal, and it is the line that reads as "the config is fine".
            count = len(self.document.pending_template_issues)
            if expansion is None:
                lines.append(
                    f"{count} path(s) contain {{mat}}; none of them can be "
                    f"substituted until the problem above is fixed.")
            else:
                lines.append(
                    f"{count} path(s) contain {{mat}}; they will be "
                    f"substituted across {expansion.configs} config(s) when "
                    f"you generate.")

        text = '\n'.join(lines) if lines else 'Config is valid.'
        if text != self.issues.toPlainText():
            self.issues.setPlainText(text)
        self._refresh_counts()
        ok = result.ok and not unsupported
        if enforce_required:
            ok = ok and not any(blocking)
        if refusals_block:
            ok = ok and not refusals
        return ok

    # --------------------------------------------------------------- events

    def closeEvent(self, event: QCloseEvent) -> None:
        """Ask before discarding an edited config or settings document.

        A quit that silently threw away unsaved work used to be the only
        behaviour on offer; this puts the same Save/Discard/Cancel choice in
        front of the user that closing an editor normally would.
        """
        self._remember_assistant_state()
        config_dirty = self.document.dirty
        settings_dirty = self.settings_document.dirty
        if not (config_dirty or settings_dirty):
            self.database.cancel()
            self._shutdown_assistant()
            event.accept()
            return

        if config_dirty and settings_dirty:
            what = 'The configuration and the settings have'
        elif settings_dirty:
            what = 'The settings have'
        else:
            what = 'The configuration has'

        choice = QMessageBox.question(
            self, 'Unsaved changes',
            f'{what} unsaved changes. Save before closing?',
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save)

        if choice == QMessageBox.Cancel:
            event.ignore()
            return
        if choice == QMessageBox.Discard:
            self.database.cancel()
            self._shutdown_assistant()
            event.accept()
            return

        # Save: route through whichever document(s) are actually dirty. Each
        # save path may itself be cancelled (a file dialog) or fail
        # (validation), leaving its document dirty -- in that case the
        # window must not close, or the edits it holds would still be lost.
        if config_dirty:
            self._save_config()
        if settings_dirty:
            self._save_settings()
        if self.document.dirty or self.settings_document.dirty:
            event.ignore()
            return
        self.database.cancel()
        self._shutdown_assistant()
        event.accept()

    def _on_family_changed(self) -> None:
        self._populate_modes()
        self._on_mode_changed()

    def _on_mode_changed(self) -> None:
        model = self.mode_box.currentData()
        if not model or model == self.document.model:
            return
        try:
            self.document.set_model(model)
        except DocumentError as exc:
            QMessageBox.warning(self, 'Cannot switch', str(exc))
            self._resync_mode_box()
            return

        # Save scroll position before rebuild
        scroll_value = self.config_scroll.verticalScrollBar().value()

        self.form.rebuild()

        # Restore scroll position after rebuild. The scrollbar range settles on a
        # deferred layout pass, so this restore relies on returning to the event
        # loop before repaint.
        self.config_scroll.verticalScrollBar().setValue(scroll_value)

        self._refresh_counts()
        # A mode switch swaps the document's model wholesale -- the largest
        # possible edit short of opening a different file -- so the badge
        # must reflect it just as it would a single field edit. `form.changed`
        # is not emitted here: `rebuild()` runs under each SectionForm's own
        # `load()`, not a user edit. `_on_config_changed` already no-ops when
        # nothing has been rendered yet.
        self._on_config_changed()

    def _resync_mode_box(self) -> None:
        """Put the selector back on the model the document is actually in.

        A refused switch that leaves the combo reading the mode it refused is
        worse than the refusal itself: `_on_mode_changed` returns early when
        the combo already matches `document.model`, so the *next* attempt to
        select that same mode -- after fixing whatever the refusal asked for --
        would silently do nothing at all.

        Restored under `blockSignals` so putting it back does not re-enter this
        handler and warn a second time.
        """
        index = self.mode_box.findData(self.document.model)
        if index < 0:
            return
        self.mode_box.blockSignals(True)
        try:
            self.mode_box.setCurrentIndex(index)
        finally:
            self.mode_box.blockSignals(False)

    def _on_commit_refused(self, message: str) -> None:
        """Show a refused commit without blocking on it.

        Deliberately not a dialog: this arrives from `_on_field_changed`,
        which fires on ordinary field edits, so a modal would interrupt typing
        -- and, with no one to dismiss it in an automated run, would hang
        rather than fail.

        The status bar is enough *here* only because nothing has been written
        or launched yet. The moment an action would persist something, the
        refusal has to stop it and be said somewhere that survives -- see
        `_record_refusals`.
        """
        logger.warning("commit refused: %s", message)
        self.statusBar().showMessage(message)

    def _record_refusals(self, action: str, refusals: List[str]) -> str:
        """Put refusals in the issues panel and return them as one message.

        The status bar cannot carry these. `_save_config` overwrites it with
        its own `Saved …` a few lines later -- behind an intervening file
        dialog -- so the only trace of the disagreement vanished and a file
        was left on disk that silently did not match the form. The panel
        survives until the next validation, and the caller adds a dialog when
        it is refusing an action the user explicitly started.
        """
        logger.warning("%s refused: %s", action, refusals)
        text = '\n'.join(refusals)
        self.issues.setPlainText(
            f"{action} refused -- the form and the config disagree, so what "
            f"would be written is not what is on screen:\n{text}")
        return text

    def _on_config_changed(self) -> None:
        self._refresh_counts()
        if self._render_snapshot is not None:
            self.viewport.set_stale(self.document.to_dict() != self._render_snapshot)

    def _open_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open configuration', str(self.workspace.configs_dir),
            'Configs (*.ini *.yaml *.yml *.json);;All files (*)')
        if not path:
            return

        model = self.mode_box.currentData() or 'afm'
        try:
            self.document = ConfigDocument.from_file(path, model)
        except DocumentError as exc:
            QMessageBox.warning(self, 'Cannot open this config', str(exc))
            return
        except Exception as exc:  # pylint: disable=broad-except
            QMessageBox.critical(self, 'Could not read the config', str(exc))
            return

        # A loaded document always has source_path set, so write_material_names
        # never actually consults this -- set for consistency with __init__
        # and _choose_workspace rather than because this document needs it.
        self.document.workspace_dir = self.workspace.configs_dir
        self.document.workspace_root = self.workspace.root
        self.form.document = self.document
        self.form.clear_highlights()
        self._end_assistant_undo()
        self.form.rebuild()
        self._validate()
        # Opening a different config swaps the whole document out from under
        # a render -- the largest staleness event there is -- yet it never
        # goes through `form.changed` (`load()` runs under `_loading`), so
        # the badge stayed hidden until the user separately touched a field.
        self._on_config_changed()
        self.statusBar().showMessage(f"Opened {path}")

    def _save_config(self) -> None:
        self.form.commit()
        # A refused commit means the document no longer matches the form.
        # Writing it anyway produces a `.ini` that disagrees with the screen
        # -- the file is the artefact, so this is a wrong-results path and it
        # stops here rather than being noted in passing.
        if self.form.refusals():
            QMessageBox.warning(
                self, 'This config was not saved',
                self._record_refusals('Save', self.form.refusals()))
            return
        suggested = self.document.source_path or \
            self.workspace.default_config_path(self.document.model)
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save configuration', str(suggested), 'Config (*.ini)')
        if not path:
            return
        try:
            written = self.document.save(path)
        except Exception as exc:  # pylint: disable=broad-except
            QMessageBox.critical(self, 'Could not save', str(exc))
            return
        # Saved: the assistant's marks have served their purpose, and an Undo
        # would now leave the Builder disagreeing with the file just written.
        self.form.clear_highlights()
        self._end_assistant_undo()
        if hasattr(self._assistant_pane, 'set_accept_enabled'):
            self._assistant_pane.set_accept_enabled(bool(self.settings_form.highlights()))
        self.statusBar().showMessage(f"Saved {written}")

    def _choose_workspace(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, 'Choose a workspace', str(self.workspace.root))
        if not chosen:
            return
        if self._assistant_width_timer.isActive():
            self._remember_assistant_state()  # a pending drag belongs to the old workspace
        self.workspace = Workspace.open(chosen)
        RecentWorkspaces().add(chosen)
        self.preview.workspace = self.workspace
        self.generator.workspace = self.workspace
        self.database.workspace = self.workspace
        self._restore_database_state()
        if hasattr(self._assistant_pane, 'new_chat'):
            self._assistant_pane.new_chat()
        self._restore_assistant_state()
        # Only meaningful for an unsaved document (write_material_names falls
        # back to source_path's own directory once there is one) -- but keep
        # it correct regardless, so switching workspaces before ever saving
        # does not leave the old workspace's directory behind.
        self.document.workspace_dir = self.workspace.configs_dir
        self.document.workspace_root = self.workspace.root
        self._update_workspace_label()
        self.statusBar().showMessage(f"Workspace: {self.workspace.root}")

    # -------------------------------------------------------------- actions

    def _render(self) -> None:
        if not self._sync_settings():
            return
        if not self._validate(enforce_required=False, refusals_block=False):
            self.statusBar().showMessage('Fix the reported problems first.')
            return

        request = PreviewRequest(
            config=self.document.to_dict(),
            model=self.document.model,
            layer=None,
            settings_file=self.settings_file,
            referenced_files=self._referenced_files(),
            base_dir=self.document.base_dir,
        )
        self.preview.start(request)

    def _referenced_files(self) -> list:
        """Files the config points at, so the cache notices when they change."""
        out = []
        for values in self.document.to_dict().values():
            if not isinstance(values, dict):
                continue
            for key, value in values.items():
                if key.endswith('_path') and isinstance(value, str):
                    out.append(Path(value))
        return out

    def _on_preview_ready(self, system) -> None:
        self._set_render_building(False)
        message = f"Rendered {system.count:,} atoms."
        # A sweep config previews exactly one member (see PreviewController);
        # say which, so what is shown is never a silent, arbitrary choice.
        if self.preview.current_label:
            message += f" {self.preview.current_label}"
        self._busy(message, False)
        # `finite_sheet` makes `2D` an island inside the substrate box rather
        # than a sheet spanning it, so there -- and only there -- a `2D`
        # reaching the periodic wall is the >=2x tip-diameter failure rather
        # than the geometry working as intended. Nothing in the assembled
        # system records which it is, so it is said here, where the config is.
        finite = list(FINITE_COMPONENTS)
        if _is_true(self.document.get('general', 'finite_sheet')):
            finite.append('2D')
        self.viewport.set_finite_components(finite)
        self.viewport.show_system(system)
        self._render_snapshot = self.document.to_dict()
        self.viewport.set_stale(False)
        if system.warnings:
            self.issues.setPlainText('\n'.join(system.warnings))

    def _on_preview_failed(self, message: str) -> None:
        self._set_render_building(False)
        self._busy('Preview failed.', False)
        self.issues.setPlainText(message)
        QMessageBox.warning(self, 'Preview failed', message)

    def _generate(self) -> None:
        if not self._sync_settings():
            return
        if not self._validate():
            self.statusBar().showMessage('Fix the reported problems first.')
            return

        # Asked before the counts, which come from the same expansion and so
        # would raise the very message this reports -- as a traceback out of a
        # Qt slot, where the user never sees it.
        preview = self.document.expansion_output_dirs()
        if preview.error:
            self.issues.setPlainText(preview.error)
            QMessageBox.warning(
                self, 'Cannot expand this config',
                f"Nothing was generated.\n\n{preview.error}")
            self.statusBar().showMessage('Fix the reported problems first.')
            return
        dirs = preview.dirs

        summary = self.document.expansion()
        message = (
            f"This will generate {summary.configs} config(s), {summary.builds} "
            f"build(s) and {summary.jobs} job(s) into:\n{self.workspace.runs_dir}"
        )

        if dirs:
            shown = dirs[:8]
            message += "\n\n" + "\n".join(str(d) for d in shown)
            if len(dirs) > 8:
                message += f"\n… and {len(dirs) - 8} more"

            duplicates = sorted({str(d) for d in dirs if dirs.count(d) > 1})
            if duplicates:
                message += (
                    "\n\nWARNING: these output directories are shared by more "
                    "than one run and would overwrite each other:\n" +
                    "\n".join(duplicates)
                )

        # Say up front that the tick box will not produce anything, rather than
        # after the build has run.
        if self.hpc_box.isChecked():
            blocker = hpc_generation_blocker(self.settings)
            if blocker:
                message += (
                    f"\n\nHPC job scripts will NOT be written.\n{blocker}\n"
                    f"You can set this on the Settings tab, under HPC."
                )

        confirm = QMessageBox.question(
            self, 'Generate simulation package', message + "\n\nContinue?")
        if confirm != QMessageBox.Yes:
            return

        self.generator.start(GenerateRequest(
            config=self.document.to_dict(),
            model=self.document.model,
            generate_hpc=self.hpc_box.isChecked(),
            settings_file=self.settings_file,
            base_dir=self.document.base_dir,
        ))

    def _on_generated(self, result: dict) -> None:
        created = len(result.get('created', []))
        jobs = result.get('jobs')
        message = f"Generated {created} simulation(s)"
        if jobs is not None:
            message += f", {jobs} job script(s)"
        self._busy(message, False)

        detail = message + f"\n\n{result.get('simulation_root')}"

        if result.get('hpc_requested'):
            written = result.get('hpc_scripts', 0)
            if written:
                detail += f"\n\nHPC scripts: {written} written to the hpc/ folder."
            else:
                detail += "\n\nHPC scripts: none written."
                if result.get('hpc_blocked'):
                    detail += f"\n{result['hpc_blocked']}"

        if result.get('failures'):
            detail += "\n\nFailures:\n" + '\n'.join(result['failures'])
        QMessageBox.information(self, 'Generation complete', detail)

    def _on_generate_failed(self, message: str) -> None:
        self._busy('Generation failed.', False)
        self.issues.setPlainText(message)
        QMessageBox.warning(self, 'Generation failed', message)


def choose_workspace(parent=None) -> Optional[Path]:
    """Ask for a workspace, defaulting to the most recent one."""
    recent = RecentWorkspaces()
    start = recent.most_recent() or Path.home()
    chosen = QFileDialog.getExistingDirectory(
        parent, 'Choose a working directory for FrictionSim2D', str(start))
    return Path(chosen) if chosen else None


def main(argv: Sequence[str] = ()) -> int:
    """Run the Qt event loop. Returns the process exit code."""
    install_message_filter()

    app = QApplication.instance() or QApplication(list(argv) or sys.argv[:1])
    app.setApplicationName(APP_NAME)

    root = choose_workspace()
    if root is None:
        return 0

    workspace = Workspace.open(root)
    RecentWorkspaces().add(root)

    window = MainWindow(workspace)
    window.show()
    status = app.exec()

    summary = suppressed_summary()
    if summary:
        print(summary, file=sys.stderr)
    return status
