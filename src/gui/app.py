"""The application window.

Split view: configuration on the left, the rendered system on the right, which
is the shape the whole thing exists for -- see what you configured before
committing it to a cluster.

Imports Qt, so it must only be reached through :func:`src.gui.main`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
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

from .controllers.document import ConfigDocument, DocumentError
from .controllers.generate import GenerateController, GenerateRequest
from .controllers.preview import PreviewController, PreviewRequest
from ..core.run import AFM_LIKE_MODELS, hpc_generation_blocker
from .controllers.settings_document import SettingsDocument
from .qt_logging import install_message_filter, suppressed_summary
from .style import CAPTION
from .preview.viewport import FINITE_COMPONENTS, ViewportPanel
from .schema.simulation_types import FAMILIES, MODES, family_for_model
from .widgets.form import ConfigForm
from .workspace import RecentWorkspaces, Workspace

logger = logging.getLogger(__name__)

APP_NAME = 'FrictionSim2D'


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

    def __init__(self, workspace: Workspace) -> None:
        super().__init__()
        self.workspace = workspace
        self.document = ConfigDocument.blank('afm')
        # An unsaved config still needs somewhere to write a material list --
        # see ConfigDocument.write_material_names' docstring for why it
        # refuses to fall back to the process's cwd.
        self.document.workspace_dir = workspace.configs_dir
        # A workspace settings.yaml is used when present, but only because it
        # is passed explicitly; nothing picks it up from the directory itself.
        workspace_settings = workspace.settings_path
        self.settings_document = SettingsDocument.load(
            workspace_settings if workspace_settings.exists() else None)
        self.settings = self.settings_document.as_settings()

        self.preview = PreviewController(workspace, self)
        self.generator = GenerateController(workspace, self)
        # What was last rendered, so an edit afterwards can be flagged as
        # stale rather than leaving the viewport showing a superseded system
        # with no indication anything changed.
        self._render_snapshot: Optional[dict] = None

        self.setWindowTitle(f"{APP_NAME} — Simulation Builder")
        self.resize(1500, 950)

        self._build_ui()
        self._connect()
        self._refresh_counts()

    # ------------------------------------------------------------------- ui

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 5)

        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.addWidget(self._build_header())
        layout.addWidget(splitter, 1)
        layout.addLayout(self._build_actions())
        self.setCentralWidget(container)

        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage(f"Workspace: {self.workspace.root}")

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
        self.statusBar().showMessage(f"Saved {written}")

    def _revert_settings(self) -> None:
        self.settings_document = SettingsDocument.defaults()
        self.settings_form.document = self.settings_document
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
        self.settings = self.settings_document.as_settings()
        self.form.set_settings(self.settings)
        self._refresh_settings_origin()
        self.statusBar().showMessage('Applied settings changes.')
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
            # The sweep cannot be counted yet -- typically a materials_list
            # whose per-material files do not resolve.
            logger.debug("expansion count failed: %s", exc)

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

        if self.document.model in AFM_LIKE_MODELS:
            # Non-blocking: a footprint under 2x the tip diameter lets the
            # tip interact with its own periodic image, but plenty of valid
            # setups (small scans, deliberately, or values still being
            # sized) fall in this range, so it is reported and not gated on.
            sheet_x = _min_numeric(self.document.get('2D', 'x'))
            sheet_y = _min_numeric(self.document.get('2D', 'y'))
            tip_r = _min_numeric(self.document.get('tip', 'r'))
            if sheet_x is not None and sheet_y is not None and tip_r is not None \
                    and min(sheet_x, sheet_y) < 4 * tip_r:
                lines.append(
                    "[tip] sheet footprint < 2× tip diameter; the tip will "
                    "interact with its own periodic image")

        # Non-blocking by construction: each of these builds and runs, it just
        # does not do what it looks like it does (see `ConfigDocument.conflicts`),
        # so it is said out loud and `ok` is left alone.
        lines += self.document.conflicts()

        if self.document.pending_template_issues:
            # Still worth saying the placeholders are pending even when the
            # count is not available; the count is the part at risk.
            target = f"{expansion.configs} config(s)" if expansion is not None \
                else "each material in the sweep"
            lines.append(
                f"{len(self.document.pending_template_issues)} path(s) contain "
                f"{{mat}}; they will be substituted across {target} when you "
                f"generate.")

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
        config_dirty = self.document.dirty
        settings_dirty = self.settings_document.dirty
        if not (config_dirty or settings_dirty):
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
        self.form.document = self.document
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
        self.statusBar().showMessage(f"Saved {written}")

    def _choose_workspace(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, 'Choose a workspace', str(self.workspace.root))
        if not chosen:
            return
        self.workspace = Workspace.open(chosen)
        RecentWorkspaces().add(chosen)
        self.preview.workspace = self.workspace
        self.generator.workspace = self.workspace
        # Only meaningful for an unsaved document (write_material_names falls
        # back to source_path's own directory once there is one) -- but keep
        # it correct regardless, so switching workspaces before ever saving
        # does not leave the old workspace's directory behind.
        self.document.workspace_dir = self.workspace.configs_dir
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
