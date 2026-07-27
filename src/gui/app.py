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
from typing import Optional, Sequence

from PySide6.QtCore import Qt
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

from ..core.config import load_settings
from .controllers.document import ConfigDocument, DocumentError
from .controllers.generate import GenerateController, GenerateRequest
from .controllers.preview import PreviewController, PreviewRequest
from .preview.viewport import ViewportPanel
from .schema.simulation_types import FAMILIES, MODES, family_for_model
from .widgets.form import ConfigForm
from .workspace import RecentWorkspaces, Workspace

logger = logging.getLogger(__name__)

APP_NAME = 'FrictionSim2D'


class MainWindow(QMainWindow):
    """Configuration on the left, rendered system on the right."""

    def __init__(self, workspace: Workspace) -> None:
        super().__init__()
        self.workspace = workspace
        self.document = ConfigDocument.blank('afm')
        self.settings = load_settings()

        self.preview = PreviewController(workspace, self)
        self.generator = GenerateController(workspace, self)

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
        self.workspace_label.setStyleSheet('color: palette(mid);')
        row.addWidget(self.workspace_label)

        change = QPushButton('Change workspace…', header)
        change.clicked.connect(self._choose_workspace)
        row.addWidget(change)

        self._populate_modes()
        self._update_workspace_label()
        return header

    def _build_left(self) -> QWidget:
        self.tabs = QTabWidget(self)

        self.form = ConfigForm(self.document, self)
        scroll = QScrollArea(self)
        scroll.setWidget(self.form)
        scroll.setWidgetResizable(True)
        self.tabs.addTab(scroll, 'Config')

        self.settings_view = QPlainTextEdit(self)
        self.settings_view.setReadOnly(True)
        self._refresh_settings_view()
        self.tabs.addTab(self.settings_view, 'Settings')

        return self.tabs

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
        self.render_button.setToolTip(
            'Build the configured components and show the assembled system. '
            'Never runs a melt-quench or a minimisation.')
        self.render_button.clicked.connect(self._render)
        row.addWidget(self.render_button)

        self.generate_button = QPushButton('Generate', self)
        self.generate_button.setToolTip(
            'Write the full simulation package into the workspace.')
        self.generate_button.clicked.connect(self._generate)
        row.addWidget(self.generate_button)

        return row

    def _connect(self) -> None:
        self.family_box.currentIndexChanged.connect(self._on_family_changed)
        self.mode_box.currentIndexChanged.connect(self._on_mode_changed)
        self.form.changed.connect(self._on_config_changed)

        self.preview.started.connect(lambda: self._busy('Building preview…', True))
        self.preview.progress.connect(self.statusBar().showMessage)
        self.preview.finished.connect(self._on_preview_ready)
        self.preview.failed.connect(self._on_preview_failed)
        self.preview.cancelled.connect(lambda: self._busy('Preview cancelled.', False))

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

    def _update_workspace_label(self) -> None:
        self.workspace_label.setText(str(self.workspace.root))

    def _refresh_settings_view(self) -> None:
        """Show the effective settings and where they came from.

        Read-only for now, and honest about it: load_settings replaces rather
        than merges, so editing must write a complete file or it would silently
        reset everything it omitted.
        """
        from ..core.config import settings_origin  # noqa: PLC0415

        try:
            origin = settings_origin()
        except Exception:  # pylint: disable=broad-except
            origin = None

        import yaml  # noqa: PLC0415

        header = f"# Settings from: {origin or 'built-in defaults'}\n\n"
        self.settings_view.setPlainText(
            header + yaml.safe_dump(self.settings.model_dump(), sort_keys=False))

    def _refresh_counts(self) -> None:
        try:
            summary = self.document.expansion()
            self.counts.setText(
                f"{summary.configs} config → {summary.builds} builds → "
                f"{summary.jobs} jobs")
        except Exception as exc:  # pylint: disable=broad-except
            self.counts.setText('—')
            logger.debug("count failed: %s", exc)

    def _validate(self) -> bool:
        """Validate and surface the result. Returns True when valid."""
        self.form.commit()
        result = self.document.validate(self.settings)
        self.form.show_errors(result.issues)

        unsupported = self.document.unsupported_lists()
        lines = [issue.render() for issue in result.issues]
        lines += [
            f"[{section}] {key}: a list here is not supported -- it would not "
            f"change the output directory, so the runs would overwrite each other"
            for section, key in unsupported
        ]

        self.issues.setPlainText('\n'.join(lines) if lines else 'Config is valid.')
        self._refresh_counts()
        return result.ok and not unsupported

    # --------------------------------------------------------------- events

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
            return
        self.form.rebuild()
        self._refresh_counts()

    def _on_config_changed(self) -> None:
        self._refresh_counts()

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

        self.form.document = self.document
        self.form.rebuild()
        self._validate()
        self.statusBar().showMessage(f"Opened {path}")

    def _save_config(self) -> None:
        self.form.commit()
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
        self._update_workspace_label()
        self.statusBar().showMessage(f"Workspace: {self.workspace.root}")

    # -------------------------------------------------------------- actions

    def _render(self) -> None:
        if not self._validate():
            self.statusBar().showMessage('Fix the reported problems first.')
            return

        request = PreviewRequest(
            config=self.document.to_dict(),
            model=self.document.model,
            layer=None,
            referenced_files=self._referenced_files(),
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
        self._busy(f"Rendered {system.count:,} atoms.", False)
        self.viewport.show_system(system)
        if system.warnings:
            self.issues.setPlainText('\n'.join(system.warnings))

    def _on_preview_failed(self, message: str) -> None:
        self._busy('Preview failed.', False)
        self.issues.setPlainText(message)
        QMessageBox.warning(self, 'Preview failed', message)

    def _generate(self) -> None:
        if not self._validate():
            self.statusBar().showMessage('Fix the reported problems first.')
            return

        summary = self.document.expansion()
        confirm = QMessageBox.question(
            self, 'Generate simulation package',
            f"This will generate {summary.configs} config(s), {summary.builds} "
            f"build(s) and {summary.jobs} job(s) into:\n{self.workspace.runs_dir}\n\n"
            f"Continue?",
        )
        if confirm != QMessageBox.Yes:
            return

        self.generator.start(GenerateRequest(
            config=self.document.to_dict(),
            model=self.document.model,
            generate_hpc=self.hpc_box.isChecked(),
        ))

    def _on_generated(self, result: dict) -> None:
        created = len(result.get('created', []))
        jobs = result.get('jobs')
        message = f"Generated {created} simulation(s)"
        if jobs is not None:
            message += f", {jobs} job script(s)"
        self._busy(message, False)

        detail = message + f"\n\n{result.get('simulation_root')}"
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
    app = QApplication.instance() or QApplication(list(argv) or sys.argv[:1])
    app.setApplicationName(APP_NAME)

    root = choose_workspace()
    if root is None:
        return 0

    workspace = Workspace.open(root)
    RecentWorkspaces().add(root)

    window = MainWindow(workspace)
    window.show()
    return app.exec()
