"""The Plots tab: a dataset picker, a plot list, a schema-driven form, a PNG.

No matplotlib import anywhere in this module, by design (G4): the figures are
rendered by `src.gui.plot_worker` in a subprocess and arrive as PNG files.
Importing matplotlib here would put a second Qt backend in the process ovito
already owns.

Every `postprocessing` import is inside a method or `__init__` for the same
reason -- `plot_config` and `presets` are matplotlib-free, but `plot_data` is
not, and module-level imports here would be one refactor away from reaching it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QListWidget, QPushButton, QScrollArea,
                               QSplitter, QVBoxLayout, QWidget)

from ...controllers.plot_document import COMMON_SECTION, PlotEditorDocument
from ...controllers.plots import PlotController, PlotFigure, PlotRequest
from ...workspace import Workspace
from ..form import ConfigForm

logger = logging.getLogger(__name__)

__all__ = ['PlotsPane', 'discover_datasets', 'DATASETS_DIRNAME']

_NO_PRESET = '(none)'

#: Where `aiida export-results --output-dir ...` is expected to drop a set.
DATASETS_DIRNAME = '_datasets'


def discover_datasets(workspace: Workspace) -> List[Path]:
    """Every directory that looks like a plottable dataset.

    A dataset is a directory with an `outputs/` holding at least one
    `output_full_*.json` -- the same thing `Plotter` accepts as a `data_dir`.
    Two places are searched: the workspace root (where `run` writes its
    simulation packages) and `<workspace>/.preview/_datasets/`, which is where
    `FrictionSim2D aiida export-results --output-dir ...` puts a set pulled
    back out of the local store.
    """
    roots = [workspace.root, workspace.preview_dir / DATASETS_DIRNAME]
    found: List[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for candidate in sorted(root.iterdir()):
            if not candidate.is_dir():
                continue
            outputs = candidate / 'outputs'
            if outputs.is_dir() and any(outputs.glob('output_full_*.json')):
                if candidate not in found:
                    found.append(candidate)
    return found


class PlotsPane(QWidget):  # pylint: disable=too-many-instance-attributes
    """Edit a plot document and render it to PNG."""

    document_changed = Signal()

    def __init__(self, workspace: Workspace,
                 parent: Optional[QWidget] = None,
                 controller: Optional[PlotController] = None) -> None:
        super().__init__(parent)
        from ....postprocessing.plot_config import (PLOT_TYPES,  # noqa: PLC0415  # pylint: disable=import-outside-toplevel
                                                    PlotDocument)
        from ....postprocessing.presets import PRESETS  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        self.workspace = workspace
        self._document = PlotDocument()
        self._editors: List[PlotEditorDocument] = []
        self._available: List[Path] = []
        self._current = -1
        self._warnings: List[str] = []
        self._exporting = False
        self.current_path: Optional[Path] = None

        # ---- left column: datasets, plots, the form ---------------------
        self.available_list = QListWidget()
        self.add_dataset_button = QPushButton('Add dataset')
        self.add_dataset_button.clicked.connect(self.add_selected_dataset)
        self.dataset_list = QListWidget()
        self.remove_dataset_button = QPushButton('Remove dataset')
        self.remove_dataset_button.clicked.connect(
            self.remove_selected_dataset)

        self.plot_list = QListWidget()
        self.plot_list.currentRowChanged.connect(self._on_plot_row_changed)
        self.type_combo = QComboBox()
        self.type_combo.addItems(list(PLOT_TYPES))
        self.type_combo.currentTextChanged.connect(self._on_type_changed)
        self.add_button = QPushButton('Add plot')
        self.add_button.clicked.connect(
            lambda: self.add_plot(self.type_combo.currentText()))
        self.remove_button = QPushButton('Remove plot')
        self.remove_button.clicked.connect(self.remove_selected_plot)

        self.form = ConfigForm(PlotEditorDocument.blank('summary'), self)
        self.form.changed.connect(self._commit_current_plot)
        self.form.refused.connect(self.status_label_set)

        left = QWidget(self)
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel('Available datasets'))
        left_layout.addWidget(self.available_list, 1)
        left_layout.addWidget(self.add_dataset_button)
        left_layout.addWidget(QLabel('In this document'))
        left_layout.addWidget(self.dataset_list, 1)
        left_layout.addWidget(self.remove_dataset_button)
        left_layout.addWidget(QLabel('Plots'))
        left_layout.addWidget(self.plot_list, 1)
        plot_buttons = QHBoxLayout()
        plot_buttons.addWidget(self.type_combo, 1)
        plot_buttons.addWidget(self.add_button)
        plot_buttons.addWidget(self.remove_button)
        left_layout.addLayout(plot_buttons)
        left_layout.addWidget(self.form, 2)

        # ---- right column: the PNG, the preset, the actions -------------
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.preview_label)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems([_NO_PRESET, *PRESETS])
        self.render_button = QPushButton('Render')
        self.render_button.clicked.connect(self.render)
        self.export_button = QPushButton('Export')
        self.export_button.clicked.connect(self.export)
        self.open_button = QPushButton('Open...')
        self.open_button.clicked.connect(self.open_document)
        self.save_button = QPushButton('Save')
        self.save_button.clicked.connect(self.save)
        self.save_as_button = QPushButton('Save as...')
        self.save_as_button.clicked.connect(self.save_as)
        self.status_label = QLabel('')
        self.status_label.setWordWrap(True)
        self.document_changed.connect(self._on_document_changed)

        right = QWidget(self)
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(scroll, 1)
        files = QHBoxLayout()
        files.addWidget(self.open_button)
        files.addWidget(self.save_button)
        files.addWidget(self.save_as_button)
        files.addStretch(1)
        right_layout.addLayout(files)

        actions = QHBoxLayout()
        actions.addWidget(QLabel('Preset'))
        actions.addWidget(self.preset_combo, 1)
        actions.addWidget(self.render_button)
        actions.addWidget(self.export_button)
        right_layout.addLayout(actions)
        right_layout.addWidget(self.status_label)

        splitter = QSplitter(Qt.Horizontal, self)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 5)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter, 1)

        # Wired after the widgets exist: `_on_started` touches the buttons.
        self.controller = controller or PlotController(workspace, self)
        self.controller.started.connect(self._on_started)
        self.controller.progress.connect(self.status_label_set)
        self.controller.warned.connect(self._on_warned)
        self.controller.finished.connect(self._on_finished)
        self.controller.failed.connect(self._on_failed)
        self.controller.cancelled.connect(self._on_cancelled)

        self.refresh_available_datasets()
        self.set_document(self._document)

    # ------------------------------------------------------------- document

    def status_label_set(self, message: str) -> None:
        self.status_label.setText(message)

    def document(self):
        """The current document, with every editor's values committed."""
        from ....postprocessing.plot_config import PlotDocument  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        self._commit_current_plot()
        return PlotDocument(
            data_dirs=list(self._document.data_dirs),
            labels=list(self._document.labels),
            dataset_display_labels=dict(self._document.dataset_display_labels),
            series_color_map=dict(self._document.series_color_map),
            plots=[editor.to_plot() for editor in self._editors])

    def set_document(self, document) -> None:
        self._document = document
        self._editors = [PlotEditorDocument.from_plot(p)
                         for p in document.plots]
        self._current = -1
        self.dataset_list.clear()
        self.dataset_list.addItems(list(document.labels))
        self.refresh_available_datasets()
        self._refresh_plot_list()
        if self._editors:
            self.plot_list.setCurrentRow(0)
        self.document_changed.emit()

    def load_document(self, path: Path) -> None:
        from ....postprocessing.plot_config import load_plot_document  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        self.set_document(load_plot_document(Path(path)))
        self.current_path = Path(path)
        self.status_label.setText(f'Opened {Path(path).name}')

    def open_document(self) -> None:
        """Open a plot document -- the same JSON `postprocess plot` reads."""
        start = str(self.current_path or self.workspace.root)
        path, _filter = QFileDialog.getOpenFileName(
            self, 'Open plot document', start, 'Plot documents (*.json)')
        if not path:
            return
        try:
            self.load_document(Path(path))
        except Exception as exc:  # pylint: disable=broad-except
            # Reported in the status label, never in a modal: an unpatched
            # QMessageBox in an offscreen test blocks forever.
            self.status_label.setText(
                f'Could not open {Path(path).name}: {exc}')

    def save(self) -> None:
        """Save to the path this document came from, or ask for one."""
        if self.current_path is None:
            self.save_as()
            return
        self.save_document(self.current_path)
        self.status_label.setText(f'Saved {self.current_path.name}')

    def save_as(self) -> None:
        start = str(self.current_path or (self.workspace.root / 'plots.json'))
        path, _filter = QFileDialog.getSaveFileName(
            self, 'Save plot document', start, 'Plot documents (*.json)')
        if not path:
            return
        self.current_path = Path(path)
        self.save_document(self.current_path)
        self.status_label.setText(f'Saved {self.current_path.name}')

    def save_document(self, path: Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            self.document().model_dump_json(indent=2, exclude_unset=True),
            encoding='utf-8')
        return target

    def _on_document_changed(self) -> None:
        """`document_changed` had no listener at all before this."""
        name = self.current_path.name if self.current_path else 'unsaved'
        self.status_label.setText(
            f'{name}: {len(self._editors)} plot(s), '
            f'{len(self._document.data_dirs)} dataset(s)')

    # ------------------------------------------------------------- datasets

    def refresh_available_datasets(self) -> None:
        """Re-scan the workspace for plottable dataset directories."""
        self._available = discover_datasets(self.workspace)
        self.available_list.clear()
        self.available_list.addItems([d.name for d in self._available])

    def add_selected_dataset(self) -> None:
        row = self.available_list.currentRow()
        if not 0 <= row < len(self._available):
            return
        directory = self._available[row]
        if str(directory) in self._document.data_dirs:
            self.status_label.setText(
                f'{directory.name} is already in the document.')
            return
        self._document.data_dirs.append(str(directory))
        self._document.labels.append(directory.name)
        self.dataset_list.addItem(directory.name)
        self.document_changed.emit()

    def remove_selected_dataset(self) -> None:
        row = self.dataset_list.currentRow()
        if not 0 <= row < len(self._document.data_dirs):
            return
        self._document.data_dirs.pop(row)
        self._document.labels.pop(row)
        self.dataset_list.takeItem(row)
        self.document_changed.emit()

    # ---------------------------------------------------------------- plots

    def _on_plot_row_changed(self, row: int) -> None:
        self._commit_current_plot()
        self._select_plot(row)

    def _on_type_changed(self, plot_type: str) -> None:
        if not 0 <= self._current < len(self._editors):
            return
        self._editors[self._current].set_plot_type(plot_type)
        self.form.document = self._editors[self._current]
        self.form.rebuild()
        self._commit_current_plot()
        self.document_changed.emit()

    def add_plot(self, plot_type: str = 'summary') -> None:
        self._commit_current_plot()
        self._editors.append(PlotEditorDocument.blank(plot_type))
        self._refresh_plot_list()
        self.plot_list.setCurrentRow(len(self._editors) - 1)
        self.document_changed.emit()

    def remove_selected_plot(self) -> None:
        row = self.plot_list.currentRow()
        if not 0 <= row < len(self._editors):
            return
        self._editors.pop(row)
        self._current = -1
        self._refresh_plot_list()
        if self._editors:
            self.plot_list.setCurrentRow(min(row, len(self._editors) - 1))
        self.document_changed.emit()

    def _refresh_plot_list(self) -> None:
        self.plot_list.blockSignals(True)
        self.plot_list.clear()
        for index, editor in enumerate(self._editors):
            filename = editor.get(COMMON_SECTION, 'filename') or '(unnamed)'
            self.plot_list.addItem(
                f'{index + 1}. {editor.plot_type} - {filename}')
        self.plot_list.blockSignals(False)

    def _select_plot(self, row: int) -> None:
        if not 0 <= row < len(self._editors):
            return
        self._current = row
        editor = self._editors[row]
        self.type_combo.blockSignals(True)
        self.type_combo.setCurrentText(editor.plot_type)
        self.type_combo.blockSignals(False)
        self.form.document = editor
        self.form.rebuild()

    def _commit_current_plot(self) -> None:
        """Push the form's edits into the list row's label.

        The editor IS the form's document, so the values are already there;
        this only refreshes the row text and reports a validation failure
        in the status label -- never in a modal, which would block a headless
        test forever with no one to click it.
        """
        if not 0 <= self._current < len(self._editors):
            return
        editor = self._editors[self._current]
        result = editor.validate()
        if result.issues:
            self.status_label.setText(
                '; '.join(f'{i.key}: {i.message}' for i in result.issues))
        filename = editor.get(COMMON_SECTION, 'filename') or '(unnamed)'
        item = self.plot_list.item(self._current)
        if item is not None:
            item.setText(
                f'{self._current + 1}. {editor.plot_type} - {filename}')

    # --------------------------------------------------------------- render

    def _preset(self) -> Optional[str]:
        preset = self.preset_combo.currentText()
        return None if preset == _NO_PRESET else preset

    def render(self) -> None:
        """Render a PNG PREVIEW.

        The format override is deliberate and applies to the preview only:
        `preview_label` shows a QPixmap, and a preset whose `export.formats`
        is `["pdf", "svg", "png"]` would otherwise write two files this pane
        cannot display for every one it can. `export()` is the action that
        honours the preset's own formats.
        """
        self._exporting = False
        row = self.plot_list.currentRow()
        # ONE plot, not all of them: a preview shows `figures[0]`, and a
        # ten-plot document rendered in full costs ten figures to display one.
        indices = [row] if 0 <= row < len(self._editors) else None
        self.controller.start(PlotRequest(
            document=self.document(), preset=self._preset(),
            settings={'export': {'formats': ['png']}},
            plot_indices=indices))

    def export(self) -> None:
        """Render with the chosen preset's OWN export formats.

        `settings=None`, so nothing overrides `export.formats`: the
        `publication` preset writes pdf, svg and png, which is the point of
        choosing it. The files land under the render's scratch directory and
        their paths are reported in the status label.
        """
        self._exporting = True
        self.controller.start(PlotRequest(
            document=self.document(), preset=self._preset(), settings=None))

    def show_figure(self, figure: PlotFigure) -> None:
        pngs = [f for f in figure.files if f.suffix.lower() == '.png']
        if not pngs:
            self.status_label.setText(
                f'{figure.plot_type}: no PNG to preview')
            return
        self.preview_label.setPixmap(QPixmap(str(pngs[0])))
        self.status_label.setText(f'{figure.plot_type}: {pngs[0].name}')

    # ----------------------------------------------------------- controller

    def _on_started(self) -> None:
        self.render_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.status_label.setText(
            'Exporting...' if self._exporting else 'Rendering...')
        self._warnings = []

    def _on_warned(self, warnings: list) -> None:
        self._warnings = list(warnings)

    def _on_finished(self, figures: list) -> None:
        self.render_button.setEnabled(True)
        self.export_button.setEnabled(True)
        if not figures:
            self.status_label.setText('Nothing was rendered.')
        elif self._exporting:
            files = [f for figure in figures for f in figure.files]
            suffixes = sorted({f.suffix.lstrip('.') for f in files})
            self.status_label.setText(
                f'Exported {len(files)} file(s): ' + ', '.join(suffixes)
                + f' -> {files[0].parent}')
        else:
            self.show_figure(figures[0])
        if self._warnings:
            self.status_label.setText(
                f'{self.status_label.text()} - '
                f'{len(self._warnings)} warning(s): {self._warnings[0]}')

    def _on_failed(self, message: str) -> None:
        self.render_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.status_label.setText(f'Render failed: {message}')

    def _on_cancelled(self) -> None:
        self.render_button.setEnabled(True)
        self.export_button.setEnabled(True)
        self.status_label.setText('Render cancelled.')
