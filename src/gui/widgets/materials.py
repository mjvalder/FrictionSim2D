"""Choosing the materials a run sweeps over.

A sweep needs three things agreeing with each other: a `.txt` of names in
``[2D] materials_list``, and `{mat}` placeholders in `mat`, `cif_path` and
`pot_path` so expansion has somewhere to substitute. Setting them by hand is
why the feature was effectively undiscoverable -- and the placeholder half was
invisible in the GUI entirely.

Both halves come from one question instead: which structure directory, and
which potential directory. The material list is the intersection, which is what
``preflight_materials`` works out at build time anyway.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..controllers.materials import (
    infer_potential_extension,
    scan_material_directories,
)
from ..schema.introspect import FieldSpec
from ..style import CAPTION_SMALL
from .fields import FieldEditor, register_editor

__all__ = ['MaterialListEditor']


class MaterialListEditor(FieldEditor):
    """A sweep toggle, two directory pickers, and the resulting material list.

    ``value()`` is the ``materials_list`` path, because that is what the config
    holds. Writing the `.txt` and the `{mat}` templates spans other fields in
    the section, so it happens in ``SectionForm.commit``.
    """

    def __init__(self, spec: FieldSpec, parent: Optional[QWidget] = None) -> None:
        super().__init__(spec, parent)
        self._path: Optional[str] = None
        self._extension: Optional[str] = None
        #: Whether the user has ever changed a checkbox themselves (directly,
        #: or via Select all / None). Until then, a rescan defaults to
        #: selecting everything usable; after, an empty selection is a real
        #: choice and must survive a rescan rather than being read as "nothing
        #: has been picked yet" and silently reset to "everything".
        self._selection_touched = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self.sweep_box = QCheckBox('Sweep over many materials', self)
        self.sweep_box.toggled.connect(self._on_sweep_toggled)
        layout.addWidget(self.sweep_box)

        self.body = QWidget(self)
        body = QVBoxLayout(self.body)
        body.setContentsMargins(16, 0, 0, 0)
        body.setSpacing(3)

        self.structure_edit, structure_row = self._directory_row('Structures')
        self.potential_edit, potential_row = self._directory_row('Potentials')
        self.extension_label = QLabel('', self.body)
        self.extension_label.setStyleSheet(CAPTION_SMALL)
        potential_row.addWidget(self.extension_label)
        body.addLayout(structure_row)
        body.addLayout(potential_row)

        self.summary = QLabel('', self.body)
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet(CAPTION_SMALL)
        body.addWidget(self.summary)

        self.listing = QListWidget(self.body)
        self.listing.setMaximumHeight(160)
        self.listing.itemChanged.connect(self._on_item_changed)
        body.addWidget(self.listing)

        buttons = QHBoxLayout()
        for label, slot in (('Select all', self.select_all),
                            ('None', self.select_none)):
            button = QPushButton(label, self.body)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        buttons.addStretch(1)
        body.addLayout(buttons)

        layout.addWidget(self.body)
        self.body.setVisible(False)

    def _directory_row(self, label: str) -> Tuple[QLineEdit, QHBoxLayout]:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel(label, self.body))

        edit = QLineEdit(self.body)
        edit.editingFinished.connect(self._rescan)
        row.addWidget(edit, 1)

        browse = QPushButton('…', self.body)
        browse.setFixedWidth(30)
        browse.clicked.connect(lambda _=False, e=edit: self._browse(e))
        row.addWidget(browse)
        return edit, row

    def _browse(self, edit: QLineEdit) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, 'Select a directory', edit.text() or '')
        if chosen:
            edit.setText(chosen)
            self._rescan()

    # ----------------------------------------------------------------- state

    def is_sweeping(self) -> bool:
        return self.sweep_box.isChecked()

    def set_sweeping(self, sweeping: bool) -> None:
        self.sweep_box.setChecked(sweeping)

    def directories(self) -> Tuple[Optional[Path], Optional[Path], Optional[str]]:
        structures = self.structure_edit.text().strip()
        potentials = self.potential_edit.text().strip()
        return (Path(structures) if structures else None,
                Path(potentials) if potentials else None,
                self._extension)

    def set_directories(self, structure_dir, potential_dir,
                        extension: Optional[str] = None) -> None:
        self.structure_edit.setText(str(structure_dir or ''))
        self.potential_edit.setText(str(potential_dir or ''))
        self._extension = extension
        self.set_sweeping(True)
        self._rescan()

    def listed_names(self) -> List[str]:
        return [self.listing.item(i).data(Qt.UserRole)
                for i in range(self.listing.count())]

    def selected_names(self) -> List[str]:
        return [
            self.listing.item(i).data(Qt.UserRole)
            for i in range(self.listing.count())
            if self.listing.item(i).checkState() == Qt.Checked
        ]

    def set_selected(self, names: List[str]) -> None:
        wanted = set(names)
        for index in range(self.listing.count()):
            item = self.listing.item(index)
            if not (item.flags() & Qt.ItemIsUserCheckable):
                continue
            item.setCheckState(
                Qt.Checked if item.data(Qt.UserRole) in wanted else Qt.Unchecked)

    def select_all(self) -> None:
        self.set_selected(
            [c for c in self.listed_names()]
        )

    def select_none(self) -> None:
        self.set_selected([])

    def summary_text(self) -> str:
        return self.summary.text()

    # --------------------------------------------------------------- scanning

    def _rescan(self) -> None:
        """Re-read the directories and rebuild the list, keeping the selection."""
        structures, potentials, _ = self.directories()
        self._extension = (self._extension
                           or infer_potential_extension(potentials))
        keep = set(self.selected_names())
        # Only "nobody has decided yet" defaults to everything. Using an empty
        # `keep` for that would also fire the first time a user re-picks a
        # directory after explicitly clicking None: their empty selection is a
        # real choice, not an unset default, and must not silently become
        # "all" again.
        select_by_default = not self._selection_touched

        # A potential file with no matching structure cannot become a material
        # at all -- there is no geometry to place in the sheet -- so it is not
        # a sweep candidate the way a structure without a potential is (that
        # one is real and just needs a potential, which is worth surfacing;
        # see MaterialCandidate.reason). Filtering it out here, rather than in
        # the controller, keeps scan_material_directories reporting the full
        # picture for anything that inspects it directly.
        candidates = [c for c in scan_material_directories(
            structures, potentials, self._extension) if c.has_structure]

        self.listing.blockSignals(True)
        try:
            self.listing.clear()
            for candidate in candidates:
                text = (candidate.name if candidate.usable
                        else f"{candidate.name}   {candidate.reason}")
                item = QListWidgetItem(text, self.listing)
                item.setData(Qt.UserRole, candidate.name)
                if candidate.usable:
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    # First scan selects everything usable; a later one keeps
                    # whatever was ticked (even nothing), so re-picking a
                    # directory does not silently undo the user's selection.
                    item.setCheckState(
                        Qt.Checked if (select_by_default or candidate.name in keep)
                        else Qt.Unchecked)
                else:
                    item.setFlags(item.flags() & ~Qt.ItemIsUserCheckable
                                  & ~Qt.ItemIsEnabled)
                    item.setCheckState(Qt.Unchecked)
        finally:
            self.listing.blockSignals(False)

        usable = sum(1 for c in candidates if c.usable)
        self.extension_label.setText(self._extension or '')
        self.summary.setText(
            f"{usable} of {len(candidates)} usable → {len(self.selected_names())} "
            f"configs" if candidates else 'Choose both directories.')
        self.changed.emit()

    def _on_item_changed(self, _item) -> None:
        # Only reached from a real click: programmatic changes during _rescan
        # happen under blockSignals(). This is what distinguishes "the user
        # has never touched the selection" from "the user touched it and
        # ended up at empty".
        self._selection_touched = True
        self._rescan_summary()
        self.changed.emit()

    def _rescan_summary(self) -> None:
        total = self.listing.count()
        usable = sum(
            1 for i in range(total)
            if self.listing.item(i).flags() & Qt.ItemIsUserCheckable)
        self.summary.setText(
            f"{usable} of {total} usable → {len(self.selected_names())} configs")

    def _on_sweep_toggled(self, sweeping: bool) -> None:
        self.body.setVisible(sweeping)
        self.changed.emit()

    # ----------------------------------------------------------------- value

    def value(self):
        """The ``materials_list`` path, or None when not sweeping."""
        return self._path if self.is_sweeping() else None

    def set_value(self, value) -> None:
        self._path = value if isinstance(value, str) and value else None
        self.set_sweeping(bool(self._path))

    def note_written(self, path: Optional[Path]) -> None:
        """Record where SectionForm.commit put the list."""
        self._path = str(path) if path else None


register_editor('material_list', MaterialListEditor)
