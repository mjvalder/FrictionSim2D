"""The Langevin thermostat band table.

``langevin_boundaries`` is ``{component: {band: [from, to]}}`` and rendered as
one line of JSON, which is unreadable -- but the real problem is that the two
halves do not mean the same thing:

* **tip** bands are absolute Angstrom measured *down from the top of the tip*
  (``f_zlo = h - fix[0]``);
* **substrate** bands are a *fraction of the substrate thickness* measured *up
  from the bottom* (``f_zlo = fix[0] * thickness``).

See :func:`src.builders.components.apply_langevin_regions`. Nothing stated this
anywhere, so the table states it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from ..style import CAPTION_SMALL, SECTION_HELP
from ..schema.introspect import FieldSpec
from .fields import FieldEditor, register_editor

__all__ = ['LangevinBoundaryEditor']

#: Components the table displays, with the unit their numbers are in.
_BLOCKS = (
    ('tip', 'Tip', "depth below the tip's top face, in Å", 'Å'),
    ('sub', 'Substrate', 'fraction of thickness, from the bottom', '×t'),
)

#: Bands within each component, in the order they are shown.
_BANDS = (('fix', 'Fixed'), ('thermo', 'Thermostat'))


class LangevinBoundaryEditor(FieldEditor):
    """A table of ``component -> band -> [from, to]``."""

    def __init__(self, spec: FieldSpec, parent: Optional[QWidget] = None) -> None:
        super().__init__(spec, parent)
        #: Everything the widget was given but does not display, re-emitted
        #: untouched. Narrowing a settings value to what one widget happens to
        #: render is how settings get silently lost.
        self._passthrough: Dict[str, Any] = {}
        self._cells: Dict[tuple, QLineEdit] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        intro = QLabel('Bands split each component into fixed / thermostatted '
                       '/ free.', self)
        intro.setWordWrap(True)
        intro.setStyleSheet(SECTION_HELP)
        layout.addWidget(intro)

        grid = QGridLayout()
        row = 0
        for component, title, unit_note, unit in _BLOCKS:
            heading = QLabel(f"{title} — {unit_note}", self)
            heading.setStyleSheet(CAPTION_SMALL)
            grid.addWidget(heading, row, 0, 1, 6)
            row += 1

            for band, band_label in _BANDS:
                grid.addWidget(QLabel(band_label, self), row, 0)
                grid.addWidget(QLabel('from', self), row, 1)
                grid.addWidget(self._cell(component, band, 0), row, 2)
                grid.addWidget(QLabel('to', self), row, 3)
                grid.addWidget(self._cell(component, band, 1), row, 4)
                grid.addWidget(QLabel(unit, self), row, 5)
                row += 1

        layout.addLayout(grid)

    def _cell(self, component: str, band: str, index: int) -> QLineEdit:
        edit = QLineEdit(self)
        edit.setMaximumWidth(70)
        edit.textEdited.connect(self.changed)
        self._cells[(component, band, index)] = edit
        return edit

    def describe(self) -> str:
        """The units, as shown. Used by tests and by the tooltip."""
        return '; '.join(f"{title}: {note}" for _, title, note, _ in _BLOCKS)

    def set_cell(self, component: str, band: str, index: int, value: Any) -> None:
        """Set one number, leaving every other cell alone."""
        cell = self._cells.get((component, band, index))
        if cell is not None:
            cell.setText(str(value))

    def value(self) -> Any:
        result: Dict[str, Any] = {
            key: val for key, val in self._passthrough.items()
        }
        for component, _title, _note, _unit in _BLOCKS:
            bands: Dict[str, List[Any]] = {}
            for band, _label in _BANDS:
                numbers = []
                for index in (0, 1):
                    text = self._cells[(component, band, index)].text().strip()
                    if not text:
                        continue
                    try:
                        numbers.append(float(text))
                    except ValueError:
                        # Hand it on as typed: validation reports it against
                        # the model rather than this widget inventing an error.
                        numbers.append(text)
                if numbers:
                    bands[band] = numbers
            if bands:
                result[component] = bands
        # `langevin_boundaries` is not Optional on the model -- it is
        # `Dict[str, Dict[str, List[float]]]` with a `default_factory`, not
        # `Optional[...]`. SettingsDocument always loads a complete model
        # dump, so the key is always already present in the document, and
        # SectionForm.commit() writes whatever value() returns straight into
        # it unconditionally (the "key not in existing" skip never applies).
        # Returning None here for an emptied table -- as the free-form
        # dict/list editors below do -- would therefore commit `None` into a
        # non-Optional dict field and fail model validation on save. An empty
        # dict is a type-valid value for this field, so return that instead.
        return result

    def set_value(self, value: Any) -> None:
        value = value or {}
        self._passthrough = {
            key: val for key, val in value.items()
            if key not in {component for component, *_ in _BLOCKS}
        }
        for component, _title, _note, _unit in _BLOCKS:
            bands = value.get(component) or {}
            for band, _label in _BANDS:
                numbers = list(bands.get(band) or [])
                for index in (0, 1):
                    text = '' if index >= len(numbers) else str(numbers[index])
                    self._cells[(component, band, index)].setText(text)


register_editor('langevin_bands', LangevinBoundaryEditor)
