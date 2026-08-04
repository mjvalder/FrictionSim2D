"""The interactive 3D view.

Wraps ovito's own viewport widget, which is a real ``QOpenGLWidget`` and brings
rotation, panning and zooming with it -- the reason this application is a
desktop one.

Adds what makes the view diagnostic rather than decorative: colouring by
element or by component, per-component visibility (sheets are what most often
generate wrongly, and you cannot see one through a substrate), and a warning
when a component meets a periodic boundary.
"""

from __future__ import annotations

import colorsys
import logging
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .assemble import AssembledSystem, build_data_collection
from ..style import CAPTION, WARNING

logger = logging.getLogger(__name__)

__all__ = ['ViewportPanel', 'COLOUR_MODES', 'distinct_colour']

#: Sentinel for "no cell has ever been framed yet", distinct from the
#: legitimate ``None`` that :meth:`ViewportPanel._frame_and_refresh` records
#: when the *framed* system itself has no cell (e.g. a multi-placement build
#: with no explicit box, see ``Assembly.effective_cell()``). Plain ``None``
#: used to serve both meanings, which made a cell-less system re-frame on
#: every single update -- see :meth:`ViewportPanel._needs_framing`.
_UNFRAMED = object()

COLOUR_BY_ELEMENT = 'Element'      # was 'Atom type', which coloured by element
COLOUR_BY_TYPE = 'LAMMPS type'
COLOUR_BY_COMPONENT = 'Component'
COLOUR_MODES = (COLOUR_BY_ELEMENT, COLOUR_BY_TYPE, COLOUR_BY_COMPONENT)


def distinct_colour(index: int):
    """Golden-angle hue walk: any count of types stays distinguishable."""
    return colorsys.hsv_to_rgb((index * 0.61803398875) % 1.0, 0.65, 0.95)

#: Distinguishable colours for components, in assignment order.
_COMPONENT_COLOURS = [
    (0.90, 0.35, 0.30),
    (0.30, 0.55, 0.85),
    (0.45, 0.75, 0.40),
    (0.90, 0.70, 0.25),
    (0.65, 0.45, 0.80),
    (0.35, 0.75, 0.75),
]

#: Colours for common elements; anything else falls back to the palette above.
_ELEMENT_COLOURS = {
    'Si': (0.55, 0.55, 0.60),
    'Mo': (0.35, 0.75, 0.70),
    'S': (0.95, 0.85, 0.30),
    'W': (0.40, 0.50, 0.75),
    'Se': (0.85, 0.55, 0.25),
    'Te': (0.75, 0.45, 0.60),
    'C': (0.30, 0.30, 0.30),
}


class ViewportPanel(QWidget):
    """An ovito viewport plus the controls that make it useful."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.system: Optional[AssembledSystem] = None
        self._pipeline = None
        self._viewport = None
        self._widget: Optional[QWidget] = None
        self._visibility: Dict[str, bool] = {}
        self._framed = False
        self._last_cell: object = _UNFRAMED

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._view_container = QVBoxLayout()
        layout.addLayout(self._view_container, 1)

        self.placeholder = QLabel(
            'Press Render to build and display the configured system.', self)
        self.placeholder.setAlignment(
            self.placeholder.alignment().__class__.AlignCenter)
        self.placeholder.setStyleSheet(CAPTION + ' padding: 40px;')
        self._view_container.addWidget(self.placeholder)

        self.stale_label = QLabel('Config changed since last render', self)
        self.stale_label.setStyleSheet(WARNING)
        self.stale_label.hide()
        layout.addWidget(self.stale_label)

        layout.addLayout(self._build_controls())

        self.legend = QLabel('', self)
        self.legend.setWordWrap(True)
        self.legend.setStyleSheet('color: palette(text); padding: 2px;')
        layout.addWidget(self.legend)

        self.warning = QLabel('', self)
        self.warning.setWordWrap(True)
        self.warning.setStyleSheet(WARNING)
        self.warning.hide()
        layout.addWidget(self.warning)

    # -------------------------------------------------------------- controls

    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()

        row.addWidget(QLabel('Colour:', self))
        self.colour_mode = QComboBox(self)
        self.colour_mode.addItems(COLOUR_MODES)
        self.colour_mode.currentTextChanged.connect(lambda _: self._apply_colours())
        row.addWidget(self.colour_mode)

        self.show_box = QCheckBox('Box', self)
        self.show_box.setChecked(True)
        self.show_box.setToolTip('Show the simulation cell')
        self.show_box.toggled.connect(self._apply_cell_visibility)
        row.addWidget(self.show_box)

        row.addSpacing(12)
        row.addWidget(QLabel('Show:', self))
        self._component_row = QHBoxLayout()
        row.addLayout(self._component_row)

        row.addStretch(1)

        for label, direction in (('Top', 'top'), ('Front', 'front'),
                                 ('Side', 'side'), ('Perspective', 'perspective')):
            button = QPushButton(label, self)
            button.setToolTip(f'{label} view')
            button.clicked.connect(lambda _=False, d=direction: self.set_camera(d))
            row.addWidget(button)

        return row

    def _rebuild_component_toggles(self) -> None:
        while self._component_row.count():
            item = self._component_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self.system:
            return

        for component in self.system.components:
            box = QCheckBox(component.name, self)
            box.setChecked(self._visibility.get(component.name, True))
            box.setToolTip(f'{component.count} atoms')
            box.toggled.connect(
                lambda checked, name=component.name: self._set_visible(name, checked))
            self._component_row.addWidget(box)

    # ----------------------------------------------------------------- state

    def show_system(self, system: AssembledSystem) -> None:
        """Display an assembled system.

        Component names that are unchanged from the previous system keep
        their visibility ticks and toggle widgets -- only a structurally
        different system (components added, removed or renamed) rebuilds
        the toggle row.
        """
        same_names = (self.system is not None and
                      [c.name for c in self.system.components] ==
                      [c.name for c in system.components])
        self.system = system
        self._visibility = {c.name: self._visibility.get(c.name, True)
                            for c in system.components}
        if not same_names:
            self._rebuild_component_toggles()
        self._update_pipeline()
        self._update_legend()
        self._update_warnings()

    def clear(self) -> None:
        self.system = None
        self._framed = False
        self._last_cell = _UNFRAMED
        self._teardown_view()
        self.placeholder.show()
        self.legend.setText('')
        self.warning.hide()
        self.stale_label.hide()

    def set_stale(self, stale: bool) -> None:
        """Show or hide the "config changed since last render" caption.

        Stays hidden while no system has been shown yet -- there is nothing
        stale relative to before the first render.
        """
        self.stale_label.setVisible(stale and self.system is not None)

    # ------------------------------------------------------------- rendering

    def _visible_mask(self) -> np.ndarray:
        assert self.system is not None
        mask = np.ones(self.system.count, dtype=bool)
        for component in self.system.components:
            if not self._visibility.get(component.name, True):
                mask[component.first:component.last] = False
        return mask

    def _update_pipeline(self) -> None:
        """Update the displayed data from the current visibility selection.

        Updates the existing pipeline's source data in place rather than
        tearing down and rebuilding the scene, so the camera the user set up
        survives a re-render. Only the first render, a :meth:`clear`, or a
        cell whose size changed materially triggers a re-frame -- see
        :meth:`_needs_framing`. If the in-place update fails for any reason
        the except clause below falls back to a full teardown, which is
        today's (pre-this-change) behaviour.

        Hidden components are dropped from the data rather than made
        transparent: it is simpler, and it keeps the atom count -- and so the
        frame rate -- proportional to what is actually on screen.
        """
        if self.system is None:
            return

        mask = self._visible_mask()
        subset = AssembledSystem(
            positions=self.system.positions[mask],
            type_ids=self.system.type_ids[mask],
            group_ids=self.system.group_ids[mask],
            cell=self.system.cell if self.show_box.isChecked() else None,
            components=self.system.components,
            legend=self.system.legend,
        )

        try:
            data = build_data_collection(subset, colours=self._colour_map())
            if self._pipeline is None:
                from ovito.pipeline import Pipeline, StaticSource  # noqa: PLC0415

                self._teardown_view()
                self._pipeline = Pipeline(source=StaticSource(data=data))
                self._ensure_widget()
                self._pipeline.add_to_scene()
                # Framing has to happen once the scene has contents; zooming
                # to an empty scene leaves the camera pointing at nothing.
                self._frame_and_refresh()
            else:
                self._pipeline.source.data = data
                if self._needs_framing():
                    self._frame_and_refresh()
                else:
                    self._refresh()
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Failed to display the assembled system")
            self._teardown_view()
            self.placeholder.setText(f"Could not display the system: {exc}")
            self.placeholder.show()

    def _ensure_widget(self) -> None:
        """Create the ovito viewport widget once."""
        if self._widget is not None:
            return

        import ovito.vis  # noqa: PLC0415

        # Perspective, not the default top-down: looking straight down puts the
        # substrate in front of everything, hiding the tip and sheet that are
        # the reason for looking at all.
        self._viewport = ovito.vis.Viewport(type=ovito.vis.Viewport.Type.Perspective)
        self._viewport.camera_dir = (-2.0, 1.0, -0.6)
        self._widget = self._viewport.create_qt_widget(parent=self)
        self._widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.placeholder.hide()
        self._view_container.addWidget(self._widget, 1)

    def _teardown_view(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.remove_from_scene()
            except Exception:  # pylint: disable=broad-except
                pass
            self._pipeline = None

    def _colour_map(self) -> Dict[int, tuple]:
        """Colour per atom type, for the selected mode.

        Computed before the pipeline is built: ovito's element types have to
        carry their colours from the start, because assigning them to a
        computed data collection writes to a copy the pipeline throws away.
        """
        if self.system is None:
            return {}

        mode = self.colour_mode.currentText()
        colours: Dict[int, tuple] = {}

        for index, type_id in enumerate(sorted(self.system.legend)):
            if mode == COLOUR_BY_COMPONENT:
                group_index = self._group_index_of(type_id)
                colours[type_id] = _COMPONENT_COLOURS[group_index % len(_COMPONENT_COLOURS)]
            elif mode == COLOUR_BY_TYPE:
                colours[type_id] = distinct_colour(index)
            else:
                element = self.system.legend.get(type_id, '')
                colours[type_id] = _ELEMENT_COLOURS.get(
                    element,
                    _COMPONENT_COLOURS[type_id % len(_COMPONENT_COLOURS)],
                )
        return colours

    def _apply_colours(self) -> None:
        """Recolour by updating the pipeline, since colours live on the type
        definitions -- the camera is untouched. Also refreshes the legend so
        its swatches follow the newly selected mode."""
        self._update_pipeline()
        self._update_legend()

    def _group_index_of(self, type_id: int) -> int:
        assert self.system is not None
        for index, component in enumerate(self.system.components):
            if type_id in component.type_ids:
                return index
        return 0

    def _apply_cell_visibility(self) -> None:
        self._update_pipeline()

    def _set_visible(self, name: str, visible: bool) -> None:
        self._visibility[name] = visible
        self._update_pipeline()

    def _fit(self) -> None:
        """Frame the whole scene."""
        if self._viewport is None:
            return
        try:
            self._viewport.zoom_all()
        except Exception:  # pylint: disable=broad-except
            logger.debug("zoom_all failed", exc_info=True)

    def _frame_and_refresh(self) -> None:
        """Frame the whole scene, remember that it has been framed (and at
        what cell size), and repaint -- the path taken on the first render
        and whenever the cell changed enough that the old camera no longer
        makes sense.

        ``_last_cell`` is anchored here rather than in every
        :meth:`_update_pipeline` call, so :meth:`_needs_framing`'s drift
        check measures against the last *framed* cell, not merely the
        immediately preceding one -- otherwise a cell drifting a few percent
        per render would never accumulate past the threshold.
        """
        self._fit()
        self._framed = True
        self._last_cell = self.system.cell if self.system is not None else None
        self._refresh()

    def _needs_framing(self) -> bool:
        """Whether the camera should be (re-)framed on this update.

        True on the first render and after :meth:`clear` (``_last_cell`` is
        still the ``_UNFRAMED`` sentinel -- nothing has been framed yet).
        Once framed, ``_last_cell`` may legitimately be ``None`` itself: that
        is what :meth:`_frame_and_refresh` records when the framed system's
        own cell is ``None`` (a multi-placement build with no explicit box).
        A cell-less system followed by another cell-less system is therefore
        a steady state, not "unknown", and must not force a re-frame -- the
        bug this sentinel exists to prevent reinstates itself the moment
        ``None`` is read as ambiguous rather than as a real, previously-framed
        value. Any other change of cell-ness (cell to no cell, or no cell to
        cell) is material and does reframe, as does the new cell's max extent
        differing from the one last framed by more than 10%. Anything smaller
        than that (recolouring, toggling a component, toggling the box
        outline) leaves the camera exactly where the user put it.
        """
        if not self._framed:
            return True
        if self._last_cell is _UNFRAMED:
            return True
        current_cell = self.system.cell if self.system is not None else None
        if current_cell is None or self._last_cell is None:
            return current_cell != self._last_cell
        previous = self._max_extent(self._last_cell)
        current = self._max_extent(current_cell)
        if previous == 0:
            return current != 0
        return abs(current - previous) / previous > 0.10

    @staticmethod
    def _max_extent(cell) -> float:
        xlo, xhi, ylo, yhi, zlo, zhi = cell
        return max(xhi - xlo, yhi - ylo, zhi - zlo)

    def set_camera(self, direction: str) -> None:
        """Point the camera along a standard axis and fit the scene."""
        if self._viewport is None:
            return

        import ovito.vis  # noqa: PLC0415

        types = {
            'top': ovito.vis.Viewport.Type.Top,
            'front': ovito.vis.Viewport.Type.Front,
            'side': ovito.vis.Viewport.Type.Left,
            'perspective': ovito.vis.Viewport.Type.Perspective,
        }
        self._viewport.type = types.get(
            direction, ovito.vis.Viewport.Type.Perspective)
        if direction == 'perspective':
            self._viewport.camera_dir = (-2.0, 1.0, -0.6)
        self._fit()
        self._refresh()

    def _refresh(self) -> None:
        if self._widget is not None:
            self._widget.update()

    # ------------------------------------------------------------- reporting

    def _update_legend(self) -> None:
        if not self.system:
            self.legend.setText('')
            return

        counts = {}
        for type_id in self.system.type_ids:
            counts[int(type_id)] = counts.get(int(type_id), 0) + 1

        colours = self._colour_map()
        self.legend.setTextFormat(Qt.RichText)
        parts = []
        for type_id, element in sorted(self.system.legend.items()):
            r, g, b = colours.get(type_id, (0.0, 0.0, 0.0))
            swatch = (f'<span style="color:#{int(r * 255):02x}'
                      f'{int(g * 255):02x}{int(b * 255):02x}">■</span>')
            parts.append(f'{swatch}&nbsp;{type_id}:{element}&nbsp;'
                         f'({counts.get(type_id, 0)})')
        total = self.system.count
        self.legend.setText(f"{total:,} atoms  —  " + '  '.join(parts))

    def _update_warnings(self) -> None:
        if not self.system:
            self.warning.hide()
            return

        messages: List[str] = list(self.system.warnings)

        touching = [name for name, hits in self.system.touches_boundary().items() if hits]
        # The sheet and substrate are meant to span the cell; a tip or flake
        # reaching the wall would interact with its own periodic image.
        unexpected = [n for n in touching if n not in ('sub', '2D')]
        if unexpected:
            messages.append(
                f"{', '.join(unexpected)} reaches the periodic boundary and will "
                f"interact with its own image; enlarge the sheet footprint."
            )

        if messages:
            self.warning.setText(' • '.join(messages))
            self.warning.show()
        else:
            self.warning.hide()
