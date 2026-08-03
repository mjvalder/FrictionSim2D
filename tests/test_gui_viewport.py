"""The viewport must update in place, not tear down and rebuild.

Rebuilding on every Render lost the camera and any component visibility ticks
the user had set. These tests exercise the state logic in isolation -- the
pipeline layer (``_update_pipeline``, ``_ensure_widget``) is monkeypatched to a
no-op, so no GL widget is ever created.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication  # noqa: E402

from src.gui.preview.assemble import AssembledSystem, ComponentSlice  # noqa: E402


@pytest.fixture(scope='module', name='qt_app')
def _qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def two_component_system():
    return AssembledSystem(
        positions=np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]]),
        type_ids=np.array([1, 1, 2, 2]),
        group_ids=np.array([0, 0, 1, 1]),
        cell=(0., 10., 0., 10., 0., 10.),
        components=[ComponentSlice('tip', 0, 0, 2, [1]),
                    ComponentSlice('sub', 1, 2, 2, [2])],
        legend={1: 'Si', 2: 'Mo'},
    )


@pytest.fixture
def panel(qt_app, monkeypatch):
    from src.gui.preview import viewport as vp
    monkeypatch.setattr(vp.ViewportPanel, '_update_pipeline', lambda self: None)
    monkeypatch.setattr(vp.ViewportPanel, '_ensure_widget', lambda self: None)
    return vp.ViewportPanel()


def test_visibility_survives_rerender(panel, two_component_system):
    panel.show_system(two_component_system)
    panel._visibility['tip'] = False
    panel.show_system(two_component_system)
    assert panel._visibility['tip'] is False


def test_framing_only_needed_once(panel, two_component_system):
    assert panel._needs_framing()
    panel.show_system(two_component_system)
    # What _frame_and_refresh sets together in the real path: _last_cell is
    # anchored there, alongside _framed, so the two never disagree about
    # whether -- and against what -- the view was last framed.
    panel._framed = True
    panel._last_cell = two_component_system.cell
    panel.show_system(two_component_system)
    assert not panel._needs_framing()


def test_unknown_last_cell_owes_a_reframe_even_if_marked_framed(panel, two_component_system):
    """`_framed` and `_last_cell` are meant to be set together by
    `_frame_and_refresh`; if they ever disagree (`_framed` True but no
    `_last_cell` on record), that is an unknown state and must not be
    silently trusted as "already correctly framed"."""
    panel.show_system(two_component_system)
    panel._framed = True
    panel._last_cell = None
    assert panel._needs_framing()


def test_cumulative_drift_from_the_last_framed_cell_triggers_a_reframe(
        panel, two_component_system):
    """The 10% threshold must be measured from the last cell that was
    actually *framed* (`_last_cell`, anchored in `_frame_and_refresh`), not
    from whatever cell happened to be displayed on the previous render --
    otherwise a handful of individually-small drifts (~4% each) could
    accumulate past 10% of the framed baseline without ever triggering a
    reframe."""
    panel.show_system(two_component_system)
    panel._framed = True
    panel._last_cell = two_component_system.cell  # max extent 10.0, the framed baseline

    def _with_extent(extent):
        return AssembledSystem(
            positions=two_component_system.positions,
            type_ids=two_component_system.type_ids,
            group_ids=two_component_system.group_ids,
            cell=(0., extent, 0., extent, 0., extent),
            components=two_component_system.components,
            legend=two_component_system.legend,
        )

    # Two ~4% steps relative to the framed baseline: individually under the
    # 10% threshold, and _last_cell must stay put (only a real
    # _frame_and_refresh moves it, which none of these renders trigger).
    for extent in (10.4, 10.8):
        panel.system = _with_extent(extent)
        assert not panel._needs_framing()
        assert panel._last_cell == two_component_system.cell

    # A third step brings cumulative drift from the framed baseline to 12%,
    # which does owe a reframe -- even though each individual step was small.
    panel.system = _with_extent(11.2)
    assert panel._needs_framing()
