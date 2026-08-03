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
    panel._framed = True          # what _frame_and_refresh sets in the real path
    panel.show_system(two_component_system)
    assert not panel._needs_framing()
