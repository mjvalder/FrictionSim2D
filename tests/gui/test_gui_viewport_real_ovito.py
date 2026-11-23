"""Regression tests that exercise real ovito and real Qt native windows.

The stubbed tests in ``test_gui_viewport.py`` monkeypatch ``_update_pipeline``,
``_ensure_widget`` and ``build_data_collection`` -- which is why regressions
shipped straight through that suite without being caught. This file
deliberately does not stub ovito (or, where noted, Qt's native window): it
uses the real ``ovito`` package and, for the window-recreation test, a real
(offscreen) Qt top-level window.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip('ovito')
pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication, QMainWindow  # noqa: E402

from frictionsim2d.gui.preview.assemble import (  # noqa: E402
    AssembledSystem,
    ComponentSlice,
    build_data_collection,
)


@pytest.fixture(scope='module', name='qt_app')
def _qt_app():
    yield QApplication.instance() or QApplication([])


def _system(count, type_id=1):
    return AssembledSystem(
        positions=np.zeros((count, 3)), type_ids=np.full(count, type_id),
        group_ids=np.zeros(count, dtype=int), cell=(0., 10., 0., 10., 0., 10.),
        components=[ComponentSlice('2D', 0, 0, count, [type_id])],
        legend={type_id: 'Mo'})


# --------------------------------------------------------------------- task 1
# _update_pipeline's in-place update (``pipeline.source.data = data``)
# silently never invalidates ovito's cache, so colour changes, component
# visibility toggles and the Box toggle never reach the screen.


def test_replacing_source_updates_what_the_pipeline_computes():
    from ovito.pipeline import Pipeline, StaticSource

    pipe = Pipeline(source=StaticSource(data=build_data_collection(_system(2))))
    assert pipe.compute().particles.count == 2

    # the production update path must make this 5
    from frictionsim2d.gui.preview.viewport import _install_data
    _install_data(pipe, build_data_collection(_system(5)))
    assert pipe.compute().particles.count == 5


def test_colour_change_reaches_the_computed_types():
    from ovito.pipeline import Pipeline, StaticSource

    from frictionsim2d.gui.preview.viewport import _install_data

    pipe = Pipeline(source=StaticSource(
        data=build_data_collection(_system(3), colours={1: (1.0, 0.0, 0.0)})))
    _install_data(pipe, build_data_collection(_system(3), colours={1: (0.0, 0.0, 1.0)}))
    t = pipe.compute().particles['Particle Type'].types[0]
    assert tuple(round(c, 3) for c in t.color) == (0.0, 0.0, 1.0)


def test_show_system_twice_updates_the_live_pipeline(qt_app):
    """End-to-end: driving the panel through ``show_system`` twice with
    different atom counts must leave the real pipeline showing the second
    system, not the first -- this is what actually broke for the user
    (colour-mode changes, component toggles and the Box toggle silently not
    appearing on screen)."""
    from frictionsim2d.gui.preview.viewport import ViewportPanel

    panel = ViewportPanel()
    panel.show_system(_system(2))
    panel.show_system(_system(5))
    assert panel._pipeline.compute().particles.count == 5


# --------------------------------------------------------------------- task 2
# _ensure_widget built the ovito GL widget lazily, on the first render --
# always after the window is already shown. Adding a QOpenGLWidget to an
# already-shown top-level window forces Qt/X11 to recreate the native
# window, which unmaps and remaps it: the whole app visibly closes and
# reopens.


def test_widget_is_created_eagerly_during_construction(qt_app):
    """The GL widget must exist right after ``__init__`` -- before any system
    is ever shown -- so that it never has to be added to a window that is
    already visible. Creating it lazily (on first render, which always
    happens after ``window.show()``) is exactly the old bug."""
    from frictionsim2d.gui.preview.viewport import ViewportPanel

    panel = ViewportPanel()
    assert panel._widget is not None


def test_native_window_is_not_recreated_by_the_first_render(qt_app):
    from frictionsim2d.gui.preview.viewport import ViewportPanel

    win = QMainWindow()
    panel = ViewportPanel(win)
    win.setCentralWidget(panel)
    win.resize(800, 600)
    win.show()
    qt_app.processEvents()

    before = int(win.winId())
    panel.show_system(_system(4))
    qt_app.processEvents()

    assert int(win.winId()) == before, "the GL widget recreated the native window"


def test_panel_constructs_when_ovito_is_unavailable(qt_app, monkeypatch):
    """The app must never fail to *start* because eager widget creation
    couldn't import ovito or initialise a GL context -- a headless machine or
    a broken GL stack must fall back to the placeholder instead."""
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == 'ovito.vis' or name.startswith('ovito.vis.'):
            raise ImportError("simulated: no ovito on this machine")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', _fake_import)

    from frictionsim2d.gui.preview.viewport import ViewportPanel

    panel = ViewportPanel()  # must not raise
    panel.show()
    qt_app.processEvents()
    assert panel._widget is None
    assert panel.placeholder.isVisible()


def test_placeholder_hides_only_once_a_system_is_shown(qt_app):
    from frictionsim2d.gui.preview.viewport import ViewportPanel

    panel = ViewportPanel()
    panel.show()
    qt_app.processEvents()
    # The widget is created eagerly, but the placeholder must still be what
    # the user sees until the first real render.
    assert panel.placeholder.isVisible()
    panel.show_system(_system(3))
    assert not panel.placeholder.isVisible()
    panel.clear()
    assert panel.placeholder.isVisible()
