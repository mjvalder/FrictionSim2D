"""The Plots tab's controller and pane (G1, G2, G4).

No pytest-qt in this environment: the event loop is driven by `_run_until`,
the same plain-QtCore pump `test_gui_preview_process.py` uses. Every widget
here is built offscreen, and nothing may open a modal -- an unpatched
QMessageBox in this directory blocks the suite forever with no one to click
it.
"""

from __future__ import annotations

import json
import os

import pytest

pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.gui.controllers.plots import (PlotController, PlotFigure,  # noqa: E402
                                       PlotRequest)
from src.gui.workspace import Workspace  # noqa: E402
from src.postprocessing.plot_config import PlotDocument  # noqa: E402
from src.postprocessing.read_data import DataReader  # noqa: E402
from tests.helpers.results_tree import make_family_run  # noqa: E402


def _run_until(signal, timeout_ms: int = 60000) -> list:
    """Pump a local event loop until ``signal`` fires or ``timeout_ms`` elapses.

    No pytest-qt in this environment (declared in pyproject.toml's ``dev``
    extra but not installed here), so this drives the real Qt event loop with
    plain QtCore -- the same primitives the app itself uses -- rather than
    pulling in a new test dependency.

    Returns the arguments the signal was emitted with, or ``[]`` on timeout
    (distinguishable from a signal genuinely emitted with no arguments,
    which none of the controller's signals are).
    """
    loop = QEventLoop()
    received: list = []

    def _capture(*args):
        received.append(args)
        loop.quit()

    signal.connect(_capture)
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(timeout_ms)

    loop.exec()

    timer.stop()
    signal.disconnect(_capture)
    return received[0] if received else []


@pytest.fixture(name='qt_app', scope='module')
def _qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(name='workspace')
def _workspace(tmp_path):
    return Workspace.open(tmp_path / 'ws')


@pytest.fixture(name='document')
def _document(tmp_path):
    root = tmp_path / 'ds'
    make_family_run(root, 'afm', loads=(10.0, 30.0, 50.0))
    reader = DataReader(results_dir=str(root))
    reader.export_full_data_to_json()
    reader.export_summary_csv()
    return PlotDocument(
        data_dirs=[str(root)], labels=['ds'],
        plots=[{'plot_type': 'summary', 'x_axis': 'force', 'y_axis': 'lf',
                'filter_size': '100x100y', 'filename': 'lf.png'}])


class TestPlotController:

    def test_a_render_reaches_finished_with_the_figures(
            self, qt_app, workspace, document):
        del qt_app
        controller = PlotController(workspace)

        controller.start(PlotRequest(
            document=document, settings={'export': {'formats': ['png']}}))
        result = _run_until(controller.finished)

        assert result, 'the "finished" signal never arrived'
        figures = result[0]
        assert [type(f) for f in figures] == [PlotFigure]
        assert figures[0].index == 0
        assert figures[0].plot_type == 'summary'
        assert figures[0].files[0].name == 'lf.png'
        assert figures[0].files[0].is_file()
        assert not controller.running

    def test_progress_lines_are_forwarded(self, qt_app, workspace, document):
        del qt_app
        controller = PlotController(workspace)
        seen = []
        controller.progress.connect(seen.append)

        controller.start(PlotRequest(document=document))
        _run_until(controller.finished)

        assert any('plot 1/1' in message for message in seen)

    def test_a_failed_to_start_process_ends_the_render(
            self, qt_app, workspace, document, monkeypatch):
        """Qt never emits `finished` when the process cannot start at all;
        left unwired, `running` stays True and the Render button stays
        disabled forever -- the same defect PreviewController had."""
        del qt_app
        import src.gui.controllers.plots as plots_module
        monkeypatch.setattr(plots_module.sys, 'executable',
                            '/nonexistent/not-a-real-interpreter-xyz')
        controller = PlotController(workspace)

        controller.start(PlotRequest(document=document))
        result = _run_until(controller.failed, timeout_ms=10000)

        assert result, 'the "failed" signal never arrived -- the render is stuck'
        assert not controller.running

    def test_a_broken_document_reaches_failed_with_the_error(
            self, qt_app, workspace, tmp_path):
        del qt_app
        controller = PlotController(workspace)
        document = PlotDocument(
            data_dirs=[str(tmp_path / 'nothing')], labels=['ds'],
            plots=[{'plot_type': 'summary', 'filter_size': '100x100y',
                    'filename': 'x.png'}])

        controller.start(PlotRequest(document=document))
        result = _run_until(controller.failed, timeout_ms=60000)

        assert result
        assert 'PlotDataError' in result[0] or 'no summary' in result[0]

    def test_warnings_arrive_before_finished(self, qt_app, workspace, document):
        del qt_app
        # StickSlipPlot, not type(document.plots[0]) -- that is SummaryPlot,
        # whose plot_type is Literal['summary'] and rejects any other value.
        from src.postprocessing.plot_config import StickSlipPlot
        document.plots.append(StickSlipPlot(
            filter_size='nope', filename='bad.png'))
        controller = PlotController(workspace)
        order = []
        controller.warned.connect(lambda w: order.append(('warned', w)))
        controller.finished.connect(lambda f: order.append(('finished', f)))

        controller.start(PlotRequest(document=document))
        _run_until(controller.finished)

        assert [step for step, _ in order] == ['warned', 'finished']
        assert 'plot 2' in order[0][1][0]

    def test_cancel_ends_a_running_render(self, qt_app, workspace, document):
        del qt_app
        controller = PlotController(workspace)
        controller.start(PlotRequest(document=document))

        controller.cancel()

        assert not controller.running

    def test_the_request_writes_its_files_under_the_plots_dir(
            self, qt_app, workspace, document):
        del qt_app
        controller = PlotController(workspace)
        request = PlotRequest(document=document)

        controller.start(request)
        _run_until(controller.finished)

        directory = controller.plots_dir / request.key()
        assert (directory / 'plots.json').is_file()
        assert (directory / 'request.json').is_file()
        assert (directory / 'result.json').is_file()
        payload = json.loads(
            (directory / 'request.json').read_text(encoding='utf-8'))
        assert set(payload) == {'document_file', 'output_dir', 'preset',
                                'settings', 'plot_indices', 'result_file'}

    def test_the_key_changes_with_the_settings(self, document):
        a = PlotRequest(document=document, preset='publication')
        b = PlotRequest(document=document, preset=None)
        c = PlotRequest(document=document, preset='publication',
                        plot_indices=[0])

        assert a.key() != b.key()
        assert a.key() != c.key()
        assert a.key() == PlotRequest(document=document,
                                      preset='publication').key()


def test_the_plots_scratch_directory_survives_preview_pruning(tmp_path):
    """`prune_previews` keeps the newest eight directories under `.preview`.
    `_plots` is one directory there, so eight new previews used to delete
    every rendered figure the Plots tab was showing."""
    workspace = Workspace.open(tmp_path / 'ws')
    plots = workspace.preview_dir / '_plots'
    (plots / 'abc').mkdir(parents=True)
    for i in range(12):
        (workspace.preview_dir / f'preview{i}').mkdir()

    removed = workspace.prune_previews(keep=4)

    assert plots.is_dir()
    assert (plots / 'abc').is_dir()
    assert plots not in removed
    assert len(removed) == 8
