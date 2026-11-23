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

from frictionsim2d.gui.controllers.plots import (PlotController, PlotFigure,  # noqa: E402
                                       PlotRequest)
from frictionsim2d.gui.workspace import Workspace  # noqa: E402
from frictionsim2d.postprocessing.plot_config import PlotDocument  # noqa: E402
from frictionsim2d.postprocessing.read_data import DataReader  # noqa: E402
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
        import frictionsim2d.gui.controllers.plots as plots_module
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
        from frictionsim2d.postprocessing.plot_config import StickSlipPlot
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


from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtGui import QColor, QPixmap  # noqa: E402
from PySide6.QtWidgets import (QComboBox, QLabel, QListWidget,  # noqa: E402
                               QPushButton, QScrollArea)

from frictionsim2d.gui.widgets.plots import PlotsPane, discover_datasets  # noqa: E402
from frictionsim2d.postprocessing.plot_config import PLOT_TYPES  # noqa: E402


class _StubController(QObject):
    """A PlotController that answers immediately, without a subprocess.

    Spec section 3 asks for the worker stubbed at the boundary. Task 30's
    tests are the one place that boundary is exercised for real -- they ARE
    the protocol seam, and stubbing there would test the stub. Everything in
    this class is about what the PANE does with an answer, so it gets one
    synchronously and the suite spends no interpreter starts here.

    Signals match `PlotController`'s, name for name and payload for payload.
    """

    started = Signal()
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    warned = Signal(list)

    def __init__(self, tmp_path, parent=None):
        super().__init__(parent)
        self.requests = []
        self.mode = 'ok'                  # 'ok' | 'fail' | 'empty'
        self.warnings = []
        self.running = False
        self._dir = tmp_path / 'figures'
        self._dir.mkdir(parents=True, exist_ok=True)
        # A real, loadable PNG made with Qt alone -- no matplotlib in this
        # process either, so the test cannot mask a G4 regression.
        pixmap = QPixmap(8, 8)
        pixmap.fill(QColor('white'))
        self.png = self._dir / 'lf.png'
        pixmap.save(str(self.png))
        self.vectors = [self._dir / 'lf.pdf', self._dir / 'lf.svg', self.png]
        for path in self.vectors[:2]:
            path.write_bytes(b'%PDF-stub')

    def start(self, request):
        from frictionsim2d.gui.controllers.plots import PlotFigure

        self.requests.append(request)
        self.started.emit()
        self.progress.emit('plot 1/1: summary')
        if self.mode == 'fail':
            self.failed.emit(
                'PlotDataError: ds: no output_full_*.json under /nope')
            return
        if self.warnings:
            self.warned.emit(list(self.warnings))
        if self.mode == 'empty':
            self.finished.emit([])
            return
        files = self.vectors if request.settings is None else [self.png]
        index = (request.plot_indices or [0])[0]
        self.finished.emit(
            [PlotFigure(index=index, plot_type='summary', files=list(files))])

    def cancel(self):
        self.cancelled.emit()


class TestPlotsPane:

    @pytest.fixture(name='stub')
    def _stub(self, tmp_path):
        return _StubController(tmp_path)

    @pytest.fixture(name='pane')
    def _pane(self, qt_app, workspace, document, stub):
        del qt_app
        pane = PlotsPane(workspace, controller=stub)
        pane.set_document(document)
        yield pane
        pane.close()

    def test_the_widgets_are_the_declared_ones(self, pane):
        assert isinstance(pane.dataset_list, QListWidget)
        assert isinstance(pane.plot_list, QListWidget)
        assert isinstance(pane.type_combo, QComboBox)
        assert isinstance(pane.preset_combo, QComboBox)
        assert isinstance(pane.preview_label, QLabel)
        assert isinstance(pane.render_button, QPushButton)
        assert pane.preview_label.parent() is not None

    def test_the_preview_is_a_scrolled_label_not_a_canvas(self, pane):
        """G4: no live matplotlib canvas anywhere in the widget tree."""
        scrolls = pane.findChildren(QScrollArea)

        assert scrolls, 'the preview must live in a scroll area'
        assert any(s.widget() is pane.preview_label for s in scrolls)
        assert not [w for w in pane.findChildren(QLabel)
                    if type(w).__name__.endswith('Canvas')]

    def test_the_type_combo_offers_every_plot_type(self, pane):
        offered = [pane.type_combo.itemText(i)
                   for i in range(pane.type_combo.count())]

        assert offered == list(PLOT_TYPES)

    def test_the_preset_combo_offers_none_plus_the_shipped_presets(self, pane):
        from frictionsim2d.postprocessing.presets import PRESETS

        offered = [pane.preset_combo.itemText(i)
                   for i in range(pane.preset_combo.count())]

        assert offered == ['(none)'] + list(PRESETS)

    def test_discover_datasets_finds_workspace_and_exported_sets(
            self, workspace, tmp_path):
        """G1: a dataset is a directory with outputs/output_full_*.json --
        either a package under the workspace root, or a set pulled out of the
        local store by `aiida export-results --output-dir
        <workspace>/.preview/_datasets/<label>`."""
        del tmp_path
        for parent in (workspace.root, workspace.preview_dir / '_datasets'):
            outputs = (parent / 'ds_a' / 'outputs'
                       if parent is workspace.root
                       else parent / 'set_b' / 'outputs')
            outputs.mkdir(parents=True)
            (outputs / 'output_full_100x100y.json').write_text(
                '{}', encoding='utf-8')
        (workspace.root / 'not_a_dataset').mkdir()
        (workspace.root / 'empty_outputs' / 'outputs').mkdir(parents=True)

        found = [d.name for d in discover_datasets(workspace)]

        assert 'ds_a' in found
        assert 'set_b' in found
        assert 'not_a_dataset' not in found
        assert 'empty_outputs' not in found

    def test_add_and_remove_a_dataset(self, pane, workspace):
        outputs = workspace.root / 'ds_a' / 'outputs'
        outputs.mkdir(parents=True)
        (outputs / 'output_full_100x100y.json').write_text(
            '{}', encoding='utf-8')
        pane.refresh_available_datasets()
        before = pane.dataset_list.count()

        pane.available_list.setCurrentRow(
            [pane.available_list.item(i).text()
             for i in range(pane.available_list.count())].index('ds_a'))
        pane.add_selected_dataset()

        assert pane.dataset_list.count() == before + 1
        assert 'ds_a' in pane.document().labels
        assert str(workspace.root / 'ds_a') in pane.document().data_dirs

        pane.dataset_list.setCurrentRow(before)
        pane.remove_selected_dataset()

        assert pane.dataset_list.count() == before
        assert 'ds_a' not in pane.document().labels

    def test_adding_the_same_dataset_twice_is_refused_quietly(self, pane,
                                                              workspace):
        outputs = workspace.root / 'ds_a' / 'outputs'
        outputs.mkdir(parents=True)
        (outputs / 'output_full_100x100y.json').write_text(
            '{}', encoding='utf-8')
        pane.refresh_available_datasets()
        row = [pane.available_list.item(i).text()
               for i in range(pane.available_list.count())].index('ds_a')
        pane.available_list.setCurrentRow(row)
        pane.add_selected_dataset()
        count = pane.dataset_list.count()

        pane.add_selected_dataset()

        assert pane.dataset_list.count() == count
        assert 'already in the document' in pane.status_label.text()
        assert QApplication.activeModalWidget() is None

    def test_open_save_and_save_as_round_trip(self, pane, tmp_path,
                                              monkeypatch):
        """G3, and the thing that makes the tab usable at all: a document
        built here has to be keepable. The dialogs are patched -- an
        unpatched modal blocks this suite forever."""
        from PySide6.QtWidgets import QFileDialog

        from frictionsim2d.postprocessing.plot_config import load_plot_document

        target = tmp_path / 'kept.json'
        monkeypatch.setattr(QFileDialog, 'getSaveFileName',
                            staticmethod(lambda *a, **k: (str(target), '')))
        pane.add_plot('correlation')

        pane.save_as()

        assert target.is_file()
        assert pane.current_path == target
        assert len(load_plot_document(target).plots) == 2
        assert 'Saved kept.json' in pane.status_label.text()

        pane.add_plot('rank_friction')
        pane.save()                                  # no dialog this time
        assert len(load_plot_document(target).plots) == 3

        monkeypatch.setattr(QFileDialog, 'getOpenFileName',
                            staticmethod(lambda *a, **k: (str(target), '')))
        pane.add_plot('timeseries')
        pane.open_document()

        assert len(pane.document().plots) == 3
        assert QApplication.activeModalWidget() is None

    def test_a_cancelled_dialog_changes_nothing(self, pane, monkeypatch):
        from PySide6.QtWidgets import QFileDialog

        monkeypatch.setattr(QFileDialog, 'getSaveFileName',
                            staticmethod(lambda *a, **k: ('', '')))
        before = len(pane.document().plots)

        pane.save_as()

        assert pane.current_path is None
        assert len(pane.document().plots) == before

    def test_an_unopenable_document_reports_in_the_status_label(
            self, pane, tmp_path, monkeypatch):
        from PySide6.QtWidgets import QFileDialog

        bad = tmp_path / 'bad.json'
        bad.write_text('{"plots": [{"plot_type": "wobble"}]}', encoding='utf-8')
        monkeypatch.setattr(QFileDialog, 'getOpenFileName',
                            staticmethod(lambda *a, **k: (str(bad), '')))

        pane.open_document()

        assert 'Could not open bad.json' in pane.status_label.text()
        assert QApplication.activeModalWidget() is None

    def test_document_changed_reaches_the_status_label(self, pane):
        """The signal was emitted and connected to nothing."""
        pane.add_plot('summary')

        assert 'plot(s)' in pane.status_label.text()

    def test_render_previews_only_the_selected_plot(self, pane, stub):
        """A ten-plot document renders ten figures to display figures[0]."""
        pane.add_plot('summary')
        pane.plot_list.setCurrentRow(1)

        pane.render()

        assert stub.requests[-1].plot_indices == [1]

    def test_export_renders_every_plot(self, pane, stub):
        pane.add_plot('summary')
        pane.plot_list.setCurrentRow(1)

        pane.export()

        assert stub.requests[-1].plot_indices is None

    def test_the_export_button_exists(self, pane):
        assert isinstance(pane.export_button, QPushButton)
        assert isinstance(pane.add_dataset_button, QPushButton)
        assert isinstance(pane.available_list, QListWidget)
        assert isinstance(pane.open_button, QPushButton)
        assert isinstance(pane.save_button, QPushButton)
        assert isinstance(pane.save_as_button, QPushButton)

    def test_the_lists_reflect_the_document(self, pane):
        assert pane.dataset_list.count() == 1
        assert pane.dataset_list.item(0).text() == 'ds'
        assert pane.plot_list.count() == 1
        assert 'summary' in pane.plot_list.item(0).text()

    def test_selecting_a_plot_rebuilds_the_form_over_its_model(self, pane):
        pane.add_plot('stick_slip_analysis')
        pane.plot_list.setCurrentRow(1)

        assert pane.form.document.plot_type == 'stick_slip_analysis'
        assert 'stick_slip_analysis' in pane.form.sections

    def test_editing_the_form_reaches_the_document(self, pane):
        from frictionsim2d.gui.controllers.plot_document import COMMON_SECTION

        pane.plot_list.setCurrentRow(0)
        pane.form.document.set(COMMON_SECTION, 'filter_layer', 2)
        pane._commit_current_plot()  # pylint: disable=protected-access

        assert pane.document().plots[0].filter_layer == 2

    def test_add_and_remove_a_plot(self, pane):
        pane.add_plot('correlation')
        assert pane.plot_list.count() == 2
        assert pane.document().plots[1].plot_type == 'correlation'

        pane.plot_list.setCurrentRow(1)
        pane.remove_selected_plot()

        assert pane.plot_list.count() == 1
        assert len(pane.document().plots) == 1

    def test_changing_the_type_keeps_the_common_fields(self, pane):
        from frictionsim2d.gui.controllers.plot_document import COMMON_SECTION

        pane.plot_list.setCurrentRow(0)
        pane.form.document.set(COMMON_SECTION, 'filter_size', '100x100y')
        pane.type_combo.setCurrentText('timeseries')

        assert pane.document().plots[0].plot_type == 'timeseries'
        assert pane.document().plots[0].filter_size == '100x100y'

    def test_save_and_load_round_trip_the_same_json(self, pane, tmp_path):
        """G3: the GUI reads and writes exactly what `postprocess plot` reads."""
        from frictionsim2d.postprocessing.plot_config import load_plot_document

        path = pane.save_document(tmp_path / 'plots.json')

        assert path.is_file()
        reloaded = load_plot_document(path)
        assert reloaded.model_dump(exclude_unset=True) == \
            pane.document().model_dump(exclude_unset=True)

        pane.add_plot('correlation')
        pane.load_document(path)
        assert len(pane.document().plots) == 1

    def test_render_shows_the_png_and_never_opens_a_modal(self, pane):
        """An unpatched modal in this directory blocks the suite forever."""
        pane.render()

        # QLabel.pixmap() returns a NULL QPixmap, never None, so isNull() is
        # the only assertion here that means anything.
        assert not pane.preview_label.pixmap().isNull()
        assert 'lf.png' in pane.status_label.text()
        assert QApplication.activeModalWidget() is None
        assert pane.render_button.isEnabled()

    def test_the_preview_asks_for_png_and_the_export_does_not(self, pane, stub):
        """G2: Export uses the preset's OWN formats. The PNG override on
        `render` is a preview concern -- a QLabel cannot show a PDF -- and
        applying it to an export would silently defeat
        `publication`'s pdf/svg/png."""
        pane.preset_combo.setCurrentText('publication')

        pane.render()
        pane.export()

        assert stub.requests[0].settings == {'export': {'formats': ['png']}}
        assert stub.requests[0].preset == 'publication'
        assert stub.requests[1].settings is None
        assert stub.requests[1].preset == 'publication'

    def test_export_reports_the_formats_it_wrote(self, pane):
        pane.preset_combo.setCurrentText('publication')

        pane.export()

        text = pane.status_label.text()
        assert 'Exported 3 file(s)' in text
        assert 'pdf' in text and 'svg' in text and 'png' in text
        assert pane.export_button.isEnabled()

    def test_a_failed_render_reports_in_the_status_label(self, pane, stub):
        stub.mode = 'fail'

        pane.render()

        assert 'failed' in pane.status_label.text().lower()
        assert 'no output_full' in pane.status_label.text()
        assert QApplication.activeModalWidget() is None
        assert pane.render_button.isEnabled()
        assert pane.export_button.isEnabled()

    def test_a_render_that_produces_nothing_says_so(self, pane, stub):
        stub.mode = 'empty'

        pane.render()

        assert 'Nothing was rendered' in pane.status_label.text()

    def test_warnings_are_shown_beside_the_figures(self, pane, stub):
        stub.warnings = [
            'plot 2: PlotDataError: filter_size nope matched nothing']

        pane.render()

        assert 'warning' in pane.status_label.text().lower()
        assert 'plot 2' in pane.status_label.text()


class TestPaneFitsOnAScreen:
    """A window cannot be resized below its layout's minimum, and a window
    manager cannot maximise it into a screen shorter than that. A tall form
    added straight to a layout therefore locks the whole application's
    vertical size -- and because QStackedWidget takes the maximum over all its
    pages, it does so whichever tab is in front.
    """

    #: Shortest screen the app is expected to run on (a 1366x768 laptop, less
    #: room for the panel and window decorations).
    SCREEN_HEIGHT = 700

    def test_the_pane_does_not_demand_more_height_than_a_screen(
            self, qt_app, workspace):
        from frictionsim2d.gui.widgets.plots.pane import PlotsPane

        pane = PlotsPane(workspace)
        assert pane.minimumSizeHint().height() <= self.SCREEN_HEIGHT, (
            "PlotsPane demands more vertical space than a laptop screen; the "
            "window manager cannot resize or maximise a window whose minimum "
            "exceeds the screen")

    def test_the_main_window_does_not_demand_more_height_than_a_screen(
            self, qt_app, workspace):
        from frictionsim2d.gui.app import MainWindow

        window = MainWindow(workspace)
        try:
            assert window.minimumSizeHint().height() <= self.SCREEN_HEIGHT, (
                "MainWindow's minimum height exceeds a laptop screen: it "
                "cannot be resized vertically or maximised")
        finally:
            window.document.dirty = False
            window.settings_document.dirty = False
            window.close()
