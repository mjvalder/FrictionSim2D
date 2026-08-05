"""End-to-end-ish: the staged config.ini a build subprocess actually reads
must contain absolute paths, not paths relative to wherever the document
used to live.

``PreviewController``/``GenerateController`` both write a copy of the config
into a directory other than the document's own (a preview cache directory, a
``_generate`` staging directory). Before ``staged_paths.absolutize_staged_paths``
existed, a relative ``cif_path``/``pot_path``/``materials_list`` -- exactly
what a material sweep configures via
``ConfigDocument.apply_material_directories`` -- silently failed to resolve
once staged, because ``search_paths`` resolves relative names against the
config file's own directory. See ``src/gui/controllers/staged_paths.py``.

These tests read the config.ini each controller actually wrote and assert
its path-like values are absolute -- they do not need the (LAMMPS) build
itself to succeed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.core.config import parse_config  # noqa: E402
from src.gui.controllers.generate import GenerateController, GenerateRequest  # noqa: E402
from src.gui.controllers.preview import PreviewController, PreviewRequest  # noqa: E402
from src.gui.workspace import Workspace  # noqa: E402


def _run_until(signal, timeout_ms: int = 5000) -> list:
    """Pump a local event loop until ``signal`` fires or ``timeout_ms`` elapses."""
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


@pytest.fixture(name='qt_app')
def _qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(name='workspace')
def _workspace(tmp_path):
    return Workspace.open(tmp_path / 'ws')


@pytest.fixture(name='document_dir')
def _document_dir(tmp_path):
    """Where the user's config, cif and potential files 'really' live --
    unrelated to `workspace`, exactly as when a config opened from an
    arbitrary directory is staged into the workspace's preview/_generate
    areas."""
    directory = tmp_path / 'myruns'
    (directory / 'cif').mkdir(parents=True)
    (directory / 'cif' / 'h_MoS2.cif').write_text('cif contents')
    (directory / 'cif' / 'h_WS2.cif').write_text('cif contents')
    (directory / 'sw_lammps').mkdir()
    (directory / 'sw_lammps' / 'h_MoS2.sw').write_text('pot contents')
    (directory / 'sw_lammps' / 'h_WS2.sw').write_text('pot contents')
    # Two materials, not one: expand_config_sweeps treats a single-material
    # list as "not really a sweep" and returns the config unexpanded (a
    # separate, pre-existing quirk, unrelated to path resolution) -- two
    # materials keeps this test on the actual sweep-expansion path.
    (directory / 'materials.txt').write_text('h_MoS2\nh_WS2\n')
    return directory


def _swept_config(document_dir: Path) -> dict:
    """What `ConfigDocument.apply_material_directories` actually writes:
    relative `cif_path`/`pot_path` templated on `{mat}`, plus a relative
    `materials_list` -- all relative to `document_dir`."""
    del document_dir
    return {
        '2D': {
            'mat': '{mat}',
            'cif_path': 'cif/{mat}.cif',
            'pot_path': 'sw_lammps/{mat}.sw',
            'materials_list': 'materials.txt',
            'x': 60,
            'y': 60,
        },
        'general': {'temp': 300},
    }


class TestPreviewControllerStagesAbsolutePaths:
    def test_staged_config_ini_has_absolute_paths(self, qt_app, workspace, document_dir,
                                                    monkeypatch):
        del qt_app
        # No real interpreter: the worker never actually runs (there is
        # nothing for it to build here), but the staged config.ini is
        # written synchronously before the process is even started, and
        # `errorOccurred` -> `failed` lets the test clean up deterministically.
        import src.gui.controllers.preview as preview_module
        monkeypatch.setattr(
            preview_module.sys, 'executable', '/nonexistent/not-a-real-interpreter-xyz')

        controller = PreviewController(workspace)
        config = _swept_config(document_dir)
        request = PreviewRequest(
            config=config, model='indent', base_dir=document_dir)

        controller.start(request)
        _run_until(controller.failed, timeout_ms=5000)

        directory = workspace.preview_path(request.key())
        written = parse_config(directory / 'config.ini')

        cif_path = written['2D']['cif_path']
        pot_path = written['2D']['pot_path']
        assert Path(cif_path).is_absolute(), cif_path
        assert Path(pot_path).is_absolute(), pot_path
        # The sweep's single member substitutes {mat} before staging, so the
        # concrete file must actually exist at the staged path.
        assert Path(cif_path).exists()
        assert Path(pot_path).exists()


class TestGenerateControllerStagesAbsolutePaths:
    def test_staged_config_ini_has_absolute_paths(self, qt_app, workspace, document_dir):
        del qt_app
        controller = GenerateController(workspace)
        config = _swept_config(document_dir)
        request = GenerateRequest(
            config=config, model='indent', base_dir=document_dir)

        controller.start(request)
        try:
            staged = workspace.preview_dir / '_generate' / 'config.ini'
            written = parse_config(staged)

            cif_path = written['2D']['cif_path']
            pot_path = written['2D']['pot_path']
            materials_list = written['2D']['materials_list']
            assert Path(cif_path).is_absolute(), cif_path
            assert Path(pot_path).is_absolute(), pot_path
            assert Path(materials_list).is_absolute(), materials_list
            assert Path(materials_list).exists()
            # Still templated -- expand_config_sweeps (run inside the
            # subprocess) substitutes {mat}, not this staging step.
            assert '{mat}' in cif_path
            assert '{mat}' in pot_path
        finally:
            controller.cancel()
