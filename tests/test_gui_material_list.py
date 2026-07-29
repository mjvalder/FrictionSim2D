"""The material sweep: reachable at all, then discoverable.

A sweep config is *supposed* to carry {mat} placeholders -- expansion
substitutes them. preflight_paths says so explicitly ("not user error"), but
the GUI treated every issue as fatal, so Render and Generate refused a
correctly configured sweep.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication  # noqa: E402

from src.gui.controllers.document import ConfigDocument  # noqa: E402
from src.gui.workspace import Workspace  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / 'examples'


@pytest.fixture(scope='module', name='qt_app')
def _qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(name='materials_file')
def _materials_file(tmp_path):
    path = tmp_path / 'mats.txt'
    path.write_text('h-MoS2\nt-MoS2\n', encoding='utf-8')
    return path


@pytest.fixture(name='swept')
def _swept(materials_file, monkeypatch):
    """An indent config with a resolving two-material sweep."""
    monkeypatch.chdir(REPO_ROOT)
    document = ConfigDocument.from_file(EXAMPLES / 'indent_config.ini', 'indent')
    document.set('2D', 'materials_list', str(materials_file))
    document.set('2D', 'mat', '{mat}')
    document.set('2D', 'cif_path', 'examples/materials/{mat}.cif')
    return document


class TestSweepIsReachable:

    def test_the_sweep_expands(self, swept):
        assert swept.expansion().configs == 2

    def test_a_templated_sweep_config_validates(self, swept):
        """The regression: this returned ok=False on the {mat} placeholder
        alone, which blocked both Render and Generate."""
        result = swept.validate()
        assert result.ok, result.render()

    def test_the_pending_substitutions_are_still_reported(self, swept):
        swept.validate()
        keys = {i.key for i in swept.pending_template_issues}
        assert 'cif_path' in keys

    def test_a_placeholder_with_no_list_is_still_an_error(self, swept):
        """Nothing will ever substitute it, so it is a genuine mistake."""
        del swept.values['2D']['materials_list']
        result = swept.validate()
        assert not result.ok
        assert any(i.kind == 'unexpanded_template' for i in result.issues)


class TestWindowIntegration:
    """MainWindow._validate() surfaces pending substitutions without blocking."""

    @pytest.fixture(name='window')
    def _window(self, qt_app, tmp_path, monkeypatch):
        """Create a MainWindow for testing."""
        monkeypatch.chdir(REPO_ROOT)
        workspace = Workspace.open(tmp_path / 'ws')
        from src.gui.app import MainWindow  # noqa: PLC0415
        window = MainWindow(workspace)
        yield window
        window.close()

    def test_valid_sweep_validates_and_reports_pending(self, window, materials_file):
        """A correctly configured sweep: _validate() returns True, issues pane
        mentions the pending substitution."""
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        window.document.set('2D', 'materials_list', str(materials_file))
        window.document.set('2D', 'mat', '{mat}')
        window.document.set('2D', 'cif_path', 'examples/materials/{mat}.cif')
        window.form.document = window.document
        window.form.rebuild()

        # _validate() must return True for a valid sweep
        assert window._validate()

        # The issues pane must mention the pending substitution
        issues_text = window.issues.toPlainText()
        assert '{mat}' in issues_text
        assert 'substituted' in issues_text

    def test_unresolvable_materials_list_does_not_raise(self, window, tmp_path, monkeypatch):
        """A materials_list naming materials whose CIF files do not exist:
        expand_config_sweeps will raise, but window._validate() must not crash
        and must still report something useful."""
        monkeypatch.chdir(REPO_ROOT)
        window.document = ConfigDocument.from_file(
            EXAMPLES / 'indent_config.ini', 'indent')
        bad_materials = tmp_path / 'bad_mats.txt'
        bad_materials.write_text('nonexistent1\nnonexistent2\n', encoding='utf-8')

        window.document.set('2D', 'materials_list', str(bad_materials))
        window.document.set('2D', 'mat', '{mat}')
        window.document.set('2D', 'cif_path', 'examples/materials/{mat}.cif')
        window.form.document = window.document
        window.form.rebuild()

        # _validate() must not raise an exception
        try:
            result = window._validate()
        except Exception as exc:  # pylint: disable=broad-except
            pytest.fail(f"_validate() raised {type(exc).__name__}: {exc}")

        # The issues pane should still mention pending substitutions
        issues_text = window.issues.toPlainText()
        assert '{mat}' in issues_text
        # When count fails, it falls back to a generic message
        assert 'substituted' in issues_text
