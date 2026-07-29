"""The material sweep: reachable at all, then discoverable.

A sweep config is *supposed* to carry {mat} placeholders -- expansion
substitutes them. preflight_paths says so explicitly ("not user error"), but
the GUI treated every issue as fatal, so Render and Generate refused a
correctly configured sweep.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.gui.controllers.document import ConfigDocument

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / 'examples'


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
