"""What the form says about itself.

The docs carry a table of which fields may hold a list. It belongs in the form,
generated from the same LOOPABILITY map the expansion uses -- not restated by
hand in the override table, where it drifts.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication  # noqa: E402

from src.gui.controllers.document import ConfigDocument  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module', name='qt_app')
def _qt_app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture(name='form')
def _form(qt_app, monkeypatch):
    from src.gui.widgets.form import ConfigForm

    monkeypatch.chdir(REPO_ROOT)
    form = ConfigForm(ConfigDocument.blank('afm'))
    yield form
    form.deleteLater()


class TestLoopabilityHints:

    def _caption(self, form, section, key):
        return form.sections[section]._captions[key].text()

    def test_a_swept_field_says_a_list_makes_separate_runs(self, form):
        assert 'separate simulation' in self._caption(form, 'general', 'temp')

    def test_a_layer_field_says_a_list_makes_separate_builds(self, form):
        assert 'L{n}' in self._caption(form, '2D', 'layers')

    def test_an_in_file_field_says_a_list_is_looped(self, form):
        assert 'looped inside' in self._caption(form, 'general', 'force')

    def test_a_scalar_field_gets_no_hint(self, form):
        """Captioning every single-value field with a negative is noise; a list
        that is actually present is already reported by validation."""
        caption = form.sections['tip']._captions.get('r')
        assert caption is not None
        assert 'list' not in caption.text()

    def test_the_hint_is_appended_not_substituted(self, form):
        """twist_angle has its own caption and no loopability; force has a
        loopability and no hand-written caption. Both must render."""
        assert self._caption(form, 'general', 'twist_angle')

    def test_settings_fields_get_no_hint(self, qt_app, monkeypatch):
        """SettingsDocument has no loopability concept at all."""
        from src.gui.controllers.settings_document import SettingsDocument
        from src.gui.widgets.form import ConfigForm

        monkeypatch.chdir(REPO_ROOT)
        form = ConfigForm(SettingsDocument.defaults())
        caption = form.sections['hpc']._captions.get('num_cpus')
        assert caption is None or 'list' not in caption.text()
        form.deleteLater()
