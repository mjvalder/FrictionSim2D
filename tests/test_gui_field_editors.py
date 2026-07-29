"""Editors must return exactly what they were given.

Every widget here replaces a line of raw JSON or a free-text box. The risk they
all share is silent narrowing: dropping a key the widget does not display, or
snapping an unrecognised value to a legal one before validation ever sees it.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication  # noqa: E402

from src.gui.schema.introspect import FieldKind, FieldSpec  # noqa: E402


@pytest.fixture(scope='module', name='qt_app')
def _qt_app():
    yield QApplication.instance() or QApplication([])


def _spec(key='stack_type', kind=FieldKind.STR, **kwargs):
    return FieldSpec(name=key, key=key, kind=kind, required=False, **kwargs)


class TestStringChoices:
    """stack_type is typed `str` on the model, but the builders only act on AA
    versus not-AA. A combo is honest about that without changing the model."""

    def test_a_string_field_with_choices_becomes_a_combo(self, qt_app):
        from src.gui.widgets.fields import build_editor

        editor = build_editor(_spec(), choices=('AA', 'AB'))
        assert hasattr(editor, 'box')

    def test_the_declared_choices_are_offered(self, qt_app):
        from src.gui.widgets.fields import build_editor

        editor = build_editor(_spec(), choices=('AA', 'AB'))
        offered = {editor.box.itemData(i) for i in range(editor.box.count())}
        assert {'AA', 'AB'} <= offered

    def test_an_unrecognised_value_is_preserved_not_snapped(self, qt_app):
        """The regression: opening a hand-written `stack_type = ABC` and saving
        used to write AA, silently changing the simulation."""
        from src.gui.widgets.fields import build_editor

        editor = build_editor(_spec(), choices=('AA', 'AB'))
        editor.set_value('ABC')
        assert editor.value() == 'ABC'

    def test_an_unrecognised_value_is_flagged(self, qt_app):
        from src.gui.widgets.fields import build_editor

        editor = build_editor(_spec(), choices=('AA', 'AB'))
        editor.set_value('ABC')
        assert editor.warning() is not None
        assert 'ABC' in editor.warning()

    def test_a_recognised_value_is_not_flagged(self, qt_app):
        from src.gui.widgets.fields import build_editor

        editor = build_editor(_spec(), choices=('AA', 'AB'))
        editor.set_value('AB')
        assert editor.warning() is None


class TestRegistry:

    def test_an_unknown_slug_falls_back_to_the_kind(self, qt_app):
        from src.gui.widgets.fields import build_editor

        editor = build_editor(_spec(kind=FieldKind.STR), editor='no_such_slug')
        assert hasattr(editor, 'edit')


class TestStackingOverride:

    def test_stacking_offers_only_aa_and_ab(self):
        from src.gui.schema.overrides import section_override

        assert section_override('2D').fields['stack_type'].choices == ('AA', 'AB')

    def test_stacking_explains_itself(self):
        from src.gui.schema.overrides import section_override

        override = section_override('2D').fields['stack_type']
        assert override.help
        assert 'AA' in override.help
