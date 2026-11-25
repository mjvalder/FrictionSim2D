"""The filter form of the database pane.

The bar produces one dict, the `filters` block of a `list` request, which
`db_worker` passes to `Friction2DDB.query` over the local AiiDA store. What
is pinned here: an untouched control
contributes nothing (None, not '(any)'), a half-filled range still works, and
free text is NOT part of the query.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication  # noqa: E402

from frictionsim2d.gui.widgets.database.filters import (  # noqa: E402
    ANY, EMPTY_FILTERS, HIGH_SENTINEL, LOW_SENTINEL, FilterBar)


@pytest.fixture(name='qt_app', scope='module')
def _qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(name='bar')
def _bar(qt_app):
    del qt_app
    bar = FilterBar()
    bar.set_choices({'materials': ['h-MoS2', 'h-WS2'], 'families': ['afm', 'hetero'],
                     'models': ['afm'], 'potentials': ['sw', 'rebomos'],
                     'layers': [1, 2], 'temperatures': [300.0],
                     'set_labels': ['s1', 's2']})
    return bar


def test_an_untouched_bar_filters_nothing(bar):
    assert bar.value() == EMPTY_FILTERS
    assert bar.text() == ''


def test_free_text_is_not_a_query_filter(bar):
    bar.search.setText('h-MoS2')

    assert bar.text() == 'h-MoS2'
    assert 'text' not in bar.value()
    assert bar.value() == EMPTY_FILTERS


def test_the_choices_populate_every_combo_with_an_any_entry(bar):
    families = [bar.family.itemText(i) for i in range(bar.family.count())]
    layers = [bar.layers.itemText(i) for i in range(bar.layers.count())]

    assert families == [ANY, 'afm', 'hetero']
    assert layers == [ANY, '1', '2']


def test_selecting_values_produces_the_worker_filter_block(bar):
    bar.family.setCurrentText('hetero')
    bar.material.setCurrentText('h-WS2')
    bar.potential.setCurrentText('sw')
    bar.layers.setCurrentText('2')
    bar.set_label.setCurrentText('s2')

    value = bar.value()

    assert value['family'] == 'hetero'
    assert value['materials'] == ['h-WS2']
    assert value['potential'] == 'sw'
    assert value['layers'] == 2
    assert value['set_label'] == 's2'


def test_a_load_range_with_only_a_minimum_uses_the_high_sentinel(bar):
    bar.load_min.setText('20')

    assert bar.value()['force_range'] == (20.0, HIGH_SENTINEL)
    assert bar.value()['pressure_range'] is None


def test_the_load_range_follows_the_unit_combo(bar):
    """A pressure-loaded family has a null force column: sending the range to
    force_range would hide every sheet-like run instead of filtering it."""
    bar.load_min.setText('1.0')
    bar.load_max.setText('2.0')
    bar.load_kind.setCurrentIndex(1)

    value = bar.value()

    assert bar.load_kind.currentData() == 'pressure'
    assert value['pressure_range'] == (1.0, 2.0)
    assert value['force_range'] is None


def test_an_empty_load_range_sets_neither_key(bar):
    bar.load_kind.setCurrentIndex(1)

    assert bar.value()['force_range'] is None
    assert bar.value()['pressure_range'] is None


def test_set_value_restores_a_pressure_range_as_a_pressure_range(bar):
    bar.set_value(dict(EMPTY_FILTERS, pressure_range=(1.0, 2.0)))

    assert bar.load_kind.currentData() == 'pressure'
    assert bar.value()['pressure_range'] == (1.0, 2.0)
    assert bar.value()['force_range'] is None


def test_a_temperature_range_with_only_a_maximum_uses_the_low_sentinel(bar):
    bar.temp_max.setText('350')

    assert bar.value()['temperature_range'] == (LOW_SENTINEL, 350.0)


def test_unparseable_range_text_is_ignored_rather_than_raising(bar):
    bar.load_min.setText('lots')

    assert bar.value()['force_range'] is None


def test_the_open_ended_sentinels_are_plus_minus_1e30(bar):
    """The sentinels are what the worker receives for a half-filled range."""
    assert (LOW_SENTINEL, HIGH_SENTINEL) == (-1e30, 1e30)


def test_set_value_restores_an_open_end_as_an_empty_box(bar):
    bar.set_value(dict(EMPTY_FILTERS, force_range=(20.0, HIGH_SENTINEL),
                       temperature_range=(LOW_SENTINEL, 350.0)))

    assert (bar.load_min.text(), bar.load_max.text()) == ('20.0', '')
    assert (bar.temp_min.text(), bar.temp_max.text()) == ('', '350.0')


def test_set_value_round_trips_a_stored_filter_block(bar):
    stored = dict(EMPTY_FILTERS, family='afm', materials=['h-MoS2'], layers=1,
                  force_range=(10.0, 40.0))

    bar.set_value(stored)

    assert bar.value() == stored


def test_a_remembered_value_survives_a_combo_with_no_choices_yet(qt_app):
    """Restoring D10 state happens at construction, before any query has
    filled the combos. A cold start must not drop the remembered filter."""
    del qt_app
    cold = FilterBar()

    cold.set_value(dict(EMPTY_FILTERS, family='hetero', set_label='s9'))

    assert cold.value()['family'] == 'hetero'
    assert cold.value()['set_label'] == 's9'


def test_facets_replace_a_remembered_value_the_store_does_not_have(bar):
    bar.set_value(dict(EMPTY_FILTERS, family='gone'))

    bar.set_choices({'families': ['afm', 'hetero']})

    assert bar.value()['family'] is None


def test_set_choices_keeps_a_selection_that_still_exists(bar):
    bar.family.setCurrentText('hetero')

    bar.set_choices({'families': ['afm', 'hetero'], 'materials': []})

    assert bar.family.currentText() == 'hetero'


def test_changing_a_control_emits_changed_once(bar):
    seen = []
    bar.changed.connect(lambda: seen.append(1))

    bar.family.setCurrentText('afm')

    assert len(seen) == 1


def test_clear_returns_the_bar_to_empty(bar):
    bar.family.setCurrentText('afm')
    bar.search.setText('x')

    bar.clear()

    assert bar.value() == EMPTY_FILTERS
    assert bar.text() == ''


def test_no_two_grid_widgets_share_a_cell(bar):
    """A row-2 regression: a widened search box once overlapped the T (K)
    controls that sit at columns 4-6 of the same row. Any two widgets
    claiming the same cell would silently clip one of them."""
    grid = bar.layout()
    covered = set()
    for i in range(grid.count()):
        row, column, row_span, col_span = grid.getItemPosition(i)
        cells = {(r, c) for r in range(row, row + row_span)
                 for c in range(column, column + col_span)}
        assert covered.isdisjoint(cells), f"grid item {i} overlaps an earlier one"
        covered |= cells
