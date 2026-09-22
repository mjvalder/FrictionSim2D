"""The first model/view pair in this codebase.

Every other table here is a `QTableWidget`, which cannot sort thousands of
rows by a raw value or filter without rebuilding itself. The rules this pins:
the visible text is formatted and unit-aware, the *sort* is on the raw value,
and a missing field renders as an em dash rather than 'None'.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip('PySide6')

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QModelIndex, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.gui.widgets.database.model import (  # noqa: E402
    CURATED_COLUMNS, SORT_ROLE, Column, ResultSortProxyModel, ResultTableModel,
    columns_for_record, display_value, record_value, sort_value)


@pytest.fixture(name='qt_app', scope='module')
def _qt_app():
    return QApplication.instance() or QApplication([])


RECORDS = [
    {'set_label': 's1', 'family': 'afm', 'materials': ['h-MoS2'], 'layers': 2,
     'load_kind': 'force', 'load': 30.0, 'angle': 0.0, 'speed': 2.0,
     'temperature': 300.0, 'mean_cof': 0.0731, 'std_cof': 0.0044,
     'is_complete': True, 'imported_at': '2026-09-01T10:00:00',
     'run_uuid': '1a2b', 'result_uuid': '9f8e'},
    {'set_label': 's1', 'family': 'hetero', 'materials': ['h-MoS2', 'h-WS2'],
     'layers': None, 'load_kind': 'pressure', 'load': 1.5, 'angle': 30.0,
     'speed': 2.0, 'temperature': 300.0, 'mean_cof': 0.0125, 'std_cof': 0.0009,
     'is_complete': False, 'imported_at': '2026-09-02T10:00:00',
     'run_uuid': '3c4d', 'result_uuid': '7b6a'},
]


class TestColumnFormatting:

    def test_a_list_renders_as_a_stack(self):
        column = next(c for c in CURATED_COLUMNS if c.key == 'materials')
        assert display_value(RECORDS[1], column) == 'h-MoS2+h-WS2'

    def test_the_load_unit_follows_load_kind(self):
        column = next(c for c in CURATED_COLUMNS if c.key == 'load')
        assert display_value(RECORDS[0], column) == '30.00 nN'
        assert display_value(RECORDS[1], column) == '1.50 GPa'

    def test_a_missing_value_is_an_em_dash_not_none(self):
        column = next(c for c in CURATED_COLUMNS if c.key == 'layers')
        assert display_value(RECORDS[1], column) == '—'

    def test_a_missing_cof_is_an_em_dash_not_a_perfect_zero(self):
        """The foundation's sentinel for "could not be computed" is None.
        Rendering it as 0.0000 would put a flawless frictionless run in the
        table -- the most expensive kind of wrong."""
        column = next(c for c in CURATED_COLUMNS if c.key == 'mean_cof')
        assert display_value({'mean_cof': None}, column) == '—'
        assert display_value({'mean_cof': 0.0}, column) == '0.0000'

    def test_a_central_row_reads_its_own_field_names(self):
        """`ResultRecord` has force_nN/pressure_gpa/scan_angle/scan_speed/
        material where a local row has load/angle/speed/materials."""
        central = {'material': 'h_MoS2', 'force_nN': 30.0, 'scan_angle': 30.0,
                   'scan_speed': 2.0}
        by_key = {c.key: c for c in CURATED_COLUMNS}

        assert display_value(central, by_key['materials']) == 'h_MoS2'
        assert display_value(central, by_key['load']) == '30.00 nN'
        assert display_value(central, by_key['angle']) == '30.0 °'
        assert display_value(central, by_key['speed']) == '2.00 Å/ps'

    def test_a_central_pressure_row_is_labelled_GPa(self):
        column = next(c for c in CURATED_COLUMNS if c.key == 'load')
        assert display_value({'pressure_gpa': 1.5}, column) == '1.50 GPa'

    def test_status_falls_back_to_is_complete_with_its_own_words(self):
        column = next(c for c in CURATED_COLUMNS if c.key == 'status')
        assert display_value(RECORDS[0], column) == 'complete'
        assert display_value(RECORDS[1], column) == 'incomplete'

    def test_status_shows_an_incomplete_suffix_for_a_local_row(self):
        """Every real local row carries its own `status` ('imported'), so
        the is_complete fallback above never fires for it -- but a local row
        can still be marked incomplete, and that must not vanish."""
        column = next(c for c in CURATED_COLUMNS if c.key == 'status')
        record = {'status': 'imported', 'is_complete': False}
        assert display_value(record, column) == 'imported (incomplete)'

    def test_status_shows_an_incomplete_suffix_for_a_central_row(self):
        """A central row's own status word is 'staged' or 'published'."""
        column = next(c for c in CURATED_COLUMNS if c.key == 'status')
        record = {'status': 'published', 'is_complete': False}
        assert display_value(record, column) == 'published (incomplete)'

    def test_the_added_column_falls_back_from_uploaded_to_imported(self):
        column = next(c for c in CURATED_COLUMNS if c.key == 'uploaded_at')
        assert column.fallbacks == ('imported_at',)
        assert display_value(RECORDS[0], column) == '2026-09-01T10:00:00'

    def test_record_value_prefers_the_primary_key(self):
        column = Column('a', 'A', fallbacks=('b',))
        assert record_value({'a': 1, 'b': 2}, column) == 1
        assert record_value({'b': 2}, column) == 2
        assert record_value({}, column) is None

    def test_sort_value_is_always_comparable(self):
        column = next(c for c in CURATED_COLUMNS if c.key == 'materials')
        assert sort_value(RECORDS[1], column) == 'h-MoS2+h-WS2'
        assert sort_value({}, column) == ''


class TestResultTableModel:

    @pytest.fixture(name='model')
    def _model(self, qt_app):
        del qt_app
        model = ResultTableModel()
        model.set_records(RECORDS)
        return model

    def test_it_shows_the_curated_columns_by_default(self, model):
        assert model.columnCount() == len(CURATED_COLUMNS)
        assert model.rowCount() == 2
        titles = [model.headerData(i, Qt.Horizontal, Qt.DisplayRole)
                  for i in range(model.columnCount())]
        assert titles[:3] == ['Set', 'Family', 'Materials']

    def test_all_columns_shows_every_field_of_a_record(self, model):
        """The curated Status/Added columns come from is_complete/imported_at
        fallbacks while is_complete/imported_at also appear as their own
        extra columns, so the column count exceeds the record's key count --
        assert the record's keys are covered and the columns are unique,
        not a length match."""
        model.set_all_columns(True)
        keys = [c.key for c in model.columns]
        assert 'run_uuid' in keys
        assert 'result_uuid' in keys
        assert set(RECORDS[0]) <= set(keys)
        assert len(keys) == len(set(keys))

    def test_all_columns_puts_the_curated_ones_first(self, model):
        model.set_all_columns(True)
        keys = [c.key for c in model.columns]
        assert keys[0] == 'set_label'
        assert keys.index('set_label') < keys.index('run_uuid')

    def test_display_role_is_formatted_and_sort_role_is_raw(self, model):
        load = [c.key for c in model.columns].index('load')
        index = model.index(0, load)
        assert model.data(index, Qt.DisplayRole) == '30.00 nN'
        assert model.data(index, SORT_ROLE) == 30.0

    def test_numbers_are_right_aligned(self, model):
        load = [c.key for c in model.columns].index('load')
        alignment = model.data(model.index(0, load), Qt.TextAlignmentRole)
        assert alignment == int(Qt.AlignRight | Qt.AlignVCenter)

    def test_record_at_hands_back_the_original_dict(self, model):
        assert model.record_at(1)['run_uuid'] == '3c4d'

    def test_a_child_index_has_no_rows(self, model):
        assert model.rowCount(model.index(0, 0)) == 0
        assert model.columnCount(model.index(0, 0)) == 0

    def test_columns_for_record_is_stable_for_an_empty_record(self):
        assert columns_for_record({}) == CURATED_COLUMNS


class TestResultSortProxyModel:

    @pytest.fixture(name='proxy')
    def _proxy(self, qt_app):
        del qt_app
        model = ResultTableModel()
        model.set_records(RECORDS)
        proxy = ResultSortProxyModel()
        proxy.setSourceModel(model)
        return proxy

    def test_sorting_uses_the_raw_value_not_the_formatted_text(self, proxy):
        cof = [c.key for c in proxy.sourceModel().columns].index('mean_cof')
        proxy.sort(cof, Qt.AscendingOrder)
        first = proxy.data(proxy.index(0, cof), SORT_ROLE)
        second = proxy.data(proxy.index(1, cof), SORT_ROLE)
        assert first == 0.0125
        assert second == 0.0731

    def test_the_text_filter_matches_any_field_case_insensitively(self, proxy):
        proxy.set_text_filter('h-ws2')
        assert proxy.rowCount() == 1
        proxy.set_text_filter('')
        assert proxy.rowCount() == 2

    def test_the_text_filter_matches_fields_no_column_shows(self, proxy):
        proxy.set_text_filter('7b6a')
        assert proxy.rowCount() == 1

    def test_an_empty_filter_accepts_every_row(self, proxy):
        assert proxy.filterAcceptsRow(0, QModelIndex()) is True


class TestStatusColour:
    """No inline hex anywhere: the three tokens live in style.py."""

    def test_status_level_maps_the_stored_vocabulary(self):
        from src.gui.widgets.database.model import status_level

        assert status_level({'status': 'completed'}) == 'ok'
        assert status_level({'status': 'published'}) == 'ok'
        assert status_level({'status': 'running'}) == 'warn'
        assert status_level({'status': 'staged'}) == 'warn'
        assert status_level({'status': 'failed'}) == 'error'
        assert status_level({'status': 'rejected'}) == 'error'

    def test_imported_the_only_local_status_is_ok(self):
        from src.gui.widgets.database.model import status_level

        assert status_level({'status': 'imported'}) == 'ok'

    def test_an_incomplete_run_without_a_status_still_warns(self):
        from src.gui.widgets.database.model import status_level

        assert status_level({'is_complete': False}) == 'warn'
        assert status_level({'is_complete': True}) == 'ok'
        assert status_level({}) == 'ok'

    def test_an_incomplete_local_import_warns_despite_its_ok_status(self):
        # N6: 'imported' + is_complete False -> 'warn', not the 'ok' the
        # status word alone would give.
        from src.gui.widgets.database.model import status_level

        assert status_level({'status': 'imported', 'is_complete': False}) == 'warn'

    def test_an_incomplete_published_row_warns_despite_its_ok_status(self):
        # N6: 'published' + is_complete False -> 'warn', same reasoning for
        # the central source's shape.
        from src.gui.widgets.database.model import status_level

        assert status_level({'status': 'published', 'is_complete': False}) == 'warn'

    def test_an_incomplete_failed_run_stays_an_error(self):
        # N6: error outranks incomplete -- demoting a failed run to 'warn'
        # would hide the more serious fact.
        from src.gui.widgets.database.model import status_level

        assert status_level({'status': 'failed', 'is_complete': False}) == 'error'

    def test_the_row_is_painted_with_the_style_token(self, qt_app):
        del qt_app
        from PySide6.QtGui import QColor
        from src.gui.style import STATUS_ERROR_COLOR

        model = ResultTableModel()
        model.set_records([dict(RECORDS[0], status='failed')])

        brush = model.data(model.index(0, 0), Qt.ForegroundRole)

        assert brush.color() == QColor(STATUS_ERROR_COLOR)

    def test_an_ok_row_is_left_to_the_palette(self, qt_app):
        del qt_app
        model = ResultTableModel()
        model.set_records([dict(RECORDS[0], status='completed')])

        assert model.data(model.index(0, 0), Qt.ForegroundRole) is None
