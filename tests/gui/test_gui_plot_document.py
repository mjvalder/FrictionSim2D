"""The plot editor's document is Qt-free and schema-driven (G1).

No PySide6 import anywhere in this file: the document duck-types
ConfigDocument the way SettingsDocument does, and it must be testable --
and importable -- on a machine with no Qt at all.
"""

from __future__ import annotations

from src.gui.controllers.plot_document import (COMMON_SECTION,
                                               MODEL_BY_PLOT_TYPE,
                                               PlotEditorDocument)
from src.gui.schema.overrides import PLOT_MODEL, overrides_for
from src.postprocessing.plot_config import PLOT_TYPES, SummaryPlot


def test_every_plot_type_has_a_model():
    assert set(MODEL_BY_PLOT_TYPE) == set(PLOT_TYPES)


def test_a_blank_document_has_two_sections():
    doc = PlotEditorDocument.blank('summary')

    keys = [s.key for s in doc.sections()]

    assert keys == [COMMON_SECTION, 'summary']
    assert doc.model == PLOT_MODEL


def test_the_common_section_holds_the_base_fields_only():
    doc = PlotEditorDocument.blank('summary')

    common = doc.section(COMMON_SECTION)
    names = {f.name for f in common.fields}

    assert 'filter_size' in names
    assert 'tier' in names
    assert 'add_linear_fit' not in names      # that is summary's own
    assert 'plot_type' not in names           # set by the type selector


def test_the_specific_section_holds_only_its_own_fields():
    doc = PlotEditorDocument.blank('stick_slip_analysis')

    specific = doc.section('stick_slip_analysis')
    names = {f.name for f in specific.fields}

    assert 'fft_peak_rank' in names
    assert 'filter_size' not in names


def test_get_and_set_route_by_section():
    doc = PlotEditorDocument.blank('summary')

    doc.set(COMMON_SECTION, 'filter_size', '100x100y')
    doc.set('summary', 'add_linear_fit', True)

    assert doc.get(COMMON_SECTION, 'filter_size') == '100x100y'
    assert doc.get('summary', 'add_linear_fit') is True
    assert doc.dirty is True


def test_setting_the_same_value_does_not_dirty_the_document():
    doc = PlotEditorDocument.blank('summary')
    doc.set(COMMON_SECTION, 'filter_size', '100x100y')
    doc.dirty = False

    doc.set(COMMON_SECTION, 'filter_size', '100x100y')

    assert doc.dirty is False


def test_to_plot_returns_the_right_model_with_only_what_was_set():
    doc = PlotEditorDocument.blank('summary')
    doc.set(COMMON_SECTION, 'filename', 'lf.png')
    doc.set('summary', 'add_linear_fit', True)

    plot = doc.to_plot()

    assert isinstance(plot, SummaryPlot)
    assert plot.filename == 'lf.png'
    assert plot.add_linear_fit is True
    assert plot.model_dump(exclude_unset=True) == {
        'plot_type': 'summary', 'filename': 'lf.png', 'add_linear_fit': True}


def test_from_plot_round_trips():
    # plot_type is passed EXPLICITLY: model_dump(exclude_unset=True) omits a
    # Literal that was merely defaulted, while to_plot() always sets it (it is
    # the key the dispatcher needs). Leaving it unset here would compare
    # {'filename': ...} against {'plot_type': 'summary', 'filename': ...}.
    plot = SummaryPlot(plot_type='summary', filename='x.png',
                       filter_size='100x100y', add_linear_fit=True)

    doc = PlotEditorDocument.from_plot(plot)

    assert doc.plot_type == 'summary'
    assert doc.get(COMMON_SECTION, 'filter_size') == '100x100y'
    assert doc.to_plot().model_dump(exclude_unset=True) == \
        plot.model_dump(exclude_unset=True)


def test_changing_the_plot_type_keeps_the_common_values():
    doc = PlotEditorDocument.blank('summary')
    doc.set(COMMON_SECTION, 'filter_size', '100x100y')
    doc.set('summary', 'add_linear_fit', True)

    doc.set_plot_type('timeseries')

    assert doc.get(COMMON_SECTION, 'filter_size') == '100x100y'
    assert [s.key for s in doc.sections()] == [COMMON_SECTION, 'timeseries']
    assert doc.to_plot().plot_type == 'timeseries'
    assert 'add_linear_fit' not in doc.to_plot().model_dump(exclude_unset=True)


def test_validate_reports_the_offending_field():
    doc = PlotEditorDocument.blank('summary')
    doc.values[COMMON_SECTION]['tier'] = 'hourly'

    result = doc.validate()

    assert result.issues
    assert any(issue.key == 'tier' for issue in result.issues)


def test_validate_passes_on_a_good_document():
    doc = PlotEditorDocument.blank('summary')
    doc.set(COMMON_SECTION, 'filename', 'x.png')

    assert doc.validate().issues == []


def test_sections_are_never_optional_and_never_unlocked():
    doc = PlotEditorDocument.blank('correlation')

    for spec in doc.sections():
        assert doc.has_section(spec.key)
        assert doc.is_section_locked(spec.key) is True
        assert doc.section_lock_reason(spec.key)


def test_the_override_table_labels_every_section():
    table = overrides_for(PLOT_MODEL)

    assert COMMON_SECTION in table
    for plot_type in PLOT_TYPES:
        assert plot_type in table, plot_type
        assert table[plot_type].label


def test_overrides_for_an_unrelated_model_is_unaffected():
    assert COMMON_SECTION not in overrides_for('afm')
