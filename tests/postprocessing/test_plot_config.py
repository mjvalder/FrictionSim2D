"""The plot schema (Q2): one model per plot type, unknown keys are errors."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.postprocessing.plot_config import (PLOT_TYPES, CofHistogramPlot,
                                            PlotDocument, StickSlipPlot,
                                            SummaryPlot, load_plot_document,
                                            schema_markdown)

PUBLICATION = (Path(__file__).resolve().parents[2] / 'documentation'
               / 'publication' / 'plots_publication.json')


def test_the_shipped_publication_document_validates():
    """The project's own published plot document is the acceptance test. A
    schema that rejects it is a schema nobody will adopt."""
    document = load_plot_document(PUBLICATION)

    assert len(document.plots) == len(
        json.loads(PUBLICATION.read_text(encoding='utf-8'))['plots'])
    assert len(document.data_dirs) == len(document.labels)
    assert {p.plot_type for p in document.plots} <= set(PLOT_TYPES)


def test_round_trip_preserves_every_key_that_was_set():
    raw = json.loads(PUBLICATION.read_text(encoding='utf-8'))

    document = PlotDocument.model_validate(raw)
    back = document.model_dump(exclude_unset=True)

    assert back['plots'] == raw['plots']
    assert back['data_dirs'] == raw['data_dirs']


def test_an_unknown_key_is_an_error():
    with pytest.raises(ValidationError, match='wobble'):
        SummaryPlot(plot_type='summary', filename='x.png', wobble=3)


def test_an_unknown_plot_type_is_an_error():
    with pytest.raises(ValidationError):
        PlotDocument(data_dirs=['a'], labels=['a'],
                     plots=[{'plot_type': 'sunburst', 'filename': 'x.png'}])


def test_the_discriminator_picks_the_right_model():
    document = PlotDocument(data_dirs=['a'], labels=['a'], plots=[
        {'plot_type': 'summary', 'filename': 's.png', 'add_linear_fit': True},
        {'plot_type': 'stick_slip_analysis', 'filename': 'k.png',
         'fft_peak_rank': 2},
        {'plot_type': 'cof_histogram', 'filename': 'h.png', 'bar_width': 0.5},
    ])

    assert isinstance(document.plots[0], SummaryPlot)
    assert isinstance(document.plots[1], StickSlipPlot)
    assert isinstance(document.plots[2], CofHistogramPlot)
    assert document.plots[1].fft_peak_rank == 2


def test_stick_slip_is_spelled_as_the_dispatcher_spells_it():
    """Plotter.generate_plot dispatches on 'stick_slip_analysis'. A model
    keyed 'stick_slip' would validate documents the Plotter then ignores."""
    assert 'stick_slip_analysis' in PLOT_TYPES
    assert 'stick_slip' not in PLOT_TYPES


def test_a_summary_key_on_a_timeseries_plot_is_refused():
    with pytest.raises(ValidationError, match='potential_markers'):
        PlotDocument(data_dirs=['a'], labels=['a'], plots=[
            {'plot_type': 'timeseries', 'filename': 't.png',
             'potential_markers': {'sw': 'o'}}])


def test_base_fields_are_available_on_every_model():
    for plot_type in PLOT_TYPES:
        document = PlotDocument(data_dirs=['a'], labels=['a'], plots=[
            {'plot_type': plot_type, 'filename': 'x.png',
             'filter_size': '100x100y', 'tier': 'window', 'family': 'afm'}])
        assert document.plots[0].filter_size == '100x100y'
        assert document.plots[0].tier == 'window'
        assert document.plots[0].family == 'afm'


def test_tier_is_constrained():
    with pytest.raises(ValidationError, match='tier'):
        SummaryPlot(plot_type='summary', filename='x.png', tier='hourly')


def test_labels_must_match_data_dirs():
    with pytest.raises(ValidationError, match='labels'):
        PlotDocument(data_dirs=['a', 'b'], labels=['a'], plots=[])


def test_external_series_and_source_specs_are_typed():
    plot = SummaryPlot(plot_type='summary', filename='x.png', external_series=[
        {'file': 'd.json', 'path': ['cofvsload'], 'x_key': 'pressure',
         'y_key': 'cof', 'label': 'SW - Hu', 'plot_style': 'line',
         'line_style': '--', 'x_transform': 'lambda x: x * 98'}])

    assert plot.external_series[0].label == 'SW - Hu'
    assert plot.external_series[0].path == ['cofvsload']


def test_schema_markdown_names_every_type_and_field():
    text = schema_markdown()

    for plot_type in PLOT_TYPES:
        assert f'`{plot_type}`' in text
    assert 'filter_size' in text
    assert 'fft_peak_rank' in text
