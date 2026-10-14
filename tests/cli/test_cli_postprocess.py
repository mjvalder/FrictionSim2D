"""The postprocess CLI over a real synthetic tree."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from src.cli import cli
from tests.helpers.results_tree import make_family_run


@pytest.fixture(name='runner')
def _runner():
    return CliRunner()


def test_read_export_writes_json_and_csv(runner, tmp_path):
    make_family_run(tmp_path, 'afm', loads=(10.0, 30.0))

    result = runner.invoke(cli, ['postprocess', 'read', str(tmp_path),
                                 '--export'])

    assert result.exit_code == 0, result.output
    outputs = tmp_path / 'outputs'
    assert (outputs / 'output_full_100x100y.json').is_file()
    assert (outputs / 'summary_100x100y.csv').is_file()
    assert 'output_full_100x100y.json' in result.output
    assert 'summary_100x100y.csv' in result.output


def test_read_without_export_writes_no_data_files(runner, tmp_path):
    make_family_run(tmp_path, 'afm')

    result = runner.invoke(cli, ['postprocess', 'read', str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert not (tmp_path / 'outputs' / 'output_full_100x100y.json').exists()


def test_read_reports_how_many_runs_it_found(runner, tmp_path):
    make_family_run(tmp_path, 'afm', loads=(10.0, 30.0), angles=(0.0, 90.0))

    result = runner.invoke(cli, ['postprocess', 'read', str(tmp_path)])

    assert '4 run' in result.output


def test_skip_fraction_reaches_the_summary(runner, tmp_path):
    import pandas as pd

    make_family_run(tmp_path, 'afm', n=50)

    runner.invoke(cli, ['postprocess', 'read', str(tmp_path), '--export',
                        '--skip-fraction', '0.5'])

    frame = pd.read_csv(tmp_path / 'outputs' / 'summary_100x100y.csv')
    assert frame['skip_fraction'].iloc[0] == 0.5


def test_read_on_an_empty_root_says_so(runner, tmp_path):
    result = runner.invoke(cli, ['postprocess', 'read', str(tmp_path),
                                 '--export'])

    assert result.exit_code == 0
    assert 'No runs found' in result.output


def test_the_exported_json_is_schema_three(runner, tmp_path):
    make_family_run(tmp_path, 'sheetonsheet')

    runner.invoke(cli, ['postprocess', 'read', str(tmp_path), '--export'])

    doc = json.loads(
        (tmp_path / 'outputs' / 'output_full_100x100y.json').read_text(
            encoding='utf-8'))
    assert doc['schema'] == 3


def _document(tmp_path, data_dir):
    doc = tmp_path / 'plots.json'
    doc.write_text(json.dumps({
        'data_dirs': [str(data_dir)], 'labels': ['ds'],
        'plots': [{'plot_type': 'summary', 'x_axis': 'force', 'y_axis': 'lf',
                   'filter_size': '100x100y', 'filename': 'lf.png'}],
    }), encoding='utf-8')
    return doc


def test_validate_accepts_a_good_document(runner, tmp_path):
    make_family_run(tmp_path / 'ds', 'afm', loads=(10.0, 30.0))
    runner.invoke(cli, ['postprocess', 'read', str(tmp_path / 'ds'), '--export'])
    doc = _document(tmp_path, tmp_path / 'ds')

    result = runner.invoke(cli, ['postprocess', 'plot', str(doc), '--validate'])

    assert result.exit_code == 0, result.output
    assert '1 plot' in result.output
    assert not (tmp_path / 'plots').exists()      # nothing rendered


def test_validate_refuses_an_unknown_key(runner, tmp_path):
    doc = tmp_path / 'bad.json'
    doc.write_text(json.dumps({
        'data_dirs': ['a'], 'labels': ['a'],
        'plots': [{'plot_type': 'summary', 'wobble': 1}]}), encoding='utf-8')

    result = runner.invoke(cli, ['postprocess', 'plot', str(doc), '--validate'])

    assert result.exit_code != 0
    assert 'wobble' in result.output


def test_validate_refuses_mismatched_labels(runner, tmp_path):
    doc = tmp_path / 'bad.json'
    doc.write_text(json.dumps({
        'data_dirs': ['a', 'b'], 'labels': ['a'], 'plots': []}),
        encoding='utf-8')

    result = runner.invoke(cli, ['postprocess', 'plot', str(doc), '--validate'])

    assert result.exit_code != 0
    assert 'labels' in result.output


def test_preset_is_applied_and_settings_win_over_it(runner, tmp_path,
                                                    monkeypatch):
    seen = {}

    class _Plotter:
        def __init__(self, data_dirs, labels, output_dir, settings=None, **kw):
            seen['settings'] = settings
            self.written = []

        def generate_plot(self, spec):
            pass

    import src.postprocessing as pp
    monkeypatch.setattr(pp, 'Plotter', _Plotter, raising=False)

    doc = _document(tmp_path, tmp_path / 'ds')
    override = tmp_path / 'override.json'
    override.write_text(json.dumps({'figure': {'dpi': 111}}), encoding='utf-8')

    result = runner.invoke(cli, ['postprocess', 'plot', str(doc),
                                 '--preset', 'publication',
                                 '--settings', str(override),
                                 '-o', str(tmp_path / 'plots')])

    assert result.exit_code == 0, result.output
    assert seen['settings']['figure']['dpi'] == 111
    assert seen['settings']['export']['formats'] == ['pdf', 'svg', 'png']
    assert seen['settings']['layout']['constrained_layout'] is True


def test_an_unknown_preset_is_refused(runner, tmp_path):
    doc = _document(tmp_path, tmp_path / 'ds')

    result = runner.invoke(cli, ['postprocess', 'plot', str(doc),
                                 '--preset', 'wallpaper', '--validate'])

    assert result.exit_code != 0
    assert 'wallpaper' in result.output


def test_init_writes_a_document_that_validates(runner, tmp_path):
    """"Rapid publication-quality plots" cannot start with "hand-write this
    JSON". --init is the scaffold."""
    from src.postprocessing.plot_config import load_plot_document

    make_family_run(tmp_path / 'ds', 'afm', loads=(10.0, 30.0))
    runner.invoke(cli, ['postprocess', 'read', str(tmp_path / 'ds'), '--export'])
    target = tmp_path / 'plots.json'

    result = runner.invoke(cli, ['postprocess', 'plot', '--init', str(target),
                                 '--data-dir', str(tmp_path / 'ds')])

    assert result.exit_code == 0, result.output
    assert target.is_file()
    document = load_plot_document(target)
    assert document.labels == ['ds']
    assert [p.plot_type for p in document.plots] == ['summary', 'summary']
    assert 'preset publication' in result.output


def test_the_document_init_writes_actually_renders(runner, tmp_path):
    make_family_run(tmp_path / 'ds', 'afm', loads=(10.0, 30.0, 50.0))
    runner.invoke(cli, ['postprocess', 'read', str(tmp_path / 'ds'), '--export'])
    target = tmp_path / 'plots.json'
    runner.invoke(cli, ['postprocess', 'plot', '--init', str(target),
                        '--data-dir', str(tmp_path / 'ds')])

    result = runner.invoke(cli, ['postprocess', 'plot', str(target),
                                 '-o', str(tmp_path / 'plots')])

    assert result.exit_code == 0, result.output
    assert (tmp_path / 'plots' / 'lf_vs_load.png').is_file()
    assert (tmp_path / 'plots' / 'cof_vs_load.png').is_file()


def test_no_document_and_no_init_says_how_to_make_one(runner):
    result = runner.invoke(cli, ['postprocess', 'plot'])

    assert result.exit_code != 0
    assert '--init' in result.output


def test_schema_prints_the_models_and_needs_no_document(runner):
    result = runner.invoke(cli, ['postprocess', 'plot', '--schema'])

    assert result.exit_code == 0
    assert '`summary`' in result.output
    assert '`stick_slip_analysis`' in result.output
    assert 'filter_size' in result.output


def test_plot_renders_and_reports_what_it_wrote(runner, tmp_path):
    make_family_run(tmp_path / 'ds', 'afm', loads=(10.0, 30.0, 50.0))
    runner.invoke(cli, ['postprocess', 'read', str(tmp_path / 'ds'), '--export'])
    doc = _document(tmp_path, tmp_path / 'ds')

    result = runner.invoke(cli, ['postprocess', 'plot', str(doc),
                                 '-o', str(tmp_path / 'plots')])

    assert result.exit_code == 0, result.output
    assert (tmp_path / 'plots' / 'lf.png').is_file()
    assert 'lf.png' in result.output
