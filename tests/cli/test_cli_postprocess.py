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


def test_the_exported_json_is_schema_two(runner, tmp_path):
    make_family_run(tmp_path, 'sheetonsheet')

    runner.invoke(cli, ['postprocess', 'read', str(tmp_path), '--export'])

    doc = json.loads(
        (tmp_path / 'outputs' / 'output_full_100x100y.json').read_text(
            encoding='utf-8'))
    assert doc['schema'] == 2
