"""`FrictionSim2D analyse` list/run/show/export, with the store monkeypatched."""
from types import SimpleNamespace

import pandas as pd
import pytest
from click.testing import CliRunner
from pydantic import BaseModel

from src.analysis import base, store
from src.analysis.base import Analysis, register
from src.cli import cli

TABLE = pd.DataFrame({'material': ['h_MoS2', 'h_WS2'], 'cof': [0.1, 0.2]})


@pytest.fixture
def probe(monkeypatch):
    """A registered throwaway analysis and a fake store around it."""
    monkeypatch.setattr(base, 'ANALYSES', dict(base.ANALYSES))

    @register
    class _Probe(Analysis):  # pylint: disable=unused-variable
        name = 'probe'
        version = '3'
        level = 'run'
        families = frozenset({'afm', 'sheetonsheet'})

        class Params(BaseModel):
            """Two parameters."""
            window: int = 5
            mode: str = 'a'

        def compute(self, source, params):
            return TABLE

    calls = []
    monkeypatch.setattr('src.aiida.load_aiida_profile', lambda profile=None: calls.append(profile))

    def fake_run(name, dataset, params=None, *, force=False):
        calls.append((name, dataset, params, force))
        return TABLE

    node = SimpleNamespace(
        uuid='uuid-1', get_content=lambda mode='rb': TABLE.to_csv(index=False).encode(),
        base=SimpleNamespace(extras=SimpleNamespace(
            all={'n_rows': 2, 'n_dropped': 1, 'version': '3', 'dataset': 'paper'})))
    monkeypatch.setattr(store, 'run_analysis', fake_run)
    monkeypatch.setattr(store, 'find_result_node', lambda *a, **k: node)
    return calls


def test_list(probe):
    result = CliRunner().invoke(cli, ['analyse', 'list'])
    assert result.exit_code == 0, result.output
    assert 'probe' in result.output and '3' in result.output and 'run' in result.output
    assert 'afm' in result.output and 'sheetonsheet' in result.output


def test_run_parses_params_and_reports(probe):
    result = CliRunner().invoke(cli, ['analyse', 'run', 'probe', 'paper', '--param', 'window=7',
                                      '--force', '--profile', 'p'])
    assert result.exit_code == 0, result.output
    assert probe[0] == 'p'
    assert probe[1] == ('probe', 'paper', {'window': 7}, True)
    assert 'rows: 2' in result.output and 'dropped: 1' in result.output
    assert 'uuid-1' in result.output


def test_run_bad_param_value_names_the_parameter(probe):
    result = CliRunner().invoke(cli, ['analyse', 'run', 'probe', 'paper', '--param', 'window=abc'])
    assert result.exit_code != 0
    assert 'window' in result.output


def test_run_unknown_param_and_malformed(probe):
    result = CliRunner().invoke(cli, ['analyse', 'run', 'probe', 'paper', '--param', 'bogus=1'])
    assert result.exit_code != 0 and 'bogus' in result.output
    result = CliRunner().invoke(cli, ['analyse', 'run', 'probe', 'paper', '--param', 'window'])
    assert result.exit_code != 0 and 'window' in result.output


def test_run_unknown_analysis(probe):
    result = CliRunner().invoke(cli, ['analyse', 'run', 'nope', 'paper'])
    assert result.exit_code != 0 and 'nope' in result.output


def test_show(probe):
    result = CliRunner().invoke(cli, ['analyse', 'show', 'probe', 'paper'])
    assert result.exit_code == 0, result.output
    assert 'uuid-1' in result.output and 'h_WS2' in result.output
    assert 'n_dropped' in result.output


def test_show_missing(probe, monkeypatch):
    monkeypatch.setattr(store, 'find_result_node', lambda *a, **k: None)
    result = CliRunner().invoke(cli, ['analyse', 'show', 'probe', 'paper'])
    assert result.exit_code != 0 and 'no stored result' in result.output


def test_export(probe, tmp_path):
    out = tmp_path / 'out.csv'
    result = CliRunner().invoke(cli, ['analyse', 'export', 'probe', 'paper', str(out)])
    assert result.exit_code == 0, result.output
    assert pd.read_csv(out).equals(TABLE)
