"""Smoke tests for the FrictionSim2D CLI.

These tests verify that all command groups and key commands are reachable and
produce sensible help text, without requiring LAMMPS, AiiDA, or a database.
"""

import json

import pytest
import yaml
from click.testing import CliRunner
from src.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def test_cli_help(runner):
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'FrictionSim2D' in result.output


def test_cli_version(runner):
    result = runner.invoke(cli, ['--version'])
    assert result.exit_code == 0
    assert '0.1.0' in result.output


# ---------------------------------------------------------------------------
# run group
# ---------------------------------------------------------------------------

def test_run_help(runner):
    result = runner.invoke(cli, ['run', '--help'])
    assert result.exit_code == 0
    assert 'afm' in result.output
    assert 'sheetonsheet' in result.output


def test_run_afm_help(runner):
    result = runner.invoke(cli, ['run', 'afm', '--help'])
    assert result.exit_code == 0
    assert 'CONFIG_FILE' in result.output
    assert '--output-dir' in result.output
    assert '--aiida' in result.output
    assert '--hpc-scripts' in result.output


def test_run_sheetonsheet_help(runner):
    result = runner.invoke(cli, ['run', 'sheetonsheet', '--help'])
    assert result.exit_code == 0
    assert 'CONFIG_FILE' in result.output


# ---------------------------------------------------------------------------
# settings group
# ---------------------------------------------------------------------------

def test_settings_help(runner):
    result = runner.invoke(cli, ['settings', '--help'])
    assert result.exit_code == 0
    assert 'show' in result.output
    assert 'init' in result.output
    assert 'reset' in result.output


def test_settings_show(runner):
    result = runner.invoke(cli, ['settings', 'show'])
    assert result.exit_code == 0
    # Should produce YAML-like output
    assert ':' in result.output


def test_settings_show_origin(runner):
    result = runner.invoke(cli, ['settings', 'show', '--origin'])
    assert result.exit_code == 0


def test_settings_init_help(runner):
    result = runner.invoke(cli, ['settings', 'init', '--help'])
    assert result.exit_code == 0
    assert '--global' in result.output
    assert '--force' in result.output


# ---------------------------------------------------------------------------
# settings group: init/reset behaviour (test-gap; cli.md findings 4-5)
# ---------------------------------------------------------------------------

def test_settings_init_writes_a_file_at_the_configured_location(runner, tmp_path, monkeypatch):
    """`--global` targets $XDG_CONFIG_HOME/FrictionSim2D/settings.yaml -- the
    conftest autouse fixture already redirects that variable for the whole
    session; monkeypatch layers a fresh, test-local value on top of it."""
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))

    result = runner.invoke(cli, ['settings', 'init', '--global'])

    assert result.exit_code == 0, result.output
    target = tmp_path / 'FrictionSim2D' / 'settings.yaml'
    assert target.exists()
    written = yaml.safe_load(target.read_text(encoding='utf-8'))
    assert written['hpc']['scheduler_type'] == 'pbs'


def test_settings_init_again_without_force_refuses(runner, tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    target = tmp_path / 'FrictionSim2D' / 'settings.yaml'

    first = runner.invoke(cli, ['settings', 'init', '--global'])
    assert first.exit_code == 0, first.output
    written_at_first = target.read_text(encoding='utf-8')

    second = runner.invoke(cli, ['settings', 'init', '--global'])

    assert second.exit_code != 0
    assert target.read_text(encoding='utf-8') == written_at_first


def test_settings_init_force_overwrites_without_prompting(runner, tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))

    runner.invoke(cli, ['settings', 'init', '--global'])
    result = runner.invoke(cli, ['settings', 'init', '--global', '--force'])

    assert result.exit_code == 0, result.output


def test_settings_reset_restores_defaults(runner, tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    target = tmp_path / 'FrictionSim2D' / 'settings.yaml'

    runner.invoke(cli, ['settings', 'init', '--global'])
    assert target.exists()

    result = runner.invoke(cli, ['settings', 'reset', '--global', '--yes'])

    assert result.exit_code == 0, result.output
    assert not target.exists()


def test_settings_show_origin_names_the_file_actually_loaded(runner, tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    target = tmp_path / 'FrictionSim2D' / 'settings.yaml'

    no_file_yet = runner.invoke(cli, ['settings', 'show', '--origin'])
    assert no_file_yet.exit_code == 0
    assert 'hardcoded defaults' in no_file_yet.output

    runner.invoke(cli, ['settings', 'init', '--global'])

    result = runner.invoke(cli, ['settings', 'show', '--origin'])
    assert result.exit_code == 0
    assert str(target) in result.output


# ---------------------------------------------------------------------------
# hpc group
# ---------------------------------------------------------------------------

def test_hpc_help(runner):
    result = runner.invoke(cli, ['hpc', '--help'])
    assert result.exit_code == 0
    assert 'generate' in result.output


def test_hpc_generate_help(runner):
    result = runner.invoke(cli, ['hpc', 'generate', '--help'])
    assert result.exit_code == 0
    assert '--scheduler' in result.output
    assert '--output-dir' in result.output


# ---------------------------------------------------------------------------
# postprocess group
# ---------------------------------------------------------------------------

def test_postprocess_help(runner):
    result = runner.invoke(cli, ['postprocess', '--help'])
    assert result.exit_code == 0
    assert 'read' in result.output
    assert 'plot' in result.output


def test_postprocess_read_help(runner):
    result = runner.invoke(cli, ['postprocess', 'read', '--help'])
    assert result.exit_code == 0
    assert 'RESULTS_DIR' in result.output
    assert '--export' in result.output


def test_postprocess_plot_help(runner):
    result = runner.invoke(cli, ['postprocess', 'plot', '--help'])
    assert result.exit_code == 0
    assert 'PLOT_CONFIG' in result.output
    assert '--output-dir' in result.output


def test_postprocess_plot_missing_keys(runner, tmp_path):
    """Config missing required keys should abort gracefully."""
    import json
    cfg = tmp_path / 'bad.json'
    cfg.write_text(json.dumps({'plots': []}), encoding='utf-8')
    result = runner.invoke(cli, ['postprocess', 'plot', str(cfg)])
    assert result.exit_code != 0


def test_postprocess_plot_mismatched_labels(runner, tmp_path):
    """Mismatched data_dirs / labels should abort gracefully."""
    import json
    cfg = tmp_path / 'bad.json'
    cfg.write_text(
        json.dumps({'data_dirs': ['a', 'b'], 'labels': ['only_one'], 'plots': [{}]}),
        encoding='utf-8',
    )
    result = runner.invoke(cli, ['postprocess', 'plot', str(cfg)])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# db group
# ---------------------------------------------------------------------------

def test_db_help(runner):
    result = runner.invoke(cli, ['db', '--help'])
    assert result.exit_code == 0
    for cmd in ('upload', 'query', 'stats', 'init', 'stage', 'publish',
                'reject', 'setup', 'create-key', 'delete'):
        assert cmd in result.output


def test_db_upload_help(runner):
    result = runner.invoke(cli, ['db', 'upload', '--help'])
    assert result.exit_code == 0
    assert '--uploader' in result.output


def test_db_query_help(runner):
    result = runner.invoke(cli, ['db', 'query', '--help'])
    assert result.exit_code == 0
    assert '--material' in result.output


def test_db_stage_help(runner):
    result = runner.invoke(cli, ['db', 'stage', '--help'])
    assert result.exit_code == 0
    assert '--uploader' in result.output
    assert '--api-key' in result.output


def test_db_setup_help(runner):
    result = runner.invoke(cli, ['db', 'setup', '--help'])
    assert result.exit_code == 0
    assert '--name' in result.output
    assert '--profile' in result.output


# ---------------------------------------------------------------------------
# api group
# ---------------------------------------------------------------------------

def test_api_help(runner):
    result = runner.invoke(cli, ['api', '--help'])
    assert result.exit_code == 0
    assert 'serve' in result.output


def test_api_serve_help(runner):
    result = runner.invoke(cli, ['api', 'serve', '--help'])
    assert result.exit_code == 0
    assert '--host' in result.output
    assert '--port' in result.output
    assert '--reload' in result.output


# ---------------------------------------------------------------------------
# aiida group (requires aiida-core)
# ---------------------------------------------------------------------------

aiida = pytest.importorskip('aiida', reason='aiida-core not installed')


def test_aiida_help(runner):
    result = runner.invoke(cli, ['aiida', '--help'])
    # exit_code may be non-zero if no AiiDA profile is configured,
    # but the help text should still show available commands.
    assert 'status' in result.output or result.exit_code == 0


def test_aiida_status_help(runner):
    result = runner.invoke(cli, ['aiida', 'status', '--help'])
    assert result.exit_code == 0 or 'status' in result.output


def test_aiida_import_help(runner):
    result = runner.invoke(cli, ['aiida', 'import', '--help'])
    assert result.exit_code == 0
    assert 'SIMULATION_FOLDER' in result.output
    assert '--label' in result.output


def test_aiida_rebuild_help(runner):
    result = runner.invoke(cli, ['aiida', 'rebuild', '--help'])
    assert result.exit_code == 0
    assert 'SET_LABEL' in result.output
    assert '--hpc-scripts' in result.output


def test_aiida_delete_help(runner):
    result = runner.invoke(cli, ['aiida', 'delete', '--help'])
    assert result.exit_code == 0
    assert 'SET_LABEL' in result.output


def test_aiida_clear_help(runner):
    result = runner.invoke(cli, ['aiida', 'clear', '--help'])
    assert result.exit_code == 0


def test_aiida_list_sets_help(runner):
    result = runner.invoke(cli, ['aiida', 'list-sets', '--help'])
    assert result.exit_code == 0
    assert '--format' in result.output


def test_aiida_query_help(runner):
    result = runner.invoke(cli, ['aiida', 'query', '--help'])
    assert result.exit_code == 0
    assert '--material' in result.output
    assert '--set-label' in result.output
    assert '--force-min' in result.output
    assert '--force-max' in result.output


def test_aiida_query_forwards_filters_to_friction2ddb(runner, monkeypatch):
    calls = {}

    class _FakeResult:
        total_count = 0

        def to_dataframe(self):
            import pandas as pd
            return pd.DataFrame()

    class _FakeDB:
        def __init__(self):
            pass

        def get_available_conditions(self):
            return {'set_labels': ['q1']}

        def query(self, **kwargs):
            calls.update(kwargs)
            return _FakeResult()

    monkeypatch.setattr('src.aiida.load_aiida_profile', lambda profile=None: None)
    monkeypatch.setattr('src.aiida.query.Friction2DDB', _FakeDB)

    result = runner.invoke(cli, [
        'aiida', 'query', '--material', 'h-MoS2', '--family', 'hetero',
        '--set-label', 'q1', '--force-min', '10', '--force-max', '30',
    ])

    assert result.exit_code == 0, result.output
    assert calls['materials'] == ['h-MoS2']
    assert calls['family'] == 'hetero'
    assert calls['set_label'] == 'q1'
    assert calls['force_range'] == (10.0, 30.0)


def test_aiida_query_unknown_set_label_raises(runner, monkeypatch):
    class _FakeDB:
        def __init__(self):
            pass

        def get_available_conditions(self):
            return {'set_labels': ['q1', 'q2']}

        def query(self, **_kwargs):
            raise AssertionError("query() must not run for an unknown set label")

    monkeypatch.setattr('src.aiida.load_aiida_profile', lambda profile=None: None)
    monkeypatch.setattr('src.aiida.query.Friction2DDB', _FakeDB)

    result = runner.invoke(cli, ['aiida', 'query', '--set-label', 'nope'])

    assert result.exit_code != 0
    assert 'nope' in result.output
    assert 'q1' in result.output


def test_aiida_export_help(runner):
    result = runner.invoke(cli, ['aiida', 'export', '--help'])
    assert result.exit_code == 0
    assert 'SET_LABEL' in result.output
    assert '--output' in result.output
    assert '--profile' in result.output
    assert '--material' not in result.output


def test_aiida_import_archive_help(runner):
    result = runner.invoke(cli, ['aiida', 'import-archive', '--help'])
    assert result.exit_code == 0
    assert 'ARCHIVE_PATH' in result.output


def test_aiida_import_archive_loads_profile(runner, monkeypatch, tmp_path):
    archive = tmp_path / 'results.aiida'
    archive.write_text('dummy', encoding='utf-8')

    calls = []

    def fake_import_archive(path, profile=None):
        calls.append(('import_archive', str(path)))
        return 0

    monkeypatch.setattr('src.aiida.export.import_archive', fake_import_archive)

    result = runner.invoke(cli, ['aiida', 'import-archive', str(archive)])

    assert result.exit_code == 0
    assert calls == [
        ('import_archive', str(archive)),
    ]


def test_aiida_package_help(runner):
    result = runner.invoke(cli, ['aiida', 'package', '--help'])
    assert result.exit_code == 0
    assert 'SIMULATION_DIR' in result.output


def test_aiida_submit_help(runner):
    result = runner.invoke(cli, ['aiida', 'submit', '--help'])
    assert result.exit_code == 0
    assert '--dry-run' in result.output
    assert '--walltime' in result.output
