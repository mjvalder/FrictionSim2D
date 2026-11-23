"""Smoke tests for the FrictionSim2D CLI.

These tests verify that all command groups and key commands are reachable and
produce sensible help text, without requiring LAMMPS, AiiDA, or a database.
"""

import json

import pytest
import yaml
from click.testing import CliRunner
from frictionsim2d.cli import cli


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
    assert '--aiida' not in result.output
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
    # Named HPC profiles were dropped: clusters switch via --settings-file.
    assert '--profile' not in result.output


GPU_SETTINGS_YAML = """\
hpc:
  scheduler_type: pbs
  queue: gpu72
  num_cpus: 8
  memory_gb: 64
  modules: [tools/prod, lammps]
  use_tmpdir: false
  gpu: {ngpus: 1, gpu_type: L40S}
  frameworks:
    deepmd: {binary: $HOME/deepmd-kit/bin/lmp}
"""


def _mixed_sim_tree(tmp_path):
    """A built tree with one classical and one DeePMD (pot_ segment) run."""
    root = tmp_path / 'sim'
    for rel in ('sheetonsheet/h-MoS2/50x_50y/K300',
                'sheetonsheet/h-MoS2/50x_50y/K300/pot_deepmd_dpa3'):
        lammps_dir = root / rel / 'lammps'
        lammps_dir.mkdir(parents=True)
        (lammps_dir / 'slide.in').write_text('# slide\n', encoding='utf-8')
    settings = tmp_path / 'settings.yaml'
    settings.write_text(GPU_SETTINGS_YAML, encoding='utf-8')
    return root, settings


def test_hpc_generate_splits_mlip_runs_into_a_gpu_script_with_the_overlay(runner, tmp_path):
    sim_dir, settings = _mixed_sim_tree(tmp_path)

    result = runner.invoke(cli, [
        'hpc', 'generate', str(sim_dir), '--settings-file', str(settings),
    ])

    assert result.exit_code == 0, result.output
    cpu = (sim_dir / 'hpc' / 'run_cpu.pbs').read_text(encoding='utf-8')
    gpu = (sim_dir / 'hpc' / 'run_gpu_deepmd.pbs').read_text(encoding='utf-8')
    assert 'ngpus' not in cpu
    assert 'mpirun lmp -in ' in cpu
    assert 'ngpus=1:gpu_type=L40S' in gpu
    assert '$HOME/deepmd-kit/bin/lmp -in ' in gpu


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
# removed groups
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('group', ['db', 'api'])
def test_the_central_database_groups_are_gone(runner, group):
    result = runner.invoke(cli, [group, '--help'])
    assert result.exit_code == 2
    assert 'No such command' in result.output


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

    monkeypatch.setattr('frictionsim2d.aiida.load_aiida_profile', lambda profile=None: None)
    monkeypatch.setattr('frictionsim2d.aiida.query.Friction2DDB', _FakeDB)

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

    monkeypatch.setattr('frictionsim2d.aiida.load_aiida_profile', lambda profile=None: None)
    monkeypatch.setattr('frictionsim2d.aiida.query.Friction2DDB', _FakeDB)

    result = runner.invoke(cli, ['aiida', 'query', '--set-label', 'nope'])

    assert result.exit_code != 0
    assert 'nope' in result.output
    assert 'q1' in result.output


def test_aiida_query_profile_failure_is_a_clean_error(runner, monkeypatch):
    """m6: the profile used to be loaded outside the try -> a raw traceback."""
    def _bad(profile=None):
        raise RuntimeError(f'profile {profile!r} does not exist')

    monkeypatch.setattr('frictionsim2d.aiida.load_aiida_profile', _bad)

    result = runner.invoke(cli, ['aiida', 'query', '--profile', 'ghost'])

    assert result.exit_code != 0
    assert "Query failed: profile 'ghost' does not exist" in result.output
    assert not isinstance(result.exception, RuntimeError)


def test_aiida_query_json_prints_the_rows(runner, monkeypatch):
    """m6: `--format json` without `-o` printed only the query parameters."""
    import json

    import pandas as pd

    class _FakeResult:
        total_count = 1
        query_params = {'family': 'afm'}

        def to_dataframe(self):
            return pd.DataFrame([{'rel_path': 'afm/x/L2', 'mean_cof': 0.05}])

    class _FakeDB:
        def get_available_conditions(self):
            return {'set_labels': []}

        def query(self, **_kwargs):
            return _FakeResult()

    monkeypatch.setattr('frictionsim2d.aiida.load_aiida_profile', lambda profile=None: None)
    monkeypatch.setattr('frictionsim2d.aiida.query.Friction2DDB', _FakeDB)

    result = runner.invoke(cli, ['aiida', 'query', '--family', 'afm',
                                 '--format', 'json'])

    assert result.exit_code == 0, result.output
    rows = json.loads(result.output[result.output.index('['):])
    assert rows == [{'rel_path': 'afm/x/L2', 'mean_cof': 0.05}]


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

    monkeypatch.setattr('frictionsim2d.aiida.export.import_archive', fake_import_archive)

    result = runner.invoke(cli, ['aiida', 'import-archive', str(archive)])

    assert result.exit_code == 0
    assert calls == [
        ('import_archive', str(archive)),
    ]


@pytest.mark.parametrize('gone', ['package', 'submit'])
def test_aiida_submission_commands_are_gone(runner, gone):
    """AiiDA is import-only: jobs go through `hpc generate` + the scheduler +
    `aiida import`; nothing submits through AiiDA any more."""
    result = runner.invoke(cli, ['aiida', gone, '--help'])
    assert result.exit_code != 0
    assert 'No such command' in result.output
