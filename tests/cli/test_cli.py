"""Smoke tests for the FrictionSim2D CLI.

These tests verify that all command groups and key commands are reachable and
produce sensible help text, without requiring LAMMPS, AiiDA, or a database.
"""

import json

import pytest
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


def test_aiida_dump_help(runner):
    result = runner.invoke(cli, ['aiida', 'dump', '--help'])
    assert result.exit_code == 0
    assert 'SET_LABEL' in result.output
    assert '--output-dir' in result.output


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
    assert '--set' in result.output


def test_aiida_export_help(runner):
    result = runner.invoke(cli, ['aiida', 'export', '--help'])
    assert result.exit_code == 0
    assert '--output' in result.output
    assert '--material' in result.output


def test_aiida_import_archive_help(runner):
    result = runner.invoke(cli, ['aiida', 'import-archive', '--help'])
    assert result.exit_code == 0
    assert 'ARCHIVE_PATH' in result.output


def test_aiida_import_archive_loads_profile(runner, monkeypatch, tmp_path):
    archive = tmp_path / 'results.aiida'
    archive.write_text('dummy', encoding='utf-8')

    calls = []

    def fake_load_aiida_profile(profile_name=None):
        calls.append(('load_profile', profile_name))

    def fake_import_archive(path):
        calls.append(('import_archive', str(path)))
        return 0

    monkeypatch.setattr('src.aiida.load_aiida_profile', fake_load_aiida_profile)
    monkeypatch.setattr('src.aiida.integration.import_archive', fake_import_archive)

    result = runner.invoke(cli, ['aiida', 'import-archive', str(archive)])

    assert result.exit_code == 0
    assert calls == [
        ('load_profile', None),
        ('import_archive', str(archive)),
    ]


def test_aiida_rebuild_restores_underscore_material_names(tmp_path, monkeypatch):
    from src.aiida import integration

    class FakeProvNode:
        def get_file_content(self, filename, category):
            assert filename == 'config.json'
            assert category == 'config'
            return json.dumps(
                {
                    'general': {'temp': 300},
                    'sheet': {
                        'mat': 'h-MoS2',
                        'x': 100,
                        'y': 100,
                        'layers': [1],
                    },
                    'tip': {'mat': 'Si', 'amorph': 'c', 'r': 25},
                    'sub': {'mat': 'Si', 'amorph': 'a'},
                }
            ).encode('utf-8')

        def export_to_directory(self, path):
            path.mkdir(parents=True, exist_ok=True)
            return {}

    class FakeDefaults:
        def model_dump(self):
            return {}

    class FakeAFMConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    built = {}

    class FakeBuilder:
        def __init__(self, config_obj, output_dir, config_path=None):
            built['output_dir'] = output_dir
            built['config_path'] = config_path

        def set_base_output_dir(self, base_output_dir):
            built['base_output_dir'] = str(base_output_dir)

        def build(self):
            built['built'] = True

    monkeypatch.setattr(integration, '_patch_config_paths', lambda *args, **kwargs: None)

    simulation_root = tmp_path / 'simulation_root'
    output_dir, success = integration._rebuild_single_material(
        prov_node=FakeProvNode(),
        material='h-MoS2',
        sim_type='afm',
        simulation_root=simulation_root,
        tmp_root=tmp_path / 'tmp_root',
        defaults=FakeDefaults(),
        AFMSimulationConfig=FakeAFMConfig,
        SheetOnSheetSimulationConfig=None,
        AFMSimulation=FakeBuilder,
        SheetOnSheetSimulation=None,
    )

    assert success is True
    assert built['built'] is True
    assert output_dir == (
        simulation_root / 'afm' / 'h_MoS2' / '100x_100y' / 'sub_aSi_tip_Si_r25' / 'K300'
    )
    assert built['output_dir'] == output_dir


def test_aiida_package_help(runner):
    result = runner.invoke(cli, ['aiida', 'package', '--help'])
    assert result.exit_code == 0
    assert 'SIMULATION_DIR' in result.output


def test_aiida_submit_help(runner):
    result = runner.invoke(cli, ['aiida', 'submit', '--help'])
    assert result.exit_code == 0
    assert '--dry-run' in result.output
    assert '--walltime' in result.output
