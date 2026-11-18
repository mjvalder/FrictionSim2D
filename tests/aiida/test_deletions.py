"""Dead run modes are gone (spec §4.3, §8)."""
import importlib
from importlib.metadata import entry_points

import pytest


def test_workchain_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('src.aiida.workchain')


def test_workchain_entry_point_is_deregistered():
    eps = list(entry_points(group='aiida.workflows'))
    # Controller ruling R23: aiida-core itself registers entry points (core.arithmetic.*),
    # so we assert that friction2d.friction is NOT present
    friction_eps = [ep for ep in eps if 'friction2d.friction' in str(ep)]
    assert not friction_eps, f"Expected no friction2d.friction entry points, found: {friction_eps}"


def test_lazy_export_is_gone():
    import src.aiida as aiida_pkg
    assert 'FrictionWorkChain' not in aiida_pkg._LAZY_EXPORTS
    with pytest.raises(AttributeError):
        _ = aiida_pkg.FrictionWorkChain


def test_ssh_transport_helpers_are_gone():
    import src.aiida.profile as profile_mod

    assert not hasattr(profile_mod, 'setup_computer_from_hpc_settings')
    assert not hasattr(profile_mod, '_TRANSPORT_MAP')
    assert not hasattr(profile_mod, 'start_rabbitmq')
    assert not hasattr(profile_mod, '_patch_rabbitmq_consumer_timeout')
    assert hasattr(profile_mod, 'setup_profile')


@pytest.mark.parametrize('name', ['setup_localhost_computer', 'setup_lammps_code',
                                  '_find_lammps_executable', 'check_daemon_status',
                                  'start_daemon'])
def test_code_and_daemon_helpers_are_gone(name):
    """Computers and Codes are made at import; nothing needs the daemon."""
    import src.aiida.profile as profile_mod

    assert not hasattr(profile_mod, name)


def test_full_setup_sets_up_the_profile_only(monkeypatch):
    import src.aiida.profile as profile_mod

    seen = []
    monkeypatch.setattr(profile_mod, 'setup_profile',
                        lambda name, use_postgres: seen.append(name) or True)
    assert profile_mod.full_setup('p1') == {'profile': True}
    assert seen == ['p1']


@pytest.mark.parametrize('module', ['src.aiida.submit', 'src.aiida.data.simulation',
                                    'src.aiida.data.simulation_set',
                                    'src.aiida.data.results'])
def test_submission_and_custom_node_modules_are_gone(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_no_friction2d_data_entry_points():
    assert not [ep for ep in entry_points(group='aiida.data')
                if ep.name.startswith('friction2d')]


def test_aiida_setup_has_no_remote_options():
    from click.testing import CliRunner

    from src.cli import cli

    out = CliRunner().invoke(cli, ['aiida', 'setup', '--help']).output

    assert '--use-remote' not in out
    assert '--hpc-config' not in out
    assert '--lammps-path' not in out
    # `aiida setup` CREATES a profile; every other command LOADS one.
    assert '--create-profile' in out
    assert '--profile ' not in out


GONE = [
    'register_simulation_batch', 'register_single_simulation',
    'import_results_to_aiida', 'import_simulation_set', 'dump_results_to_json',
    'rebuild_simulation_set', 'export_archive', 'import_archive',
    '_detect_simulation_type', '_find_material_provenance_dirs',
    '_rebuild_single_material', '_patch_config_paths', '_resolve_set_node',
    '_load_manifest', '_update_manifest_entry',
]
KEPT = ['list_simulation_sets', 'delete_simulation_set', 'clear_all_nodes']


@pytest.mark.parametrize('name', GONE)
def test_superseded_function_is_gone(name):
    import src.aiida.integration as integ
    assert not hasattr(integ, name), name


@pytest.mark.parametrize('name', KEPT)
def test_surviving_function_is_kept(name):
    import src.aiida.integration as integ
    assert callable(getattr(integ, name))


def test_lazy_exports_point_at_the_new_modules():
    import src.aiida as aiida_pkg

    exports = aiida_pkg._LAZY_EXPORTS
    assert exports['import_root'][0].endswith('.ingest')
    assert 'ingest_root' not in exports
    assert exports['export_results'][0].endswith('.export')
    assert exports['export_archive'][0].endswith('.export')
    assert exports['import_archive'][0].endswith('.export')
    for dead in ('register_simulation_batch', 'register_single_simulation',
                 'import_results_to_aiida', 'import_simulation_set',
                 'dump_results_to_json', 'rebuild_simulation_set',
                 'FrictionProvenanceData', 'FrictionWorkChain',
                 'FrictionSimulationData', 'FrictionSimulationSetData',
                 'FrictionResultsData', 'run_with_aiida', 'smart_submit'):
        assert dead not in exports
