"""Dead run modes are gone (spec §4.3, §8)."""
import importlib
from importlib.metadata import entry_points

import pytest


def test_workchain_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('frictionsim2d.aiida.workchain')


def test_workchain_entry_point_is_deregistered():
    eps = list(entry_points(group='aiida.workflows'))
    # Controller ruling R23: aiida-core itself registers entry points (core.arithmetic.*),
    # so we assert that friction2d.friction is NOT present
    friction_eps = [ep for ep in eps if 'friction2d.friction' in str(ep)]
    assert not friction_eps, f"Expected no friction2d.friction entry points, found: {friction_eps}"


def test_ssh_transport_helpers_are_gone():
    import frictionsim2d.aiida.profile as profile_mod

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
    import frictionsim2d.aiida.profile as profile_mod

    assert not hasattr(profile_mod, name)


def test_full_setup_sets_up_the_profile_only(monkeypatch):
    import frictionsim2d.aiida.profile as profile_mod

    seen = []
    monkeypatch.setattr(profile_mod, 'setup_profile',
                        lambda name, use_postgres: seen.append(name) or True)
    assert profile_mod.full_setup('p1') == {'profile': True}
    assert seen == ['p1']


@pytest.mark.parametrize('module', ['frictionsim2d.aiida.submit', 'frictionsim2d.aiida.data.simulation',
                                    'frictionsim2d.aiida.data.simulation_set',
                                    'frictionsim2d.aiida.data.results',
                                    # merged into profile, layout, rulings, query + steps
                                    'frictionsim2d.aiida.setup', 'frictionsim2d.aiida.preflight',
                                    'frictionsim2d.aiida.data_keys', 'frictionsim2d.aiida.decks',
                                    'frictionsim2d.aiida.retention', 'frictionsim2d.aiida.policy',
                                    'frictionsim2d.aiida.ledger', 'frictionsim2d.aiida.integration'])
def test_submission_and_custom_node_modules_are_gone(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_no_friction2d_data_entry_points():
    assert not [ep for ep in entry_points(group='aiida.data')
                if ep.name.startswith('friction2d')]


def test_aiida_setup_has_no_remote_options():
    from click.testing import CliRunner

    from frictionsim2d.cli import cli

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


@pytest.mark.parametrize('name', GONE)
def test_superseded_function_is_gone(name):
    import frictionsim2d.aiida.query as query_mod
    import frictionsim2d.aiida.steps as steps_mod
    assert not hasattr(query_mod, name), name
    assert not hasattr(steps_mod, name), name


@pytest.mark.parametrize('module, name', [
    ('frictionsim2d.aiida.query', 'list_simulation_sets'),
    ('frictionsim2d.aiida.steps', 'delete_simulation_set'),
    ('frictionsim2d.aiida.steps', 'clear_all_nodes')])
def test_surviving_function_is_kept(module, name):
    assert callable(getattr(importlib.import_module(module), name))
