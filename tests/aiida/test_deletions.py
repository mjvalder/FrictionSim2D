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
    import src.aiida.setup as setup_mod

    assert not hasattr(setup_mod, 'setup_computer_from_hpc_settings')
    assert not hasattr(setup_mod, '_TRANSPORT_MAP')
    assert not hasattr(setup_mod, 'start_rabbitmq')
    assert not hasattr(setup_mod, '_patch_rabbitmq_consumer_timeout')
    assert hasattr(setup_mod, 'setup_localhost_computer')
    assert hasattr(setup_mod, 'setup_lammps_code')


def test_aiida_setup_has_no_remote_options():
    from click.testing import CliRunner

    from src.cli import cli

    out = CliRunner().invoke(cli, ['aiida', 'setup', '--help']).output

    assert '--use-remote' not in out
    assert '--hpc-config' not in out
    assert '--lammps-path' in out
    # `aiida setup` CREATES a profile; every other command LOADS one.
    assert '--create-profile' in out
    assert '--profile ' not in out
