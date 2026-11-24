from types import SimpleNamespace
from unittest.mock import patch

import frictionsim2d.interfaces.lammps as lmp_mod

import logging
import subprocess
import types
from pathlib import Path

import pytest

from frictionsim2d.interfaces import lammps as lmp_iface

# The autouse conftest fixture replaces the module attribute; keep the real one.
REAL_PAIR_STYLES = lmp_iface.lammps_pair_styles



def test_run_lammps_commands_executes_and_closes():
    fake_lmp_instance = SimpleNamespace()
    fake_lmp_instance.command_calls = []

    def _command(cmd):
        fake_lmp_instance.command_calls.append(cmd)

    fake_lmp_instance.command = _command
    fake_lmp_instance.close_called = False

    def _close():
        fake_lmp_instance.close_called = True

    fake_lmp_instance.close = _close

    fake_module = SimpleNamespace(lammps=lambda cmdargs: fake_lmp_instance)

    with patch.object(lmp_mod, "_import_lammps_module", return_value=fake_module):
        lmp_mod.run_lammps_commands(["units metal", "run 0"])

    assert fake_lmp_instance.command_calls == ["units metal", "run 0"]
    assert fake_lmp_instance.close_called is True





def test_run_lammps_commands_closes_and_reraises_on_failure(monkeypatch, caplog):
    events = []
    class FakeLmp:
        def __init__(self, cmdargs=None): events.append(('open', cmdargs))
        def command(self, cmd):
            events.append(('cmd', cmd))
            if cmd == 'boom':
                raise RuntimeError('LAMMPS says no')
        def close(self): events.append(('close',))
    fake_mod = types.SimpleNamespace(lammps=FakeLmp)
    monkeypatch.setattr(lmp_iface, '_import_lammps_module', lambda: fake_mod)
    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match='says no'):
        lmp_iface.run_lammps_commands(['units metal', 'boom', 'never'])
    assert events[-1] == ('close',)
    assert ('cmd', 'never') not in events
    assert 'LAMMPS execution failed' in caplog.text


def test_create_lammps_instance_is_headless():
    seen = {}

    def _fake_lammps(cmdargs):
        seen["cmdargs"] = cmdargs
        return "instance"

    with patch.object(lmp_mod, "_import_lammps_module",
                      return_value=SimpleNamespace(lammps=_fake_lammps)):
        assert lmp_mod.create_lammps_instance() == "instance"
    assert seen["cmdargs"] == ["-log", "none", "-screen", "none", "-nocite"]


def test_import_lammps_module_returns_the_lammps_module():
    import lammps
    assert lmp_iface._import_lammps_module() is lammps


@pytest.fixture
def fresh_pair_styles():
    REAL_PAIR_STYLES.cache_clear()
    yield
    REAL_PAIR_STYLES.cache_clear()


def test_pair_styles_none_when_lmp_missing(monkeypatch, fresh_pair_styles):
    def missing(*a, **k):
        raise FileNotFoundError('lmp')
    monkeypatch.setattr(lmp_iface.subprocess, 'run', missing)
    assert REAL_PAIR_STYLES() is None


def test_pair_styles_none_when_lmp_exits_nonzero(monkeypatch, fresh_pair_styles):
    def failing(*a, **k):
        raise subprocess.CalledProcessError(1, 'lmp')
    monkeypatch.setattr(lmp_iface.subprocess, 'run', failing)
    assert REAL_PAIR_STYLES() is None


def test_pair_styles_none_without_a_pair_styles_block(monkeypatch, fresh_pair_styles):
    out = 'Fix styles:\n  nve  nvt\nBond styles:\n  harmonic\n'
    monkeypatch.setattr(lmp_iface.subprocess, 'run',
                        lambda *a, **k: types.SimpleNamespace(stdout=out))
    assert REAL_PAIR_STYLES() is None


def test_run_on_data_file_wraps_commands_between_read_and_write(monkeypatch):
    seen = []
    monkeypatch.setattr(lmp_iface, 'run_lammps_commands', seen.extend)
    lmp_iface.run_on_data_file(Path('in.data'), 'atomic', ['replicate 2 2 1'], Path('out.data'))
    assert seen == ['clear', 'units metal', 'atom_style atomic',
                    'read_data "in.data"', 'replicate 2 2 1', 'write_data "out.data"']


def test_replicate_data_file_replaces_an_existing_output(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(lmp_iface, 'run_lammps_commands', seen.extend)
    out = tmp_path / 'out.data'
    out.write_text('stale')
    lmp_iface.replicate_data_file(tmp_path / 'in.data', out, 2, 3, 1)
    assert not out.exists()
    assert 'replicate 2 3 1' in seen and seen[-1] == f'write_data "{out}"'
