from types import SimpleNamespace
from unittest.mock import patch

import frictionsim2d.interfaces.lammps as lmp_mod

import logging
import sys
import time
import types

import pytest

from frictionsim2d.interfaces import lammps as lmp_iface


def test_import_fallback_restores_strptime():
    fake_module = SimpleNamespace(lammps=object())

    with patch.object(lmp_mod, "_ORIGINAL_STRPTIME", lmp_mod.time.strptime):
        calls = {"n": 0}

        def _fake_import(name):
            assert name == "lammps"
            calls["n"] += 1
            if calls["n"] == 1:
                raise ValueError("unconverted data remains: .2.0")
            return fake_module

        original = lmp_mod.time.strptime
        with patch.dict(lmp_mod.sys.modules, dict(lmp_mod.sys.modules), clear=True):
            lmp_mod.sys.modules.pop("lammps", None)
            with patch.object(lmp_mod.importlib, "import_module", side_effect=_fake_import):
                module = lmp_mod._import_lammps_module()

        assert module is fake_module
        assert calls["n"] == 2
        assert lmp_mod.time.strptime is original


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


def test_compat_strptime_trims_suffix_only_for_ymd_format():
    assert lmp_iface._compat_strptime('2022.06.23.2.0', '%Y.%m.%d')[:3] == (2022, 6, 23)


def test_compat_strptime_leaves_other_formats_alone():
    with pytest.raises(ValueError):
        lmp_iface._compat_strptime('2022.06.23.2.0', '%Y-%m-%d')


def test_import_reuses_an_already_imported_module(monkeypatch):
    fake = types.ModuleType('lammps')
    monkeypatch.setitem(sys.modules, 'lammps', fake)
    assert lmp_iface._import_lammps_module() is fake


def test_import_reraises_unrelated_value_errors(monkeypatch):
    monkeypatch.delitem(sys.modules, 'lammps', raising=False)
    def bad_import(name):
        raise ValueError('something else entirely')
    monkeypatch.setattr(lmp_iface.importlib, 'import_module', bad_import)
    with pytest.raises(ValueError, match='something else'):
        lmp_iface._import_lammps_module()
    assert time.strptime is lmp_iface._ORIGINAL_STRPTIME


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
