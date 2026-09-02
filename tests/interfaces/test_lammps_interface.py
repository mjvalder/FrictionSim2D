from types import SimpleNamespace
from unittest.mock import patch

import src.interfaces.lammps as lmp_mod


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
