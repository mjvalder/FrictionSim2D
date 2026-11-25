"""``run_lmp_serial`` never leaves a child LAMMPS behind."""

from __future__ import annotations

import subprocess

import pytest

from frictionsim2d.builders.construction import amorphous


class TestRunLmpSerialCleanup:

    def test_an_exception_while_waiting_kills_the_child(self, tmp_path, monkeypatch):
        killed = []

        class Child:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def wait(self):
                if not killed:
                    raise KeyboardInterrupt
                return -9

            def kill(self):
                killed.append(1)

        monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: Child())
        with pytest.raises(KeyboardInterrupt):
            amorphous.run_lmp_serial('d.in', 'd.log', tmp_path)
        assert killed == [1] and amorphous._ACTIVE_LMP is None  # pylint: disable=protected-access
