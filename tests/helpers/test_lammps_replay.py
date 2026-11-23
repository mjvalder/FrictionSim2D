"""Tests for the LAMMPS record/replay helper."""
from pathlib import Path

import pytest

from tests.helpers import lammps_replay as lr


def test_replay_returns_recorded_files(tmp_path, monkeypatch):
    fixtures = tmp_path / "fx"
    script = "units metal\nvelocity all create 300 4711\nrun 0\n"
    key_dir = fixtures / lr.replay_key(script)
    (key_dir / "d0").mkdir(parents=True)
    (key_dir / "d0" / "out.data").write_text("DATA")
    (key_dir / "stdout.txt").write_text("log")
    work = tmp_path / "work"
    work.mkdir()
    script_path = work / "in.lmp"
    script_path.write_text(script)

    def real(*a):
        raise AssertionError("real LAMMPS must not run")

    run = lr.make_run_lammps(real, record=False, fixture_dir=fixtures)
    assert run(work, script_path) == "log"
    assert (work / "out.data").read_text() == "DATA"


def test_missing_key_raises_lookup_error(tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    script_path = work / "in.lmp"
    script_path.write_text("run 0\n")
    run = lr.make_run_lammps(lambda *a: "", record=False, fixture_dir=tmp_path / "none")
    with pytest.raises(LookupError, match="no recorded LAMMPS output for .*--record-lammps"):
        run(work, script_path)


def test_key_ignores_seed_and_paths():
    a = "read_data /home/alice/x/a.lmp\nvelocity all create 300 111 dist gaussian\n"
    b = "read_data /tmp/pytest-9/b/a.lmp\nvelocity all create 300 999 dist gaussian\n"
    assert lr.replay_key(a) == lr.replay_key(b)


def test_record_then_replay_roundtrip(tmp_path):
    fixtures = tmp_path / "fx"
    work = tmp_path / "work"
    work.mkdir()
    script_path = work / "in.lmp"
    script_path.write_text("run 0\n")

    def real(workdir, script):
        (workdir / "made.data").write_text("X")
        return "stdout"

    lr.make_run_lammps(real, True, fixtures)(work, script_path)
    (work / "made.data").unlink()
    out = lr.make_run_lammps(real, False, fixtures)(work, script_path)
    assert out == "stdout" and (work / "made.data").read_text() == "X"
