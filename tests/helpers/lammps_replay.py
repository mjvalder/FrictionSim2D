"""Record/replay of the three build-time LAMMPS call sites.

Replay serves recorded output files (keyed by the sha256 of the normalised
script) so fast tests need no LAMMPS. Record mode runs real LAMMPS and stores
the files the call wrote.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import types
from pathlib import Path
from typing import Callable, Dict, List

from .deck_snapshot import normalise_deck

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "lammps_replay"
_ABS_PATH = re.compile(r"(?<![\w.])/[^\s\"']+")


def replay_key(script_text: str) -> str:
    return hashlib.sha256(normalise_deck(script_text).encode()).hexdigest()


def _missing(key: str) -> LookupError:
    return LookupError(f"no recorded LAMMPS output for {key}; rerun with --record-lammps")


def _snapshot(directory: Path) -> Dict[Path, tuple]:
    out = {}
    if directory.is_dir():
        for p in directory.rglob("*"):
            if p.is_file():
                st = p.stat()
                out[p.relative_to(directory)] = (st.st_mtime_ns, st.st_size)
    return out


def _store(key_dir: Path, watch: List[Path], before: List[Dict[Path, tuple]]) -> None:
    if key_dir.exists():
        shutil.rmtree(key_dir)
    key_dir.mkdir(parents=True)
    for i, (d, snap) in enumerate(zip(watch, before)):
        for rel, sig in _snapshot(d).items():
            if snap.get(rel) != sig:
                dest = key_dir / f"d{i}" / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(d / rel, dest)


def _restore(key_dir: Path, watch: List[Path]) -> None:
    for i, d in enumerate(watch):
        src = key_dir / f"d{i}"
        if src.is_dir():
            for f in src.rglob("*"):
                if f.is_file():
                    dest = d / f.relative_to(src)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest)


def _command_dirs(commands: List[str]) -> List[Path]:
    """Directories of the absolute paths a command list references."""
    dirs = set()
    for cmd in commands:
        for m in _ABS_PATH.findall(cmd):
            dirs.add(Path(m).parent)
    return sorted(d for d in dirs if d.is_dir())


def make_run_lammps_commands(real: Callable, record: bool, fixture_dir: Path = FIXTURE_DIR):
    def run(commands: List[str]) -> None:
        key = replay_key("\n".join(commands))
        watch = _command_dirs(commands)
        key_dir = fixture_dir / key
        if record:
            before = [_snapshot(d) for d in watch]
            real(commands)
            _store(key_dir, watch, before)
            return
        if not key_dir.is_dir():
            raise _missing(key)
        _restore(key_dir, watch)
    return run


def make_run_lammps(real: Callable, record: bool, fixture_dir: Path = FIXTURE_DIR):
    def run(workdir: Path, script_path: Path) -> str:
        workdir = Path(workdir)
        key = replay_key(Path(script_path).read_text(encoding="utf-8"))
        key_dir = fixture_dir / key
        if record:
            before = [_snapshot(workdir)]
            out = real(workdir, script_path)
            _store(key_dir, [workdir], before)
            (key_dir / "stdout.txt").write_text(out, encoding="utf-8")
            return out
        if not key_dir.is_dir():
            raise _missing(key)
        _restore(key_dir, [workdir])
        stdout = key_dir / "stdout.txt"
        return stdout.read_text(encoding="utf-8") if stdout.exists() else ""
    return run


def install(monkeypatch, record: bool = False, fixture_dir: Path = FIXTURE_DIR) -> None:
    """Patch the three call sites where they are looked up."""
    from src.builders.construction import components, hetero_stack
    from src.builders.sheetonsheet import flake_slide

    monkeypatch.setattr(
        components, "run_lammps_commands",
        make_run_lammps_commands(components.run_lammps_commands, record, fixture_dir))
    run = make_run_lammps(hetero_stack._run_lammps, record, fixture_dir)
    monkeypatch.setattr(hetero_stack, "_run_lammps", run)
    monkeypatch.setattr(flake_slide, "_run_lammps", run)

    if not record:
        def no_melt_quench(cmd, *args, **kwargs):
            text = str(cmd)
            m = re.search(r"-in\s+(\S+)", text)
            script = Path(m.group(1)).read_text(encoding="utf-8") if m else text
            raise _missing(replay_key(script))

        shim = types.SimpleNamespace(**{n: getattr(components.subprocess, n)
                                        for n in dir(components.subprocess)
                                        if not n.startswith("__")})
        shim.run = no_melt_quench
        monkeypatch.setattr(components, "subprocess", shim)
