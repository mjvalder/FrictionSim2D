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
from typing import Callable, Dict, Iterable, List, Tuple

from .deck_snapshot import normalise_deck

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "lammps_replay"


_INPUT_RE = re.compile(r"(?m)^[ \t]*(?:read_data|include|read_restart)[ \t]+(\S+)")
_OUTPUT_RE = re.compile(r"(?m)^[ \t]*(?:write_data|write_restart)[ \t]+(\S+)")


def replay_key(script_text: str, input_files: Iterable[Path] = ()) -> str:
    """sha256 of the normalised script plus the normalised content of its input files."""
    digest = hashlib.sha256(normalise_deck(script_text).encode())
    parts = sorted(
        hashlib.sha256(normalise_deck(Path(f).read_text(encoding="utf-8")).encode()).hexdigest()
        for f in input_files
    )
    for part in parts:
        digest.update(b"\n" + part.encode())
    return digest.hexdigest()


def _missing(key: str) -> LookupError:
    return LookupError(f"no recorded LAMMPS output for {key}; rerun with --record-lammps")


def _resolve(token: str, cwd: Path) -> Path:
    p = Path(token)
    return p if p.is_absolute() else Path(cwd) / p


def _literal(tokens: Iterable[str]) -> List[str]:
    return [t for t in tokens if not any(c in t for c in "${}")]


def _inputs(text: str, cwd: Path) -> List[Path]:
    paths = [_resolve(t, cwd) for t in _literal(_INPUT_RE.findall(text))]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"LAMMPS input {p} is missing; cannot key the replay")
    return paths


def _outputs(text: str, cwd: Path) -> List[Tuple[str, Path]]:
    seen: Dict[Path, None] = {}
    for t in _literal(_OUTPUT_RE.findall(text)):
        seen.setdefault(_resolve(t, cwd))
    return [(f"{i}_{p.name}", p) for i, p in enumerate(seen)]


def _store(key_dir: Path, outputs: List[Tuple[str, Path]]) -> None:
    if key_dir.exists():
        shutil.rmtree(key_dir)
    key_dir.mkdir(parents=True)
    for label, path in outputs:
        if path.is_file():
            shutil.copy2(path, key_dir / label)


def _restore(key_dir: Path, outputs: List[Tuple[str, Path]]) -> None:
    for label, path in outputs:
        src = key_dir / label
        if src.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, path)


def make_run_lammps_commands(real: Callable, record: bool, fixture_dir: Path = FIXTURE_DIR):
    def run(commands: List[str]) -> None:
        text = "\n".join(commands)
        cwd = Path.cwd()
        key = replay_key(text, _inputs(text, cwd))
        outputs = _outputs(text, cwd)
        key_dir = fixture_dir / key
        if record:
            real(commands)
            _store(key_dir, outputs)
            return
        if not key_dir.is_dir():
            raise _missing(key)
        _restore(key_dir, outputs)
    return run


def make_run_lammps(real: Callable, record: bool, fixture_dir: Path = FIXTURE_DIR):
    def run(workdir: Path, script_path: Path) -> str:
        workdir = Path(workdir)
        text = Path(script_path).read_text(encoding="utf-8")
        key = replay_key(text, _inputs(text, workdir))
        outputs = _outputs(text, workdir)
        key_dir = fixture_dir / key
        if record:
            out = real(workdir, script_path)
            _store(key_dir, outputs)
            (key_dir / "stdout.txt").write_text(normalise_deck(out), encoding="utf-8")
            return out
        if not key_dir.is_dir():
            raise _missing(key)
        _restore(key_dir, outputs)
        stdout = key_dir / "stdout.txt"
        return stdout.read_text(encoding="utf-8") if stdout.exists() else ""
    return run


def install(monkeypatch, record: bool = False, fixture_dir: Path = FIXTURE_DIR) -> None:
    """Patch the three call sites where they are looked up."""
    from frictionsim2d.builders.construction import components, hetero_stack
    from frictionsim2d.builders.sheetonsheet import flake_slide

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
