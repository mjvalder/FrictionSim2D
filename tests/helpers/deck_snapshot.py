"""Deck normalisation: strip run-specific noise so decks compare and hash stably."""
from __future__ import annotations

import re
from pathlib import Path

_REPO = str(Path(__file__).resolve().parents[2])

_SEED_PATTERNS = [
    re.compile(r"(?m)^(\s*velocity\s+\S+\s+create\s+\S+\s+)\d+"),
    re.compile(r"(?m)^(\s*fix\b.*\blangevin\s+\S+\s+\S+\s+\S+\s+)\d+"),
    re.compile(r"(?mi)^(\s*variable\s+\w*seed\w*\s+\w+\s+)\d+"),
]
_UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
                   r"|\b[0-9a-f]{32}\b")
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")
_GIT = re.compile(r"\b[0-9a-f]{40}\b")
_TAIL = r"(?:/[^\s\"'),:;]*)?"


def _abs_paths(text: str) -> str:
    text = text.replace(_REPO, "<ABS>")
    for prefix in (r"/(?:var/)?tmp", r"/var/folders", r"/home/[^/\s]+"):
        text = re.sub(prefix + _TAIL, "<ABS>", text)
    return re.sub(r"<ABS>" + _TAIL, "<ABS>", text)


def normalise_deck(text: str) -> str:
    """Replace seeds, uuids, timestamps, commit hashes and absolute paths."""
    for pat in _SEED_PATTERNS:
        text = pat.sub(r"\g<1><SEED>", text)
    text = _TIMESTAMP.sub("<TIMESTAMP>", text)
    text = _UUID.sub("<UUID>", text)
    text = _GIT.sub("<GITHASH>", text)
    return _abs_paths(text)


# ---------------------------------------------------------------- rendering
import contextlib
import datetime as _dt
import difflib
import hashlib
import os
import random
import uuid
from dataclasses import dataclass
from typing import Dict, Iterator, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = _REPO_ROOT / "tests" / "fixtures" / "deck_snapshot"
#: Task 20: append scenario-matrix configs here (``RenderCase(name, run_type, ini)``).
EXTRA_CASES: List["RenderCase"] = []
_BIG_FILE = 200_000
HASH_BIG = False  # flip if the snapshot outgrows 15 MB
_FIXED_NOW = _dt.datetime(2026, 1, 1, 12, 0, 0)


@dataclass(frozen=True)
class RenderCase:
    name: str
    run_type: str
    ini: Path


def render_set() -> List[RenderCase]:
    """Every ``examples/*.ini`` that is a run config, plus ``EXTRA_CASES``."""
    from tests.helpers.examples import CASES, CAMPAIGNS, HETERO, EXAMPLES
    cases = [RenderCase(Path(n).stem, m, EXAMPLES / n) for n, m, _ in CASES + CAMPAIGNS + [HETERO]]
    return cases + list(EXTRA_CASES)


class _FixedDatetime(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return _FIXED_NOW


@contextlib.contextmanager
def _deterministic() -> Iterator[None]:
    import sys
    counter = iter(range(1, 10**9))
    patched = []
    for mod in list(sys.modules.values()):
        if getattr(mod, "__name__", "").startswith("src.") and getattr(mod, "datetime", None) is _dt.datetime:
            patched.append(mod)
            mod.datetime = _FixedDatetime
    real_uuid4 = uuid.uuid4
    uuid.uuid4 = lambda: uuid.UUID(int=next(counter))
    random.seed(20260101)
    try:
        yield
    finally:
        uuid.uuid4 = real_uuid4
        for mod in patched:
            mod.datetime = _dt.datetime


def render(case: RenderCase, out: Path) -> Dict[str, str]:
    """Build ``case`` into ``out``; map relative path to normalised text (text files only)."""
    from src.run.artifacts import generate_package
    from tests.helpers.examples import CAMPAIGN_NAMES, one_material_copy
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ini = one_material_copy(case.ini.name, out) if case.ini.name in CAMPAIGN_NAMES else case.ini
    cwd = os.getcwd()
    os.chdir(_REPO_ROOT)
    try:
        with _deterministic():
            generate_package(ini, case.run_type, output_root=out, generate_hpc=True,
                             allow_generate_amorphous=False, simulation_root_name="sim")
    finally:
        os.chdir(cwd)
    files: Dict[str, str] = {}
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        rel = path.relative_to(out).as_posix()
        if rel == ini.name or rel == "one_material.txt":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        files[rel] = normalise_deck(text)
    return files


def stored_form(text: str) -> str:
    """What the snapshot keeps for a file: its text (digest only when SNAPSHOT_HASH_BIG is set)."""
    if HASH_BIG and len(text) > _BIG_FILE:
        return "sha256:" + hashlib.sha256(text.encode()).hexdigest() + "\n"
    return text


def write_snapshot(case: RenderCase, files: Dict[str, str]) -> None:
    import shutil
    target = SNAPSHOT_DIR / case.name
    shutil.rmtree(target, ignore_errors=True)
    for rel, text in files.items():
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(stored_form(text), encoding="utf-8")


def read_snapshot(case: RenderCase) -> Dict[str, str]:
    root = SNAPSHOT_DIR / case.name
    return {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
            for p in sorted(root.rglob("*")) if p.is_file()}


def diff_files(expected: Dict[str, str], actual: Dict[str, str], limit: int = 3) -> str:
    """Unified diff of the first ``limit`` differing files (missing/extra files named)."""
    chunks, shown = [], 0
    for rel in sorted(set(expected) | set(actual)):
        want, got = expected.get(rel), actual.get(rel)
        if want == got:
            continue
        if shown >= limit:
            break
        shown += 1
        chunks.append("".join(difflib.unified_diff(
            (want or "").splitlines(True), (got or "").splitlines(True),
            f"snapshot/{rel}" if want is not None else "/dev/null",
            f"render/{rel}" if got is not None else "/dev/null")))
    return "\n".join(chunks)
