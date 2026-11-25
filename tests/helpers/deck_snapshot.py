"""Deck normalisation: strip run-specific noise so decks compare and hash stably."""
from __future__ import annotations

import contextlib
import datetime as _dt
import difflib
import hashlib
import os
import random
import re
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List

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
_GIT_DIRTY = re.compile(r'("git_dirty":\s*)(?:true|false)')
_VERSION = re.compile(r'("frictionsim2d_version":\s*)"[^"]*"')
_PYTEST_TMP = re.compile(r"(?:/private)?/(?:var/)?tmp/pytest-of-[^/\s]+/pytest-\d+/[^/\s\"']*?\d+(?=[/\s\"'),:;]|$)")
_TMP_HEX = re.compile(r"(?<=_)[0-9a-f]{8}(?=_|\.)")


def _abs_paths(text: str) -> str:
    """Replace only the root prefix of a path; the relative tail stays visible."""
    text = text.replace(_REPO, "<REPO>")
    text = _PYTEST_TMP.sub("<BUILD>", text)
    text = re.sub(r"/(?:var/)?tmp(?=/)", "<TMP>", text)
    text = re.sub(r"/var/folders/[^/\s]+/[^/\s]+/T", "<TMP>", text)
    text = re.sub(r"/home/[^/\s]+", "<HOME>", text)
    return _TMP_HEX.sub("<HEX>", text)


def normalise_deck(text: str) -> str:
    """Replace seeds, uuids, timestamps, commit hashes and path roots (tails stay)."""
    for pat in _SEED_PATTERNS:
        text = pat.sub(r"\g<1><SEED>", text)
    text = _TIMESTAMP.sub("<TIMESTAMP>", text)
    text = _UUID.sub("<UUID>", text)
    text = _GIT.sub("<GITHASH>", text)
    text = _GIT_DIRTY.sub(r"\g<1><GITDIRTY>", text)
    text = _VERSION.sub(r'\g<1>"<VERSION>"', text)
    return _abs_paths(text)


# ---------------------------------------------------------------- rendering
_REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = _REPO_ROOT / "tests" / "fixtures" / "deck_snapshot"
_BIG_FILE = 200_000
HASH_BIG = False  # flip if the snapshot outgrows 15 MB
_FIXED_NOW = _dt.datetime(2026, 1, 1, 12, 0, 0)


@dataclass(frozen=True)
class RenderCase:
    name: str
    run_type: str
    ini: Path


_CASE_DIR = _REPO_ROOT / "tests" / "fixtures" / "deck_cases"
#: Builder paths no shipped example reaches, one small config each.
EXTRA_CASES: List[RenderCase] = [
    RenderCase(n, m, _CASE_DIR / f"{n}.ini") for n, m in [
        ("afm_crystalline_sub", "afm"),
        ("afm_hetero_matched_sub", "afm"),
        ("afm_hetero_finite", "afm"),
        ("sheet_five_layers", "sheetonsheet"),
        ("sheet_hetero_alternating", "sheetonsheet"),
        ("sheet_hetero_twist", "sheetonsheet"),
        ("sheet_hetero_ilp", "sheetonsheet"),
        ("flake_slide_hetero_flake", "sheetonsheet"),
        # `run quench` (write_quench_root): melt_temp 3000 so no shipped block
        # satisfies it; a 16 A slab keeps the crystal block small.
        ("quench_si", "quench"),
    ]
]


_SCENARIO_DIR = _REPO_ROOT / "tests" / "scenarios" / "configs"
#: The scenario tier's matrix (``tests/scenarios/test_scenarios.py``): one row
#: per (family, potential, condition). Substitutions for the planned matrix:
#: - afm-hetero (lj = the default interlayer), sheet-hetero twist and quench
#:   are the existing ``EXTRA_CASES`` afm_hetero_finite, sheet_hetero_twist and
#:   quench_si, reused rather than copied so the snapshot holds each build once;
#: - afm-freestanding + tersoff is a freestanding ``pes_tip`` run (the only
#:   freestanding builder) on b-SiGe with 1989_SiGe.tersoff;
#: - pes-sheet + kc is pes_sheet_ljg (lj_g): kc has bundled C/H parameters
#:   only, and no carbon material ships;
#: - MLIP (mace) reads configs/fake_mace.json, junk bytes nothing parses.
#: Four rows skip ``aiida import`` in the tier (its ``AIIDA_ROWS``):
#: afm_rebomos_angle_finite, afm_freestanding_tersoff, sheet_sw_ilp and
#: afm_hetero_finite.
SCENARIO_CASES: List[RenderCase] = [
    RenderCase(n, m, _SCENARIO_DIR / f"{n}.ini") for n, m in [
        ("afm_sw_load", "afm"),
        ("afm_rebomos_angle_finite", "afm"),
        ("afm_freestanding_tersoff", "pes_tip"),   # afm-freestanding + tersoff
        ("indent_sw", "indent"),
        ("sheet_reaxff_speed", "sheetonsheet"),
        ("sheet_sw_ilp", "sheetonsheet"),
        ("flake_slide_sw", "sheetonsheet"),
        ("pes_tip_sw", "pes_tip"),
        ("pes_sheet_ljg", "pes_sheet"),            # stands in for pes-sheet + kc
        ("sheet_mace_d3", "sheetonsheet"),         # fake model file
    ]
] + [c for c in EXTRA_CASES if c.name in ("afm_hetero_finite", "sheet_hetero_twist", "quench_si")]


def render_set() -> List[RenderCase]:
    """Every frozen ``examples/*.ini`` copy (``tests/fixtures/examples``) that is
    a run config, plus ``EXTRA_CASES`` and the scenario matrix's own configs."""
    from tests.helpers.examples import CASES, CAMPAIGNS, HETERO, FROZEN_EXAMPLES
    cases = [RenderCase(Path(n).stem, m, FROZEN_EXAMPLES / n)
             for n, m, _ in CASES + CAMPAIGNS + [HETERO]]
    extra = list(EXTRA_CASES)
    return cases + extra + [c for c in SCENARIO_CASES if c not in extra]


class _FixedDatetime(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return _FIXED_NOW


@contextlib.contextmanager
def _deterministic() -> Iterator[None]:
    counter = iter(range(1, 10**9))
    patched = []
    for mod in list(sys.modules.values()):
        if getattr(mod, "__name__", "").startswith("frictionsim2d.") and getattr(mod, "datetime", None) is _dt.datetime:
            patched.append(mod)
            mod.datetime = _FixedDatetime
    real_uuid4 = uuid.uuid4
    pid = (os.getpid() & 0xFFFF) << 112
    # pid and counter in the top bits: the 8 hex digits `_tmp_path` keeps differ per
    # process, so concurrent runs do not share scratch files.
    uuid.uuid4 = lambda: uuid.UUID(int=pid | (next(counter) << 96))
    random.seed(20260101)
    try:
        yield
    finally:
        uuid.uuid4 = real_uuid4
        for mod in patched:
            mod.datetime = _dt.datetime


def render(case: RenderCase, out: Path) -> Dict[str, str]:
    """Build ``case`` into ``out``; map relative path to normalised text (text files only)."""
    from frictionsim2d.run.artifacts import generate_package
    from tests.helpers.examples import CAMPAIGN_NAMES, one_material_copy
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ini = (one_material_copy(case.ini.name, out, case.ini.parent)
           if case.ini.name in CAMPAIGN_NAMES else case.ini)
    cwd = os.getcwd()
    os.chdir(_REPO_ROOT)
    try:
        with _deterministic():
            result = generate_package(ini, case.run_type, output_root=out, generate_hpc=True,
                                      simulation_root_name="sim")
    finally:
        os.chdir(cwd)
    if result.failures:
        raise RuntimeError(f"{case.name}: build failed: {result.failures[0].error!r}")
    files: Dict[str, str] = {}
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        rel = path.relative_to(out).as_posix()
        if rel == ini.name or rel == "one_material.txt":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            files[rel] = f"<binary sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}>\n"
            continue
        files[rel] = normalise_deck(text.replace(str(out), "<BUILD>"))
    return files


def stored_form(text: str) -> str:
    """What the snapshot keeps for a file: its text (digest only when HASH_BIG is set)."""
    if HASH_BIG and len(text) > _BIG_FILE:
        return "sha256:" + hashlib.sha256(text.encode()).hexdigest() + "\n"
    return text


def write_snapshot(case: RenderCase, files: Dict[str, str]) -> None:
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
