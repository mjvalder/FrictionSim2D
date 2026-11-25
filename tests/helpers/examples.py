"""The shipped ``examples/<mode>/*.ini`` set, and how to build one for a test.

Two sets live here. The LIVE examples are discovered, never listed:
:func:`discover` walks ``examples/<mode>/*.ini`` and derives what a test needs
from the folder name and the file itself, so a new example is tested the moment
it exists. The FROZEN set (``tests/fixtures/examples/``, flat names) is what the
deck snapshot and every exact-value test read, so rewriting ``examples/`` never
moves them.

Shared by every module in ``tests/integration/``: each config is built at most
once per session (the ``build_example`` fixture in that package's conftest holds
the cache), because ``generate_package`` is the slow part of the tier.
"""
from __future__ import annotations

import configparser
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from frictionsim2d.core.mlip import is_mlip
from frictionsim2d.run.artifacts import generate_package

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / 'examples'
#: A frozen copy of ``examples/`` as of the Task 22 rewrite (2026-11-25): the
#: deck snapshot and every test that asserts an example's exact values read
#: this, so rewriting ``examples/`` never moves them. Identical to the
#: originals except that ``materials_list`` names the copy's own list file
#: (repo-root relative, as the build resolves it against the working directory).
FROZEN_EXAMPLES = REPO_ROOT / 'tests' / 'fixtures' / 'examples'

#: The ``FrictionSim2D run`` subcommands; one ``examples/`` folder each.
MODES: Tuple[str, ...] = ('afm', 'sheetonsheet', 'pes-tip', 'pes-sheet', 'indent', 'quench')

#: A list longer than this makes an example a campaign: built one-material,
#: its full member count asserted with ``count_expansion`` (Phase 3 decision D4).
CAMPAIGN_MIN_ENTRIES = 10

_SETTINGS_RE = re.compile(r'^#\s*settings:\s*(\S+)', re.M)
_MATERIALS_LIST_RE = re.compile(r'^\s*materials_list\s*=\s*([^#\s]+)', re.M)
_PAIR_SECTION_RE = re.compile(r'-[12]$')


@dataclass(frozen=True)
class Example:
    """One live example and what a test needs to know to build it."""
    path: Path
    rel: str
    mode: str
    model: str
    settings: Optional[Path]
    hetero: bool
    mlip: bool
    campaign: bool


def list_entries(text: str) -> int:
    """Non-blank lines of the list a config's ``materials_list`` names (0 without one)."""
    match = _MATERIALS_LIST_RE.search(text)
    if not match:
        return 0
    listed = REPO_ROOT / match.group(1)
    if not listed.is_file():
        return 0
    return sum(1 for line in listed.read_text(encoding='utf-8').splitlines() if line.strip())


def _example(path: Path, root: Path) -> Example:
    text = path.read_text(encoding='utf-8')
    mode = path.parent.name
    settings = None
    match = _SETTINGS_RE.search(text)
    if match:
        settings = REPO_ROOT / match.group(1)
        if not settings.is_file():
            raise ValueError(f"{path}: '# settings: {match.group(1)}' does not exist")
    parser = configparser.ConfigParser(inline_comment_prefixes=('#',), interpolation=None)
    parser.read_string(text)
    return Example(
        path=path,
        rel=path.relative_to(root).as_posix(),
        mode=mode,
        model=mode.replace('-', '_'),
        settings=settings,
        hetero=any(_PAIR_SECTION_RE.search(s) for s in parser.sections()),
        mlip=any(is_mlip(parser[s].get('pot_type', '')) for s in parser.sections()),
        campaign=list_entries(text) > CAMPAIGN_MIN_ENTRIES,
    )


def discover(root: Path = EXAMPLES) -> List[Example]:
    """Every ``<root>/<mode>/*.ini``, sorted by relative path.

    Raises:
        ValueError: if a ``# settings:`` header names a file that does not exist
            -- a silent fall-back to default settings would test the wrong run.
    """
    return [_example(p, root) for p in sorted(root.glob('*/*.ini'))]


def friction_examples() -> List[Example]:
    """:func:`discover` minus ``quench/`` (no friction model reads a [quench] config)."""
    return [e for e in discover() if e.mode != 'quench']


#: The FROZEN set: (config file, model, expected members). "Members" is
#: ``GenerationResult.created`` -- one per expanded config, not per layer.
FROZEN_CASES: List[Tuple[str, str, int]] = [
    ('afm_config.ini', 'afm', 1),
    ('afm_flake_config.ini', 'afm', 1),
    ('afm_freestanding_pes_config.ini', 'pes_tip', 1),
    ('indent_config.ini', 'indent', 1),
    ('pes_sheet_config.ini', 'pes_sheet', 1),
    ('pes_tip_config.ini', 'pes_tip', 1),
    ('sheet_config.ini', 'sheetonsheet', 1),
    ('sheet_config_interlayer.ini', 'sheetonsheet', 3),
]

#: (config file, model, members of the REAL file) in the frozen set.
FROZEN_CAMPAIGNS: List[Tuple[str, str, int]] = [
    ('afm_freestanding_pes_campaign.ini', 'pes_tip', 156),
    ('indent_campaign.ini', 'indent', 156),
]

#: Needs ``lmp_serial`` at BUILD time (``hetero_stack.run_lammps``).
FROZEN_HETERO: Tuple[str, str, int] = ('hetero_pair_sweep.ini', 'sheetonsheet', 3)

CAMPAIGN_NAMES = {name for name, _, _ in FROZEN_CAMPAIGNS}


def one_material_copy(name: str, dest: Path, source: Path = EXAMPLES) -> Path:
    """A campaign config restricted to the first material of its list.

    Only ``materials_list`` changes; every other line, including the ``{mat}``
    markers, is the original's, so the member this builds is the one the real
    campaign would build first.
    """
    original = (source / name).read_text(encoding='utf-8')
    listed = REPO_ROOT / _MATERIALS_LIST_RE.search(original).group(1)
    first = next(line for line in
                 listed.read_text(encoding='utf-8').splitlines() if line.strip())
    dest.mkdir(parents=True, exist_ok=True)
    short_list = dest / 'one_material.txt'
    short_list.write_text(f"{first}\n", encoding='utf-8')
    trimmed = re.sub(r'^materials_list\s*=.*$',
                     f'materials_list = {short_list}', original,
                     count=1, flags=re.M)
    assert trimmed != original, f"{name}: no materials_list line to trim"
    out = dest / Path(name).name
    out.write_text(trimmed, encoding='utf-8')
    return out


def build(name: str, model: str, root: Path, source: Path = FROZEN_EXAMPLES, **overrides):
    """Generate one example into ``root`` and return the ``GenerationResult``.

    ``source`` is the folder the config is read from: the live ``examples/``
    (the parses-and-renders contract) or :data:`FROZEN_EXAMPLES` (a test that
    asserts the example's exact values). ``overrides`` are passed straight through to ``generate_package`` (e.g.
    ``settings_file=``); the defaults below are what the tier wants everywhere.

    ``os.chdir`` to the repo root is required, not cosmetic: the example configs
    spell ``materials_list`` relative to it, the same way
    ``tests/run/test_orchestration.py`` does it.
    """
    config = (one_material_copy(name, root, source) if name in CAMPAIGN_NAMES
              else source / name)
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        result = generate_package(
            config, model,
            output_root=root,
            generate_hpc=False,
            simulation_root_name='sim',
            **overrides,
        )
    finally:
        os.chdir(cwd)
    assert not result.failures, "; ".join(f.render() for f in result.failures)
    return result


def build_example_file(example: Example, root: Path, **overrides):
    """Generate one LIVE example into ``root``: its settings file, campaigns one-material."""
    if example.settings is not None:
        overrides.setdefault('settings_file', example.settings)
    if example.campaign:
        config = one_material_copy(example.rel, root / 'campaign_copy', EXAMPLES)
        return build(config.name, example.model, root, source=config.parent, **overrides)
    return build(example.rel, example.model, root, source=EXAMPLES, **overrides)
