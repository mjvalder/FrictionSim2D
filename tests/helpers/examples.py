"""The shipped ``examples/*.ini`` set, and how to build one for a test.

Shared by every module in ``tests/integration/``: each config is built at most
once per session (the ``build_example`` fixture in that package's conftest holds
the cache), because ``generate_package`` is the slow part of the tier.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

from frictionsim2d.run.artifacts import generate_package

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / 'examples'

#: (config file, model, expected members). "Members" is
#: ``GenerationResult.created`` -- one per expanded config, not per layer.
CASES: List[Tuple[str, str, int]] = [
    ('afm_config.ini', 'afm', 1),
    ('afm_flake_config.ini', 'afm', 1),
    ('afm_freestanding_pes_config.ini', 'pes_tip', 1),
    ('indent_config.ini', 'indent', 1),
    ('pes_sheet_config.ini', 'pes_sheet', 1),
    ('pes_tip_config.ini', 'pes_tip', 1),
    ('sheet_config.ini', 'sheetonsheet', 1),
    ('sheet_config_interlayer.ini', 'sheetonsheet', 3),
]

#: (config file, model, members of the REAL file). Built one-material: 156 real
#: members cost 31 s and 98 s and render the same script shape, so the count is
#: asserted with ``count_expansion`` instead (Phase 3 decision D4).
CAMPAIGNS: List[Tuple[str, str, int]] = [
    ('afm_freestanding_pes_campaign.ini', 'pes_tip', 156),
    ('indent_campaign.ini', 'indent', 156),
]

#: Needs ``lmp_serial`` at BUILD time (``hetero_stack.run_lammps``), so every
#: test that uses it must carry an ``lmp_serial`` skipif -- ``conftest.py``
#: derives the ``lammps`` marker from that guard.
HETERO: Tuple[str, str, int] = ('hetero_pair_sweep.ini', 'sheetonsheet', 3)

CAMPAIGN_NAMES = {name for name, _, _ in CAMPAIGNS}


def one_material_copy(name: str, dest: Path) -> Path:
    """A campaign config restricted to the first material of its list.

    Only ``materials_list`` changes; every other line, including the ``{mat}``
    markers, is the original's, so the member this builds is the one the real
    campaign would build first.
    """
    original = (EXAMPLES / name).read_text(encoding='utf-8')
    first = next(line for line in
                 (EXAMPLES / 'material_list.txt').read_text(
                     encoding='utf-8').splitlines() if line.strip())
    short_list = dest / 'one_material.txt'
    short_list.write_text(f"{first}\n", encoding='utf-8')
    trimmed = re.sub(r'^materials_list\s*=.*$',
                     f'materials_list = {short_list}', original,
                     count=1, flags=re.M)
    assert trimmed != original, f"{name}: no materials_list line to trim"
    out = dest / name
    out.write_text(trimmed, encoding='utf-8')
    return out


def build(name: str, model: str, root: Path, **overrides):
    """Generate one example into ``root`` and return the ``GenerationResult``.

    ``overrides`` are passed straight through to ``generate_package`` (e.g.
    ``settings_file=``); the defaults below are what the tier wants everywhere.

    ``os.chdir`` to the repo root is required, not cosmetic: the example configs
    spell ``cif_path``/``pot_path`` relative to it, the same way
    ``tests/run/test_orchestration.py`` does it.
    """
    config = (one_material_copy(name, root) if name in CAMPAIGN_NAMES
              else EXAMPLES / name)
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        result = generate_package(
            config, model,
            output_root=root,
            generate_hpc=False,
            allow_generate_amorphous=False,
            simulation_root_name='sim',
            **overrides,
        )
    finally:
        os.chdir(cwd)
    assert not result.failures, "; ".join(f.render() for f in result.failures)
    return result
