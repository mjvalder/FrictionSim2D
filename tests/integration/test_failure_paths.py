"""Cross-script failures must fail loudly, early, and for their own reason.

Spec Phase 3, item 6. Each of these is a mistake a user makes at the ``.ini``
level and only finds out about after the build -- or, worse, not at all. The
value here is not that an exception is raised (unit tests cover that) but that
it is raised *before anything is written*: a half-built tree is worse than none,
because ``hpc generate`` will happily package it and the cluster will run it.

Every case therefore asserts two things -- the message names the problem, and
the output root holds no rendered script.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import pytest

from frictionsim2d.core.config import GlobalSettings
from frictionsim2d.run.artifacts import generate_package
from frictionsim2d.run.hpc_dispatch import generate_hpc_scripts_for_root
from tests.helpers.examples import EXAMPLES, REPO_ROOT

pytestmark = pytest.mark.integration

AFM_BASE, AFM_MODEL = 'afm_flake_config.ini', 'afm'


def _edited(tmp_path: Path, base: str, section: str, key: str,
            value: Optional[str]) -> Path:
    """A copy of ``base`` with ``[section] key`` replaced (or removed).

    The replacement is scoped to the named section, since ``mat``, ``cif_path``
    and ``pot_path`` all appear in several.
    """
    text = (EXAMPLES / base).read_text(encoding='utf-8')
    start = text.index(f"[{section}]")
    nxt = text.find('\n[', start + 1)
    end = len(text) if nxt == -1 else nxt
    body = text[start:end]
    pattern = re.compile(rf'^{re.escape(key)}\s*=.*$', re.M)
    assert pattern.search(body), f"[{section}] has no {key}"
    body = pattern.sub('' if value is None else f"{key} = {value}", body, count=1)
    out = tmp_path / base
    out.write_text(text[:start] + body + text[end:], encoding='utf-8')
    return out


def _generate(config: Path, output_root: Path, model: str = AFM_MODEL):
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        return generate_package(config, model, output_root=output_root,
                                generate_hpc=False,
                                simulation_root_name='sim')
    finally:
        os.chdir(cwd)


def _nothing_written(output_root: Path) -> bool:
    return not list(output_root.rglob('*.in'))


def _expect_rejection(config: Path, output_root: Path, match: str,
                      model: str = AFM_MODEL):
    """Either it raises, or it reports the member as failed -- never both, and
    never silence. ``generate_package`` catches per-member errors and returns
    ``ok=False``; a whole-config error propagates."""
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        result = _generate(config, output_root, model)
    except (ValueError, FileNotFoundError, OSError) as exc:
        assert re.search(match, str(exc), re.I), str(exc)
    else:
        assert not result.ok, result.render()
        assert result.failures
        messages = '\n'.join(str(f.error) for f in result.failures)
        assert re.search(match, messages, re.I), messages
    assert _nothing_written(output_root), sorted(
        str(p) for p in output_root.rglob('*.in'))


def test_a_missing_material_file_is_refused_by_name(tmp_path):
    """The message must carry the path that was not found. A generic "could not
    read the CIF" is what a caller gets if `get_material_path` stops raising and
    hands the missing path downstream, and it does not say which of the three
    `cif_path` entries in an AFM config is wrong."""
    config = _edited(tmp_path, AFM_BASE, '2D', 'cif_path',
                     'examples/materials/not_a_real_material.cif')
    _expect_rejection(config, tmp_path / 'out',
                      r'not_a_real_material\.cif.*not found')


def test_a_missing_potential_file_is_refused_by_name(tmp_path):
    config = _edited(tmp_path, AFM_BASE, '2D', 'pot_path',
                     'examples/potentials/sw/not_a_real_potential.sw')
    _expect_rejection(config, tmp_path / 'out',
                      r'not_a_real_potential\.sw.*not found')


def test_an_unknown_potential_type_is_refused_by_name(tmp_path):
    """It used to be accepted, and landed verbatim in
    `pair_style hybrid sw sw sw unobtainium lj/cut 11.0` -- the build succeeded,
    the tree looked correct, and LAMMPS rejected the run with "Unknown pair
    style" after the job had been transferred, queued and started. The same
    value was already rejected on the potential-SWEEP route
    (`resolve_pot_type`); the plain `[section] pot_type` route reached
    `PotentialManager.is_sheet_lj`, which warned and guessed. Both routes now
    use the one `KNOWN_TYPES` gate.
    """
    config = _edited(tmp_path, AFM_BASE, '2D', 'pot_type', 'unobtainium')
    _expect_rejection(config, tmp_path / 'out',
                      r"'unobtainium' is not a known LAMMPS pair style")


def test_a_footprint_smaller_than_twice_the_tip_diameter_is_refused(tmp_path):
    """The afm-tip-box-sizing rule. A sheet this small lets the tip interact
    with its own periodic image, so the recorded forces are wrong -- and the run
    would otherwise complete and produce plausible-looking numbers."""
    config = _edited(tmp_path, AFM_BASE, '2D', 'x', '40')
    _expect_rejection(config, tmp_path / 'out',
                      r'footprint \(40\.0 x 100\.0\).*smaller than 2x the tip diameter')


def test_a_config_that_builds_can_still_be_refused_by_hpc_generation(tmp_path):
    """The other shape of failure: the tree is complete and correct, and only
    the packaging step refuses. It must leave no half-written `hpc/` behind for
    a later `submit_jobs.sh` to pick up."""
    out = tmp_path / 'out'
    result = _generate(EXAMPLES / 'indent_config.ini', out, 'indent')
    assert result.ok, result.render()
    root = Path(result.simulation_root)

    reason = generate_hpc_scripts_for_root(root, GlobalSettings())
    assert reason is not None
    assert 'Modules' in reason
    assert not (root / 'hpc').exists()
    assert sorted(root.rglob('lammps/*.in')), "the build itself must survive"
