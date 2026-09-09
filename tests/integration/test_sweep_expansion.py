"""A sweep must produce one real directory per value, and a bad one nothing.

Spec Phase 3, item 4. ``tests/run/test_run.py`` covers sweep expansion
thoroughly at the *config* level -- how many dicts come out of
``expand_config_sweeps``, and which keys raise. What it does not do is build:
two expanded configs that map to the same output directory silently overwrite
each other, and the config-level count still reads correctly. That failure mode
is real -- a list-valued ``scan_angle_force`` did exactly this until the
``LAMMPS_LOOP_PARAMS`` derivation in ``run/naming.py`` was tied to
``LOOPABILITY`` (see its comment).

``pes_sheet_config.ini`` is the vehicle: 0.1 s per member, so a three-value
sweep is still the cheapest thing in this tier.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.run.artifacts import generate_package
from tests.helpers.examples import EXAMPLES, REPO_ROOT

pytestmark = pytest.mark.integration

BASE, MODEL = 'pes_sheet_config.ini', 'pes_sheet'
TEMPS = [100, 200, 300]


def _config_with(tmp_path: Path, section: str, key: str, value: str) -> Path:
    """A copy of the base example with one ``[section] key`` set or replaced.

    Paths inside stay repo-relative, and :func:`_generate` runs from the repo
    root, so the copy resolves its CIF and potential exactly as the original does.
    """
    text = (EXAMPLES / BASE).read_text(encoding='utf-8')
    line = f"{key} = {value}"
    pattern = re.compile(rf'^{re.escape(key)}\s*=.*$', re.M)
    if pattern.search(text):
        text = pattern.sub(line, text, count=1)
    else:
        text = re.sub(rf'^\[{re.escape(section)}\]$', f"[{section}]\n{line}",
                      text, count=1, flags=re.M)
        assert line in text, f"no [{section}] section to add {key} to"
    out = tmp_path / BASE
    out.write_text(text, encoding='utf-8')
    return out


def _generate(config: Path, output_root: Path):
    import os
    cwd = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        return generate_package(config, MODEL, output_root=output_root,
                                generate_hpc=False,
                                allow_generate_amorphous=False,
                                simulation_root_name='sim')
    finally:
        os.chdir(cwd)


def test_an_n_value_sweep_writes_n_distinct_directories(tmp_path):
    config = _config_with(tmp_path, 'general', 'temp', str(TEMPS))
    result = _generate(config, tmp_path / 'out')
    assert result.ok, result.render()
    assert len(result.created) == len(TEMPS)
    assert len({str(p) for p in result.created}) == len(TEMPS), [
        str(p) for p in result.created]


def test_each_swept_directory_carries_its_own_value_in_its_name(tmp_path):
    """The token is what keeps two members apart on disk. Without it they
    collide and the last build silently wins."""
    config = _config_with(tmp_path, 'general', 'temp', str(TEMPS))
    result = _generate(config, tmp_path / 'out')
    names = {str(Path(p).relative_to(result.simulation_root))
             for p in result.created}
    for temp in TEMPS:
        assert any(f"K{temp}" in name for name in names), (temp, names)


def test_each_swept_directory_holds_its_own_rendered_script(tmp_path):
    config = _config_with(tmp_path, 'general', 'temp', str(TEMPS))
    result = _generate(config, tmp_path / 'out')
    for member in result.created:
        scripts = sorted(Path(member).rglob('lammps/*.in'))
        assert scripts, member


def test_a_non_loopable_parameter_sweep_is_rejected_by_name(tmp_path):
    """`driving_spring` is not in `run/naming.py`'s LOOPABILITY table, so a list
    for it is neither a sweep nor an in-script loop. Silently taking the first
    value, or silently sweeping it into colliding directories, are the two
    failure modes the error exists to prevent -- so the message must name the
    key."""
    config = _config_with(tmp_path, 'general', 'driving_spring', '[10, 20]')
    with pytest.raises(ValueError, match='driving_spring'):
        _generate(config, tmp_path / 'out')


def test_a_rejected_sweep_writes_nothing_at_all(tmp_path):
    """Rejection has to happen before any directory is created; a half-written
    tree is worse than none, because `hpc generate` would then package it."""
    config = _config_with(tmp_path, 'general', 'driving_spring', '[10, 20]')
    output_root = tmp_path / 'out'
    output_root.mkdir()
    with pytest.raises(ValueError):
        _generate(config, output_root)
    assert list(output_root.rglob('*.in')) == []
