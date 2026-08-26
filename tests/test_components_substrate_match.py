"""Tests for D2b crystalline-substrate matching helpers in components.py.

Covers ``extract_substrate_surface_cell`` (the 2x2 in-plane surface cell + a
one-layer atom set used as the coincidence-search input) and
``build_matched_crystalline_substrate`` (strain the substrate surface cell onto
the fixed hetero box + replicate to thickness, no seam).
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.core.config import SubstrateConfig
from src.builders.construction.components import extract_substrate_surface_cell

MAT, POT = "examples/materials", "examples/potentials/sw"
LMP = shutil.which("lmp_serial")


def _si_sub():
    # NB: SubstrateConfig requires `thickness` (the plan draft wrote `z=12.0`,
    # which is not a field and fails validation) -> use thickness=12.0.
    return SubstrateConfig(mat="Si", cif_path=f"{MAT}/Si.cif", pot_path=f"{POT}/Si.sw",
                           pot_type="sw", amorph="c", thickness=12.0)


def test_extract_surface_cell_is_2x2_and_one_layer(tmp_path):
    cell2x2, mono_path = extract_substrate_surface_cell(_si_sub(), tmp_path)
    assert cell2x2.shape == (2, 2)
    assert np.linalg.det(cell2x2) > 0            # non-degenerate surface cell
    assert Path(mono_path).exists()
    from ase.io import read
    z = read(str(mono_path), format="lammps-data").get_positions()[:, 2]
    assert z.ptp() < 3.0                          # a single surface layer, not a bulk slab


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_matched_crystalline_substrate_equals_hetero_box(tmp_path):
    from ase.io import read
    from src.builders.construction.components import build_matched_crystalline_substrate
    # NB: the plan draft used a 38x44 box, but that is incommensurate with Si's
    # 10.887 A (2x2) surface cell (38/10.887 = 3.49 -> no integer supercell within
    # 2*strain_tol) and would fail loud by design. 33x43 is reachable by a 3x4
    # supercell within tolerance (residual ~0.013), so it exercises the strain path.
    hetero_box = np.array([[33.0, 0.0], [0.0, 43.0]])
    sub = _si_sub()
    p = build_matched_crystalline_substrate(hetero_box, sub, thickness=12.0, build_dir=tmp_path)
    atoms = read(str(p), format="lammps-data")
    cell = atoms.get_cell()
    assert abs(cell[0, 0] - 33.0) < 1e-4 and abs(cell[1, 1] - 43.0) < 1e-4   # box == hetero box
    zspan = atoms.get_positions()[:, 2].ptp()
    assert 8.0 <= zspan <= 16.0                                             # ~12 A thickness
