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
from src.builders.components import extract_substrate_surface_cell

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
