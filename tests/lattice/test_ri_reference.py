"""Characterisation: the registry-index scan reproduces the recorded reference grids.

The reference (``data/ri_reference.npz``) was recorded from the original
pure-Python loop implementation, before it was vectorised.
"""
from pathlib import Path

import numpy as np
import pytest

from frictionsim2d.lattice.matching import scan_registry_index

from ._ri_cases import cases

_REF = np.load(Path(__file__).parent / "data" / "ri_reference.npz")


@pytest.mark.parametrize("name", list(cases()))
def test_ri_grid_and_argmin_match_reference(name):
    cart_a, cart_b, cell, n_grid = cases()[name]
    r = scan_registry_index(cart_a, cart_b, cell, n_grid=n_grid)
    np.testing.assert_allclose(r.ri_grid, _REF[f"{name}_grid"], rtol=0, atol=1e-12)
    idx = np.array([np.unravel_index(np.argmax(r.ri_grid), r.ri_grid.shape),
                    np.unravel_index(np.argmin(r.ri_grid), r.ri_grid.shape)])
    np.testing.assert_array_equal(idx, _REF[f"{name}_idx"])
    np.testing.assert_allclose(r.aa_shift, _REF[f"{name}_aa"], atol=1e-12)
    np.testing.assert_allclose(r.ab_shift, _REF[f"{name}_ab"], atol=1e-12)
