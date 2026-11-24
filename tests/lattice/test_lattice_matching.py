"""Tests for frictionsim2d/lattice/matching.py — Zur-McGill coincidence lattice."""
import math
from typing import Tuple

import numpy as np
import pytest

from frictionsim2d.core.config import LatticeMatchConfig
from frictionsim2d.lattice.matching import (
    MatchResult,
    StackingResult,
    cell_from_params,
    find_coincidence_lattice,
    scan_registry_index,
    max_strain,
    _enumerate_matrices,
    _circle_overlap_area,
    relative_rotation_strain,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hex_cell(a: float) -> np.ndarray:
    """Hexagonal unit cell with a=b, γ=120°."""
    return cell_from_params(a, a, 120.0)

def _rect_cell(a: float, b: float) -> np.ndarray:
    """Orthorhombic cell with γ=90°."""
    return cell_from_params(a, b, 90.0)


# ---------------------------------------------------------------------------
# cell_from_params
# ---------------------------------------------------------------------------

class TestCellFromParams:
    def test_orthorhombic_axes(self):
        cell = cell_from_params(3.0, 4.0, 90.0)
        # a-vector along x
        assert math.isclose(cell[0, 0], 3.0, abs_tol=1e-10)
        assert math.isclose(cell[0, 1], 0.0, abs_tol=1e-10)
        # b-vector perpendicular
        assert math.isclose(cell[1, 0], 0.0, abs_tol=1e-6)
        assert math.isclose(cell[1, 1], 4.0, abs_tol=1e-6)

    def test_hexagonal_angle(self):
        a = 3.18
        cell = cell_from_params(a, a, 120.0)
        # |a| and |b| both equal a
        assert math.isclose(np.linalg.norm(cell[0]), a, rel_tol=1e-9)
        assert math.isclose(np.linalg.norm(cell[1]), a, rel_tol=1e-9)
        # angle between rows ≈ 120°
        cos_angle = np.dot(cell[0], cell[1]) / (a * a)
        assert math.isclose(cos_angle, math.cos(math.radians(120.0)), abs_tol=1e-9)

    def test_area_equals_determinant(self):
        cell = cell_from_params(3.0, 4.0, 90.0)
        assert math.isclose(abs(np.linalg.det(cell)), 12.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# max_strain
# ---------------------------------------------------------------------------

class TestMaxStrain:
    def test_identical_cells_zero_strain(self):
        cell = _hex_cell(3.18)
        assert math.isclose(max_strain(cell, cell), 0.0, abs_tol=1e-9)

    def test_scaled_cell_gives_strain(self):
        cell = _rect_cell(3.0, 4.0)
        stretched = _rect_cell(3.03, 4.0)   # 1% strain along a
        s = max_strain(cell, stretched)
        assert 0.005 < s < 0.02

    def test_singular_cell_returns_inf(self):
        good = _hex_cell(3.18)
        singular = np.zeros((2, 2))
        assert max_strain(good, singular) == math.inf


# ---------------------------------------------------------------------------
# _enumerate_matrices
# ---------------------------------------------------------------------------

class TestEnumerateMatrices:
    def test_identity_always_included(self):
        identity = np.eye(2, dtype=int)
        matrices = list(_enumerate_matrices(5))
        assert any(np.array_equal(m, identity) for m in matrices)

    def test_all_positive_determinant(self):
        for m in _enumerate_matrices(4):
            assert np.linalg.det(m) > 0

    def test_upper_triangular(self):
        for m in _enumerate_matrices(4):
            assert m[1, 0] == 0, f"Not upper-triangular: {m}"

    def test_count_grows_with_n(self):
        n3 = len(list(_enumerate_matrices(3)))
        n5 = len(list(_enumerate_matrices(5)))
        assert n5 > n3


def test_enumerate_matrices_n3_exact_determinant_multiset():
    """Pins the exact set of determinants produced for n=3 and their shape.

    `_enumerate_matrices` iterates a, c in [1, n], so det = a*c ranges over
    [1, n^2] by construction -- a test that only checks that bound can never
    fail. This instead pins the concrete determinant multiset for n=3 (a
    real change to the a/c ranges or the `det = a * c` computation would
    change it) and that every yielded matrix stays upper-triangular with a
    positive determinant.
    """
    n = 3
    mats = list(_enumerate_matrices(n))
    dets = [int(round(np.linalg.det(mat))) for mat in mats]
    assert sorted(set(dets)) == [1, 2, 3, 4, 6, 9]
    for mat in mats:
        assert mat[1, 0] == 0, f"Not upper-triangular: {mat}"
        assert np.linalg.det(mat) > 0


# ---------------------------------------------------------------------------
# find_coincidence_lattice — identical cells
# ---------------------------------------------------------------------------

class TestFindCoincidenceLatticeIdentical:
    """When both cells are the same material, the 1×1 identity should match."""

    def test_same_cell_finds_match(self):
        cell = _hex_cell(3.18)
        result = find_coincidence_lattice(cell, cell, strain_tol=0.01)
        assert result is not None

    def test_same_cell_near_zero_strain(self):
        cell = _hex_cell(3.18)
        result = find_coincidence_lattice(cell, cell, strain_tol=0.01)
        assert result.strain_a < 1e-6
        assert result.strain_b < 1e-6

    def test_same_cell_identity_matrices(self):
        cell = _hex_cell(3.18)
        result = find_coincidence_lattice(cell, cell, strain_tol=0.01)
        assert np.array_equal(result.matrix_a, np.eye(2, dtype=int))
        assert np.array_equal(result.matrix_b, np.eye(2, dtype=int))

    def test_area_matches_unit_cell(self):
        cell = _hex_cell(3.18)
        expected_area = abs(np.linalg.det(cell))
        result = find_coincidence_lattice(cell, cell, strain_tol=0.01)
        assert math.isclose(result.area, expected_area, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# find_coincidence_lattice — known commensurate pairs
# ---------------------------------------------------------------------------

class TestFindCoincidenceLatticePairs:
    def test_graphene_mos2_finds_match(self):
        """Graphene (a=2.46 Å) on MoS2 (a=3.18 Å): known ~14% strain pair."""
        cell_gr = _hex_cell(2.46)
        cell_mos2 = _hex_cell(3.18)
        # With generous tolerances a match should be found
        result = find_coincidence_lattice(cell_gr, cell_mos2, strain_tol=0.15, max_supercell=8)
        assert result is not None
        assert result.strain_a <= 0.15
        assert result.strain_b <= 0.15

    def test_small_mismatch_finds_small_supercell(self):
        """Two nearly-identical hexagonal cells should find a 1×1 match."""
        cell_a = _hex_cell(3.18)
        cell_b = _hex_cell(3.20)   # 0.6% mismatch
        result = find_coincidence_lattice(cell_a, cell_b, strain_tol=0.01, max_supercell=5)
        assert result is not None
        # Expect trivial 1×1 since the mismatch is within tolerance
        assert result.matrix_a[0, 0] == 1 and result.matrix_b[0, 0] == 1

    def test_no_match_for_tight_tolerance(self):
        """Very tight tolerance and unrelated cells should return None."""
        cell_a = _hex_cell(2.46)
        cell_b = _rect_cell(4.0, 6.0)
        result = find_coincidence_lattice(cell_a, cell_b, strain_tol=1e-4, max_supercell=3)
        assert result is None

    def test_result_area_positive(self):
        cell = _hex_cell(3.18)
        result = find_coincidence_lattice(cell, cell, strain_tol=0.01)
        assert result.area > 0.0

    def test_supercell_matrices_shape(self):
        cell = _hex_cell(3.18)
        result = find_coincidence_lattice(cell, cell, strain_tol=0.01)
        assert result.matrix_a.shape == (2, 2)
        assert result.matrix_b.shape == (2, 2)
        assert result.supercell_a.shape == (2, 2)
        assert result.supercell_b.shape == (2, 2)


def test_find_coincidence_lattice_skips_degenerate_sub_angstrom_cells():
    """A sub-Å² cell area (finding 8's `area < 1.0` guard) rules out every
    candidate regardless of tolerance, so no match is ever found."""
    tiny = np.array([[0.1, 0.0], [0.0, 0.1]])
    assert find_coincidence_lattice(tiny, np.eye(2) * 3.0, max_supercell=2) is None


# ---------------------------------------------------------------------------
# LatticeMatchConfig
# ---------------------------------------------------------------------------

class TestLatticeMatchConfig:
    def test_defaults(self):
        cfg = LatticeMatchConfig()
        assert cfg.method == 'zur_mcgill'
        assert cfg.strain_tol == 0.02
        assert cfg.max_supercell == 10
        assert cfg.area_tol == 0.10

    def test_custom_values(self):
        cfg = LatticeMatchConfig(strain_tol=0.05, max_supercell=6)
        assert cfg.strain_tol == 0.05
        assert cfg.max_supercell == 6

    def test_invalid_strain_tol_rejected(self):
        with pytest.raises(Exception):
            LatticeMatchConfig(strain_tol=2.0)   # > 1.0 is invalid

    def test_invalid_max_supercell_rejected(self):
        with pytest.raises(Exception):
            LatticeMatchConfig(max_supercell=0)   # must be >= 1


# ---------------------------------------------------------------------------
# _circle_overlap_area
# ---------------------------------------------------------------------------

class TestCircleOverlapArea:
    def test_identical_centres_full_overlap(self):
        r = 1.5
        assert math.isclose(_circle_overlap_area(0.0, r), math.pi * r * r, rel_tol=1e-9)

    def test_no_overlap_beyond_cutoff(self):
        assert _circle_overlap_area(3.1, 1.5) == 0.0

    def test_at_cutoff_zero(self):
        assert _circle_overlap_area(3.0, 1.5) == 0.0

    def test_partial_overlap_monotone(self):
        r = 1.5
        a1 = _circle_overlap_area(0.5, r)
        a2 = _circle_overlap_area(1.5, r)
        a3 = _circle_overlap_area(2.5, r)
        assert a1 > a2 > a3


# ---------------------------------------------------------------------------
# scan_registry_index — qualitative tests on a hexagonal monolayer
# ---------------------------------------------------------------------------

def _graphene_positions() -> Tuple[np.ndarray, np.ndarray]:
    """Return Cartesian positions of graphene A-B sublattices in a unit cell."""
    a = 2.46
    cell = cell_from_params(a, a, 120.0)
    frac = np.array([[0.0, 0.0], [1/3, 2/3]])
    cart = frac @ cell
    return cart, cell


class TestScanRegistryIndex:
    def test_returns_stacking_result(self):
        cart, cell = _graphene_positions()
        result = scan_registry_index(cart, cart, cell, n_grid=10)
        assert isinstance(result, StackingResult)

    def test_grid_shape(self):
        cart, cell = _graphene_positions()
        result = scan_registry_index(cart, cart, cell, n_grid=12)
        assert result.ri_grid.shape == (12, 12)

    def test_aa_ri_is_maximum(self):
        cart, cell = _graphene_positions()
        result = scan_registry_index(cart, cart, cell, n_grid=15)
        assert math.isclose(result.aa_ri, result.ri_grid.max(), rel_tol=1e-9)

    def test_ab_ri_is_minimum(self):
        cart, cell = _graphene_positions()
        result = scan_registry_index(cart, cart, cell, n_grid=15)
        assert math.isclose(result.ab_ri, result.ri_grid.min(), rel_tol=1e-9)

    def test_ri_normalised_to_one(self):
        cart, cell = _graphene_positions()
        result = scan_registry_index(cart, cart, cell, n_grid=15)
        assert math.isclose(result.aa_ri, 1.0, abs_tol=1e-9)

    def test_zero_shift_is_aa_registry_for_same_layer(self):
        """Placing B directly on A (u=0) should give maximum registry (RI=1)."""
        cart, cell = _graphene_positions()
        result = scan_registry_index(cart, cart, cell, n_grid=20)
        # Find the grid point closest to shift=(0,0) and check RI is close to max.
        i0 = np.unravel_index(
            np.argmin(result.shifts_x**2 + result.shifts_y**2), result.ri_grid.shape
        )
        assert result.ri_grid[i0] > 0.9

    def test_ab_ri_less_than_aa_ri(self):
        cart, cell = _graphene_positions()
        result = scan_registry_index(cart, cart, cell, n_grid=15)
        assert result.ab_ri < result.aa_ri

    def test_shifts_x_y_same_shape_as_grid(self):
        cart, cell = _graphene_positions()
        result = scan_registry_index(cart, cart, cell, n_grid=10)
        assert result.shifts_x.shape == result.ri_grid.shape
        assert result.shifts_y.shape == result.ri_grid.shape

    def test_all_zero_grid_is_returned_unscaled(self):
        """raw_max == 0 must not divide by zero and must return a valid,
        un-normalised (all-zero) grid (finding 11). atomic_radius=0.0 makes
        every pairwise disk-overlap zero regardless of the atoms' positions,
        so this reaches the `raw_max <= 0.0` fallback (matching.py's
        `scan_registry_index`) without needing artificially separated atoms."""
        cart, cell = _graphene_positions()
        result = scan_registry_index(cart, cart, cell, n_grid=4, atomic_radius=0.0)
        assert result.ri_grid.shape == (4, 4)
        assert np.all(result.ri_grid == 0.0)
        assert not np.isnan(result.ri_grid).any()
        assert result.aa_ri == 0.0 and result.ab_ri == 0.0


# --- D2b: fixed-reference substrate match (match_to_fixed_box) ---

def test_match_to_fixed_box_pins_box_and_strains_other():
    from frictionsim2d.lattice.matching import match_to_fixed_box, FixedBoxMatch
    # box = a 20x22 orthogonal hetero cell; other = a ~4.1x4.5 crystalline surface cell.
    # (A perfect 4x4 cell is genuinely incommensurate with 22 -> 22/4 = 5.5, min
    # residual 0.083 > 0.05; these near-commensurate lengths let a 5x5 supercell
    # strain onto the box within tol while still exercising a nonzero strain.)
    box = np.array([[20.0, 0.0], [0.0, 22.0]])
    other = np.array([[4.1, 0.0], [0.0, 4.5]])
    m = match_to_fixed_box(box, other, strain_tol=0.05, max_supercell=8)
    assert isinstance(m, FixedBoxMatch)
    # box_cell returned unchanged
    assert np.allclose(m.box_cell, box)
    # ASSERT THE ALGEBRA: the integer supercell of `other`, once strain-closed, lands on `box`.
    # Use the module's confirmed row-vector convention sc = M @ cell (matching.py:163).
    sup = m.supercell_matrix @ other                       # 2x2 supercell BEFORE strain
    from frictionsim2d.lattice.matching import max_strain as _ms
    assert abs(_ms(box, sup) - m.residual_strain) < 1e-9   # residual is the box<-sup strain
    # and the residual is within tol -> the strained supercell equals the box within that strain
    assert m.residual_strain <= 0.05 + 1e-9


def test_match_to_fixed_box_fails_loud_when_incommensurate():
    from frictionsim2d.lattice.matching import match_to_fixed_box
    box = np.array([[20.0, 0.0], [0.0, 22.0]])
    other = np.array([[7.31, 0.0], [0.0, 3.17]])   # no low-order supercell within tol
    with pytest.raises(ValueError, match=r"strain"):
        match_to_fixed_box(box, other, strain_tol=0.001, max_supercell=3)


# =============================================================================
# D3: commensurate twist search + orthorhombic wrapper
# =============================================================================

def _hex_cell(a=3.16):
    return np.array([[a, 0.0], [a * 0.5, a * np.sqrt(3) / 2]])


def test_find_commensurate_twist_emits_orthorhombic_box_near_target():
    from frictionsim2d.lattice.matching import find_commensurate_twist, TwistMatch
    A = _hex_cell(); B = _hex_cell()
    m = find_commensurate_twist(A, B, target_angle=7.0, angle_tol=3.0,
                                strain_tol=0.03, max_supercell=12)
    assert isinstance(m, TwistMatch)
    assert abs(m.achieved_angle_deg - 7.0) <= 3.0        # within tol
    assert m.residual_strain <= 0.03
    assert m.moire_period > 0
    # the EMITTED box is orthorhombic (downstream needs orthogonal): xy tilt ~ 0
    assert abs(m.moire_cell[1, 0]) < 1e-6 * abs(m.moire_cell[0, 0])
    assert m.ortho_tiling is not None                     # a real rhombus->rectangle wrapper was applied


def test_find_commensurate_twist_fails_loud_names_nearest():
    from frictionsim2d.lattice.matching import find_commensurate_twist
    A = _hex_cell(); B = _hex_cell()
    with pytest.raises(ValueError, match=r"nearest|achievable"):
        find_commensurate_twist(A, B, target_angle=7.0, angle_tol=0.01,
                                strain_tol=0.001, max_supercell=4)


def test_orthorhombic_wrapper_tiles_rhombic_moire():
    from frictionsim2d.lattice.matching import orthorhombic_wrapper
    rhombus = _hex_cell(a=30.0)                            # a moire-scale rhombus (60 deg)
    ortho, tiling = orthorhombic_wrapper(rhombus)
    assert abs(ortho[1, 0]) < 1e-6                         # rectangular
    # the wrapper area is an integer multiple of the rhombus area (genuine periodic tiling)
    ratio = abs(np.linalg.det(ortho)) / abs(np.linalg.det(rhombus))
    assert abs(ratio - round(ratio)) < 1e-6 and round(ratio) >= 1


def test_rotation_angle_between_recovers_the_applied_rotation():
    from frictionsim2d.lattice.matching import rotation_angle_between
    A = _hex_cell()
    theta = 15.0
    r = math.radians(theta)
    R = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    B = A @ R.T                                            # A rotated by +15 deg
    assert abs(rotation_angle_between(A, B) - (-theta)) < 1e-6 \
        or abs(rotation_angle_between(A, B) - theta) < 1e-6


def test_relative_rotation_strain_singular_supercell_returns_zero_angle_inf_strain():
    """A degenerate (zero-area) sc_a makes the polar stretch tensor singular,
    so the LinAlgError fallback (finding 9) fires: (0.0, inf)."""
    angle, strain = relative_rotation_strain(np.zeros((2, 2)), np.eye(2))
    assert angle == 0.0 and math.isinf(strain)


def test_scan_registry_index_reports_the_ab_offset_and_aa_at_the_origin():
    cart, cell = _graphene_positions()
    result = scan_registry_index(cart, cart, cell, n_grid=12)
    frac_aa = np.linalg.solve(cell.T, result.aa_shift) % 1.0
    frac_ab = np.linalg.solve(cell.T, result.ab_shift) % 1.0
    assert np.allclose(np.minimum(frac_aa, 1.0 - frac_aa), 0.0, atol=1e-9)
    # The 12-point scan minimum of this disk-overlap model is at (1/2, 7/12); the
    # (1/3, 1/3) site the sublattice offset suggests has a clearly higher RI.
    assert np.allclose(frac_ab, [0.5, 7 / 12])
    assert result.ri_grid[4, 4] > result.ab_ri + 0.2


def test_orthorhombic_wrapper_raises_when_no_tiling_fits():
    from frictionsim2d.lattice.matching import orthorhombic_wrapper
    skew = np.array([[1.0, 0.0], [math.sqrt(2.0) * 0.5, 1.0]])
    with pytest.raises(ValueError, match="no orthogonalising integer tiling"):
        orthorhombic_wrapper(skew, max_tiling=1)


def test_gauss_reduce_2d_returns_a_right_handed_integer_transform_of_the_cell():
    from frictionsim2d.lattice.matching import _gauss_reduce_2d
    for cell in (np.array([[3.0, 0.0], [1.5, 2.598]]),
                 np.array([[3.0, 0.0], [10.5, 2.598]]),     # skewed basis of the same lattice
                 np.array([[5.0, 1.0], [14.0, 3.5]])):
        reduced, unimod = _gauss_reduce_2d(cell)
        assert np.allclose(reduced, unimod @ cell)
        assert round(abs(np.linalg.det(unimod))) == 1
        assert np.linalg.det(reduced) > 0
