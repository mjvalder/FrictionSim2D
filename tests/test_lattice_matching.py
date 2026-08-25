"""Tests for src/lattice/matching.py — Zur-McGill coincidence lattice."""
import math
from typing import Tuple

import numpy as np
import pytest

from src.core.config import LatticeMatchConfig
from src.lattice.matching import (
    MatchResult,
    StackingResult,
    cell_from_params,
    find_coincidence_lattice,
    find_coincidence_lattice_from_cif,
    generate_supercell_positions,
    scan_registry_index,
    rotation_angle_from_match,
    _max_strain,
    _enumerate_matrices,
    _circle_overlap_area,
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
# _max_strain
# ---------------------------------------------------------------------------

class TestMaxStrain:
    def test_identical_cells_zero_strain(self):
        cell = _hex_cell(3.18)
        assert math.isclose(_max_strain(cell, cell), 0.0, abs_tol=1e-9)

    def test_scaled_cell_gives_strain(self):
        cell = _rect_cell(3.0, 4.0)
        stretched = _rect_cell(3.03, 4.0)   # 1% strain along a
        s = _max_strain(cell, stretched)
        assert 0.005 < s < 0.02

    def test_singular_cell_returns_inf(self):
        good = _hex_cell(3.18)
        singular = np.zeros((2, 2))
        assert _max_strain(good, singular) == math.inf


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

    def test_mean_strain_is_average(self):
        cell = _hex_cell(3.18)
        result = find_coincidence_lattice(cell, cell, strain_tol=0.01)
        assert math.isclose(result.mean_strain, (result.strain_a + result.strain_b) / 2)

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


# ---------------------------------------------------------------------------
# find_coincidence_lattice_from_cif — parameter tuple path
# ---------------------------------------------------------------------------

class TestFindCoincidenceLatticeFromCif:
    def test_tuple_input_matches(self):
        result = find_coincidence_lattice_from_cif(
            lat_a=(3.18, 3.18, 120.0),
            lat_b=(3.18, 3.18, 120.0),
            strain_tol=0.01,
        )
        assert result is not None
        assert result.strain_a < 1e-6

    def test_cif_path_input(self, tmp_path):
        cif = tmp_path / "test.cif"
        cif.write_text("""\
data_test
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   3.18
_cell_length_b   3.18
_cell_length_c   15.0
_cell_angle_alpha   90
_cell_angle_beta    90
_cell_angle_gamma   120

loop_
  _atom_site_type_symbol
  _atom_site_label
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  Mo Mo1 0.333 0.667 0.5
""", encoding='utf-8')
        result = find_coincidence_lattice_from_cif(
            cif_a=str(cif), cif_b=str(cif), strain_tol=0.01
        )
        assert result is not None

    def test_raises_without_input(self):
        with pytest.raises(ValueError):
            find_coincidence_lattice_from_cif()


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
# generate_supercell_positions
# ---------------------------------------------------------------------------

class TestGenerateSupercellPositions:
    def test_identity_matrix_preserves_atoms(self):
        unit_cell = cell_from_params(3.18, 3.18, 120.0)
        frac = np.array([[0.0, 0.0], [1/3, 2/3]])
        matrix = np.eye(2, dtype=int)
        cart = generate_supercell_positions(frac, unit_cell, matrix)
        assert cart.shape == (2, 2)

    def test_2x1_supercell_doubles_atoms(self):
        unit_cell = cell_from_params(3.0, 4.0, 90.0)
        frac = np.array([[0.0, 0.0], [0.5, 0.5]])
        matrix = np.array([[2, 0], [0, 1]], dtype=int)
        cart = generate_supercell_positions(frac, unit_cell, matrix)
        # det=2 → 2×2=4 atoms
        assert cart.shape[0] == 4

    def test_singular_matrix_raises(self):
        unit_cell = cell_from_params(3.0, 4.0, 90.0)
        frac = np.array([[0.0, 0.0]])
        matrix = np.zeros((2, 2), dtype=int)
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            generate_supercell_positions(frac, unit_cell, matrix)


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


# ---------------------------------------------------------------------------
# rotation_angle_from_match
# ---------------------------------------------------------------------------

class TestRotationAngleFromMatch:
    def test_identical_cells_zero_rotation(self):
        """Two identical hexagonal cells should give 0° rotation."""
        cell = _hex_cell(3.18)
        match = MatchResult(
            matrix_a=np.eye(2, dtype=int),
            matrix_b=np.eye(2, dtype=int),
            supercell_a=cell,
            supercell_b=cell,
            strain_a=0.0,
            strain_b=0.0,
            area=float(np.abs(np.linalg.det(cell))),
        )
        angle = rotation_angle_from_match(match)
        assert abs(angle) < 1e-6

    def test_none_match_returns_zero(self):
        """rotation_angle_from_match(None) must return 0.0."""
        assert rotation_angle_from_match(None) == 0.0

    def test_45_degree_rotation(self):
        """A deliberately rotated supercell should give the correct angle."""
        theta = math.radians(45.0)
        R = np.array([[math.cos(theta), -math.sin(theta)],
                      [math.sin(theta),  math.cos(theta)]])
        base = _rect_cell(3.0, 3.0)
        rotated = base @ R.T  # row-vector convention
        match = MatchResult(
            matrix_a=np.eye(2, dtype=int),
            matrix_b=np.eye(2, dtype=int),
            supercell_a=base,
            supercell_b=rotated,
            strain_a=0.0,
            strain_b=0.0,
            area=float(np.abs(np.linalg.det(base))),
        )
        angle = rotation_angle_from_match(match)
        assert abs(angle - 45.0) < 0.1



# --- D2b: fixed-reference substrate match (match_to_fixed_box) ---

def test_match_to_fixed_box_pins_box_and_strains_other():
    from src.lattice.matching import match_to_fixed_box, FixedBoxMatch
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
    from src.lattice.matching import _max_strain as _ms
    assert abs(_ms(box, sup) - m.residual_strain) < 1e-9   # residual is the box<-sup strain
    # and the residual is within tol -> the strained supercell equals the box within that strain
    assert m.residual_strain <= 0.05 + 1e-9


def test_match_to_fixed_box_fails_loud_when_incommensurate():
    from src.lattice.matching import match_to_fixed_box
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
    from src.lattice.matching import find_commensurate_twist, TwistMatch
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
    from src.lattice.matching import find_commensurate_twist
    A = _hex_cell(); B = _hex_cell()
    with pytest.raises(ValueError, match=r"nearest|achievable"):
        find_commensurate_twist(A, B, target_angle=7.0, angle_tol=0.01,
                                strain_tol=0.001, max_supercell=4)


def test_orthorhombic_wrapper_tiles_rhombic_moire():
    from src.lattice.matching import orthorhombic_wrapper
    rhombus = _hex_cell(a=30.0)                            # a moire-scale rhombus (60 deg)
    ortho, tiling = orthorhombic_wrapper(rhombus)
    assert abs(ortho[1, 0]) < 1e-6                         # rectangular
    # the wrapper area is an integer multiple of the rhombus area (genuine periodic tiling)
    ratio = abs(np.linalg.det(ortho)) / abs(np.linalg.det(rhombus))
    assert abs(ratio - round(ratio)) < 1e-6 and round(ratio) >= 1


def test_rotation_angle_between_matches_from_match():
    """The factored-out raw-cell helper agrees with rotation_angle_from_match."""
    from src.lattice.matching import rotation_angle_between, rotation_angle_from_match
    A = _hex_cell()
    theta = 15.0
    r = math.radians(theta)
    R = np.array([[math.cos(r), -math.sin(r)], [math.sin(r), math.cos(r)]])
    B = A @ R.T                                            # A rotated by +15 deg
    assert abs(rotation_angle_between(A, B) - (-theta)) < 1e-6 \
        or abs(rotation_angle_between(A, B) - theta) < 1e-6
    match = MatchResult(matrix_a=np.eye(2, dtype=int), matrix_b=np.eye(2, dtype=int),
                        supercell_a=A, supercell_b=B, strain_a=0.0, strain_b=0.0, area=1.0)
    assert abs(rotation_angle_between(A, B) - rotation_angle_from_match(match)) < 1e-9
