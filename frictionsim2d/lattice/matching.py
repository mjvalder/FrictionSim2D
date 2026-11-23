"""Zur-McGill coincidence lattice matching and Hod Registry Index for heterostructures.

This module provides two complementary tools for 2D material heterostructure setup:

1. **Zur-McGill lattice matching** — finds the smallest commensurate supercell for two
   2D lattices by enumerating integer transformation matrices and minimising strain.
   Answers: *"What supercell box should I build?"*

   Reference:
       Zur & McGill, J. Appl. Phys. 55, 378 (1984).

2. **Hod Registry Index (RI)** — scans the geometric overlap of electron-density disks
   between two layers as a function of lateral shift.  RI = 1 at AA stacking (full
   registry, highest friction); RI = 0 at the optimal AB anti-registry site
   (superlubricity regime).

   Reference:
       Hod, O. ChemPhysChem 14, 2376 (2013).
       "The Registry Index: A Quantitative Measure of Materials' Interfacial
        Commensurability"
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Tuple

import numpy as np


@dataclass
class MatchResult:
    """Coincidence-lattice match between two 2D material cells.

    All matrices are 2×2 integer transformation matrices M such that
    ``supercell = M @ unit_cell`` (row-vector convention).

    Attributes:
        matrix_a: Integer supercell transformation matrix for material A.
        matrix_b: Integer supercell transformation matrix for material B.
        supercell_a: 2×2 supercell lattice vectors for material A (Å).
        supercell_b: 2×2 supercell lattice vectors for material B (Å).
        strain_a: Max absolute principal strain applied to A to reach the common cell.
        strain_b: Max absolute principal strain applied to B to reach the common cell.
        area: Supercell area (Å²).
        mean_strain: Average of ``strain_a`` and ``strain_b``.
    """
    matrix_a: np.ndarray
    matrix_b: np.ndarray
    supercell_a: np.ndarray
    supercell_b: np.ndarray
    strain_a: float
    strain_b: float
    area: float
    mean_strain: float = field(init=False)

    def __post_init__(self) -> None:
        self.mean_strain = (self.strain_a + self.strain_b) / 2.0

    def __repr__(self) -> str:  # pragma: no cover
        ma = self.matrix_a.tolist()
        mb = self.matrix_b.tolist()
        return (
            f"MatchResult(M_A={ma}, M_B={mb}, "
            f"strain_A={self.strain_a:.4f}, strain_B={self.strain_b:.4f}, "
            f"area={self.area:.2f} Å²)"
        )


def cell_from_params(lat_a: float, lat_b: float, ang_gamma_deg: float) -> np.ndarray:
    """Build a 2×2 row-vector lattice matrix from cell parameters.

    Args:
        lat_a: Lattice parameter a (Å).
        lat_b: Lattice parameter b (Å).
        ang_gamma_deg: In-plane angle γ (degrees).

    Returns:
        2×2 array where rows are lattice vectors::

            [[ax, ay],
             [bx, by]]
    """
    gamma = math.radians(ang_gamma_deg)
    return np.array([
        [lat_a, 0.0],
        [lat_b * math.cos(gamma), lat_b * math.sin(gamma)],
    ])


def _max_strain(cell_ref: np.ndarray, cell_def: np.ndarray) -> float:
    """Return the maximum absolute principal strain deforming *cell_def* → *cell_ref*.

    Both cells must have the same number of atoms / area sign so that the
    deformation gradient deform = cell_ref @ inv(cell_def) is meaningful.

    Args:
        cell_ref: Target 2×2 cell (rows = vectors).
        cell_def: Starting 2×2 cell (rows = vectors).

    Returns:
        Max |principal strain - 1|.  Zero means perfect match.
    """
    try:
        deform = cell_ref @ np.linalg.inv(cell_def)
    except np.linalg.LinAlgError:
        return math.inf
    # Polar decomposition: deform = rot @ stretch.  cauchy_green = deform.T @ deform
    # is the right Cauchy-Green tensor (FᵀF); its eigenvalues are *squared* stretch
    # ratios -- take the square root below to get the stretches.
    cauchy_green = deform.T @ deform
    eigvals = np.linalg.eigvalsh(cauchy_green)
    # principal strains: λ_i = sqrt(eigenvalue) - 1
    stretches = np.sqrt(np.maximum(eigvals, 0.0))
    return float(np.max(np.abs(stretches - 1.0)))


def _enumerate_matrices(n: int):
    """Yield all 2×2 integer matrices with determinant in [1, n^2] and |entries| <= n.

    Only upper-triangular forms are generated to avoid counting rotationally
    equivalent supercells multiple times (Hermite Normal Form subset).

    Yields:
        2×2 np.ndarray of dtype int with positive determinant.
    """
    for a in range(1, n + 1):
        for c in range(1, n + 1):
            if a * c > n * n:
                break
            for b in range(0, c):
                det = a * c
                if 1 <= det <= n * n:
                    yield np.array([[a, b], [0, c]], dtype=int)


def find_coincidence_lattice(
    cell_a: np.ndarray,
    cell_b: np.ndarray,
    strain_tol: float = 0.02,
    max_supercell: int = 10,
    area_tol: float = 0.10,
) -> Optional[MatchResult]:
    """Find the smallest commensurate supercell for two 2D material cells.

    Implements the Zur-McGill coincidence-lattice algorithm: enumerates integer
    transformation matrices for both cells up to ``max_supercell`` repeat units,
    and accepts pairs whose mutual strain is below ``strain_tol``.  The best
    candidate is the one with the smallest supercell area.

    Args:
        cell_a: 2×2 lattice matrix for material A (row vectors, Å).
        cell_b: 2×2 lattice matrix for material B (row vectors, Å).
        strain_tol: Maximum allowed strain for either material (default 2 %).
        max_supercell: Maximum supercell repeat along each axis (default 10).
        area_tol: Relative tolerance on supercell area agreement (default 10 %).

    Returns:
        :class:`MatchResult` with the best (smallest-area, within-tolerance) match,
        or ``None`` if no match is found within the constraints.
    """
    best: Optional[MatchResult] = None

    for ma in _enumerate_matrices(max_supercell):
        sc_a = ma @ cell_a  # type: ignore[operator]
        area_a = abs(float(np.linalg.det(sc_a)))
        if area_a < 1.0:
            continue

        for mb in _enumerate_matrices(max_supercell):
            sc_b = mb @ cell_b  # type: ignore[operator]
            area_b = abs(float(np.linalg.det(sc_b)))

            # Areas must agree within area_tol before computing full strain.
            if abs(area_a - area_b) / max(area_a, area_b) > area_tol:
                continue

            # Compute strain: how much must each supercell deform to reach a
            # common mean cell?
            mean_cell = (sc_a + sc_b) / 2.0
            sa = _max_strain(mean_cell, sc_a)
            sb = _max_strain(mean_cell, sc_b)

            if sa > strain_tol or sb > strain_tol:
                continue

            area = abs(float(np.linalg.det(mean_cell)))

            if best is None or area < best.area:
                best = MatchResult(
                    matrix_a=ma.copy(),
                    matrix_b=mb.copy(),
                    supercell_a=sc_a.copy(),
                    supercell_b=sc_b.copy(),
                    strain_a=sa,
                    strain_b=sb,
                    area=area,
                )

    return best


@dataclass
class FixedBoxMatch:
    """A fixed-reference coincidence match: box pinned, other cell strained.

    Unlike :class:`MatchResult` (which strains both cells onto a mean cell),
    this pins ``box_cell`` and strains only the supercell of the other cell
    onto it. See the D2b crystalline-substrate design (§4.1).

    Attributes:
        supercell_matrix: Integer 2×2 matrix applied to ``other_cell``
            (row-vector convention ``supercell = M @ other_cell``).
        residual_strain: ``_max_strain(box_cell, other_supercell)`` — the
            strain the supercell absorbs to land on the pinned box.
        box_cell: The pinned reference cell (returned unchanged).
    """
    supercell_matrix: np.ndarray
    residual_strain: float
    box_cell: np.ndarray


def match_to_fixed_box(
    box_cell: np.ndarray,
    other_cell: np.ndarray,
    strain_tol: float = 0.02,
    max_supercell: int = 10,
    area_tol: float = 0.10,
) -> FixedBoxMatch:
    """Pin ``box_cell``; strain a supercell of ``other_cell`` onto it.

    Enumerates integer supercells of ``other_cell`` (via :func:`_enumerate_matrices`,
    with the same area pre-filter as :func:`find_coincidence_lattice`) and returns
    the one whose supercell, strained onto the fixed ``box_cell``, needs the
    smallest ``_max_strain(box_cell, other_supercell)``. Accepts only if that
    residual strain is ``<= 2 * strain_tol``.

    Deliberately diverges from :func:`find_coincidence_lattice` (which strains
    both cells onto a mean cell): here the box is load-bearing and never strained.
    See the D2b spec §4.1. Reuses ``_enumerate_matrices``/``_max_strain`` so the
    matrix convention matches sheet–sheet matching exactly.

    Args:
        box_cell: 2×2 pinned reference cell (row vectors, Å) — never strained.
        other_cell: 2×2 lattice matrix to supercell + strain onto the box.
        strain_tol: Base strain tolerance; the accept bound is ``2 * strain_tol``.
        max_supercell: Maximum supercell repeat along each axis.
        area_tol: Relative tolerance on supercell/box area agreement (pre-filter).

    Returns:
        :class:`FixedBoxMatch` with the least-strain supercell.

    Raises:
        ValueError: If no supercell matches within ``2 * strain_tol`` — names
            the best achievable strain so the user can fall back to amorphous.
    """
    box_cell = np.asarray(box_cell, dtype=float)
    other_cell = np.asarray(other_cell, dtype=float)
    box_area = abs(float(np.linalg.det(box_cell)))

    best: Optional[Tuple[np.ndarray, float]] = None
    best_eps = math.inf
    for matrix in _enumerate_matrices(max_supercell):
        sup = matrix @ other_cell  # type: ignore[operator]
        sup_area = abs(float(np.linalg.det(sup)))
        if sup_area < 1.0:
            continue
        # Area pre-filter: box must be reachable by strain (mirror find_coincidence_lattice).
        if abs(sup_area - box_area) / max(sup_area, box_area) > area_tol:
            continue
        eps = _max_strain(box_cell, sup)
        if eps < best_eps:
            best = (matrix.copy(), eps)
            best_eps = eps

    if best is None or best[1] > 2.0 * strain_tol:
        got = ("no supercell within max_supercell/area_tol"
               if best is None else f"best residual strain {best[1]:.4f}")
        raise ValueError(
            f"Substrate cannot be matched onto the fixed hetero box within "
            f"{2 * strain_tol:.4f} strain ({got}). Use an amorphous substrate "
            f"([sub] amorph = a) or widen match_strain_tol."
        )
    return FixedBoxMatch(
        supercell_matrix=best[0],
        residual_strain=best[1],
        box_cell=box_cell,
    )


def _relative_rotation_strain(sc_a: np.ndarray, sc_b: np.ndarray) -> Tuple[float, float]:
    """Relative rotation (deg) and residual stretch strain between two 2×2 cells.

    Uses the **left** transform ``rel_transform = inv(sc_b) @ sc_a``
    (``sc_a = sc_b @ rel_transform``), whose singular values (and polar-rotation)
    are invariant under rotating *either* cell (a right/left orthogonal factor).
    This is the correct frame to separate a rigid relative rotation (the twist)
    from a pure shape mismatch (the residual strain): for ``sc_a = sc_b @ rotᵀ``
    (rigid rotation, no strain) ``rel_transform = rotᵀ`` exactly, so the strain
    is 0 and the angle is the twist -- unlike the right transform
    ``sc_a @ inv(sc_b)``, which reports spurious strain for rotated
    non-orthogonal cells.

    Returns:
        ``(angle_deg, max_strain)`` where ``angle_deg`` is the polar-rotation of
        ``rel_transform`` (CCW positive) and ``max_strain`` is ``max|σ_i - 1|``
        over ``rel_transform``'s singular values. ``(0.0, inf)`` if
        ``rel_transform`` is singular.
    """
    try:
        rel_transform = np.linalg.inv(np.asarray(sc_b, float)) @ np.asarray(sc_a, float)
        cauchy_green_b = rel_transform.T @ rel_transform
        eigvals, eigvecs = np.linalg.eigh(cauchy_green_b)
        eigvals = np.maximum(eigvals, 0.0)
        stretches = np.sqrt(eigvals)
        stretch = eigvecs @ np.diag(stretches) @ eigvecs.T
        rot = rel_transform @ np.linalg.inv(stretch)
        angle = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
        strain = float(np.max(np.abs(stretches - 1.0)))
        return angle, strain
    except np.linalg.LinAlgError:
        return 0.0, math.inf


def rotation_angle_between(sc_a: np.ndarray, sc_b: np.ndarray) -> float:
    """Rotation angle (degrees) carrying raw 2×2 cell ``sc_b`` onto ``sc_a``.

    The relative rotation is the polar-rotation component of the left transform
    ``G = inv(sc_b) @ sc_a`` (see :func:`_relative_rotation_strain`), which is
    rotation-invariant and so gives the twist directly. Operates on **raw** 2×2
    lattice matrices (row-vector convention), so it can be called on candidate
    supercell pairs during a twist search -- unlike
    :func:`rotation_angle_from_match`, which unpacks a :class:`MatchResult`.

    Args:
        sc_a: 2×2 reference cell (rows = lattice vectors, Å).
        sc_b: 2×2 cell to measure the rotation of, onto ``sc_a``.

    Returns:
        Rotation angle in degrees (counter-clockwise positive). Returns 0.0 when
        the two cells are already aligned or the decomposition is singular.
    """
    return _relative_rotation_strain(sc_a, sc_b)[0]


def _gauss_reduce_2d(cell: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Lagrange–Gauss reduce a 2×2 lattice basis to its two shortest vectors.

    Returns ``(reduced, unimod)`` where ``reduced`` is a right-handed
    (positive-determinant) reduced basis (rows = the two shortest lattice
    vectors) and ``unimod`` is the integer unimodular transform with
    ``reduced == unimod @ cell`` (so the reduction can be composed with a
    supercell matrix to recover the exact integer tiling of the original
    monolayer).

    Two bases of lattices that are rigid rotations of each other reduce to bases
    related by the lattice point group (rotations by multiples of the
    reduced-cell angle + reflections), so their relative rotation is well-defined
    modulo that symmetry -- the twist search folds the result onto the hexagonal
    wedge. Basis reduction is essential: two *raw* supercell matrices for the
    same sublattice differ by an integer unimodular factor, which would otherwise
    masquerade as large strain.
    """
    v1, v2 = np.asarray(cell, float)[0].copy(), np.asarray(cell, float)[1].copy()
    unimod = np.eye(2)
    swap = np.array([[0.0, 1.0], [1.0, 0.0]])
    if v1 @ v1 > v2 @ v2:
        v1, v2 = v2.copy(), v1.copy()
        unimod = swap @ unimod
    for _ in range(1000):
        m = round((v1 @ v2) / (v1 @ v1))
        v2 = v2 - m * v1
        unimod = np.array([[1.0, 0.0], [-m, 1.0]]) @ unimod
        if v2 @ v2 >= v1 @ v1:
            break
        v1, v2 = v2.copy(), v1.copy()
        unimod = swap @ unimod
    # Canonicalise into a single deterministic representative so that two bases
    # of lattices related by a rotation reduce to bases that are the SAME
    # rotation of each other (otherwise a hexagonal lattice's 60° vs 120°
    # shortest-vector rhombi -- both valid reductions -- read as a spurious
    # shear). Every decision below is rotation-invariant (dot signs, det sign,
    # lengths), so canonicalize(rot·L) = rot·canonicalize(L).
    if v1 @ v2 < 0.0:            # acute angle between v1 and v2 (pick 60° image)
        v2 = -v2
        unimod = np.array([[1.0, 0.0], [0.0, -1.0]]) @ unimod
    reduced = np.array([v1, v2])
    if np.linalg.det(reduced) < 0.0:   # keep right-handed (swap, preserves the
        reduced = reduced[::-1]        # acute angle: dot is symmetric)
        unimod = swap @ unimod
    return reduced, np.rint(unimod).astype(int)


def rotation_angle_from_match(match: MatchResult) -> float:
    """Extract the rotation angle (degrees) that aligns material B to material A.

    Thin wrapper over :func:`rotation_angle_between` that unpacks the two
    supercells of a :class:`MatchResult` (the rotation component of
    ``F = supercell_a @ inv(supercell_b)``).

    Args:
        match: A :class:`MatchResult` from :func:`find_coincidence_lattice`.

    Returns:
        Rotation angle in degrees (counter-clockwise positive).  Returns 0.0
        when the two supercells are already aligned.
    """
    if match is None:
        return 0.0
    return rotation_angle_between(match.supercell_a, match.supercell_b)


@dataclass
class TwistMatch:
    """A commensurate twisted-moiré approximant emitted on an orthorhombic box.

    Produced by :func:`find_commensurate_twist`. The two integer supercell
    matrices tile each layer's monolayer into the shared moiré cell; the layer B
    monolayer is then physically rotated by ``achieved_angle_deg`` (via
    :func:`~frictionsim2d.lattice.apply.rotate_layer_to_angle`) so the two tilings
    coincide up to ``residual_strain``.

    Commensurate moiré cells are generically rhombic (60°/120°), but downstream
    postprocessing/AiiDA/HPC/plotting all assume an **orthogonal** simulation
    box. ``moire_cell`` is therefore the minimal orthorhombic super-supercell
    (the rhombus→rectangle wrapper, ``ortho_tiling`` rhombi per box) actually
    emitted, in LAMMPS-canonical form (first vector along +x, xy tilt ≈ 0).

    Attributes:
        supercell_a: Integer 2×2 supercell matrix tiling layer A's monolayer
            directly into the emitted **orthorhombic** box:
            ``supercell_a @ cell_a`` equals ``moire_cell`` up to a rigid frame
            rotation and the residual strain (it already folds in the
            rhombus→rectangle ``ortho_tiling`` and the basis reduction). Feed it
            to ``ase.build.make_supercell`` then ``set_cell(moire_cell)``.
        supercell_b: Integer 2×2 supercell matrix tiling layer B's monolayer
            into the same orthorhombic box; because it selects the θ-rotated
            sublattice, tiling B with it (vs A with ``supercell_a``) and mapping
            both onto ``moire_cell`` realises the relative twist.
        achieved_angle_deg: The realised relative twist (nearest commensurate
            angle to the target -- reported, never silently replaced by target).
        residual_strain: Max per-material strain to close the approximant onto
            the shared cell (≤ ``strain_tol``); ≈ 0 for same-material moiré.
        moire_period: Moiré periodicity length (Å) = shorter emitted box edge.
        moire_cell: The emitted 2×2 orthorhombic box (LAMMPS-canonical).
        ortho_tiling: Integer 2×2 tiling matrix mapping the rhombic moiré cell
            to the orthorhombic box (``|det|`` = rhombi per box, ≥ 1).
    """
    supercell_a: np.ndarray
    supercell_b: np.ndarray
    achieved_angle_deg: float
    residual_strain: float
    moire_period: float
    moire_cell: np.ndarray
    ortho_tiling: np.ndarray


def orthorhombic_wrapper(
    rhombic_cell: np.ndarray,
    max_tiling: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """Minimal orthorhombic super-supercell tiling an integer number of rhombi.

    Commensurate moiré cells are generically rhombic (60°/120°). Downstream
    tooling requires an orthogonal box, so rather than *filter* to zero-tilt
    supercells (which would reject essentially every hexagonal input -- a 60°
    cell has no zero-tilt HNF supercell), this *wraps* the rhombus in the
    smallest rectangular cell whose edges are still superlattice vectors of the
    rhombic lattice (so the box stays exactly periodic). For the standard 60°
    hexagonal rhombus this is the well-known c-rectangular doubling (2 rhombi
    per box).

    Searches integer 2×2 tiling matrices ``tiling`` with entries in
    ``[-max_tiling, max_tiling]`` and returns the ``tiling`` of least positive
    determinant whose ``tiling @ rhombic_cell`` has orthogonal rows.

    Args:
        rhombic_cell: 2×2 rhombic (or already-orthogonal) cell (row vectors, Å).
        max_tiling: Bound on integer tiling-matrix entries searched.

    Returns:
        ``(ortho_cell, tiling)`` where ``ortho_cell`` is the orthorhombic box in
        LAMMPS-canonical form (``[[lx, 0], [0, ly]]``, xy tilt 0) and ``tiling``
        is the integer 2×2 matrix (``|det tiling|`` = rhombi per box, ≥ 1).

    Raises:
        ValueError: If no orthogonalising tiling is found within ``max_tiling``.
    """
    a = np.asarray(rhombic_cell, dtype=float)
    best: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
    best_det = math.inf
    rng = range(-max_tiling, max_tiling + 1)
    for p in rng:
        for q in rng:
            for r in rng:
                for s in rng:
                    det = p * s - q * r
                    if det <= 0:
                        continue
                    candidate_tiling = np.array([[p, q], [r, s]], dtype=float)
                    b = candidate_tiling @ a
                    n0 = float(np.linalg.norm(b[0]))
                    n1 = float(np.linalg.norm(b[1]))
                    if n0 < 1e-9 or n1 < 1e-9:
                        continue
                    if abs(float(b[0] @ b[1])) > 1e-6 * n0 * n1:
                        continue  # rows not orthogonal
                    if det < best_det - 1e-9:
                        best = (float(det), candidate_tiling.copy(), b.copy())
                        best_det = det
    if best is None:
        raise ValueError(
            f"orthorhombic_wrapper: no orthogonalising integer tiling (|entries| "
            f"<= {max_tiling}) found for the rhombic moiré cell {a.tolist()}. "
            f"Widen max_tiling."
        )
    _det, tiling, ortho_rows = best
    # Emit in LAMMPS-canonical orthogonal form: rows are orthogonal, so the
    # canonical box is diag(|b0|, |b1|) with xy tilt exactly 0.
    lx = float(np.linalg.norm(ortho_rows[0]))
    ly = float(np.linalg.norm(ortho_rows[1]))
    ortho_cell = np.array([[lx, 0.0], [0.0, ly]])
    return ortho_cell, tiling.astype(int)


def find_commensurate_twist(
    cell_a: np.ndarray,
    cell_b: np.ndarray,
    target_angle: float,
    angle_tol: float = 2.0,
    strain_tol: float = 0.02,
    max_supercell: int = 12,
    area_tol: float = 0.10,
    max_tiling: int = 6,
) -> TwistMatch:
    """Find the commensurate twisted-moiré approximant nearest ``target_angle``.

    Enumerates integer supercell pairs ``(M_a, M_b)`` (via
    :func:`_enumerate_matrices`), forms each candidate's supercells
    ``sc_a = M_a @ cell_a`` / ``sc_b = M_b @ cell_b``, and for area-compatible
    pairs measures the **achieved** relative rotation
    (:func:`rotation_angle_between`, rotation-only) and the residual strain to
    close ``sc_b`` onto ``sc_a`` (:func:`_max_strain`, rotation-invariant, so it
    is the pure stretch mismatch). The best candidate minimises
    ``|achieved - target_angle|`` among those within ``angle_tol`` and
    ``strain_tol``. Its rhombic moiré cell is then wrapped in its minimal
    orthorhombic super-supercell (:func:`orthorhombic_wrapper`) so the emitted
    box is orthogonal.

    The achieved angle is always reported (commensurate angles are discrete, so
    it generally differs from ``target_angle``).

    Args:
        cell_a: 2×2 lattice matrix for layer A (row vectors, Å).
        cell_b: 2×2 lattice matrix for layer B (equals ``cell_a`` for a
            same-material twist).
        target_angle: Desired relative twist (deg).
        angle_tol: Max ``|achieved - target|`` accepted (deg).
        strain_tol: Max residual strain accepted to close the approximant.
        max_supercell: Max supercell repeat along each axis for the search.
        area_tol: Relative supercell-area agreement pre-filter.
        max_tiling: Forwarded to :func:`orthorhombic_wrapper`.

    Returns:
        :class:`TwistMatch` (achieved angle, residual strain, moiré period, the
        emitted orthorhombic ``moire_cell`` + its ``ortho_tiling``).

    Raises:
        ValueError: If no candidate lands within ``angle_tol`` / ``strain_tol``
            inside ``max_supercell`` -- naming the nearest achievable angle and
            its atom-count cost (rhombi-per-cell) so the caller can widen the
            budget or pick a reachable angle.
    """
    cell_a = np.asarray(cell_a, dtype=float)
    cell_b = np.asarray(cell_b, dtype=float)

    # Pre-compute candidate supercells + areas once per layer.
    cand_a = []
    for ma in _enumerate_matrices(max_supercell):
        sc = ma @ cell_a
        cand_a.append((ma, sc, abs(float(np.linalg.det(sc)))))
    cand_b = []
    for mb in _enumerate_matrices(max_supercell):
        sc = mb @ cell_b
        cand_b.append((mb, sc, abs(float(np.linalg.det(sc)))))

    # In-tolerance candidates: (sort_key, matrix_a, reduced_a, matrix_b,
    # achieved_folded, residual, ncells). The orthogonal-box wrapper is a HARD
    # feasibility constraint (downstream needs an orthogonal box), so we keep ALL
    # in-tolerance candidates and, after sorting best-angle-first, return the
    # first whose reduced moiré cell actually wraps into an orthorhombic box
    # within the tiling budget -- skipping oblique approximants that would need
    # an exploded (or non-existent) orthogonal super-supercell.
    candidates: List[
        Tuple[Tuple[float, int], np.ndarray, np.ndarray, np.ndarray, float, float, int]
    ] = []
    # Nearest *achievable* commensurate angle seen at all (for a helpful raise),
    # tracked independently of angle_tol so we can name it when nothing qualifies.
    nearest = None       # (angle_err, achieved, residual, ncells)
    nearest_err = math.inf
    nearest_res = math.inf

    for ma, sc_a, area_a in cand_a:
        if area_a < 1.0:
            continue
        for mb, sc_b, area_b in cand_b:
            if area_b < 1.0:
                continue
            if abs(area_a - area_b) / max(area_a, area_b) > area_tol:
                continue
            # Reduce both supercells to their shortest-vector bases first, so the
            # measured rotation/strain reflect the sublattices' actual geometry
            # (not an integer-unimodular basis skew).
            ra, _ua = _gauss_reduce_2d(sc_a)
            rb, _ub = _gauss_reduce_2d(sc_b)
            achieved, residual = _relative_rotation_strain(ra, rb)
            # Fold onto the hexagonal irreducible wedge [0, 60): twist is
            # defined modulo the 60° lattice symmetry, so compare in [0, 60).
            achieved_folded = abs(achieved) % 60.0
            if achieved_folded > 30.0:
                achieved_folded = 60.0 - achieved_folded   # nearest wedge image
            angle_err = abs(achieved_folded - target_angle)
            ncells = int(round(area_a / abs(float(np.linalg.det(cell_a)))))
            if (angle_err < nearest_err - 1e-9
                    or (abs(angle_err - nearest_err) < 1e-9 and residual < nearest_res)):
                nearest = (angle_err, achieved_folded, residual, ncells)
                nearest_err, nearest_res = angle_err, residual
            if angle_err > angle_tol or residual > strain_tol:
                continue
            # Prefer the closest angle; break ties by the smaller moiré cell.
            key = (round(angle_err, 6), ncells)
            candidates.append((key, ma, mb, achieved_folded, residual, ncells))

    # Best angle first, then smaller cell. Return the first candidate whose
    # reduced moiré cell wraps into an orthorhombic box within the tiling budget.
    candidates.sort(key=lambda c: c[0])
    for _key, matrix_a, matrix_b, achieved, residual, ncells in candidates:
        # Reduce each layer's rhombic moiré supercell, capturing the integer
        # reduction transform (unimod) so the FULL orthorhombic tiling matrix of
        # the *original* monolayer is S = tiling @ unimod @ M (exact integers,
        # no rounding).
        reduced_a, u_a = _gauss_reduce_2d(matrix_a @ cell_a)
        _reduced_b, u_b = _gauss_reduce_2d(matrix_b @ cell_b)
        try:
            ortho_cell, tiling = orthorhombic_wrapper(reduced_a[:2, :2], max_tiling=max_tiling)
        except ValueError:
            continue  # oblique approximant with no small orthogonal wrapper -> skip
        s_a = (tiling @ u_a @ matrix_a).astype(int)
        s_b = (tiling @ u_b @ matrix_b).astype(int)
        # Moiré periodicity = the reduced rhombus edge length (the genuine moiré
        # wavelength), not the wrapped-box edge.
        moire_period = float(np.linalg.norm(reduced_a[0]))
        return TwistMatch(
            supercell_a=s_a,
            supercell_b=s_b,
            achieved_angle_deg=float(achieved),
            residual_strain=float(residual),
            moire_period=moire_period,
            moire_cell=ortho_cell,
            ortho_tiling=tiling,
        )

    # Nothing qualified (or nothing wrapped within budget): fail loud, naming the
    # nearest achievable angle + its atom-count cost.
    near_msg = (
        f"nearest achievable angle {nearest[1]:.3f} deg "
        f"(|Δ|={nearest[0]:.3f} deg, residual strain {nearest[2]:.4%}, "
        f"{nearest[3]} unit cells/layer)"
        if nearest is not None else "no area-compatible supercell pair"
    )
    extra = "" if not candidates else (
        " (in-tolerance candidates existed but none wrapped into an orthorhombic "
        "box within the tiling budget -- moiré-cell explosion; widen max_supercell "
        "or pick a cleaner angle)"
    )
    raise ValueError(
        f"find_commensurate_twist: no commensurate moiré within angle_tol="
        f"{angle_tol:.3f} deg and strain_tol={strain_tol:.4%} of "
        f"target_angle={target_angle:.3f} deg inside max_supercell="
        f"{max_supercell}. The {near_msg}.{extra} Widen angle_tol/strain_tol/"
        f"max_supercell or pick a reachable angle."
    )


# =============================================================================
# Hod Registry Index (RI)
# =============================================================================

class StackingResult(NamedTuple):
    """Optimal stacking offsets derived from a Registry Index scan.

    All shifts are Cartesian (Å) displacements of layer B relative to layer A.

    Attributes:
        aa_shift: Shift that maximises the RI (AA-like registry, highest friction).
        ab_shift: Shift that minimises the RI (AB anti-registry, superlubricity).
        aa_ri: Registry Index value at the AA site (≈ 1.0).
        ab_ri: Registry Index value at the AB site (≈ 0.0).
        ri_grid: 2-D array of RI values over the scan grid (shape n_grid × n_grid).
        shifts_x: Cartesian x-coordinates of the grid points (Å).
        shifts_y: Cartesian y-coordinates of the grid points (Å).
    """
    aa_shift: np.ndarray
    ab_shift: np.ndarray
    aa_ri: float
    ab_ri: float
    ri_grid: np.ndarray
    shifts_x: np.ndarray
    shifts_y: np.ndarray


def _circle_overlap_area(d: float, r: float) -> float:
    """Area of intersection of two circles of equal radius *r* with centres distance *d* apart.

    Returns the analytical two-disk overlap (Hod eq. A1).
    """
    if d >= 2.0 * r:
        return 0.0
    if d <= 0.0:
        return math.pi * r * r
    half_d = d / 2.0
    arg = min(half_d / r, 1.0)
    return 2.0 * r * r * math.acos(arg) - half_d * math.sqrt(max(4.0 * r * r - d * d, 0.0))


def _ri_raw(
    cart_a: np.ndarray,
    cart_b: np.ndarray,
    cell: np.ndarray,
    shift_cart: np.ndarray,
    radius: float,
) -> float:
    """Unnormalised overlap integral at a single shift.

    Sums circle-overlap areas for all (i, j) pairs within 2*radius using minimum
    image convention in the common supercell ``cell``.

    Args:
        cart_a: (N_a, 2) Cartesian positions of layer A atoms (Å).
        cart_b: (N_b, 2) Cartesian positions of layer B atoms (Å).
        cell: 2×2 common supercell lattice (row vectors, Å).
        shift_cart: (2,) Cartesian shift applied to layer B (Å).
        radius: Effective atomic radius for disk-overlap model (Å).

    Returns:
        Raw overlap sum (Å²).
    """
    cell_inv = np.linalg.inv(cell)
    b_shifted = cart_b + shift_cart

    total = 0.0
    cutoff = 2.0 * radius

    for pa in cart_a:
        diff = b_shifted - pa                          # (N_b, 2) Cartesian
        frac_diff = diff @ cell_inv                    # fractional in supercell
        frac_diff -= np.round(frac_diff)               # minimum image
        cart_diff = frac_diff @ cell
        dists = np.linalg.norm(cart_diff, axis=1)      # (N_b,)
        nearby = dists[dists < cutoff]
        for d in nearby:
            total += _circle_overlap_area(d, radius)

    return total


def scan_registry_index(
    cart_a: np.ndarray,
    cart_b: np.ndarray,
    cell: np.ndarray,
    n_grid: int = 30,
    atomic_radius: float = 1.5,
) -> StackingResult:
    """Scan the Hod Registry Index over a grid of lateral shifts.

    The RI is normalised so that the maximum value (AA registry) equals 1 and the
    minimum (optimal AB anti-registry) approaches 0.

    Args:
        cart_a: (N_a, 2) Cartesian positions of layer A atoms (Å).
        cart_b: (N_b, 2) Cartesian positions of layer B atoms (Å).
        cell: 2×2 common supercell lattice matrix (row vectors, Å).
        n_grid: Number of grid points along each cell vector (default 30).
        atomic_radius: Effective atomic radius for the disk-overlap model (Å).
            A value of 1.5 Å works well for most 2D materials; adjust for heavier
            atoms (e.g., 1.8 Å for Bi or Te).

    Returns:
        :class:`StackingResult` with AA/AB shifts, RI values, and the full grid.
    """
    # Build fractional grid [0, 1) × [0, 1) and convert to Cartesian.
    fx = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    fy = np.linspace(0.0, 1.0, n_grid, endpoint=False)

    ri_grid = np.zeros((n_grid, n_grid))
    raw_grid = np.zeros((n_grid, n_grid))

    for i, xi in enumerate(fx):
        for j, yj in enumerate(fy):
            shift = np.array([xi, yj]) @ cell       # Cartesian
            raw_grid[i, j] = _ri_raw(cart_a, cart_b, cell, shift, atomic_radius)

    raw_max = raw_grid.max()
    if raw_max > 0.0:
        ri_grid = raw_grid / raw_max
    else:
        ri_grid = raw_grid.copy()

    # Cartesian coordinates of each grid point.
    shifts_cart = np.array([[xi, yj] for xi in fx for yj in fy]) @ cell
    sx = shifts_cart[:, 0].reshape(n_grid, n_grid)
    sy = shifts_cart[:, 1].reshape(n_grid, n_grid)

    aa_idx = np.unravel_index(np.argmax(ri_grid), ri_grid.shape)
    ab_idx = np.unravel_index(np.argmin(ri_grid), ri_grid.shape)

    aa_shift = np.array([sx[aa_idx], sy[aa_idx]])
    ab_shift = np.array([sx[ab_idx], sy[ab_idx]])

    return StackingResult(
        aa_shift=aa_shift,
        ab_shift=ab_shift,
        aa_ri=float(ri_grid[aa_idx]),
        ab_ri=float(ri_grid[ab_idx]),
        ri_grid=ri_grid,
        shifts_x=sx,
        shifts_y=sy,
    )
