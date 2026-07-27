"""Tests for src.builders.hetero.build_matched_supercells.

Covers:

* A real MoS2/WS2 pair (built via the real monolayer builder), whose
  orthogonalized cells match at IDENTITY -- verifies the two per-material
  supercells share exactly one in-plane box, the larger-area material is left
  unstrained, and the smaller is strained onto it within tolerance.
* A genuinely SHEARED, non-identity coincidence (both match matrices have a
  non-zero off-diagonal), fed as synthetic monolayer data files -- verifies the
  two BUILT supercell boxes coincide within a tight tolerance at the true (low)
  strain. This is the case the identity-only MoS2/WS2 test cannot exercise:
  applying a match to the wrong cell, or straining in an independently
  re-canonicalized frame, would produce non-coinciding boxes / spurious strain.
* Fail-loud behavior when no low-strain coincidence exists.
"""
import re
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from src.builders import components
from src.builders.hetero import (
    _detect_atom_style,
    _run_lammps,
    assemble_hetero_stack,
    build_hetero_structure,
    build_matched_supercells,
    register_hetero_potentials,
)
from src.core.config import SheetConfig, SheetOnSheetSimulationConfig, load_settings
from src.core.lattice_matching import find_coincidence_lattice

MOS2_CIF = "examples/materials/h-MoS2.cif"
WS2_CIF = "examples/materials/h-WS2.cif"
MOS2_POT = "examples/potentials/sw/MoS2_wen.sw"
# ws2_jrmidas.sw defines a third element ('C', for graphene coupling) that is
# not present in h-WS2.cif and is rejected by get_potential_element_order, so
# a plain WS2-only SW potential is used instead. Both this file and
# MoS2_wen.sw assign exactly one LAMMPS type per element (no sublattice
# splitting), so both monolayers' orthogonalized unit cells are a plain 1x2
# doubling of their CIF primitive cell -- i.e. directly comparable to each
# other and matched (internally) from the real orthogonalized cells.
WS2_POT = "examples/potentials/sw/sw_lammps/t-WS2.sw"


def _sheet_config(mat: str, cif_path: str, pot_path: str) -> SheetConfig:
    """Build a minimal single-monolayer :class:`SheetConfig` for a material.

    x/y = 4.0 Å is smaller than one duplicated unit cell (~6.4 Å) for both
    materials, so build_monolayer's target-size rounding keeps dup=(1, 1):
    the smallest structure it can produce. For potential registration the
    in-plane size is irrelevant (only cif_path/pot_path/pot_type are read),
    but keeping a single, shared constructor mirrors the real build path.
    """
    return SheetConfig(
        mat=mat,
        pot_type="sw",
        pot_path=pot_path,
        cif_path=cif_path,
        x=4.0,
        y=4.0,
        layers=[1],
    )


def _build_small_monolayer(mat: str, cif_path: str, pot_path: str):
    """Build a minimal (single orthogonalized unit cell) monolayer."""
    config = _sheet_config(mat, cif_path, pot_path)
    path, _dims, _pot_counts, _total_types, _supercell_dims = components.build_monolayer(config)
    return path


def test_hetero_type_blocks_disjoint_and_potentials_complete(tmp_path):
    """Each stacked material becomes its own PotentialManager component with a
    disjoint, contiguous atom-type block, its own many-body ``sw`` in a
    ``pair_style hybrid``, and UFF ``lj/cut`` cross-terms for every A-B element
    pair -- all emitted into ``system.in.settings``.

    Guards the old branch's half-done type mapping: the two materials' global
    type-ID sets must be disjoint (no overlap) AND together contiguous/complete
    (cover every registered type, 1..N, no gaps). Type IDs are read from the PM
    type registry, never parsed from text.
    """
    sheets = [
        _sheet_config("h-MoS2", MOS2_CIF, MOS2_POT),
        _sheet_config("h-WS2", WS2_CIF, WS2_POT),
    ]
    settings = load_settings()

    hp = register_hetero_potentials(sheets, settings, workdir=tmp_path)

    text = open(hp.settings_path).read()

    # --- hybrid pair_style with one many-body (sw) entry per material ---
    assert "pair_style hybrid" in text
    style_line = next(l for l in text.splitlines() if l.startswith("pair_style hybrid"))
    assert style_line.split().count("sw") == len(sheets)

    # --- UFF lj/cut cross-terms for every A-B element pair ---
    assert "lj/cut" in text
    cross_lines = [
        l for l in text.splitlines()
        if "lj/cut" in l and l.strip().startswith("pair_coeff")
    ]
    els_a = hp.pm.types.elements_in_component("sheet_1")
    els_b = hp.pm.types.elements_in_component("sheet_2")
    expected_pairs = {frozenset((x, y)) for x in els_a for y in els_b}
    got_pairs = set()
    for line in cross_lines:
        m = re.search(r"#\s*(\w+)\(sheet_1\)-(\w+)\(sheet_2\)", line)
        assert m is not None, f"unexpected cross-interaction line: {line!r}"
        got_pairs.add(frozenset((m.group(1), m.group(2))))
    assert got_pairs == expected_pairs  # e.g. {Mo-W, Mo-S, S-W, S-S}

    # --- Type registry: disjoint AND contiguous/complete (from the PM, not text) ---
    assert len(hp.type_ids) == len(sheets)
    id_sets = [set(ids) for ids in hp.type_ids]
    a_ids, b_ids = id_sets
    assert a_ids and b_ids
    assert a_ids.isdisjoint(b_ids)                      # no overlap
    union = a_ids | b_ids
    assert union == set(range(1, max(union) + 1))       # contiguous from 1, no gaps

    # --- Emitted masses are correct for EVERY type (range-emission + override) ---
    # `get_masses_string` emits ranges (e.g. `mass 2*5 #S`) that can transiently
    # engulf a type a later line overrides (`mass 3 #W`). A run-0 energy does NOT
    # validate mass values, so assert the FINAL mass per type (last matching
    # `mass` line wins, LAMMPS semantics) equals its element's standard mass.
    from ase.data import atomic_masses, atomic_numbers
    n_types = len(union)
    final_mass: dict[int, float] = {}
    final_elem: dict[int, str] = {}
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("mass"):
            continue
        parts = s.split()
        spec, val = parts[1], float(parts[2])
        elem = s.split("#", 1)[1].strip().split()[0] if "#" in s else None
        if "*" in spec:
            lo_s, hi_s = spec.split("*")
            lo, hi = (int(lo_s) if lo_s else 1), (int(hi_s) if hi_s else n_types)
        else:
            lo = hi = int(spec)
        for t in range(lo, hi + 1):
            final_mass[t], final_elem[t] = val, elem
    assert set(final_mass) == set(range(1, n_types + 1))   # every type got a mass
    for t in range(1, n_types + 1):
        el = final_elem[t]
        assert el is not None and final_mass[t] > 0
        expected = atomic_masses[atomic_numbers[el]]
        assert abs(final_mass[t] - expected) < 0.5, (
            f"type {t} ({el}) got mass {final_mass[t]}, expected ~{expected:.2f}"
        )
    assert len(union) == len(hp.pm.types)               # covers every registered type

    # Cross-check against the live registry and the read_data offsets.
    for name, ids in zip(hp.component_names, hp.type_ids):
        assert hp.pm.types.ids_by_component(name) == ids
    assert hp.type_offsets == [ids[0] - 1 for ids in hp.type_ids]


def _sw_element_list(pair_coeff_line):
    """Extract the trailing element list from a hybrid ``pair_coeff * * sw ...``
    line, i.e. every token after the potential-file path (the token containing a
    ``/`` or ``.``), stripped of any comment. For ``pair_coeff * * sw 1
    provenance/potentials/MoS2_wen.sw Mo S NULL NULL`` this returns
    ``['Mo', 'S', 'NULL', 'NULL']`` -- one entry per box atom type, which is what
    LAMMPS requires for a many-body pair_coeff.
    """
    main = pair_coeff_line.split("#", 1)[0].strip()
    tokens = main.split()
    path_idx = next(i for i, t in enumerate(tokens) if "/" in t or "." in t)
    return tokens[path_idx + 1:]


def test_reserve_virtual_type_sizes_sw_element_lists_and_default_unchanged(tmp_path):
    """``reserve_virtual_type=True`` registers ONE extra non-interacting virtual
    type BEFORE the many-body self-interactions (the homogeneous drive-atom
    ordering), so every ``pair_coeff * * sw ...`` element list is sized to the
    full box type count (real types + a trailing NULL for the virtual type) the
    first time -- LAMMPS-acceptable, no post-hoc rebuild -- and the settings
    carry the virtual type's zero-LJ ``pair_coeff``. The default
    (``reserve_virtual_type=False``) is the real-types-only spine: no virtual
    type, no ``1e-100`` term, sw lists sized to the real count.
    """
    sheets = [
        _sheet_config("h-MoS2", MOS2_CIF, MOS2_POT),
        _sheet_config("h-WS2", WS2_CIF, WS2_POT),
    ]
    settings = load_settings()

    # --- default: NO virtual type, sw lists sized to the real count ----------
    hp_def = register_hetero_potentials(sheets, settings, workdir=tmp_path / "default")
    real_count = sum(len(ids) for ids in hp_def.type_ids)
    assert len(hp_def.pm.types) == real_count            # no extra type
    def_text = Path(hp_def.settings_path).read_text()
    assert "1e-100" not in def_text                       # no virtual pair_coeff
    assert "Virtual" not in def_text
    def_sw = [l for l in def_text.splitlines() if l.strip().startswith("pair_coeff * * sw")]
    assert len(def_sw) == len(sheets)
    for line in def_sw:
        assert len(_sw_element_list(line)) == real_count

    # --- reserve_virtual_type=True: exactly ONE extra type -------------------
    hp = register_hetero_potentials(
        sheets, settings, workdir=tmp_path / "virtual", reserve_virtual_type=True
    )
    # type_ids/component_names still describe the REAL materials only ...
    assert sum(len(ids) for ids in hp.type_ids) == real_count
    assert hp.component_names == hp_def.component_names
    # ... while pm.types holds one extra (virtual) type.
    assert len(hp.pm.types) == real_count + 1
    n_box_types = len(hp.pm.types)

    text = Path(hp.settings_path).read_text()
    # (a) the virtual type's zero-LJ pair_coeff + mass are present on type N.
    assert f"pair_coeff * {n_box_types} lj/cut 1e-100 1e-100" in text
    assert "Virtual atom" in text                         # mass N 1.0 #Virtual atom
    # (b) EVERY many-body sw element list is sized to the full box type count
    #     (real types + trailing NULL for the virtual type) -> LAMMPS accepts it.
    sw_lines = [l for l in text.splitlines() if l.strip().startswith("pair_coeff * * sw")]
    assert len(sw_lines) == len(sheets)
    for line in sw_lines:
        elems = _sw_element_list(line)
        assert len(elems) == n_box_types, (
            f"sw element list has {len(elems)} entries, expected {n_box_types} "
            f"(one per box type incl. trailing NULL for the virtual type): {line!r}"
        )
        # the reserved virtual type's slot (last) is a non-interacting NULL.
        assert elems[-1] == "NULL"


def _write_synthetic_monolayer(path, cell_2x2, charges=None):
    """Write a minimal 2-atom monolayer LAMMPS data file with a given in-plane cell.

    The cell must be LAMMPS-canonical (a1 along +x). A 20 Å out-of-plane vector
    plus two placeholder atoms give ASE/LAMMPS a valid structure to tile; only
    the in-plane cell shape matters for the coincidence match and box geometry.
    """
    cell_2x2 = np.asarray(cell_2x2, dtype=float)
    cell3 = np.array([
        [cell_2x2[0, 0], 0.0, 0.0],
        [cell_2x2[1, 0], cell_2x2[1, 1], 0.0],
        [0.0, 0.0, 20.0],
    ])
    atoms = Atoms(symbols=["H", "He"], positions=[[0.0, 0.0, 10.0], [0.4, 0.3, 10.0]],
                  cell=cell3, pbc=True)
    style = "atomic"
    if charges is not None:
        atoms.set_initial_charges(charges)
        style = "charge"
    write(str(path), atoms, format="lammps-data", atom_style=style, specorder=["H", "He"])
    return path


def test_matched_supercells_are_symmetrically_strained_onto_mean_cell(tmp_path):
    """Both materials are strained symmetrically onto the shared mean cell
    ``mean_cell = (sc_a + sc_b) / 2`` -- neither is left as an unstrained
    reference (design §8.3). Each per-material strain is logged separately and
    is bounded by ``strain_tol``; the two on-disk boxes coincide and equal the
    mean cell.
    """
    mono_a = _build_small_monolayer("h-MoS2", MOS2_CIF, MOS2_POT)
    mono_b = _build_small_monolayer("h-WS2", WS2_CIF, WS2_POT)

    ms = build_matched_supercells(
        mono_a, mono_b, workdir=tmp_path, strain_tol=0.05, max_supercell=6,
    )

    # Both materials strained (neither is an unstrained reference).
    assert ms.strain_a > 0.0 and ms.strain_b > 0.0
    assert ms.strain_a <= 0.05 + 1e-9 and ms.strain_b <= 0.05 + 1e-9

    # Both supercells share ONE box == mean_cell.
    ca = np.array(read(str(ms.supercell_a), format="lammps-data").cell)[:2, :2]
    cb = np.array(read(str(ms.supercell_b), format="lammps-data").cell)[:2, :2]
    assert np.allclose(ca, cb, atol=1e-3)                    # shared periodic box
    assert np.allclose(ca, ms.mean_cell, atol=1e-3)          # box IS the mean cell

    # Symmetric: neither strain is 0 (the old one-sided policy pinned one to 0);
    # the two strains are comparable, not a 0-vs-full split.
    assert abs(ms.strain_a - ms.strain_b) < max(ms.strain_a, ms.strain_b)


def test_matched_supercells_fail_loud_when_either_strain_exceeds_tol(tmp_path):
    """No silent incommensurate fallback: a pair with no low-strain symmetric
    coincidence must raise ``ValueError`` mentioning the strain, not emit a
    quietly-mismatched stack.
    """
    # Nearly-golden-ratio length ratio: no small commensurate supercell exists
    # under a tight symmetric strain budget.
    cell_a = np.array([[3.0, 0.0], [0.0, 3.0]])
    cell_b = np.array([[3.0 * 1.6180339887, 0.0], [0.0, 3.0]])
    mono_a = _write_synthetic_monolayer(tmp_path / "s_inc_a.lmp", cell_a)
    mono_b = _write_synthetic_monolayer(tmp_path / "s_inc_b.lmp", cell_b)

    with pytest.raises(ValueError, match=r"strain"):
        build_matched_supercells(
            mono_a, mono_b, workdir=tmp_path, strain_tol=0.001,
        )


# A genuinely sheared, non-identity coincidence between two canonical (a1 along
# +x) in-plane cells: find_coincidence_lattice matches these at
#   matrix_a = [[1, 1], [0, 2]]   (sheared, det 2)
#   matrix_b = [[1, 2], [0, 3]]   (sheared, det 3)
# with a true one-sided strain of ~1.39 %. Both matrices have a non-zero
# off-diagonal, so make_supercell tilts each supercell off-axis; the two built
# boxes only coincide if the strain is applied in the shared co-oriented frame
# (not by independently re-canonicalizing each supercell).
_SHEAR_CELL_A = np.array([[2.889595214489674, 0.0], [-1.5942583249814066, 4.747923033943079]])
_SHEAR_CELL_B = np.array([[3.148752463798015, 0.0], [-1.106026312939178, 2.9183412862744404]])


def test_sheared_nonidentity_match_builds_coinciding_boxes(tmp_path):
    mono_a = _write_synthetic_monolayer(tmp_path / "shear_a.lmp", _SHEAR_CELL_A, charges=[0.5, -0.5])
    mono_b = _write_synthetic_monolayer(tmp_path / "shear_b.lmp", _SHEAR_CELL_B, charges=[0.6, -0.6])

    # Sanity: the coincidence really is sheared and non-identity.
    match = find_coincidence_lattice(_SHEAR_CELL_A, _SHEAR_CELL_B,
                                     strain_tol=0.02, max_supercell=3, area_tol=0.06)
    assert match is not None
    assert match.matrix_a[0, 1] != 0 or match.matrix_b[0, 1] != 0, "expected a sheared match"
    assert not (np.array_equal(match.matrix_a, np.eye(2, dtype=int))
                and np.array_equal(match.matrix_b, np.eye(2, dtype=int)))

    stack = build_matched_supercells(
        mono_a, mono_b, workdir=tmp_path, strain_tol=0.02, max_supercell=3, area_tol=0.06,
    )

    ca = np.array(read(str(stack.supercell_a), format="lammps-data").cell)[:2, :2]
    cb = np.array(read(str(stack.supercell_b), format="lammps-data").cell)[:2, :2]

    # The two BUILT boxes must coincide to high precision (the whole point).
    assert np.allclose(ca, cb, atol=1e-4)
    # Both materials strained onto the shared mean cell at the true, low strain
    # -- not the ~12 %/10 % spurious shear an independent-canonicalization build
    # produces on a sheared match (each ~half the one-sided ~1.4 %).
    assert stack.strain_a < 0.02 and stack.strain_b < 0.02
    assert stack.strain_a > 0.0 and stack.strain_b > 0.0
    assert np.allclose(ca, stack.mean_cell, atol=1e-4)
    # The match was built via genuinely non-scalar/sheared matrices.
    assert stack.match.matrix_a[0, 1] != 0 or stack.match.matrix_b[0, 1] != 0

    # Metadata (per-atom charges) survives the build for both materials.
    qa = read(str(stack.supercell_a), format="lammps-data", atom_style="charge").get_initial_charges()
    qb = read(str(stack.supercell_b), format="lammps-data", atom_style="charge").get_initial_charges()
    assert np.allclose(np.abs(qa), np.abs(qa[0]))
    assert np.allclose(np.abs(qb), np.abs(qb[0]))
    assert np.any(qa != 0) and np.any(qb != 0)


def test_incommensurate_pair_raises_instead_of_silently_mismatching(tmp_path):
    """Two monolayers with no low-strain coincidence (within the given repeat/
    strain limits) must be rejected loudly rather than silently written out as
    a "matched" (but actually incommensurate) pair of supercells.
    """
    # Nearly-golden-ratio length ratio: no small commensurate supercell exists
    # under a tight strain budget and a small max repeat.
    cell_a = np.array([[3.0, 0.0], [0.0, 3.0]])
    cell_b = np.array([[3.0 * 1.6180339887, 0.0], [0.0, 3.0]])
    mono_a = _write_synthetic_monolayer(tmp_path / "inc_a.lmp", cell_a)
    mono_b = _write_synthetic_monolayer(tmp_path / "inc_b.lmp", cell_b)

    with pytest.raises(ValueError):
        build_matched_supercells(
            mono_a, mono_b, workdir=tmp_path, strain_tol=0.005, max_supercell=2, area_tol=0.02,
        )


def _wrap_vec_mod_cell(vec, cell_2x2):
    """Wrap a 2-D Cartesian vector into the primitive cell (nearest-image)."""
    inv = np.linalg.inv(np.asarray(cell_2x2, dtype=float))
    frac = np.asarray(vec, dtype=float) @ inv
    frac -= np.round(frac)
    return frac @ np.asarray(cell_2x2, dtype=float)


def test_hetero_stack_is_periodic_and_complete(tmp_path):
    """Assemble two matched supercells into ONE periodic LAMMPS cell and verify
    the built geometry: correct atom count, a physical (ordered, non-overlapping,
    non-exploded) z-extent, a single shared in-plane box (no seam), and -- the
    key regression guard -- the cross-material RI offset SURVIVES placement (the
    old branch re-centered both materials' centroids, cancelling it).
    """
    # The two pre-aligned supercells are already in AA (max-RI) registry at zero
    # shift, so AA would give a (0, 0) offset that cannot exercise the survival
    # guard. AB (anti-registry) is a genuine, non-zero half-cell shift -- use it
    # so the guard is meaningful. Only the non-reference (top) material's
    # stack_type drives the cross-material offset; set both for robustness.
    sheets = [
        _sheet_config("h-MoS2", MOS2_CIF, MOS2_POT).model_copy(update={"stack_type": "AB"}),
        _sheet_config("h-WS2", WS2_CIF, WS2_POT).model_copy(update={"stack_type": "AB"}),
    ]
    settings = load_settings()

    mono_a = _build_small_monolayer("h-MoS2", MOS2_CIF, MOS2_POT)
    mono_b = _build_small_monolayer("h-WS2", WS2_CIF, WS2_POT)

    matched = build_matched_supercells(
        mono_a, mono_b, workdir=tmp_path, strain_tol=0.05, max_supercell=6,
    )
    hp = register_hetero_potentials(sheets, settings, workdir=tmp_path)

    stack = assemble_hetero_stack(matched, sheets, hp, "grouped", settings, workdir=tmp_path)

    # --- total atoms == sum of both matched supercells (1 layer each) ---
    na = len(read(str(matched.supercell_a), format="lammps-data"))
    nb = len(read(str(matched.supercell_b), format="lammps-data"))
    expected_total = na + nb

    at = read(str(stack.data_path), format="lammps-data")
    assert len(at) == expected_total

    # --- distinct, ordered, non-overlapping, non-exploded z stacking ---
    zc = at.get_positions()[:, 2]
    zext = zc.max() - zc.min()
    assert 4.0 < zext < 20.0

    # --- single shared periodic in-plane box (matches the matched supercell) ---
    matched_box_2x2 = np.array(read(str(matched.supercell_a), format="lammps-data").cell)[:2, :2]
    assert np.allclose(np.array(at.cell)[:2, :2], matched_box_2x2, atol=1e-4)

    # --- RI-offset-survival regression guard ---------------------------------
    # Identify (via the stack's own bookkeeping) which stacked layers belong to
    # the reference vs the displaced non-reference material, split the atoms by
    # z accordingly, and check the applied lateral offset actually survives.
    ref_z = np.mean([l.z for l in stack.layers if l.material_index == stack.reference_index])
    non_z = np.mean([l.z for l in stack.layers if l.material_index == stack.nonreference_index])
    # The two materials form two well-separated z-clusters; split the ACTUAL
    # atom z at the midpoint of their range. The material placed at the lower
    # z-shift is the lower atom cluster.
    z_split = 0.5 * (zc.min() + zc.max())
    xy = at.get_positions()[:, :2]
    low_xy, high_xy = xy[zc < z_split], xy[zc >= z_split]
    if non_z < ref_z:
        non_xy, ref_xy = low_xy, high_xy
    else:
        non_xy, ref_xy = high_xy, low_xy

    diff = non_xy.mean(axis=0) - ref_xy.mean(axis=0)
    offset = np.asarray(stack.ri_offset, dtype=float)

    # (a) the non-reference material sits at the intended RI offset (mod cell)...
    assert np.linalg.norm(_wrap_vec_mod_cell(diff - offset, matched_box_2x2)) < 1e-2
    # (b) ...and that offset is genuinely non-zero -- NOT centroid-cancelled.
    assert np.linalg.norm(_wrap_vec_mod_cell(offset, matched_box_2x2)) > 0.5


# =============================================================================
# Task 6: end-to-end structure smoke -- build_hetero_structure -> real LAMMPS
# =============================================================================


def _two_material_config() -> SheetOnSheetSimulationConfig:
    """A real 2-material (MoS2/WS2) SheetOnSheetSimulationConfig via the config path.

    Mirrors the raw-dict [2D-1]/[2D-2] shape of
    ``test_config.py::test_two_material_sections_fold_into_sheets_list`` but with
    the REAL CIFs + SW potentials (so ``build_monolayer`` actually runs) and one
    layer each. ``x``/``y`` are in ANGSTROMS; 4.0 Å is smaller than one
    duplicated unit cell (~6.4 Å) for both materials, so ``build_monolayer``'s
    target-size rounding keeps each supercell at the minimal single ortho cell
    (6 atoms) -- the fastest structure that still exercises the full
    match -> register -> assemble -> LAMMPS path. (30 Å, by contrast, rounds up
    to a ~9x5 supercell of 270 atoms per material -- needlessly slow for a smoke
    test whose point is structural self-consistency, not size.)
    """
    raw = {
        "general": {"temp": 300, "scan_speed": 1, "hetero_stacking": "grouped"},
        "2D-1": {"mat": "h-MoS2", "cif_path": MOS2_CIF, "pot_path": MOS2_POT,
                 "pot_type": "sw", "x": 4.0, "y": 4.0, "layers": [1]},
        "2D-2": {"mat": "h-WS2", "cif_path": WS2_CIF, "pot_path": WS2_POT,
                 "pot_type": "sw", "x": 4.0, "y": 4.0, "layers": [1]},
    }
    return SheetOnSheetSimulationConfig(**raw, settings=load_settings())


def _lammps_run0_energy(data_path, settings_path) -> float:
    """Drive real LAMMPS on the produced files and return the ``run 0`` energy.

    Emits a tiny self-contained script (``units metal``, ``atom_style`` detected
    from the data file, ``boundary p p p``, ``read_data`` + ``include`` the real
    settings, ``run 0``) and runs ``lmp_serial`` headless with ``cwd = workdir``
    -- mandatory, because the settings' ``pair_coeff`` lines reference the staged
    potentials by the run-dir-relative prefix ``provenance/potentials/...``. Both
    ``data_path`` and ``settings_path`` are emitted as absolute paths so they
    resolve independent of cwd. Asserts the log is free of "Lost atoms" (a lost
    atom means the structure is not self-consistent / not genuinely periodic) and
    returns the parsed potential energy.
    """
    data_path = Path(data_path).resolve()
    settings_path = Path(settings_path).resolve()
    workdir = data_path.parent  # hetero.data lives directly in the run dir
    atom_style = _detect_atom_style(data_path)

    script = f"""units           metal
atom_style      {atom_style}
boundary        p p p
read_data       {data_path}
include         {settings_path}
run             0
variable        pe equal pe
variable        na equal atoms
print           "RESULT_PE ${{pe}}"
print           "RESULT_NATOMS ${{na}}"
"""
    script_path = workdir / "run0_smoke.lmp"
    script_path.write_text(script, encoding="utf-8")
    out = _run_lammps(workdir, script_path)

    assert "Lost atoms" not in out, f"LAMMPS reported lost atoms:\n{out[-2000:]}"

    energy = None
    for line in out.splitlines():
        if line.startswith("RESULT_PE"):
            energy = float(line.split()[1])
    assert energy is not None, f"could not parse RESULT_PE from LAMMPS:\n{out[-2000:]}"
    return energy


def test_end_to_end_structure_assembles_in_lammps(tmp_path):
    """Capstone: the single ``build_hetero_structure`` entrypoint ties Tasks 3-5
    together and yields a ``hetero.data`` + ``system.in.settings`` that assemble
    in LAMMPS. A finite, physical ``run 0`` energy with no lost atoms proves the
    structure + atom types + potentials are self-consistent and the box is
    genuinely periodic (p p p).
    """
    cfg = _two_material_config()
    settings = cfg.settings

    stack = build_hetero_structure(cfg, settings, workdir=tmp_path)

    # The entrypoint returns a ready stack pointing at real, existing files.
    assert Path(stack.data_path).exists()
    assert Path(stack.settings_path).exists()

    at = read(str(stack.data_path), format="lammps-data")

    # --- atom count == sum over placed layers of their source-supercell atoms ---
    expected_total = sum(
        len(read(str(layer.source), format="lammps-data")) for layer in stack.layers
    )
    assert len(at) == expected_total

    # --- genuinely 3D-periodic box (all pbc, positive volume) -------------------
    assert bool(np.all(at.pbc))
    assert at.get_volume() > 0.0

    # --- real LAMMPS run 0: finite, physical energy; no lost atoms (in helper) --
    energy = _lammps_run0_energy(stack.data_path, stack.settings_path)
    assert np.isfinite(energy) and abs(energy) < 1e6  # physical, not exploded (~1e9 = failure)


def test_lattice_match_config_is_threaded_into_build(tmp_path):
    """``build_hetero_structure`` must thread the ``[lattice_match]`` section into
    the coincidence search (design §5), not use hardcoded defaults.

    Proof: a ``strain_tol`` far tighter than the ~0.03% MoS2/WS2 mismatch turns
    the otherwise-successful match into a fail-loud, and the raised error reports
    *that* tolerance -- so the config value reached the matcher (with the 2%
    default the match would succeed). Fails at the match step, before any LAMMPS
    run, so it is fast.
    """
    raw = {
        "general": {"temp": 300, "scan_speed": 1, "hetero_stacking": "grouped"},
        "2D-1": {"mat": "h-MoS2", "cif_path": MOS2_CIF, "pot_path": MOS2_POT,
                 "pot_type": "sw", "x": 4.0, "y": 4.0, "layers": [1]},
        "2D-2": {"mat": "h-WS2", "cif_path": WS2_CIF, "pot_path": WS2_POT,
                 "pot_type": "sw", "x": 4.0, "y": 4.0, "layers": [1]},
        "lattice_match": {"strain_tol": 1e-6, "max_supercell": 3},
    }
    cfg = SheetOnSheetSimulationConfig(**raw, settings=load_settings())
    assert cfg.lattice_match is not None and cfg.lattice_match.strain_tol == 1e-6

    with pytest.raises(ValueError, match=r"strain_tol=0\.0001%"):
        build_hetero_structure(cfg, cfg.settings, workdir=tmp_path)


# =============================================================================
# D3: twisted commensurate-moiré supercells + same-material re-enable
# =============================================================================
# NOTE ON ANGLE CHOICE: the plan's example used twist_angle=7.0, but for a
# hexagonal lattice no CLEAN commensurate cell exists near 7 deg within
# max_supercell=12 (the reachable exact angles are ~16.4 / 21.8 deg; 7 deg only
# yields a large, ~120-cell strained approximant -- the "moiré-cell explosion"
# risk the design flags). These tests use 21.8 deg (the (1,2) moiré, achieved
# ~21.787 deg, N=7 rhombus -> x2 orthorhombic wrapper = 14 cells/layer), a
# cheap CLEAN commensurate cell that exercises the identical code paths.

def test_build_twisted_supercells_shares_one_orthorhombic_box(tmp_path):
    from src.builders.hetero import build_twisted_supercells
    mono_a = _build_small_monolayer("h-MoS2", MOS2_CIF, MOS2_POT)
    mono_b = _build_small_monolayer("h-WS2", WS2_CIF, WS2_POT)
    ms = build_twisted_supercells(mono_a, mono_b, workdir=tmp_path, twist_angle=21.8,
                                  angle_tol=3.0, strain_tol=0.03, max_supercell=12)
    assert ms.twist_angle_achieved is not None
    assert abs(ms.twist_angle_achieved - 21.8) <= 3.0
    assert ms.moire_period is not None and ms.moire_period > 0
    ca = read(str(ms.supercell_a), format="lammps-data").get_cell()
    cb = read(str(ms.supercell_b), format="lammps-data").get_cell()
    assert np.allclose(np.array(ca)[:2, :2], np.array(cb)[:2, :2], atol=1e-3)  # shared box
    assert abs(ca[1, 0]) < 1e-3 * abs(ca[0, 0])                                # orthorhombic


def test_same_material_twist_synthesises_two_sheets(tmp_path):
    raw = {  # a SINGLE [2D] material + twist -> same-material moiré
        "general": {"temp": 300, "scan_speed": 1, "hetero_stacking": "grouped",
                    "twist_angle": 21.8},
        # x=y=4 keeps build_monolayer at its PRIMITIVE orthogonalized cell so the
        # moiré is built from the primitive lattice (a larger footprint would
        # pre-tile the monolayer and inflate the moiré to hundreds of atoms).
        "2D": {"mat": "h-MoS2", "cif_path": MOS2_CIF, "pot_path": MOS2_POT,
               "pot_type": "sw", "x": 4.0, "y": 4.0, "layers": [1]},
    }
    cfg = SheetOnSheetSimulationConfig(**raw, settings=load_settings())
    assert len(cfg.sheets) == 1                          # a single [2D] material
    stack = build_hetero_structure(cfg, cfg.settings, workdir=tmp_path)
    assert stack.twist_angle_achieved is not None        # twist recorded
    assert stack.moire_period is not None and stack.moire_period > 0
    assert stack.total_types >= 2                        # two material blocks even for same material


def _read_hetero_masses_and_counts(data_path):
    """Parse a written ``hetero.data``: return (type->mass, type->n_atoms).

    Reads the resolved per-type mass from the Masses section and counts atoms
    per type from the Atoms section -- the physics the type/registry checks
    alone cannot see.
    """
    from collections import Counter
    type_mass: dict[int, float] = {}
    type_count: Counter = Counter()
    section = None
    for raw_line in Path(data_path).read_text().splitlines():
        s = raw_line.strip()
        if s == "Masses":
            section = "M"; continue
        if s.startswith("Atoms"):
            section = "A"; continue
        if s in ("Velocities", "Bonds", "Angles", "Dihedrals", "Impropers"):
            section = None; continue
        if not s:
            continue
        parts = s.split()
        if not parts[0].lstrip("-").isdigit():
            continue
        if section == "M":
            type_mass[int(parts[0])] = float(parts[1])
        elif section == "A":
            type_count[int(parts[1])] += 1
    return type_mass, type_count


def test_same_material_twist_masses_preserve_metal(tmp_path):
    """The same-material twist path must keep MoS2's stoichiometry AND masses.

    Regression for the overlapping-mass-range bug: the second synthesised sheet's
    METAL sublattice was emitted with sulfur's mass (net MoS5), because
    ``get_masses_string`` collapsed the non-contiguous Mo type list [1, 3] into
    the LAMMPS range ``1*3`` and the later S range ``2*4`` overwrote type 3 (Mo).
    The type registry looked correct; only the resolved per-atom masses in
    ``hetero.data`` were wrong -- the physics the registry checks cannot see.

    Asserts on the built ``hetero.data``: (a) exactly two distinct mass species,
    which must be {32.06 (S), 95.95 (Mo)}; (b) the S:metal mass-count ratio is
    2:1 over the WHOLE cell (both layers), i.e. every Mo atom carries 95.95 and
    the honeycomb 1:2 metal:chalcogen ratio holds in mass space.
    """
    raw = {
        "general": {"temp": 300, "scan_speed": 1, "hetero_stacking": "grouped",
                    "twist_angle": 21.8},
        "2D": {"mat": "h-MoS2", "cif_path": MOS2_CIF, "pot_path": MOS2_POT,
               "pot_type": "sw", "x": 4.0, "y": 4.0, "layers": [1]},
    }
    cfg = SheetOnSheetSimulationConfig(**raw, settings=load_settings())
    stack = build_hetero_structure(cfg, cfg.settings, workdir=tmp_path)

    type_mass, type_count = _read_hetero_masses_and_counts(stack.data_path)
    assert set(type_mass) == set(type_count), "every atom type needs a mass"

    # (a) exactly two mass species, and they are S and Mo (not a stray sulfur mass).
    species = sorted(set(round(m, 2) for m in type_mass.values()))
    assert species == [32.06, 95.95], (
        f"expected only S(32.06) and Mo(95.95) masses, got {species} -- a metal "
        f"sublattice is carrying the wrong mass (MoS5 instead of MoS2)."
    )

    # (b) aggregate atom counts by mass species over the whole (two-layer) cell.
    metal_atoms = sum(n for t, n in type_count.items() if round(type_mass[t], 2) == 95.95)
    chalc_atoms = sum(n for t, n in type_count.items() if round(type_mass[t], 2) == 32.06)
    assert metal_atoms > 0 and chalc_atoms > 0
    assert chalc_atoms == 2 * metal_atoms, (
        f"S:Mo mass-count ratio must be 2:1 (MoS2), got {chalc_atoms}:{metal_atoms}."
    )

    # BOTH synthesised layers must contribute Mo atoms in the 1:2 metal:chalc
    # ratio -- proves the fix is not a single-layer accident.
    per_layer_metal = metal_atoms // 2
    assert metal_atoms == 2 * per_layer_metal            # equal Mo per layer
    assert chalc_atoms == 4 * per_layer_metal            # 2 layers x 2 S per Mo


def test_get_masses_string_no_overlapping_ranges_across_components(tmp_path):
    """``get_masses_string`` must never let one element's range overwrite another
    element's type when the same element spans multiple components.

    Locks the invariant for BOTH the same-material duplicate (Mo/S + Mo/S, where
    the bug bit: Mo=[1,3], S=[2,4]) and the genuine hetero pair (Mo/S + W/S, the
    path that only *accidentally* survived because a later W line re-corrected the
    engulfed type). Applies LAMMPS last-wins semantics to the emitted string and
    requires every type's FINAL mass to equal its registered element's mass.
    """
    from ase.data import atomic_masses, atomic_numbers

    def _final_masses(masses_str, n_types):
        final: dict[int, float] = {}
        for line in masses_str.splitlines():
            parts = line.split()
            if len(parts) < 3 or parts[0] != "mass":
                continue
            spec, val = parts[1], float(parts[2])
            if "*" in spec:
                lo_s, hi_s = spec.split("*")
                lo = int(lo_s) if lo_s else 1
                hi = int(hi_s) if hi_s else n_types
            else:
                lo = hi = int(spec)
            for t in range(lo, hi + 1):
                final[t] = val
        return final

    for sheets in (
        [_sheet_config("h-MoS2", MOS2_CIF, MOS2_POT),
         _sheet_config("h-MoS2", MOS2_CIF, MOS2_POT)],          # same-material
        [_sheet_config("h-MoS2", MOS2_CIF, MOS2_POT),
         _sheet_config("h-WS2", WS2_CIF, WS2_POT)],             # hetero pair
    ):
        hp = register_hetero_potentials(sheets, load_settings(), workdir=tmp_path)
        n_types = len(hp.pm.types)
        final = _final_masses(hp.pm.get_masses_string(), n_types)
        assert set(final) == set(range(1, n_types + 1))
        for atype in hp.pm.types:
            expected = atomic_masses[atomic_numbers[atype.element]]
            assert abs(final[atype.type_id] - expected) < 0.5, (
                f"type {atype.type_id} ({atype.element}) resolved to "
                f"{final[atype.type_id]}, expected ~{expected:.2f}"
            )


# =============================================================================
# B3: edge-atom typing for finite hetero sheets
# =============================================================================


def test_register_hetero_potentials_edge_regions_disjoint(tmp_path):
    """With ``regions=[None, 'edge']`` each material's types double into an
    interleaved core/edge pair (mirroring ``apply_sheet_edge_types``), and the
    per-material blocks stay disjoint.
    """
    sheets = [
        _sheet_config("h-MoS2", MOS2_CIF, MOS2_POT),
        _sheet_config("h-WS2", WS2_CIF, WS2_POT),
    ]
    hp = register_hetero_potentials(
        sheets, load_settings(), workdir=tmp_path, regions=[None, "edge"]
    )
    c1 = hp.pm.types.ids_by_component_region("sheet_1", "edge")
    c2 = hp.pm.types.ids_by_component_region("sheet_2", "edge")
    core1 = hp.pm.types.ids_by_component_region("sheet_1", None)
    assert c1 and c2  # each material has edge types
    assert set(c1).isdisjoint(core1)  # edge disjoint from core
    assert set(hp.pm.types.ids_by_component("sheet_1")).isdisjoint(
        hp.pm.types.ids_by_component("sheet_2")
    )


def test_register_hetero_potentials_no_regions_unchanged(tmp_path):
    """Default ``regions=None`` registers no edge types (byte-identical to the
    periodic path)."""
    sheets = [
        _sheet_config("h-MoS2", MOS2_CIF, MOS2_POT),
        _sheet_config("h-WS2", WS2_CIF, WS2_POT),
    ]
    hp = register_hetero_potentials(sheets, load_settings(), workdir=tmp_path)
    assert hp.pm.types.ids_by_component_region("sheet_1", "edge") == []
    assert hp.pm.types.ids_by_component_region("sheet_2", "edge") == []


import shutil  # noqa: E402  (B3 real-LAMMPS tests below)

LMP = shutil.which("lmp_serial")


def _finite_hetero_cfg(tmp_path, layers=None):
    """A finite (edge-typed) two-material hetero config.

    Reuses the smallest MoS2/WS2 coincidence pair (``x=y=4`` -> dup (1,1) ->
    the ~19x22 A coincidence supercell), with ``general.finite_sheet=True`` so
    :func:`build_hetero_structure` takes the edge-typing branch.
    """
    lyr = layers if layers is not None else [1]
    raw = {
        "general": {"temp": 300, "scan_speed": 1, "hetero_stacking": "grouped",
                    "finite_sheet": True},
        "2D-1": {"mat": "h-MoS2", "cif_path": MOS2_CIF, "pot_path": MOS2_POT,
                 "pot_type": "sw", "x": 20.0, "y": 20.0, "layers": lyr},
        "2D-2": {"mat": "h-WS2", "cif_path": WS2_CIF, "pot_path": WS2_POT,
                 "pot_type": "sw", "x": 20.0, "y": 20.0, "layers": lyr},
    }
    return SheetOnSheetSimulationConfig(**raw, settings=load_settings())


def _parse_lammps_atoms(path):
    """Return ``(type_ids, xy)`` read straight from the data file's Atoms section.

    Column 2 is the LAMMPS type id for ``atom_style atomic`` (id type x y z);
    columns 3-4 are x, y. ASE's lammps-data reader maps types to chemical Z
    (all even for Mo/S/W), so the even/odd edge rule needs the RAW type column,
    which only a direct parse gives.
    """
    lines = Path(path).read_text().splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().split("#")[0].strip() == "Atoms":
            start = i
            break
    assert start is not None, f"no Atoms section in {path}"
    type_ids = []
    xy = []
    for line in lines[start + 1:]:
        s = line.strip()
        if not s:
            if type_ids:  # blank line after the block ends it
                break
            continue  # blank line(s) between the header and the first row
        if s.split("#")[0].strip() == "":
            continue
        parts = s.split()
        type_ids.append(int(parts[1]))
        xy.append((float(parts[2]), float(parts[3])))
    return np.array(type_ids, dtype=int), np.array(xy, dtype=float)


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_finite_hetero_builds_with_doubled_types(tmp_path):
    """The finite branch edge-types the assembled patch, so its data file
    carries the doubled (core/edge) atom types -- exactly twice the single
    (periodic) spine's type count. (The plan's ``>= 4`` cannot fail on the
    un-implemented periodic path, whose spine here is already 5 types, so we
    assert the exact doubling against a reference single registration.)"""
    cfg = _finite_hetero_cfg(tmp_path)
    single = len(
        register_hetero_potentials(
            list(cfg.sheets), cfg.settings, workdir=tmp_path / "ref_spine"
        ).pm.types
    )
    stack = build_hetero_structure(cfg, cfg.settings, workdir=tmp_path)
    text = Path(stack.data_path).read_text()
    m = re.search(r"(\d+)\s+atom types", text)
    assert m and int(m.group(1)) == 2 * single  # every real type split core/edge


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_finite_hetero_emits_2D_edge_union(tmp_path):
    """The finite build emits per-material edge groups plus the combined
    ``2D_edge`` union and a ``2D`` union (so the finite run templates'
    ``group sheet_edge intersect 2D 2D_edge`` fires unchanged)."""
    cfg = _finite_hetero_cfg(tmp_path)
    stack = build_hetero_structure(cfg, cfg.settings, workdir=tmp_path)
    settings_text = Path(stack.settings_path).read_text()
    assert "group sheet_1_edge" in settings_text
    assert "group sheet_2_edge" in settings_text
    assert "group 2D_edge union sheet_1_edge sheet_2_edge" in settings_text
    assert "group 2D union" in settings_text


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_edge_atoms_on_true_perimeter_not_interior(tmp_path):
    """Regression guard for the ordering invariant: because edge typing runs
    AFTER matching + stacking (on the assembled patch), the edge-typed atoms sit
    on the patch's TRUE perimeter -- NOT buried in the supercell interior (the
    failure the pre-tiling order would produce)."""
    cfg = _finite_hetero_cfg(tmp_path)
    stack = build_hetero_structure(cfg, cfg.settings, workdir=tmp_path)
    type_id, xy = _parse_lammps_atoms(stack.data_path)
    ew = cfg.settings.geometry.finite_sheet_edge_width
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    # apply_sheet_edge_types numbering: core -> 2(t-1)+1 (odd), edge -> 2(t-1)+2 (even).
    edge_mask = (type_id % 2 == 0)
    assert edge_mask.any(), "no edge-typed atoms found"
    e = xy[edge_mask]
    near_perimeter = ((e[:, 0] - lo[0] < ew) | (hi[0] - e[:, 0] < ew) |
                      (e[:, 1] - lo[1] < ew) | (hi[1] - e[:, 1] < ew))
    assert near_perimeter.mean() > 0.8, (
        f"only {near_perimeter.mean():.0%} of edge atoms are within {ew} A of the "
        f"patch perimeter -- edge band buried in the interior (ordering regression)"
    )
