import shutil

import numpy as np
import pytest
from ase import Atoms
from ase.io import write, read
from frictionsim2d.lattice.apply import strain_monolayer_to_cell
from tests.helpers.lattice import _flake_lattice_orientation


def _write_square_monolayer(path, a=3.0):
    # 2x2 atoms on a square lattice, thin slab in z
    pos = [(0, 0, 5), (a, 0, 5), (0, a, 5), (a, a, 5)]
    atoms = Atoms("H4", positions=pos, cell=[2 * a, 2 * a, 20], pbc=[True, True, False])
    write(str(path), atoms, format="lammps-data")


def test_strain_scales_cell_and_positions(tmp_path):
    src = tmp_path / "in.data"
    out = tmp_path / "out.data"
    _write_square_monolayer(src, a=3.0)                 # in-plane cell = 6.0 x 6.0
    target = np.array([[6.3, 0.0], [0.0, 6.3]])          # +5% isotropic
    strain = strain_monolayer_to_cell(str(src), str(out), target)

    got = read(str(out), format="lammps-data")
    cell = np.array(got.cell)[:2, :2]
    assert np.allclose(cell, target, atol=1e-6)          # box now equals target
    assert 0.049 < strain < 0.051                        # ~5% principal strain
    # z-height unchanged
    assert abs(np.array(got.cell)[2, 2] - 20.0) < 1e-6


def test_strain_is_zero_when_target_equals_source(tmp_path):
    src = tmp_path / "in.data"
    out = tmp_path / "out.data"
    _write_square_monolayer(src, a=3.0)                 # cell 6.0 x 6.0
    target = np.array([[6.0, 0.0], [0.0, 6.0]])          # identical
    strain = strain_monolayer_to_cell(str(src), str(out), target)
    assert strain < 1e-9


def _write_charged_square_monolayer(path, a=3.0, charges=(0.1, -0.1, 0.2, -0.2)):
    # 2x2 atoms on a square lattice, atom_style=charge with distinct per-atom
    # charges -- a bare ASE lammps-data round-trip silently drops these.
    pos = [(0, 0, 5), (a, 0, 5), (0, a, 5), (a, a, 5)]
    atoms = Atoms("H4", positions=pos, cell=[2 * a, 2 * a, 20], pbc=[True, True, False])
    atoms.set_initial_charges(list(charges))
    write(str(path), atoms, format="lammps-data", atom_style="charge")


def test_strain_preserves_charges_and_atom_style(tmp_path):
    src = tmp_path / "in_charge.data"
    out = tmp_path / "out_charge.data"
    charges = (0.1, -0.1, 0.2, -0.2)
    _write_charged_square_monolayer(src, a=3.0, charges=charges)   # cell 6.0 x 6.0
    target = np.array([[6.3, 0.0], [0.0, 6.3]])                    # +5% isotropic

    strain = strain_monolayer_to_cell(str(src), str(out), target)
    assert 0.049 < strain < 0.051

    # The output must still declare atom_style=charge in its Atoms header.
    out_text = out.read_text(encoding="utf-8")
    header_line = next(l for l in out_text.splitlines() if l.strip().startswith("Atoms"))
    assert "charge" in header_line

    got = read(str(out), format="lammps-data", atom_style="charge")
    got_charges = np.array(got.get_initial_charges())
    # Charges are carried per-atom; matching by value (order may permute) is
    # sufficient to prove they were not dropped or zeroed.
    assert np.allclose(sorted(got_charges), sorted(charges), atol=1e-9)


def _write_hex_monolayer(path, a=3.16):
    # one atom, hexagonal cell a1=(a,0), a2=(-a/2, a*sqrt(3)/2) -- a1 is already
    # along +x (LAMMPS-canonical) but a2 gives the cell a nonzero xy tilt.
    a2 = np.array([-a / 2, a * np.sqrt(3) / 2])
    cell = [[a, 0, 0], [a2[0], a2[1], 0], [0, 0, 20]]
    write(str(path), Atoms("C", positions=[(0, 0, 10)], cell=cell, pbc=[True, True, False]),
          format="lammps-data")


def test_strain_hex_cell_with_xy_tilt_preserves_z_and_returns_strain(tmp_path):
    src = tmp_path / "hex.data"
    out = tmp_path / "hex_s.data"
    _write_hex_monolayer(src, a=3.16)
    # Target: the same hex shape scaled +3% (a1 along x, a2 tilted). It is
    # already LAMMPS-canonical (target[0][1] == 0.0) -- no canonicalisation
    # helper needed -- and target[1][0] != 0 alone is what drives
    # strain_monolayer_to_cell's triclinic (needs_xy_clause) branch.
    a = 3.16 * 1.03
    target = np.array([[a, 0.0], [-a / 2, a * np.sqrt(3) / 2]])
    strain = strain_monolayer_to_cell(str(src), str(out), target)

    got = read(str(out), format="lammps-data")
    got_cell = np.array(got.cell)
    assert abs(got_cell[2, 2] - 20.0) < 1e-6                # z untouched
    assert 0.02 < strain < 0.04                              # ~3%
    # The xy tilt clause was actually exercised (not skipped as a no-op):
    # the written box carries the target's nonzero xy tilt.
    assert abs(got_cell[1, 0]) > 1e-6
    assert np.allclose(got_cell[:2, :2], target, atol=1e-6)  # box now equals the tilted target


def test_strain_an_orthogonal_cell_onto_a_tilted_target_makes_the_box_triclinic(tmp_path):
    """An orthogonal source has no tilt to change: the output only gets the
    target's xy if the box is converted to triclinic first."""
    src = tmp_path / "in.data"
    out = tmp_path / "tilted.data"
    _write_square_monolayer(src, a=3.0)                  # orthogonal 6 x 6
    target = np.array([[6.0, 0.0], [1.5, 6.0]])          # same lengths, sheared
    strain_monolayer_to_cell(str(src), str(out), target)

    got_cell = np.array(read(str(out), format="lammps-data").cell)
    assert got_cell[1, 0] == pytest.approx(1.5, abs=1e-6)
    assert np.allclose(got_cell[:2, :2], target, atol=1e-6)


# --- D3b: rotate_flake_in_plane (finite-flake atoms-only twist, cell fixed) ---

def test_rotate_flake_in_plane_twists_lattice_and_keeps_box(tmp_path):
    from ase.io import read
    from frictionsim2d.core.config import FlakeConfig
    from frictionsim2d.builders.construction.components import build_flake_component
    from frictionsim2d.lattice.apply import rotate_flake_in_plane

    MAT, POT = "frictionsim2d/data/materials", "frictionsim2d/data/potentials/sw"
    flake_cfg = FlakeConfig(shape="hexagon", edge_length=14.0, corner_radius=3.0,
                            mat="h-MoS2", cif_path=f"{MAT}/h-MoS2.cif",
                            pot_path=f"{POT}/MoS2_wen.sw", pot_type="sw")
    src_path, _ = build_flake_component(flake_cfg, tmp_path)
    a0 = read(str(src_path), format="lammps-data")

    out = tmp_path / "twisted.data"
    shutil.copy(src_path, out)
    rotate_flake_in_plane(out, 15.0)
    a1 = read(str(out), format="lammps-data")

    # atoms conserved; box (cell) unchanged (only atoms rotated, not the frame)
    assert len(a1) == len(a0)
    assert np.allclose(np.array(a1.cell), np.array(a0.cell), atol=1e-6)

    rel = (_flake_lattice_orientation(a1) - _flake_lattice_orientation(a0)) % 60.0
    rel = min(rel, 60.0 - rel)
    assert abs(rel - 15.0) < 1.0        # the LATTICE actually twisted by ~15 deg (a real twist, not a no-op)


def test_strain_without_a_masses_section_falls_back_to_a_placeholder_mass(tmp_path):
    src = tmp_path / "bare.data"
    src.write_text(
        "bare\n\n2 atoms\n1 atom types\n\n0 6 xlo xhi\n0 6 ylo yhi\n0 20 zlo zhi\n\n"
        "Atoms # atomic\n\n1 1 0 0 5\n2 1 3 3 5\n", encoding="utf-8")
    out = tmp_path / "out.data"
    strain = strain_monolayer_to_cell(str(src), str(out), np.array([[6.3, 0.0], [0.0, 6.3]]))
    assert 0.049 < strain < 0.051
    assert "Masses" in out.read_text(encoding="utf-8")


def test_strain_rejects_a_target_not_along_x(tmp_path):
    src = tmp_path / "in.data"
    _write_square_monolayer(src)
    with pytest.raises(ValueError, match="first vector along x"):
        strain_monolayer_to_cell(str(src), str(tmp_path / "o.data"),
                                 np.array([[6.0, 1.0], [0.0, 6.0]]))
