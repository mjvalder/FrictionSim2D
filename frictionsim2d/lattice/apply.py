"""Apply lattice-matching results to LAMMPS monolayer data files.

Bridges the pure-math ``lattice_matching`` module to on-disk structures. Kept
separate so ``lattice_matching`` stays dependency-free (math only).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..builders.utils.lammps_data import (
    detect_atom_style, monolayer_to_atoms, read_monolayer, write_supercell,
)
from .matching import _max_strain, rotation_angle_between

_TILT_EPS = 1e-9


def _has_masses_section(path: str) -> bool:
    """Return True if the data file already defines a ``Masses`` section."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip() == "Masses":
                return True
    return False


def strain_monolayer_to_cell(input_data_path: str,
                             output_data_path: str,
                             target_cell_2x2: np.ndarray) -> float:
    """Affinely strain a monolayer so its in-plane cell becomes ``target_cell_2x2``.

    Drives native LAMMPS ``change_box ... remap`` to apply the strain, so atom
    motion is a pure affine transform (fractional coordinates preserved)
    while every other data-file field -- atom_style, per-atom charges, type
    IDs, masses, velocities, etc. -- round-trips through LAMMPS's own
    read_data/write_data untouched. This avoids the metadata loss of a bare
    ASE read/write round-trip (which silently drops per-atom charges on
    ``atom_style=charge`` files and can mangle multi-element type fidelity).

    The out-of-plane (z) vector and any xz/yz tilt are left unchanged. Only
    the in-plane box (x, y extents and the xy tilt factor) and atom x/y
    coordinates are modified.

    Args:
        input_data_path: Path to the source LAMMPS data file.
        output_data_path: Path to write the strained LAMMPS data file.
        target_cell_2x2: 2x2 target in-plane cell (row vectors, Å), in
            LAMMPS-canonical form (first vector along x, i.e. ``target[0][1]``
            must be ~0).

    Returns:
        Max absolute principal strain applied (deforming the source in-plane
        cell to ``target_cell_2x2``).
    """
    target = np.asarray(target_cell_2x2, dtype=float)
    assert abs(target[0][1]) < _TILT_EPS, \
        "target_cell_2x2 must have its first vector along x (LAMMPS-canonical form)"

    atom_style = detect_atom_style(input_data_path)
    needs_mass_fallback = not _has_masses_section(input_data_path)

    from ..interfaces.lammps import create_lammps_instance  # pylint: disable=import-outside-toplevel

    lmp = create_lammps_instance()
    try:
        lmp.command("clear")
        lmp.command("units metal")
        lmp.command(f"atom_style {atom_style}")
        lmp.command(f'read_data "{input_data_path}"')

        if needs_mass_fallback:
            # write_data refuses to run unless every type has a mass. Files
            # without a Masses section (e.g. bare test fixtures) never had
            # real per-type masses to preserve, so a placeholder is safe --
            # it adds nothing that was meaningfully present before.
            lmp.command("mass * 1.0")

        boxlo, boxhi, xy, _yz, _xz, _periodicity, _box_change = lmp.extract_box()
        xlo, ylo, _zlo = boxlo
        xhi, yhi, _zhi = boxhi
        source_2x2 = np.array([[xhi - xlo, 0.0], [xy, yhi - ylo]])

        lx = float(target[0][0])
        ly = float(target[1][1])
        xy_final = float(target[1][0])
        needs_xy_clause = abs(xy) > _TILT_EPS or abs(xy_final) > _TILT_EPS

        cmd = "change_box all "
        if needs_xy_clause:
            # Converting box style to triclinic is safe even if it already is
            # (LAMMPS treats a repeated `triclinic` keyword as a no-op).
            cmd += "triclinic "
        cmd += f"x final {xlo} {xlo + lx} y final {ylo} {ylo + ly} "
        if needs_xy_clause:
            cmd += f"xy final {xy_final} "
        cmd += "remap units box"
        lmp.command(cmd)

        lmp.command(f'write_data "{output_data_path}"')
    finally:
        lmp.close()

    return _max_strain(target, source_2x2)


def rotate_layer_to_angle(mono_path: str,
                          angle_deg: float,
                          out_path: str) -> float:
    """Rigidly reorient a monolayer's atoms + cell about z by ``angle_deg``.

    ``change_box`` (used by :func:`strain_monolayer_to_cell`) is affine and
    **cannot rotate**, so the reorientation is done in ASE: the atoms and the
    lattice vectors are rotated together about z (``atoms.rotate(angle_deg,
    'z', rotate_cell=True)``), then written back to a LAMMPS data file. The
    write re-canonicalises the box to the LAMMPS frame (first vector along +x),
    preserving every LAMMPS type as a distinct placeholder element, the real
    ``Masses`` section, per-atom charges, and the ``atom_style`` -- reusing
    :func:`~frictionsim2d.builders.construction.hetero_stack.build_matched_supercells`' metadata
    machinery,
    so nothing is dropped the way a bare ASE round-trip would.

    Note on periodicity: a rigid rotation of a *periodic* cell is a
    lattice-symmetry-equivalent operation, so after re-canonicalisation the
    reoriented monolayer's on-disk box matches the input's -- the reorientation
    only becomes physically observable **relative to another layer**. The
    twisted heterostructure realises the relative twist by tiling this layer
    with its commensurate moiré supercell matrix into the shared moiré box (see
    :func:`~frictionsim2d.builders.construction.hetero_stack.build_twisted_supercells`); this
    function is the
    reorientation primitive and the achieved-angle reporter.

    Args:
        mono_path: Path to the source monolayer LAMMPS data file.
        angle_deg: In-plane rotation about z (degrees, CCW positive).
        out_path: Path to write the reoriented monolayer data file.

    Returns:
        The **applied** in-plane rotation (deg), measured off the reoriented
        lattice vectors before the LAMMPS-frame canonicalisation (equals
        ``angle_deg`` up to numerical precision). Reported explicitly because
        commensurate twist angles are discrete and the achieved angle must
        never be silently assumed.
    """
    mono = read_monolayer(Path(mono_path))
    atoms = monolayer_to_atoms(mono)

    cell_in = np.array(atoms.cell)[:2, :2].copy()
    atoms.rotate(angle_deg, "z", rotate_cell=True)
    cell_rot = np.array(atoms.cell)[:2, :2].copy()

    write_supercell(
        atoms,
        Path(out_path),
        atom_style=str(mono["atom_style"]),
        n_types=int(mono["n_types"]),
        masses=mono["masses"],  # type: ignore[arg-type]
    )

    # Applied rotation carrying the input cell onto the reoriented cell.
    return rotation_angle_between(cell_in, cell_rot)


def rotate_flake_in_plane(flake_path, angle_deg, out_path=None):
    """Twist a FINITE flake by rotating its atoms about z in the FIXED box frame.

    Unlike :func:`rotate_layer_to_angle` (which rotates the cell too and
    re-canonicalises -- a lattice-symmetry no-op for a layer in a shared box),
    this rotates ONLY the atom xy positions, about the flake centroid, leaving
    the cell unchanged. The flake is finite, so this is a genuine twist relative
    to whatever it is later placed on (the periodic sheet / the tip contact).
    Metadata (types/masses/charges/atom_style) is preserved by reusing the same
    metadata-preserving I/O helpers as :func:`rotate_layer_to_angle`.

    Args:
        flake_path: Path to the source flake LAMMPS data file.
        angle_deg: In-plane twist about z (degrees, CCW positive).
        out_path: Where to write the twisted flake; defaults to overwriting
            ``flake_path`` in place.

    Returns:
        The output path (a :class:`~pathlib.Path`).
    """
    out_path = Path(out_path) if out_path is not None else Path(flake_path)
    mono = read_monolayer(Path(flake_path))
    atoms = monolayer_to_atoms(mono)
    cxy = atoms.get_positions()[:, :2].mean(axis=0)
    atoms.rotate(angle_deg, "z", center=(cxy[0], cxy[1], 0.0), rotate_cell=False)
    write_supercell(
        atoms,
        out_path,
        atom_style=str(mono["atom_style"]),
        n_types=int(mono["n_types"]),
        masses=mono["masses"],  # type: ignore[arg-type]
    )
    return out_path
