#!/usr/bin/env python3
"""Generate the three tiny MoS2 LAMMPS data files used by the Milestone-0 decks.

Standalone by design: ASE only, no FrictionSim2D imports, so it runs on a bare
GPU workstation with nothing but `pip install ase`.

    python make_geometries.py [--outdir .]

Writes three `atom_style atomic` data files (metal units, Angstrom):

  bilayer_mos2.data          24 atoms, 4 atom types   (2H-MoS2, 2x2x2 layers)
  trilayer_mos2.data         36 atoms, 6 atom types   (2H-MoS2, 2x2x3 layers)
  bilayer_plus_virtual.data  25 atoms, 5 atom types   (bilayer + 1 virtual atom)

ATOM TYPE LAYOUT (the whole point of this script)
-------------------------------------------------
`ase.io.write(..., format='lammps-data')` assigns one atom type per chemical
species, which would give MoS2 exactly 2 types no matter how many layers there
are. The Milestone-0 decks need **one Mo type and one S type per layer**, so
that a per-layer `pair_style hybrid` NULL map (`deepmd 1 Mo S NULL NULL ...`)
can address a single layer. We therefore set an explicit per-atom `type` array,
which ASE's writer honours (`ase/io/lammpsdata.py::_get_types`), giving:

  layer 1 -> types 1 (Mo), 2 (S)
  layer 2 -> types 3 (Mo), 4 (S)
  layer 3 -> types 5 (Mo), 6 (S)
  virtual -> the last type (5 in bilayer_plus_virtual.data)

The written `Masses` section carries the true Mo/S masses per type; the decks
repeat them explicitly so each deck stays readable and self-contained.

GEOMETRY
--------
Built from `ase.build.mx2(formula='MoS2', kind='2H')`:

  in-plane lattice constant a  = 3.18 A
  S-S layer thickness          = 3.19 A  (S at Mo_z +/- 1.595 A)
  interlayer spacing           = 6.15 A  (bulk 2H-MoS2 c/2)
  in-plane supercell           = orthogonal 2 x 1 (12 atoms per layer)
  2H stacking                  = each successive layer shifted by (2/3, 1/3)
                                 of the *primitive* in-plane vectors, so Mo of
                                 layer n+1 sits above S of layer n

The hexagonal primitive cell is first converted to its **orthogonal**
rectangular representation (P = [[1,0,0],[1,2,0],[0,0,1]], 6 atoms) and then
repeated 2x along x. This is deliberate: the tilted hexagonal 2x2 supercell has
|xy| = a exactly equal to half the box length, i.e. sitting precisely on
LAMMPS' "triclinic box skew is too large" threshold. An orthogonal box sidesteps
that entirely and needs no `box tilt large`.

Cell vectors of every file (orthogonal, so LAMMPS gets no `xy xz yz` line):

  a1 = ( 6.36,      0.0,      0.0 )
  a2 = (  0.0,  5.50792157,   0.0 )
  a3 = (  0.0,      0.0,      c   )   c = slab extent + 2 * 12 A vacuum

`bilayer_plus_virtual.data` uses **exactly** the bilayer cell and the same 24
bilayer positions, with one extra atom appended. That is required by the M0.4
PASS criterion: the total energy of the virtual-atom run must differ from the
plain bilayer run by a *constant* only, which is only meaningful if nothing
else about the cell changed. The vacuum padding is what makes room for the
virtual atom 3 A above the top S plane without it touching the periodic image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from ase import Atom, Atoms
from ase.build import make_supercell, mx2
from ase.io import write

# --- geometry constants (Angstrom) ------------------------------------------
LATTICE_A = 3.18          # in-plane lattice constant of 2H-MoS2
THICKNESS = 3.19          # S-S vertical separation within one layer
INTERLAYER = 6.15         # layer-to-layer spacing (bulk 2H-MoS2 c/2)
NX, NY = 2, 1             # repeats of the orthogonal cell -> 12 atoms per layer
VACUUM = 12.0             # vacuum padding added on each side along z
VIRTUAL_HEIGHT = 3.0      # virtual atom sits this far above the top S plane

# Placeholder species for the virtual atom. It only fixes the mass written into
# the data file (the deck overrides it with `mass <n> 1.0`); in the pair_coeff
# lines the virtual type is mapped to a real element present in the model
# (spec 1.4 placeholder-element mapping), and neigh_modify excludes it anyway.
VIRTUAL_SYMBOL = "H"
VIRTUAL_MASS = 1.0

# 2H stacking shift, in fractional coordinates of the in-plane lattice vectors.
STACK_SHIFT = (2.0 / 3.0, 1.0 / 3.0)


def _primitive() -> Atoms:
    """The hexagonal 3-atom 2H-MoS2 primitive layer, centred on z = 0."""
    return mx2(
        formula="MoS2",
        kind="2H",
        a=LATTICE_A,
        thickness=THICKNESS,
        size=(1, 1, 1),
        vacuum=None,
    )


def _monolayer() -> tuple[Atoms, np.ndarray]:
    """One orthogonal 2x1 2H-MoS2 layer (12 atoms) plus the 2H stacking shift.

    Returns the layer and the Cartesian in-plane vector by which each
    successive layer must be translated to reproduce 2H (AA') stacking.
    """
    prim = _primitive()
    shift = STACK_SHIFT[0] * np.array(prim.cell[0]) + STACK_SHIFT[1] * np.array(
        prim.cell[1]
    )

    # make_supercell needs a non-singular cell, and mx2(vacuum=None) leaves
    # c = 0. The placeholder height is replaced in _stack().
    prim.cell[2] = [0.0, 0.0, 1.0]
    ortho = make_supercell(prim, [[1, 0, 0], [1, 2, 0], [0, 0, 1]])
    layer = ortho.repeat((NX, NY, 1))
    return layer, shift


def _stack(n_layers: int) -> Atoms:
    """Stack `n_layers` 2H-MoS2 layers with the proper (2/3, 1/3) shift."""
    unit, shift = _monolayer()
    cell = np.array(unit.cell)

    stack = Atoms(cell=cell, pbc=True)
    for i in range(n_layers):
        layer = unit.copy()
        offset = i * shift + np.array([0.0, 0.0, i * INTERLAYER])
        layer.translate(offset)
        stack += layer

    # z extent: the slab plus symmetric vacuum. Set the cell height first, then
    # centre the slab in it, so the atoms sit comfortably inside the LAMMPS box.
    slab_height = (n_layers - 1) * INTERLAYER + THICKNESS
    stack.cell[2] = [0.0, 0.0, slab_height + 2.0 * VACUUM]
    stack.pbc = (True, True, True)
    stack.center(axis=2)

    # Wrap in-plane only: the shifted layers otherwise stick out of the cell
    # along a1/a2, which LAMMPS rejects when reading the data file.
    stack.wrap(pbc=(True, True, False))

    # Wrapping leaves ~1e-15 A negative coordinates on atoms that land exactly
    # on the cell face. Harmless physically, but they read as "outside the box";
    # snap them to zero.
    positions = np.round(stack.positions, 10)
    stack.positions = np.where(np.abs(positions) < 1e-10, 0.0, positions)
    return stack


def _assign_layer_types(atoms: Atoms, n_layers: int) -> Atoms:
    """Set the per-atom `type` array: Mo/S of layer i -> types 2i+1, 2i+2."""
    per_layer = len(atoms) // n_layers
    types = np.zeros(len(atoms), dtype=int)
    symbols = atoms.get_chemical_symbols()
    for idx in range(len(atoms)):
        layer = idx // per_layer
        types[idx] = 2 * layer + (1 if symbols[idx] == "Mo" else 2)
    atoms.arrays["type"] = types
    return atoms


def _add_virtual_atom(bilayer: Atoms) -> Atoms:
    """Append one massless drive atom above the slab, as its own atom type."""
    atoms = bilayer.copy()
    cell = np.array(atoms.cell)

    # Centre of the in-plane cell, VIRTUAL_HEIGHT above the topmost atom.
    xy = 0.5 * (cell[0] + cell[1])
    z = atoms.positions[:, 2].max() + VIRTUAL_HEIGHT
    position = (xy[0], xy[1], z)

    virtual_type = int(atoms.arrays["type"].max()) + 1
    types = np.append(atoms.arrays["type"], virtual_type)
    # `arrays` must stay consistent while appending, so drop and re-attach it.
    del atoms.arrays["type"]
    atoms.append(Atom(VIRTUAL_SYMBOL, position=position, mass=VIRTUAL_MASS))
    atoms.arrays["type"] = types

    if z >= cell[2][2]:
        raise RuntimeError(
            f"virtual atom at z={z:.3f} A is outside the box (c={cell[2][2]:.3f} A); "
            "increase VACUUM"
        )
    return atoms


def _write(atoms: Atoms, path: Path) -> None:
    write(str(path), atoms, format="lammps-data", atom_style="atomic", masses=True)
    n_types = int(atoms.arrays["type"].max())
    print(f"wrote {path}  ({len(atoms)} atoms, {n_types} atom types)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the Milestone-0 MoS2 LAMMPS data files.",
    )
    parser.add_argument(
        "--outdir",
        default=".",
        type=Path,
        help="directory to write the .data files into (default: current dir)",
    )
    args = parser.parse_args()
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    bilayer = _assign_layer_types(_stack(2), 2)
    trilayer = _assign_layer_types(_stack(3), 3)
    bilayer_virtual = _add_virtual_atom(bilayer)

    _write(bilayer, outdir / "bilayer_mos2.data")
    _write(trilayer, outdir / "trilayer_mos2.data")
    _write(bilayer_virtual, outdir / "bilayer_plus_virtual.data")


if __name__ == "__main__":
    main()
