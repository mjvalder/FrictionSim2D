"""LAMMPS monolayer data-file I/O: parse, rebuild as ASE ``Atoms``, write a supercell.

ASE is imported inside the functions that need it, and this module is not
registered in ``interfaces/__init__.py``.
"""
import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from ase import Atoms


def restore_masses_section(
    path: Path,
    masses: Dict[int, Tuple[float, str]],
    n_types: Optional[int] = None,
) -> None:
    """Replace the Masses section of a LAMMPS data file with ``type mass  # element`` lines.

    ``ase.io.write`` omits the masses; this restores them so the file is usable
    by LAMMPS and by downstream element parsing. A type missing from ``masses``
    falls back to the standard mass of the placeholder element ``Z = type``.

    Args:
        path: LAMMPS data file (modified in-place).
        masses: Mapping type-id -> (mass, element).
        n_types: Number of types to write; defaults to ``max(masses)``.
    """
    from ase.data import chemical_symbols  # pylint: disable=import-outside-toplevel

    if n_types is None:
        n_types = max(masses, default=0)
    if n_types < 1:
        return

    kept: List[str] = []
    in_masses = False
    for line in path.read_text(encoding='utf-8').splitlines(keepends=True):
        stripped = line.strip()
        if stripped == 'Masses':
            in_masses = True
            continue
        if in_masses:
            # Drop the existing Masses block up to the next section header.
            if stripped and not stripped[0].isdigit():
                in_masses = False
            else:
                continue
        kept.append(line)

    from .ase import element_mass  # pylint: disable=import-outside-toplevel

    block = "Masses\n\n"
    for t in range(1, n_types + 1):
        if t in masses:
            mass, elem = masses[t]
        else:
            elem = chemical_symbols[t]
            mass = element_mass(elem)
        block += f"{t} {mass:.6f}  # {elem}\n"
    block += "\n"

    text = re.sub(r'(^|\n)(Atoms)', r'\1' + block + r'\2', ''.join(kept), count=1)
    path.write_text(text, encoding='utf-8')


def detect_atom_style(path: Union[str, Path]) -> str:
    """Detect the LAMMPS ``atom_style`` from a data file's ``Atoms`` header."""
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("Atoms"):
                if "#" in stripped:
                    style = stripped.split("#", 1)[1].strip()
                    return style or "atomic"
                return "atomic"
    return "atomic"


def read_monolayer(path: Union[str, Path]) -> Dict[str, object]:
    """Parse a LAMMPS monolayer data file, preserving all metadata.

    Reads the box (incl. tilt), per-atom types + Cartesian positions, per-atom
    charges (``atom_style charge`` only), the Masses section (type -> mass,
    element), and the detected atom_style. This lets the supercell tiling carry
    every LAMMPS type through as a *distinct* placeholder element (so sublattice
    types that share a chemical element are not collapsed by ASE), and restore
    the real masses / charges / atom_style afterwards -- i.e. no metadata is
    lost the way a bare ASE read/write would drop it.

    Returns:
        Dict with keys: ``atom_style`` (str), ``cell`` (3x3 float array, rows =
        lattice vectors, LAMMPS-canonical), ``types`` (List[int]), ``positions``
        ((N,3) float, box-origin-relative), ``charges`` (List[float] or None),
        ``masses`` ({type: (mass, element)}), ``n_types`` (int).
    """
    path = Path(path)
    atom_style = detect_atom_style(path)
    is_charge = "charge" in atom_style

    bounds: Dict[str, float] = {}
    tilt = {"xy": 0.0, "xz": 0.0, "yz": 0.0}
    masses: Dict[int, Tuple[float, str]] = {}
    types: List[int] = []
    positions: List[Tuple[float, float, float]] = []
    charges: List[float] = []

    section: Optional[str] = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.split("#", 1)[0].strip()
        if low.endswith(("xlo xhi", "ylo yhi", "zlo zhi")):
            parts = low.split()
            key = parts[2][0]
            bounds[f"{key}lo"] = float(parts[0])
            bounds[f"{key}hi"] = float(parts[1])
            continue
        if low.endswith("xy xz yz"):
            parts = low.split()
            tilt["xy"], tilt["xz"], tilt["yz"] = float(parts[0]), float(parts[1]), float(parts[2])
            continue
        if line == "Masses" or line == "Atoms" or line.startswith("Atoms"):
            section = "Masses" if line == "Masses" else "Atoms"
            continue
        if line in ("Velocities", "Bonds", "Angles", "Dihedrals", "Impropers"):
            section = None
            continue
        if section == "Masses":
            parts = line.split()
            if parts[0].lstrip("-").isdigit():
                elem = line.split("#")[-1].strip().split()[0] if "#" in line else "X"
                masses[int(parts[0])] = (float(parts[1]), elem)
        elif section == "Atoms":
            parts = low.split()
            if is_charge:
                if len(parts) >= 6 and parts[0].lstrip("-").isdigit():
                    types.append(int(parts[1]))
                    charges.append(float(parts[2]))
                    positions.append((float(parts[3]), float(parts[4]), float(parts[5])))
            else:
                if len(parts) >= 5 and parts[0].lstrip("-").isdigit():
                    types.append(int(parts[1]))
                    positions.append((float(parts[2]), float(parts[3]), float(parts[4])))

    lx = bounds["xhi"] - bounds["xlo"]
    ly = bounds["yhi"] - bounds["ylo"]
    lz = bounds["zhi"] - bounds["zlo"]
    cell = np.array([
        [lx, 0.0, 0.0],
        [tilt["xy"], ly, 0.0],
        [tilt["xz"], tilt["yz"], lz],
    ])
    origin = np.array([bounds["xlo"], bounds["ylo"], bounds["zlo"]])
    pos = np.array(positions, dtype=float) - origin
    n_types = max([*types, *masses.keys()], default=1)

    return {
        "atom_style": atom_style,
        "cell": cell,
        "types": types,
        "positions": pos,
        "charges": charges if is_charge else None,
        "masses": masses,
        "n_types": int(n_types),
    }


def monolayer_to_atoms(mono: Dict[str, object]) -> "Atoms":
    """Build an ASE ``Atoms`` from parsed monolayer data using placeholder elements.

    Each distinct LAMMPS type ``t`` is carried as chemical element number ``t``
    (H, He, Li, ...) so that ``make_supercell`` never collapses two LAMMPS types
    that happen to share a real element. Per-atom charges are attached when the
    source used ``atom_style charge``.
    """
    from ase import Atoms  # pylint: disable=import-outside-toplevel
    from ase.data import chemical_symbols  # pylint: disable=import-outside-toplevel

    types: List[int] = mono["types"]  # type: ignore[assignment]
    placeholders = [chemical_symbols[t] for t in types]
    atoms = Atoms(
        symbols=placeholders,
        positions=mono["positions"],
        cell=mono["cell"],
        pbc=True,
    )
    if mono["charges"] is not None:
        atoms.set_initial_charges(mono["charges"])  # type: ignore[arg-type]
    return atoms


def write_supercell(
    atoms: "Atoms",
    out_path: Path,
    atom_style: str,
    n_types: int,
    masses: Dict[int, Tuple[float, str]],
) -> np.ndarray:
    """Write an ASE supercell as a LAMMPS data file, preserving metadata.

    The type-ordered placeholder elements are written via ``specorder`` so the
    original integer LAMMPS type ids are reproduced exactly; ``atom_style`` and
    per-atom charges round-trip through ASE; the real Masses section is restored
    afterwards. ASE canonicalizes the box to LAMMPS form (a1 along +x) on write.

    Returns:
        The written in-plane 2x2 cell (LAMMPS-canonical), read back from disk.
    """
    from ase.data import chemical_symbols  # pylint: disable=import-outside-toplevel
    from ase.io import read, write  # pylint: disable=import-outside-toplevel

    if out_path.exists():
        out_path.unlink()
    specorder = [chemical_symbols[t] for t in range(1, n_types + 1)]
    write(str(out_path), atoms, format="lammps-data", atom_style=atom_style, specorder=specorder)
    restore_masses_section(out_path, masses, n_types)
    return np.array(read(str(out_path), format="lammps-data").cell)[:2, :2]
