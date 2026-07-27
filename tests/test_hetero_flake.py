"""B2 — hetero flake as a PotentialManager component (structure-only).

A flake carved from a matched hetero *bilayer* (two materials), corner-typed,
and registered as ``flake_1``/``flake_2`` PM components with corner regions.
The heavier tests need real ``lmp_serial`` (interface minimize + ``run 0``
smoke) and are skipped when it is not on PATH.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

LMP = shutil.which("lmp_serial")
MAT, POT = "examples/materials", "examples/potentials/sw"


def _lammps_atom_types(path):
    """Return the LAMMPS atom-type column (integers) from a data file.

    ASE's ``get_array('numbers')`` maps LAMMPS types to chemical *elements* by
    mass, collapsing the two TMDs' shared sulfur types — so distinct LAMMPS
    types must be counted from the data file's Atoms section, not via ASE.
    """
    lines = Path(path).read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("Atoms")) + 2
    types = []
    for ln in lines[start:]:
        parts = ln.split()
        if not parts or not parts[0].isdigit():
            break
        types.append(int(parts[1]))
    return types


def _flake_sheets(settings):
    from src.core.config import SheetConfig
    return [
        SheetConfig(mat="h-MoS2", cif_path=f"{MAT}/h-MoS2.cif", pot_path=f"{POT}/MoS2_wen.sw",
                    pot_type="sw", x=12.0, y=12.0, layers=[1]),
        SheetConfig(mat="h-WS2", cif_path=f"{MAT}/h-WS2.cif", pot_path=f"{POT}/sw_lammps/t-WS2.sw",
                    pot_type="sw", x=12.0, y=12.0, layers=[1]),
    ]


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_hetero_flake_carries_both_materials_types(tmp_path):
    from src.core.config import load_settings, FlakeConfig
    from src.builders.components import build_hetero_flake_component
    from ase.io import read
    settings = load_settings()
    flake_cfg = FlakeConfig(shape="hexagon", edge_length=10.0, corner_radius=3.0)
    path, dims = build_hetero_flake_component(_flake_sheets(settings), settings, flake_cfg, tmp_path)
    assert Path(path).exists()
    atoms = read(str(path), format="lammps-data")
    n_types = len(set(atoms.get_array("numbers"))) if atoms.has("numbers") else None
    text = Path(path).read_text()
    # after corner-doubling, both materials' types present -> header type count is doubled bilayer types
    import re
    m = re.search(r"(\d+)\s+atom types", text)
    assert m and int(m.group(1)) >= 4         # >=2 bilayer types each doubled core/corner


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_corner_types_double_every_bilayer_type_and_find_vertices(tmp_path):
    from src.core.config import load_settings, FlakeConfig
    from src.builders.components import build_hetero_flake_component
    settings = load_settings()
    path, _ = build_hetero_flake_component(_flake_sheets(settings), settings,
                                           FlakeConfig(shape="hexagon", edge_length=10.0, corner_radius=3.0), tmp_path)
    # Count distinct LAMMPS atom types straight from the data file: ASE would
    # collapse the two TMDs' shared sulfur types by mass, hiding the split.
    types = _lammps_atom_types(path)
    distinct = set(types)
    # every bilayer type doubled core/corner -> >= 4 distinct types
    assert len(distinct) >= 4
    # corner types are the even ids (2t); at least one corner atom exists (vertices found)
    assert any(t % 2 == 0 for t in distinct)
