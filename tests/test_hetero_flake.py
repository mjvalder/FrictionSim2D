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


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_register_hetero_flake_potentials_corner_groups(tmp_path):
    from src.core.config import load_settings
    from src.builders.hetero import register_hetero_flake_potentials
    settings = load_settings()
    hp = register_hetero_flake_potentials(_flake_sheets(settings), settings, tmp_path)
    # disjoint contiguous blocks
    ids1 = hp.pm.types.ids_by_component('flake_1')
    ids2 = hp.pm.types.ids_by_component('flake_2')
    assert set(ids1).isdisjoint(ids2)
    # per-material corner ids
    c1 = hp.pm.types.ids_by_component_region('flake_1', 'corner')
    c2 = hp.pm.types.ids_by_component_region('flake_2', 'corner')
    assert c1 and c2
    # combined flake_corner group = union
    assert set(hp.flake_corner_group.split()) >= {str(i) for i in c1 + c2}
    settings_text = hp.settings_path.read_text()
    assert "flake_1_corner" in settings_text and "flake_2_corner" in settings_text
    assert "flake_corner" in settings_text


@pytest.mark.skipif(LMP is None, reason="lmp_serial not on PATH")
def test_hetero_flake_run0_smoke(tmp_path):
    from src.core.config import load_settings, FlakeConfig
    from src.builders.components import build_hetero_flake_component
    from src.builders.hetero import register_hetero_flake_potentials
    settings = load_settings()
    path, _ = build_hetero_flake_component(_flake_sheets(settings), settings,
                                           FlakeConfig(shape="hexagon", edge_length=10.0, corner_radius=3.0), tmp_path)
    hp = register_hetero_flake_potentials(_flake_sheets(settings), settings, tmp_path)
    # Absolute read_data/include with cwd = run dir, mirroring
    # test_hetero_builder._lammps_run0_energy: the settings' pair_coeff lines
    # reference the staged potentials by the run-dir-relative prefix
    # 'provenance/potentials/...', which only resolves when cwd is the run dir.
    data_path = Path(path).resolve()
    settings_path = Path(hp.settings_path).resolve()
    script = tmp_path / "run0.lmp"
    script.write_text(
        "units metal\natom_style atomic\nboundary p p p\n"
        f"read_data {data_path}\n"
        f"include {settings_path}\n"
        "run 0\n")
    out = subprocess.run([LMP, "-in", str(script)], cwd=str(tmp_path),
                         capture_output=True, text=True)
    combined = out.stdout + out.stderr
    assert out.returncode == 0, combined         # all pair coeffs set (no "All pair coeffs are not set")
    assert "Lost atoms" not in combined
