"""Unit tests for the small pure helpers and guards of builders/construction/hetero_stack.py.

None of these run LAMMPS: the guards fire before the first LAMMPS call, and the
helpers only read and write data files.
"""
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from frictionsim2d.builders.construction.hetero_stack import (
    _strain_onto_and_write,
    assemble_hetero_stack,
    box_region_cmd,
    detect_inplane_repeat,
)
from frictionsim2d.interfaces.lammps_monolayer import monolayer_to_atoms, read_monolayer


def _write_data(path: Path, atoms: Atoms, style: str = "atomic") -> Path:
    write(str(path), atoms, format="lammps-data", atom_style=style)
    return path


def _tiled_sheet(nx: int, ny: int) -> Atoms:
    """An orthogonal two-species sheet tiled ``nx`` x ``ny`` from a 3 x 5 A cell."""
    cell = np.diag([3.0 * nx, 5.0 * ny, 10.0])
    positions, symbols = [], []
    for i in range(nx):
        for j in range(ny):
            positions += [(3.0 * i + 0.5, 5.0 * j + 1.0, 5.0),
                          (3.0 * i + 2.0, 5.0 * j + 3.5, 5.0)]
            symbols += ["Mo", "S"]
    return Atoms(symbols, positions=np.array(positions), cell=cell, pbc=True)


def test_detect_inplane_repeat_finds_the_tiling(tmp_path):
    path = _write_data(tmp_path / "tiled.data", _tiled_sheet(4, 2))
    (px, py), (nx, ny) = detect_inplane_repeat(path)
    assert (nx, ny) == (4, 2)
    assert px == pytest.approx(3.0) and py == pytest.approx(5.0)


def test_detect_inplane_repeat_falls_back_to_the_box_without_a_repeat(tmp_path):
    path = _write_data(tmp_path / "irregular.data", _tiled_sheet(1, 1)
                       + Atoms("S", positions=[(1.1, 2.3, 5.0)], cell=[3.0, 5.0, 10.0]))
    (px, py), (nx, ny) = detect_inplane_repeat(path)
    assert (nx, ny) == (1, 1)
    assert (px, py) == pytest.approx((3.0, 5.0))


def test_detect_inplane_repeat_rejects_a_flat_box(tmp_path):
    path = tmp_path / "flat.data"
    path.write_text(
        "LAMMPS data\n\n1 atoms\n1 atom types\n\n"
        "0.0 0.0 xlo xhi\n0.0 5.0 ylo yhi\n0.0 10.0 zlo zhi\n\n"
        "Masses\n\n1 95.95\n\nAtoms # atomic\n\n1 1 0.0 0.0 5.0\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="Non-positive in-plane box"):
        detect_inplane_repeat(path)


def test_box_region_cmd_block_and_prism():
    assert box_region_cmd(10.0, 8.0, 0.0, -1.0, 5.0) == (
        "region box block 0.0 10.000000 0.0 8.000000 -1.000000 5.000000")
    assert box_region_cmd(10.0, 8.0, 2.0, 0.0, 5.0) == (
        "region box prism 0.0 10.000000 0.0 8.000000 0.000000 5.000000 2.000000 0.0 0.0")


def test_strain_onto_and_write_keeps_fractional_coordinates_and_metadata(tmp_path):
    src = _write_data(tmp_path / "mono.lmp", _tiled_sheet(1, 1))
    mono = read_monolayer(src)
    mono["masses"] = {1: (95.95, "Mo"), 2: (32.06, "S")}
    mono["n_types"] = 2
    atoms = monolayer_to_atoms(mono)
    frac = atoms.get_scaled_positions()
    out = tmp_path / "out.lmp"
    cell = _strain_onto_and_write(atoms, np.array([[3.3, 0.0], [0.0, 5.5]]), out, mono)
    assert np.allclose(cell, [[3.3, 0.0], [0.0, 5.5]])
    back = read(str(out), format="lammps-data")
    assert back.cell[2, 2] == pytest.approx(10.0)
    assert np.allclose(back.get_scaled_positions(), frac)
    written = read_monolayer(out)
    assert written["atom_style"] == mono["atom_style"]
    assert written["masses"] == {1: (95.95, "Mo"), 2: (32.06, "S")}


def _stack_inputs(tmp_path, *, cell_b=None, style_b="atomic", layers=(1, 1), n_sheets=2):
    a = _write_data(tmp_path / "a.lmp", _tiled_sheet(1, 1))
    atoms_b = _tiled_sheet(1, 1)
    if cell_b is not None:
        atoms_b.set_cell([cell_b[0], cell_b[1], 10.0], scale_atoms=True)
    b = _write_data(tmp_path / "b.lmp", atoms_b, style=style_b)
    sheets = [
        SimpleNamespace(mat="m", layers=[n], stack_type="AA", pot_type="sw", lat_c=None)
        for n in layers[:n_sheets]
    ]
    matched = SimpleNamespace(supercell_a=a, supercell_b=b)
    spine = SimpleNamespace(type_offsets=[0, 2], settings_path=tmp_path / "s.in",
                            pm=SimpleNamespace(types=[1, 2, 3, 4],
                                               get_masses_string=lambda: ""))
    return matched, sheets, spine


def _assemble(tmp_path, stacking="grouped", **kwargs):
    matched, sheets, spine = _stack_inputs(tmp_path, **kwargs)
    return assemble_hetero_stack(matched, sheets, spine, stacking,
                                 SimpleNamespace(geometry=SimpleNamespace(lat_c_default=6.0)),
                                 tmp_path / "work")


@pytest.mark.parametrize("kwargs, stacking, message", [
    ({}, "zigzag", "hetero_stacking must be"),
    ({"layers": (1, 1, 1), "n_sheets": 3}, "grouped", "exactly two"),
    ({"style_b": "charge"}, "grouped", "differing atom_style"),
    ({"cell_b": (3.4, 5.0)}, "grouped", "do not share one in-plane box"),
])
def test_assemble_hetero_stack_guards(tmp_path, kwargs, stacking, message):
    with pytest.raises(ValueError, match=message):
        _assemble(tmp_path, stacking, **kwargs)


def test_assemble_hetero_stack_rejects_a_sheet_with_no_layers(tmp_path):
    matched, sheets, spine = _stack_inputs(tmp_path)
    sheets[1].layers = []
    with pytest.raises(ValueError, match="empty layers"):
        assemble_hetero_stack(matched, sheets, spine, "grouped", SimpleNamespace(),
                              tmp_path / "work")


def test_assemble_hetero_stack_mlip_interface_needs_a_lat_c(tmp_path):
    matched, sheets, spine = _stack_inputs(tmp_path)
    for sheet in sheets:
        sheet.pot_type = "deepmd"
    settings = SimpleNamespace(geometry=SimpleNamespace(lat_c_default=6.0))
    with pytest.raises(ValueError, match=re.escape("needs an explicit [2D] lat_c")):
        assemble_hetero_stack(matched, sheets, spine, "grouped", settings, tmp_path / "work")


def test_alternating_stacking_interleaves_the_materials(lammps_replay, tmp_path):
    from tests.helpers.deck_snapshot import EXTRA_CASES, render
    case = next(c for c in EXTRA_CASES if c.name == "sheet_hetero_alternating")
    files = render(case, tmp_path)
    script = next(t for rel, t in files.items() if rel.endswith("stack_hetero.lmp"))
    groups = re.findall(r"group (_m\d+l\d+) ", script)
    assert groups == ["_m0l0", "_m1l0", "_m0l1", "_m1l1"]
    z = [float(v) for v in re.findall(r"\(z = (-?[\d.]+)\)", script)]
    assert z == sorted(z) and len(set(z)) == 4


def test_mlip_interface_spacing_is_the_larger_lat_c(tmp_path, monkeypatch):
    from frictionsim2d.builders.construction import hetero_stack
    matched, sheets, spine = _stack_inputs(tmp_path)
    for sheet, lat_c in zip(sheets, (6.1, 6.6)):
        sheet.pot_type = "deepmd"
        sheet.lat_c = lat_c
    spine.pm.has_mlip = lambda: True
    spine.pm.write_surrogate_file = lambda path: path
    monkeypatch.setattr(
        hetero_stack, "run_lammps",
        lambda workdir, script: (Path(workdir) / "hetero.data").write_text("x"))
    settings = SimpleNamespace(geometry=SimpleNamespace(lat_c_default=6.0))
    stack = assemble_hetero_stack(matched, sheets, spine, "grouped", settings, tmp_path / "work")
    assert stack.interface_spacing == 6.6


def test_mlip_assembly_writes_nocoeff(tmp_path, monkeypatch):
    """``pair zero`` serialises ``Pair Coeffs`` the run decks cannot read back."""
    from frictionsim2d.builders.construction import hetero_stack
    matched, sheets, spine = _stack_inputs(tmp_path)
    for sheet in sheets:
        sheet.pot_type = "deepmd"
        sheet.lat_c = 6.1
    spine.pm.has_mlip = lambda: True
    spine.pm.write_surrogate_file = lambda path: path
    monkeypatch.setattr(
        hetero_stack, "run_lammps",
        lambda workdir, script: (Path(workdir) / "hetero.data").write_text("x"))
    settings = SimpleNamespace(geometry=SimpleNamespace(lat_c_default=6.0))
    assemble_hetero_stack(matched, sheets, spine, "grouped", settings, tmp_path / "work")
    script = (tmp_path / "work" / "stack_hetero.lmp").read_text(encoding="utf-8")
    write = next(l for l in script.splitlines() if l.startswith("write_data"))
    assert write.endswith(" nocoeff")


def test_classical_assembly_template_keeps_the_plain_write_data():
    from frictionsim2d.interfaces.jinja import render_template
    ctx = {"atom_style": "atomic", "box_region": "r", "total_types": 2, "layers": [],
           "masses_block": "", "settings_path": "s", "output_path": "hetero.data"}
    write = lambda c: next(l for l in render_template("hetero/stack_hetero.lmp", c).splitlines()
                           if l.startswith("write_data"))
    assert "nocoeff" not in write({**ctx, "nocoeff": False})
    assert write({**ctx, "nocoeff": True}).endswith(" nocoeff")
