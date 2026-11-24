"""Unit tests for frictionsim2d.interfaces.lammps_monolayer."""
import numpy as np
import pytest

from frictionsim2d.interfaces.lammps_monolayer import (
    detect_atom_style, monolayer_to_atoms, read_monolayer, restore_masses_section, write_supercell)


_DECK_WITH_MASSES = (
    "t\n\n2 atoms\n3 atom types\n\n0 5 xlo xhi\n\nMasses\n\n1 1.0\n2 2.0\n\n"
    "Atoms # atomic\n\n1 1 0 0 0\n2 2 1 1 1\n"
)


def test_restore_masses_section_replaces_existing_block(tmp_path):
    path = tmp_path / "d.lmp"
    path.write_text(_DECK_WITH_MASSES)
    restore_masses_section(path, {1: (95.95, "Mo"), 2: (32.06, "S"), 3: (1.0, "X")})
    text = path.read_text()
    assert text.count("Masses") == 1
    assert "1 95.950000  # Mo\n2 32.060000  # S\n3 1.000000  # X\n" in text
    assert text.index("Masses") < text.index("Atoms")


def test_restore_masses_section_falls_back_to_the_placeholder_element(tmp_path):
    path = tmp_path / "d.lmp"
    path.write_text(_DECK_WITH_MASSES)
    restore_masses_section(path, {1: (95.95, "Mo")}, n_types=3)
    lines = path.read_text().splitlines()
    assert "1 95.950000  # Mo" in lines
    assert any(line.endswith("# He") and line.startswith("2 4.002") for line in lines)
    assert any(line.endswith("# Li") and line.startswith("3 6.9") for line in lines)


def test_restore_masses_section_without_types_leaves_the_file(tmp_path):
    path = tmp_path / "d.lmp"
    path.write_text(_DECK_WITH_MASSES)
    restore_masses_section(path, {})
    assert path.read_text() == _DECK_WITH_MASSES


# --- detect_atom_style ---

@pytest.mark.parametrize('header, style', [
    ('Atoms # charge', 'charge'), ('Atoms #', 'atomic'), ('Atoms', 'atomic'),
])
def test_detect_atom_style_reads_the_atoms_header_comment(tmp_path, header, style):
    f = tmp_path / 'x.data'; f.write_text(f'LAMMPS data\n\n2 atoms\n\n{header}\n\n1 1 0 0 0\n')
    assert detect_atom_style(str(f)) == style


def test_detect_atom_style_defaults_to_atomic_without_an_atoms_section(tmp_path):
    f = tmp_path / 'x.data'; f.write_text('LAMMPS data\n\n0 atoms\n')
    assert detect_atom_style(str(f)) == 'atomic'


def test_read_monolayer_write_supercell_round_trip_keeps_types_masses_and_style(tmp_path):
    src = tmp_path / 'mono.lmp'
    src.write_text(
        "t\n\n2 atoms\n2 atom types\n\n0 4 xlo xhi\n0 5 ylo yhi\n0 10 zlo zhi\n\n"
        "Masses\n\n1 95.95  # Mo\n2 32.06  # S\n\n"
        "Atoms # atomic\n\n1 1 0 0 5\n2 2 2 2.5 5\n", encoding='utf-8')
    mono = read_monolayer(src)
    assert mono['atom_style'] == 'atomic' and mono['types'] == [1, 2]
    assert mono['masses'] == {1: (95.95, 'Mo'), 2: (32.06, 'S')}
    atoms = monolayer_to_atoms(mono)
    assert atoms.get_chemical_symbols() == ['H', 'He']
    out = tmp_path / 'out.lmp'
    cell = write_supercell(atoms, out, 'atomic', 2, mono['masses'])
    assert np.allclose(cell, [[4.0, 0.0], [0.0, 5.0]])
    again = read_monolayer(out)
    assert again['types'] == [1, 2] and again['masses'] == mono['masses']
