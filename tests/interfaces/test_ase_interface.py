"""src.interfaces.ase — the one adapter that must stay lazy (CLAUDE.md, import weight)."""
import sys

import pytest

from src.interfaces import ase as ase_iface


def test_read_structure_takes_the_first_image_of_a_trajectory(monkeypatch, materials_path):
    import ase.io
    real_read = ase.io.read
    monkeypatch.setattr(ase.io, 'read', lambda p, **k: [real_read(p), real_read(p)])
    data = ase_iface.read_structure(materials_path / 'h-MoS2.cif')
    assert len(data['cellpar']) == 6


def test_element_mass_known_symbol():
    assert ase_iface.element_mass('Mo') == pytest.approx(95.95, abs=0.1)


def test_element_mass_unknown_symbol_raises_key_error():
    with pytest.raises(KeyError):
        ase_iface.element_mass('Xx')


def test_importing_the_adapter_does_not_import_ase_io(monkeypatch):
    import importlib
    stale = [m for m in sys.modules if m == 'ase.io' or m.startswith('ase.io.')]
    for m in stale:
        monkeypatch.delitem(sys.modules, m, raising=False)
    try:
        importlib.reload(ase_iface)
        assert 'ase.io' not in sys.modules
    finally:
        importlib.import_module('ase.io')


def test_read_lammps_data_returns_cell_and_positions(tmp_path):
    """Minimal 2-atom, atomic-style, orthogonal 10x10x10 LAMMPS data file."""
    data_file = tmp_path / 'minimal.data'
    data_file.write_text(
        "LAMMPS data file\n"
        "\n"
        "2 atoms\n"
        "1 atom types\n"
        "\n"
        "0.0 10.0 xlo xhi\n"
        "0.0 10.0 ylo yhi\n"
        "0.0 10.0 zlo zhi\n"
        "\n"
        "Masses\n"
        "\n"
        "1 12.011\n"
        "\n"
        "Atoms # atomic\n"
        "\n"
        "1 1 1.0 1.0 1.0\n"
        "2 1 5.0 5.0 5.0\n"
    )

    data = ase_iface.read_lammps_data(data_file)

    assert data.cell.diagonal().tolist() == [10.0, 10.0, 10.0]
    assert len(data.positions) == 2
