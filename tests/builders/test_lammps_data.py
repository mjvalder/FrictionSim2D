"""Unit tests for src.builders.utils.lammps_data.

Moved verbatim from tests/core/test_utils.py (C-4): these classes exercise
LAMMPS data-file surgery, not core.
"""
import pytest
import tempfile
from pathlib import Path

from src.builders.utils.lammps_data import atomic2charge, atomic2molecular, charge2atom
from src.builders.utils.lammps_data import get_model_dimensions


class TestGetModelDimensions:
    """Test cases for get_model_dimensions function."""

    @pytest.fixture
    def lammps_file_valid(self):
        """Create a valid LAMMPS data file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as f:
            f.write("""LAMMPS data file
 10 atoms
 4 atom types
 0.0 10.0 xlo xhi
 0.0 10.0 ylo yhi
 0.0 10.0 zlo zhi

Atoms

1 1 1.0 2.0 3.0
2 2 2.0 3.0 4.0
""")
            return Path(f.name)

    @pytest.fixture
    def lammps_file_missing_dims(self):
        """Create a LAMMPS file with missing dimension markers."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as f:
            f.write("""LAMMPS data file
 10 atoms
 0.0 10.0 xlo xhi

Atoms

1 1 1.0 2.0 3.0
""")
            return Path(f.name)

    def test_get_model_dimensions_valid(self, lammps_file_valid):
        """Test reading valid LAMMPS dimensions."""
        dims = get_model_dimensions(lammps_file_valid)
        assert dims['xlo'] == 0.0
        assert dims['xhi'] == 10.0
        assert dims['ylo'] == 0.0
        assert dims['yhi'] == 10.0
        assert dims['zlo'] == 0.0
        assert dims['zhi'] == 10.0
        lammps_file_valid.unlink()

    def test_get_model_dimensions_missing_file(self):
        """Test with non-existent file."""
        with pytest.raises(FileNotFoundError):
            get_model_dimensions("/nonexistent/path/file.lmp")

    def test_get_model_dimensions_missing_markers(self, lammps_file_missing_dims, caplog):
        """Test with missing dimension markers."""
        dims = get_model_dimensions(lammps_file_missing_dims)
        assert dims['xlo'] == 0.0
        assert dims['xhi'] == 10.0
        assert dims['ylo'] is None  # Missing ylo yhi line
        assert dims['yhi'] is None
        assert dims['zlo'] is None  # Missing zlo zhi line
        assert dims['zhi'] is None
        # Should have warning about missing dimensions
        assert "missing dimension markers" in caplog.text.lower()
        lammps_file_missing_dims.unlink()


class TestAtomicToCharge:
    """Test cases for atomic2charge conversion."""

    @pytest.fixture
    def lammps_atomic_file(self):
        """Create a LAMMPS file in atomic format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as f:
            f.write("""LAMMPS data file
 2 atoms

Atoms # atomic

1 1 0.0 0.0 0.0
2 2 1.0 1.0 1.0

Velocities

1 0.0 0.0 0.0
""")
            return Path(f.name)

    def test_atomic_to_charge_conversion(self, lammps_atomic_file):
        """Test conversion from atomic to charge format."""
        atomic2charge(lammps_atomic_file)
        with open(lammps_atomic_file, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "Atoms # charge" in content
        assert "1 1 0.0 0.0 0.0 0.0" in content or "1 1 0.0 0.0 0.0" in content
        assert "Velocities" in content
        lammps_atomic_file.unlink()


class TestAtomicToMolecular:
    """Test cases for atomic2molecular conversion."""

    @pytest.fixture
    def lammps_atomic_file(self):
        """Create a LAMMPS file in atomic format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as f:
            f.write("""LAMMPS data file
 2 atoms

Atoms # atomic

1 1 0.0 0.0 0.0
2 2 1.0 1.0 1.0

Velocities

1 0.0 0.0 0.0
""")
            return Path(f.name)

    def test_atomic_to_molecular_conversion(self, lammps_atomic_file):
        """Test conversion from atomic to molecular format."""
        atomic2molecular(lammps_atomic_file)
        with open(lammps_atomic_file, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "Atoms # molecular" in content
        assert "0 1" in content  # Molecule ID 0
        assert "Velocities" in content


class TestChargeToAtomic:
    """Test cases for charge2atom conversion."""

    @pytest.fixture
    def lammps_charge_file(self):
        """Create a LAMMPS file in charge format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as f:
            f.write("""LAMMPS data file
 2 atoms

Atoms # charge

1 1 -0.1 0.0 0.0 0.0
2 2 0.2 1.0 1.0 1.0

Velocities

1 0.0 0.0 0.0
""")
            return Path(f.name)

    def test_charge_to_atomic_conversion(self, lammps_charge_file):
        """Test conversion from charge to atomic format."""
        charge2atom(lammps_charge_file)
        with open(lammps_charge_file, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "Atoms # atomic" in content
        assert "1 1 0.0 0.0 0.0" in content
        assert "2 2 1.0 1.0 1.0" in content
        assert "Velocities" in content

        lammps_charge_file.unlink()

    def test_atomic_to_molecular_preserves_image_flags(self):
        """Image flags should be preserved when converting style."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as f:
            f.write("""LAMMPS data file
 1 atoms

Atoms # atomic

1 1 0.0 0.0 0.0 1 -1 2

Velocities

1 0.0 0.0 0.0
""")
            path = Path(f.name)

        atomic2molecular(path)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "Atoms # molecular" in content
        assert "1 0 1 0.0 0.0 0.0 1 -1 2" in content
        path.unlink()
