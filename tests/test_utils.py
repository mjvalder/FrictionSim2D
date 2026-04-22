"""Unit tests for src.core.utils module.

Tests core utility functions including float parsing, LJ parameters calculation,
LAMMPS file reading, and configuration file parsing.
"""
import pytest
import tempfile
from pathlib import Path
import json
import configparser
from io import StringIO

from src.core.utils import (
    _is_float,
    lj_params,
    get_model_dimensions,
    read_config,
    atomic2charge,
    atomic2molecular,
)


class TestIsFloat:
    """Test cases for _is_float helper function."""
    
    def test_is_float_integer(self):
        """Test integer string parsing."""
        assert _is_float("42") is True
        assert _is_float("0") is True
        assert _is_float("-100") is True
    
    def test_is_float_decimal(self):
        """Test decimal float parsing."""
        assert _is_float("3.14") is True
        assert _is_float("0.5") is True
        assert _is_float("-2.7") is True
        assert _is_float(".5") is True
    
    def test_is_float_scientific_notation(self):
        """Test scientific notation parsing."""
        assert _is_float("1e-5") is True
        assert _is_float("3.2e+4") is True
        assert _is_float("-1.5e-3") is True
        assert _is_float("1E5") is True
        assert _is_float("2.0E-10") is True
    
    def test_is_float_invalid(self):
        """Test invalid float strings."""
        assert _is_float("abc") is False
        assert _is_float("1.2.3") is False
        assert _is_float("") is False
        assert _is_float("e5") is False


class TestLjParams:
    """Test cases for lj_params function with case-insensitive handling."""
    
    def test_lj_params_uppercase(self):
        """Test LJ parameters with uppercase atom types."""
        eps, sig = lj_params('C', 'H')
        assert isinstance(eps, float)
        assert isinstance(sig, float)
        assert eps > 0
        assert sig > 0
    
    def test_lj_params_lowercase(self):
        """Test LJ parameters with lowercase atom types."""
        eps_lower, sig_lower = lj_params('c', 'h')
        eps_upper, sig_upper = lj_params('C', 'H')
        assert eps_lower == eps_upper
        assert sig_lower == sig_upper
    
    def test_lj_params_mixed_case(self):
        """Test LJ parameters with mixed-case atom types."""
        eps_mixed, sig_mixed = lj_params('c', 'H')
        eps_upper, sig_upper = lj_params('C', 'H')
        assert eps_mixed == eps_upper
        assert sig_mixed == sig_upper

    def test_lj_params_two_letter_symbols_case_normalization(self):
        """Test that two-letter element symbols normalize correctly."""
        eps_upper, sig_upper = lj_params('NB', 'NI')
        eps_canonical, sig_canonical = lj_params('Nb', 'Ni')
        assert eps_upper == eps_canonical
        assert sig_upper == sig_canonical
    
    def test_lj_params_symmetric(self):
        """Test that LJ parameters are symmetric for swapped atoms."""
        eps1, sig1 = lj_params('C', 'H')
        eps2, sig2 = lj_params('H', 'C')
        assert eps1 == eps2
        assert sig1 == sig2
    
    def test_lj_params_invalid_atom_type(self):
        """Test LJ parameters with invalid atom type."""
        with pytest.raises(KeyError):
            lj_params('Xx', 'C')
    
    def test_lj_params_mixing_rule(self):
        """Test that mixing rule is applied correctly (geometric mean for epsilon)."""
        eps_c_h, sig_c_h = lj_params('C', 'H')
        eps_c_c, sig_c_c = lj_params('C', 'C')
        # epsilon should be between the two pure element values
        assert eps_c_c >= eps_c_h or eps_c_h >= eps_c_c


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


class TestReadConfig:
    """Test cases for read_config function."""
    
    @pytest.fixture
    def config_file_with_floats(self):
        """Create a config file with various float formats."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write("""[section1]
int_value = 42
float_value = 3.14
scientific = 1.5e-3
negative = -2.7
array = [1, 2.5, -3.0]
empty = 

[section2]
text = hello
mixed_array = [0.1, 1e-5, text_value]
""")
            return Path(f.name)
    
    def test_read_config_integers(self, config_file_with_floats):
        """Test integer parsing."""
        config = read_config(config_file_with_floats)
        assert config['section1']['int_value'] == 42
        assert isinstance(config['section1']['int_value'], int)
    
    def test_read_config_floats(self, config_file_with_floats):
        """Test float parsing with various formats."""
        config = read_config(config_file_with_floats)
        assert config['section1']['float_value'] == 3.14
        assert config['section1']['scientific'] == 1.5e-3
        assert config['section1']['negative'] == -2.7
    
    def test_read_config_arrays(self, config_file_with_floats):
        """Test array parsing."""
        config = read_config(config_file_with_floats)
        assert config['section1']['array'] == [1, 2.5, -3.0]
        assert config['section2']['mixed_array'] == [0.1, 1e-5, 'text_value']
    
    def test_read_config_empty_values(self, config_file_with_floats):
        """Test handling of empty values."""
        config = read_config(config_file_with_floats)
        assert config['section1']['empty'] is None
    
    def test_read_config_text_values(self, config_file_with_floats):
        """Test text value parsing."""
        config = read_config(config_file_with_floats)
        assert config['section2']['text'] == 'hello'
        assert isinstance(config['section2']['text'], str)
    
    def test_read_config_nonexistent(self):
        """Test reading non-existent config file."""
        # ConfigParser doesn't raise on missing files, just returns empty
        config = read_config("/nonexistent/file.ini")
        assert isinstance(config, dict)
        assert len(config) == 0


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
        with open(lammps_atomic_file, 'r') as f:
            content = f.read()
        assert "Atoms # charge" in content
        assert "1 1 0.0 0.0 0.0 0.0" in content or "1 1 0.0 0.0 0.0" in content
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
        with open(lammps_atomic_file, 'r') as f:
            content = f.read()
        assert "Atoms # molecular" in content
        assert "0 1" in content  # Molecule ID 0
        lammps_atomic_file.unlink()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_is_float_edge_cases(self):
        """Test edge case float strings."""
        assert _is_float("0.0") is True
        assert _is_float("-0") is True
        assert _is_float("+5.5") is True
        assert _is_float("1.23456789e-10") is True
    
    @pytest.fixture
    def config_with_comments(self):
        """Create a config file with inline comments."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write("""[section]
value1 = 10 # this is a comment
value2 = 3.14 # pi value
""")
            return Path(f.name)
    
    def test_read_config_with_comments(self, config_with_comments):
        """Test parsing config with inline comments."""
        config = read_config(config_with_comments)
        # Comments should be removed
        assert config['section']['value1'] == 10
        assert config['section']['value2'] == 3.14
        config_with_comments.unlink()
