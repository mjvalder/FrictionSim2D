"""Unit tests for frictionsim2d.core.utils module.

Tests core utility functions including float parsing and configuration file
parsing. The LAMMPS data-file and potentials-vocabulary classes that used to
live here moved to tests/builders/test_lammps_data.py and
tests/potentials/test_potentials_utils.py (C-4).
"""
import pytest
import tempfile
from pathlib import Path

from frictionsim2d.core.config import read_config
from frictionsim2d.core.utils.tokens import is_float


class TestIsFloat:
    """Test cases for is_float helper function."""
    
    def test_is_float_integer(self):
        """Test integer string parsing."""
        assert is_float("42") is True
        assert is_float("0") is True
        assert is_float("-100") is True
    
    def test_is_float_decimal(self):
        """Test decimal float parsing."""
        assert is_float("3.14") is True
        assert is_float("0.5") is True
        assert is_float("-2.7") is True
        assert is_float(".5") is True
    
    def test_is_float_scientific_notation(self):
        """Test scientific notation parsing."""
        assert is_float("1e-5") is True
        assert is_float("3.2e+4") is True
        assert is_float("-1.5e-3") is True
        assert is_float("1E5") is True
        assert is_float("2.0E-10") is True
    
    def test_is_float_invalid(self):
        """Test invalid float strings."""
        assert is_float("abc") is False
        assert is_float("1.2.3") is False
        assert is_float("") is False
        assert is_float("e5") is False


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


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_is_float_edge_cases(self):
        """Test edge case float strings."""
        assert is_float("0.0") is True
        assert is_float("-0") is True
        assert is_float("+5.5") is True
        assert is_float("1.23456789e-10") is True
    
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

