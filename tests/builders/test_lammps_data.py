"""Unit tests for frictionsim2d.builders.utils.lammps_data.

Moved verbatim from tests/core/test_utils.py (C-4): these classes exercise
LAMMPS data-file surgery, not core.
"""
import pytest
import tempfile
from pathlib import Path

from frictionsim2d.builders.utils.lammps_data import atomic2charge, atomic2molecular, charge2atom
from frictionsim2d.builders.utils.lammps_data import get_model_dimensions
from frictionsim2d.builders.utils.lammps_data import (
    _is_lammps_section_header, atom_count_and_z_extent, get_num_atom_types, shift_atoms_to_z_zero)


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


class TestIsLammpsSectionHeader:
    """Direct branch coverage for the section-header/data-line classifier.

    Exercised only indirectly by the conversion tests above (whose fixtures
    never hit the empty-line or comment-line branches), so covered here
    directly instead.
    """

    def test_empty_string_is_not_a_header(self):
        assert _is_lammps_section_header('') is False

    def test_comment_line_is_not_a_header(self):
        assert _is_lammps_section_header('# a comment') is False

    def test_digit_leading_line_is_not_a_header(self):
        assert _is_lammps_section_header('1 1 0.0 0.0 0.0') is False

    def test_signed_or_decimal_leading_line_is_not_a_header(self):
        assert _is_lammps_section_header('-1.5 2 3') is False
        assert _is_lammps_section_header('+1.5 2 3') is False
        assert _is_lammps_section_header('.5 2 3') is False

    def test_letter_leading_line_is_a_header(self):
        assert _is_lammps_section_header('Velocities') is True


class TestMalformedAtomLinesArePreserved:
    """A too-short Atoms line can't be converted, so it is left untouched.

    _to_charge/_to_atomic/_to_molecular all return None for a line with too
    few fields; _convert_atoms_style then writes the raw line back verbatim
    instead of the transformed one. None of the existing fixtures exercise
    this -- every atom line in them has enough fields to convert.
    """

    def test_atomic2charge_preserves_a_short_line(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as f:
            f.write("""LAMMPS data file
 2 atoms

Atoms # atomic

1 1 0.0
2 2 1.0 1.0 1.0
""")
            path = Path(f.name)
        atomic2charge(path)
        content = path.read_text(encoding='utf-8')
        assert "1 1 0.0\n" in content  # too short to convert: unchanged
        assert "2 2 0.0 1.0 1.0 1.0" in content  # well-formed: converted
        path.unlink()

    def test_charge2atom_preserves_a_short_line(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as f:
            f.write("""LAMMPS data file
 2 atoms

Atoms # charge

1 1 -0.1
2 2 0.2 1.0 1.0 1.0
""")
            path = Path(f.name)
        charge2atom(path)
        content = path.read_text(encoding='utf-8')
        assert "1 1 -0.1\n" in content
        assert "2 2 1.0 1.0 1.0" in content
        path.unlink()

    def test_atomic2molecular_preserves_a_short_line(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as f:
            f.write("""LAMMPS data file
 2 atoms

Atoms # atomic

1 1 0.0
2 2 1.0 1.0 1.0
""")
            path = Path(f.name)
        atomic2molecular(path)
        content = path.read_text(encoding='utf-8')
        assert "1 1 0.0\n" in content
        assert "2 0 2 1.0 1.0 1.0" in content
        path.unlink()


class TestShiftAtomsToZZero:
    """Cover the two early-exit branches and both unparsable-z guards."""

    def test_no_atom_lines_is_a_no_op(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as f:
            f.write("""LAMMPS data file
 0 atoms

Atoms

""")
            path = Path(f.name)
        before = path.read_text(encoding='utf-8')
        shift_atoms_to_z_zero(path)
        after = path.read_text(encoding='utf-8')
        assert after == before  # z_values empty -> early return
        path.unlink()

    def test_already_at_zero_is_a_no_op(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as f:
            f.write("""LAMMPS data file
 2 atoms

Atoms

1 1 0.0 0.0 0.0
2 2 1.0 1.0 0.0
""")
            path = Path(f.name)
        before = path.read_text(encoding='utf-8')
        shift_atoms_to_z_zero(path)
        after = path.read_text(encoding='utf-8')
        assert after == before  # z_min already 0 -> early return
        path.unlink()

    def test_unparsable_z_is_skipped_in_both_passes(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as f:
            f.write("""LAMMPS data file
 2 atoms

      0.0      10.0  zlo zhi

Atoms

1 1 0.0 0.0 abc
2 2 1.0 1.0 5.0
""")
            path = Path(f.name)
        shift_atoms_to_z_zero(path)
        content = path.read_text(encoding='utf-8')
        assert "1 1 0.0 0.0 abc" in content  # unparsable z: left untouched
        assert "2  2  1.0  1.0  0.000000000000000" in content  # shifted to 0
        path.unlink()


class TestGetModelDimensionsMalformedLine:

    def test_unparsable_dimension_line_logs_a_warning_and_leaves_it_none(self, caplog):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as f:
            f.write("""LAMMPS data file
 1 atoms
 abc def xlo xhi
 0.0 10.0 ylo yhi
 0.0 10.0 zlo zhi

Atoms

1 1 0.0 0.0 0.0
""")
            path = Path(f.name)
        dims = get_model_dimensions(path)
        assert dims['xlo'] is None
        assert dims['ylo'] == 0.0
        assert "Failed to parse dimension line" in caplog.text
        path.unlink()


class TestGetNumAtomTypes:

    def test_defaults_to_one_when_no_header_present(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lmp', delete=False) as f:
            f.write("""LAMMPS data file
 1 atoms

Atoms

1 1 0.0 0.0 0.0
""")
            path = Path(f.name)
        assert get_num_atom_types(path) == 1
        path.unlink()


class TestAtomCountAndZExtent:
    HEADER = "t\n\n3 atoms\n1 atom types\n\n0 5 xlo xhi\n0 5 ylo yhi\n0 9 zlo zhi\n\nMasses\n\n1 1.0\n\n"

    def test_atom_count_and_z_extent(self, tmp_path):
        f = tmp_path / "a.data"
        f.write_text(self.HEADER + "Atoms # atomic\n\n1 1 0 0 0.0\n2 1 1 1 1.5\n3 1 2 2 3.1\n")
        n, dz = atom_count_and_z_extent(f)
        assert n == 3
        assert dz == pytest.approx(3.1)

    def test_charge_style(self, tmp_path):
        f = tmp_path / "c.data"
        f.write_text(self.HEADER + "Atoms # charge\n\n1 1 0.0 0 0 0.0\n2 1 0.0 1 1 1.5\n3 1 0.0 2 2 3.1\n")
        n, dz = atom_count_and_z_extent(f)
        assert n == 3
        assert dz == pytest.approx(3.1)

    def test_image_flags_are_ignored(self, tmp_path):
        f = tmp_path / "i.data"
        f.write_text(self.HEADER + "Atoms # atomic\n\n1 1 0 0 0.0 0 0 0\n2 1 1 1 1.5 0 0 1\n3 1 2 2 3.1 1 0 0\n")
        assert atom_count_and_z_extent(f)[0] == 3
        assert atom_count_and_z_extent(f)[1] == pytest.approx(3.1)


def test_atoms_section_start_reports_first_atom_line_and_style():
    from frictionsim2d.builders.utils.lammps_data import atoms_section_start
    lines = ["t", "", "Masses", "", "1 1.0", "", "Atoms # charge", "", "1 1 0.0 0 0 0"]
    assert atoms_section_start(lines) == (8, "charge")
    assert atoms_section_start(["Atoms", "", "1 1 0 0 0"]) == (2, "atomic")
    assert atoms_section_start(["Masses", "", "1 1.0"]) is None


def test_snap_small_tilts_drops_only_negligible_tilts(tmp_path):
    from frictionsim2d.builders.utils.lammps_data import snap_small_tilts
    small, large = tmp_path / "s.lmp", tmp_path / "l.lmp"
    small.write_text("0 5 xlo xhi\n0.004 0.0 -0.002 xy xz yz\n")
    large.write_text("0 5 xlo xhi\n1.2 0.0 0.0 xy xz yz\n")
    snap_small_tilts(small)
    snap_small_tilts(large)
    assert "xy xz yz" not in small.read_text()
    assert "xy xz yz" in large.read_text()
