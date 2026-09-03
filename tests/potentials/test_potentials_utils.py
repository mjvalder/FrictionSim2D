"""Unit tests for src.potentials.utils.

Moved verbatim from tests/core/test_utils.py (C-4): these classes exercise
potential-type vocabulary and resolution, not core.
"""
import pytest
from pathlib import Path

from src.potentials.utils import lj_params, get_potential_element_order


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
        eps_c_h, _ = lj_params('C', 'H')
        eps_c_c, _ = lj_params('C', 'C')
        # epsilon should be between the two pure element values
        assert eps_c_c >= eps_c_h or eps_c_h >= eps_c_c


class TestPotentialElementOrder:
    """Test cases for deriving potential element order from a CIF."""

    def test_get_potential_element_order_single_type(self, tmp_path: Path) -> None:
        """Single-type potentials should return one type per element."""
        pot_file = tmp_path / "test.rebo"
        pot_file.write_text(
            "# Test REBO file\nC H# dummy",
            encoding="utf-8",
        )

        order, counts = get_potential_element_order(
            pot_file, "rebo", ["C", "H"]
        )

        assert order == ["C", "H"]
        assert counts == {"C": 1, "H": 1}

    def test_get_potential_element_order_multi_type(self, tmp_path: Path) -> None:
        """Multi-type potentials should detect type counts from element tokens."""
        pot_file = tmp_path / "test.sw"
        pot_file.write_text(
            """# Stillinger-Weber potential
Mo1 Mo2 S1
Mo1 Mo1 12.0
S1 S1 15.0
""",
            encoding="utf-8",
        )

        order, counts = get_potential_element_order(
            pot_file, "sw", ["Mo", "S"]
        )

        assert order == ["Mo", "S"]
        assert counts["Mo"] == 2
        assert counts["S"] == 1

    def test_get_potential_element_order_mismatched_elements(self, tmp_path: Path) -> None:
        """Mismatched CIF and potential elements should raise ValueError."""
        pot_file = tmp_path / "test.sw"
        pot_file.write_text(
            """# Potential with Au and Ag
Au1 Au2 Ag1
Au1 Au1 12.0
Ag1 Ag1 15.0
""",
            encoding="utf-8",
        )

        with pytest.raises(
            ValueError,
            match="Potential contains element.*not in CIF",
        ):
            get_potential_element_order(
                pot_file, "sw", ["Mo", "S"]
            )

    def test_get_potential_element_order_missing_cif_element(self, tmp_path: Path) -> None:
        """CIF element missing from potential should raise ValueError."""
        pot_file = tmp_path / "test.sw"
        pot_file.write_text(
            """# Potential with only Mo
Mo1 Mo2
Mo1 Mo1 12.0
""",
            encoding="utf-8",
        )

        with pytest.raises(
            ValueError,
            match="CIF has element.*but potential has no definition",
        ):
            get_potential_element_order(
                pot_file, "sw", ["Mo", "S"]
            )
