"""Unit tests for src.potentials.utils.

Moved verbatim from tests/core/test_utils.py (C-4): these classes exercise
potential-type vocabulary and resolution, not core.
"""
import pytest
from pathlib import Path

from src.potentials.utils import (
    KNOWN_TYPES,
    PotentialTypeError,
    get_potential_element_order,
    lj_params,
    normalize_potential_type,
    pot_dir_token,
    resolve_pot_type,
)


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


@pytest.mark.parametrize('pot_type, path, token', [
    ('sw', 'sw/h-MoS2.sw', 'pot_sw_h-MoS2'),
    ('sw/mod', 'x/y.sw', 'pot_sw-mod_y'),
    ('reaxff', 'reaxff/ffield_ost.reax.MoS', 'pot_reaxff_ffield_ost.reax.MoS'),
    ('tersoff', 'a b.tersoff', 'pot_tersoff_a-b'),
])
def test_pot_dir_token_is_a_single_safe_segment(pot_type, path, token):
    assert pot_dir_token(pot_type, path) == token


def test_resolve_pot_type_directory_route_wins_over_extension(tmp_path):
    r = resolve_pot_type('tersoff/thing.sw', tmp_path)
    assert (r.pot_type, r.route) == ('tersoff', 'directory')


def test_resolve_pot_type_never_infers_a_variant_style(tmp_path):
    """A 'sw/mod'-shaped directory path never resolves to the variant string
    'sw/mod' — resolve_pot_type matches directory segments individually (it
    walks the path from its topmost component down, per src/potentials/utils.py's
    resolution-order comment and the already-passing
    tests/run/test_potential_sweep.py::test_directory_inference, which resolves
    'sw/sw_lammps/h-MoS2.sw' to 'sw' the same way), so the topmost 'sw' segment
    matches and wins before 'mod' is ever considered. A variant type can only be
    reached via an explicit pot_type (test_explicit_overrides_directory in that
    same file) — never inferred, only ever the base style."""
    r = resolve_pot_type('sw/mod/thing.xyz', tmp_path)
    assert (r.pot_type, r.route) == ('sw', 'directory')


def test_resolve_pot_type_unknown_extension_and_directory_fails(tmp_path):
    with pytest.raises(PotentialTypeError):
        resolve_pot_type('mystery/thing.xyz', tmp_path)


def test_known_types_contains_the_three_documented_families():
    assert {'sw', 'rebomos', 'reaxff'} <= set(KNOWN_TYPES)


def test_normalize_potential_type_is_case_and_space_insensitive():
    assert normalize_potential_type(' SW ') == 'sw'
