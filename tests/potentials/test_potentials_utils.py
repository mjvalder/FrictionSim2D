"""Unit tests for frictionsim2d.potentials.utils.

Moved verbatim from tests/core/test_utils.py (C-4): these classes exercise
potential-type vocabulary and resolution, not core.
"""
import pytest
from pathlib import Path

from frictionsim2d.potentials.utils import (
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
        eps_h_h, _ = lj_params('H', 'H')
        # Geometric mean of two distinct positive values sits strictly between
        # them, so the mixed epsilon must lie between the two like-pair values.
        assert min(eps_c_c, eps_h_h) < eps_c_h < max(eps_c_c, eps_h_h)


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
    r = resolve_pot_type('tersoff/thing.sw')
    assert (r.pot_type, r.route) == ('tersoff', 'directory')


def test_resolve_pot_type_never_infers_a_variant_style(tmp_path):
    """A 'sw/mod'-shaped directory path never resolves to the variant string
    'sw/mod' — resolve_pot_type matches directory segments individually (it
    walks the path from its topmost component down, per frictionsim2d/potentials/utils.py's
    resolution-order comment and the already-passing
    tests/run/test_potential_sweep.py::test_directory_inference, which resolves
    'sw/sw_lammps/h-MoS2.sw' to 'sw' the same way), so the topmost 'sw' segment
    matches and wins before 'mod' is ever considered. A variant type can only be
    reached via an explicit pot_type (test_explicit_overrides_directory in that
    same file) — never inferred, only ever the base style."""
    r = resolve_pot_type('sw/mod/thing.xyz')
    assert (r.pot_type, r.route) == ('sw', 'directory')


def test_resolve_pot_type_unknown_extension_and_directory_fails(tmp_path):
    with pytest.raises(PotentialTypeError):
        resolve_pot_type('mystery/thing.xyz')


def test_known_types_contains_the_three_documented_families():
    assert {'sw', 'rebomos', 'reaxff'} <= set(KNOWN_TYPES)


def test_normalize_potential_type_is_case_and_space_insensitive():
    assert normalize_potential_type(' SW ') == 'sw'


class TestCustomPotType:
    """A style the installed LAMMPS lists is accepted even if not in KNOWN_TYPES."""

    def test_custom_type_accepted_with_warning(self, monkeypatch, caplog):
        from frictionsim2d.potentials import utils
        monkeypatch.setattr('frictionsim2d.interfaces.lammps.lammps_pair_styles',
                            lambda: frozenset({'sw', 'sw/mod', 'mlip'}))
        with caplog.at_level('WARNING'):
            assert utils.validate_pot_type('mlip', source='x') == 'mlip'
        assert 'mlip' in caplog.text
        with pytest.raises(PotentialTypeError):
            utils.validate_pot_type('swx', source='x')

    def test_no_binary_accepts_with_warning(self, monkeypatch, caplog):
        from frictionsim2d.potentials import utils
        monkeypatch.setattr('frictionsim2d.interfaces.lammps.lammps_pair_styles', lambda: None)
        with caplog.at_level('WARNING'):
            assert utils.validate_pot_type('swx', source='x') == 'swx'
        assert 'swx' in caplog.text

    def test_explicit_sweep_route(self, monkeypatch):
        from frictionsim2d.potentials import utils
        monkeypatch.setattr('frictionsim2d.interfaces.lammps.lammps_pair_styles',
                            lambda: frozenset({'mlip'}))
        res = utils.resolve_pot_type({'path': 'a.bin', 'pot_type': 'mlip'})
        assert res.pot_type == 'mlip'


# --- potential/CIF compatibility, internal cutoffs, D3 slug ------------------

_MATERIALS = Path(__file__).resolve().parents[2] / 'examples' / 'materials'


def _potential(tmp_path, *triplets):
    path = tmp_path / 'p.sw'
    path.write_text('\n'.join(f'{a} {b} {c} 1.0' for a, b, c in triplets) + '\n',
                    encoding='utf-8')
    return path


def test_cif_compatibility_consistent_multiples(tmp_path):
    from frictionsim2d.potentials.utils import check_potential_cif_compatibility
    cif = _MATERIALS / 'h-MoS2.cif'                       # Mo 1, S 2 per cell
    plain = _potential(tmp_path, ('Mo', 'S', 'S'))
    assert check_potential_cif_compatibility(cif, plain) == 1
    doubled = _potential(tmp_path, ('Mo1', 'S1', 'S3'), ('Mo2', 'S2', 'S4'))
    assert check_potential_cif_compatibility(cif, doubled) == 2


def test_cif_compatibility_inconsistent_ratios_raise(tmp_path):
    from frictionsim2d.potentials.utils import check_potential_cif_compatibility
    cif = _MATERIALS / 'h-MoS2.cif'
    mixed = _potential(tmp_path, ('Mo1', 'S1', 'S2'), ('Mo2', 'S1', 'S2'))
    with pytest.raises(ValueError, match='Inconsistent atom type multiplier'):
        check_potential_cif_compatibility(cif, mixed)


_POTS = Path(__file__).resolve().parents[2] / 'examples' / 'potentials'


@pytest.mark.parametrize('pot_type, expected', [
    ('airebo', 3.0 * 3.4),
    ('comb', 7.5),
    ('comb3', 7.5),
    ('deepmd', 7.5),
    ('mace', 7.5),
    ('sw', 0.0),
])
def test_internal_pair_cutoff_by_type(pot_type, expected):
    from frictionsim2d.potentials.utils import internal_pair_cutoff
    assert internal_pair_cutoff(pot_type, 'unused', 7.5) == expected


def test_internal_pair_cutoff_reaxff_variants(tmp_path):
    from frictionsim2d.potentials.utils import internal_pair_cutoff
    wide = tmp_path / 'wide.reax'
    wide.write_text('header\n' + ''.join(f'{v} x\n' for v in
                                        [1] * 13 + [14.5, 0, 0]), encoding='utf-8')
    assert internal_pair_cutoff('reaxff', wide, 11.0) == 14.5   # swb beats QEq 10.0
    broken = tmp_path / 'broken.reax'
    broken.write_text('too short\n', encoding='utf-8')
    assert internal_pair_cutoff('reaxff', broken, 11.0) == 10.0  # unparseable: QEq


def test_internal_pair_cutoff_rebomos_without_sigma_raises(tmp_path):
    from frictionsim2d.potentials.utils import internal_pair_cutoff
    empty = tmp_path / 'x.rebomos'
    empty.write_text('1.0 epsilon_MM\n', encoding='utf-8')
    with pytest.raises(ValueError, match='no sigma_'):
        internal_pair_cutoff('rebomos', empty, 11.0)


def test_pot_dir_token_d3_slug_only_for_mlips():
    assert pot_dir_token('mace', 'm.json', dispersion='d3') == 'pot_mace-d3_m'
    assert pot_dir_token('mace', 'm.json', dispersion='none') == 'pot_mace_m'
    assert pot_dir_token('mace', 'm.json') == 'pot_mace_m'
    assert pot_dir_token('sw', 'MoS2.sw', dispersion='d3') == 'pot_sw_MoS2'


def test_lj_params_reports_the_missing_element():
    with pytest.raises(KeyError, match="'Xx' not found"):
        lj_params('Mo', 'Xx')


def test_pot_dir_token_float32_slug_only_for_mlips():
    assert pot_dir_token('mace', 'm.json', precision='float32') == 'pot_mace-f32_m'
    assert pot_dir_token('mace', 'm.json', dispersion='d3', precision='float32') == 'pot_mace-d3-f32_m'
    assert pot_dir_token('mace', 'm.json', precision='float64') == 'pot_mace_m'
    assert pot_dir_token('sw', 'MoS2.sw', precision='float32') == 'pot_sw_MoS2'
    assert pot_dir_token('sw/mod', 'MoS2.sw', precision='float32') == 'pot_sw-mod_MoS2'
