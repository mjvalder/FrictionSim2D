"""Interlayer-style vocabulary, file resolution and config validation."""
import copy
from pathlib import Path

import pytest
from pydantic import ValidationError

from frictionsim2d.core.config import (AFMSimulationConfig, GeneralConfig,
                             SheetOnSheetSimulationConfig, load_settings)
from frictionsim2d.potentials.interlayer import (EXAMPLES_POTENTIALS_DIR,
                                       default_interlayer_file, gaussian_g,
                                       ilp_substyle, mixed_g,
                                       resolve_interlayer_file)

from frictionsim2d.interfaces.lammps import lammps_pair_styles as REAL_PAIR_STYLES  # before the autouse patch

MATERIALS = Path(__file__).resolve().parents[2] / 'examples' / 'materials'


def test_default_file_tmd():
    assert default_interlayer_file('ilp', {'Mo', 'S'}) == 'TMD.ILP'
    assert default_interlayer_file('ilp', {'W', 'Se'}) == 'TMD.ILP'
    assert default_interlayer_file('ilp', {'B', 'N'}) == 'BNCH.ILP'
    assert default_interlayer_file('kc', {'Mo', 'S'}) is None
    assert default_interlayer_file('kc', {'C'}) == 'CC.KC-full'
    assert default_interlayer_file('kc', {'C', 'H'}) == 'CH.KC'


def test_ilp_substyle():
    assert ilp_substyle({'Mo', 'S'}) == 'ilp/tmd'
    assert ilp_substyle({'C'}) == 'ilp/graphene/hbn'


def test_resolve_prefers_working_dir(tmp_path):
    (tmp_path / 'with').mkdir()
    local = tmp_path / 'with' / 'TMD.ILP'
    local.write_text('x', encoding='utf-8')
    assert resolve_interlayer_file('TMD.ILP', 'ilp', tmp_path / 'with') == local
    empty = tmp_path / 'empty'
    empty.mkdir()
    got = resolve_interlayer_file('TMD.ILP', 'ilp', empty)
    assert got == EXAMPLES_POTENTIALS_DIR / 'ilp' / 'TMD.ILP'
    assert got.exists()
    assert resolve_interlayer_file('CC.KC-full', 'kc', empty).exists()


def test_resolve_unknown_raises(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        resolve_interlayer_file('nope.ILP', 'ilp', tmp_path)
    assert str(tmp_path) in str(exc.value)
    assert str(EXAMPLES_POTENTIALS_DIR / 'ilp') in str(exc.value)


def test_gaussian_g_lookup():
    assert gaussian_g('h_MoS2', None) == 175.68
    assert gaussian_g('h-MoS2', None) == 175.68
    assert gaussian_g('h_MoS2', 200.0) == 200.0
    assert gaussian_g('t_CoTe2', None) is None
    assert mixed_g(175.68, 94.87) == pytest.approx((175.68 * 94.87) ** 0.5)


def _afm_raw(tmp_path, mat='MoS2', cif='h-MoS2.cif', pot_type='sw', general=None):
    pot = tmp_path / 'p.sw'
    pot.write_text('# p', encoding='utf-8')
    dummy = tmp_path / 'dummy.cif'
    dummy.write_text('# c', encoding='utf-8')
    return {
        'general': {'temp': 300.0, 'force': 10.0, **(general or {})},
        'tip': {'mat': 'Si', 'pot_type': 'sw', 'pot_path': str(pot),
                'cif_path': str(dummy), 'r': 10.0, 'dspring': 0.1, 's': 1.0},
        'sub': {'mat': 'Si', 'pot_type': 'sw', 'pot_path': str(pot),
                'cif_path': str(dummy), 'thickness': 10.0, 'amorph': 'a'},
        '2D': {'mat': mat, 'pot_type': pot_type, 'pot_path': str(pot),
               'cif_path': str(MATERIALS / cif), 'x': 100.0, 'y': 100.0,
               'layers': [2]},
    }


def _sos_raw(tmp_path, general=None, sheet=None):
    raw = _afm_raw(tmp_path, general=general)
    raw.pop('tip')
    raw.pop('sub')
    if sheet:
        raw['2D'].update(sheet)
    return raw


def test_config_interlayer_defaults(tmp_path, mock_settings):
    cfg = AFMSimulationConfig(**_afm_raw(tmp_path), settings=mock_settings)
    assert cfg.general.interlayer_style == 'lj'
    assert cfg.general.interlayer_file is None
    assert cfg.general.interlayer_comparison is False
    assert cfg.sheet.interlayer_g is None


def test_nonlj_with_constraint_mode_rejected(tmp_path, mock_settings):
    settings = mock_settings.model_copy(deep=True)
    settings.simulation.constraint_mode = 'atom_bonds'
    raw = _sos_raw(tmp_path, general={'interlayer_style': 'ilp'})
    with pytest.raises(ValidationError, match='constraint_mode'):
        SheetOnSheetSimulationConfig(**raw, settings=settings)


def test_interlayer_file_must_match_style(tmp_path, mock_settings):
    raw = _afm_raw(tmp_path, general={'interlayer_style': 'ilp',
                                      'interlayer_file': 'CC.KC-full'})
    with pytest.raises(ValidationError, match='separate configs'):
        AFMSimulationConfig(**raw, settings=mock_settings)
    raw = _afm_raw(tmp_path, general={'interlayer_style': 'kc',
                                      'interlayer_file': 'TMD.ILP'})
    with pytest.raises(ValidationError, match='separate configs'):
        AFMSimulationConfig(**raw, settings=mock_settings)


def test_kc_on_mos2_needs_file(tmp_path, mock_settings):
    raw = _afm_raw(tmp_path, general={'interlayer_style': 'kc'})
    with pytest.raises(ValidationError, match='interlayer_file'):
        AFMSimulationConfig(**raw, settings=mock_settings)
    raw = _afm_raw(tmp_path, general={'interlayer_style': 'kc',
                                      'interlayer_file': 'my.KC'})
    AFMSimulationConfig(**raw, settings=mock_settings)


def test_ilp_default_file_ok_on_mos2(tmp_path, mock_settings):
    raw = _afm_raw(tmp_path, general={'interlayer_style': 'ilp'})
    cfg = AFMSimulationConfig(**raw, settings=mock_settings)
    assert cfg.general.interlayer_style == 'ilp'


def test_ljg_needs_g(tmp_path, mock_settings):
    raw = _afm_raw(tmp_path, mat='t_CoTe2', cif='t-CoTe2.cif',
                   general={'interlayer_style': 'lj_g'})
    with pytest.raises(ValidationError, match='interlayer_g'):
        AFMSimulationConfig(**raw, settings=mock_settings)
    raw['2D']['interlayer_g'] = 150
    AFMSimulationConfig(**raw, settings=mock_settings)


def test_ljg_known_material_ok(tmp_path, mock_settings):
    raw = _afm_raw(tmp_path, mat='h_MoS2', general={'interlayer_style': 'lj_g'})
    AFMSimulationConfig(**raw, settings=mock_settings)


def test_nonlj_on_internal_potential_rejected(tmp_path, mock_settings):
    raw = _afm_raw(tmp_path, pot_type='rebomos',
                   general={'interlayer_style': 'ilp'})
    with pytest.raises(ValidationError, match='rebomos'):
        AFMSimulationConfig(**raw, settings=mock_settings)


def test_unknown_style_rejected():
    with pytest.raises(ValidationError):
        GeneralConfig(interlayer_style='bogus')


def test_lammps_pair_styles_parses_help(monkeypatch):
    import subprocess
    from types import SimpleNamespace
    out = ("* Pair styles:\n\nlj/cut   sw   ilp/tmd\nkolmogorov/crespi/z\n\n"
           "* Bond styles:\n\nharmonic\n")
    REAL_PAIR_STYLES.cache_clear()
    try:
        monkeypatch.setattr(subprocess, 'run',
                            lambda *a, **k: SimpleNamespace(stdout=out))
        styles = REAL_PAIR_STYLES()
        assert styles == frozenset({'lj/cut', 'sw', 'ilp/tmd', 'kolmogorov/crespi/z'})
        REAL_PAIR_STYLES.cache_clear()

        def boom(*a, **k):
            raise FileNotFoundError
        monkeypatch.setattr(subprocess, 'run', boom)
        assert REAL_PAIR_STYLES() is None
    finally:
        REAL_PAIR_STYLES.cache_clear()


def test_write_gaussian_kc_file(tmp_path):
    from frictionsim2d.potentials.interlayer import write_gaussian_kc_file
    path = write_gaussian_kc_file(tmp_path / 'g.KC', {('Mo', 'S'): 175.68}, -1)
    rows = [ln.split() for ln in path.read_text().splitlines()
            if ln.strip() and not ln.startswith('#')]
    assert ['Mo', 'S', '3.0', '-87.84', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '1.0'] in rows
    assert ['S', 'Mo', '3.0', '-87.84', '0.0', '0.0', '0.0', '1.0', '0.0', '0.0', '1.0'] in rows
    with pytest.raises(ValueError):
        write_gaussian_kc_file(tmp_path / 'h.KC', {('Mo', 'S'): 1.0}, 2)


@pytest.mark.parametrize("name, expected", [
    ("sheet", True), ("flake", True),
    ("sheet_1", True), ("sheet_2", True), ("flake_1", True), ("flake_12", True),
    ("tip", False), ("sub", False), ("virtual", False),
    ("sheet_", False), ("sheet_x", False), ("my_sheet_1", False),
])
def test_splits_chalcogens_covers_hetero_components(name, expected):
    """The hetero components (register_sheet_block's sheet_N, a hetero
    flake's flake_N) are build_monolayer data too, so they split as well."""
    from frictionsim2d.potentials.interlayer import splits_chalcogens
    assert splits_chalcogens(name) is expected


def test_hetero_component_splits_under_ilp(tmp_path):
    """A registered ``sheet_1`` gets the ilp/tmd S top/bottom types."""
    from frictionsim2d.core.config import GlobalSettings, SheetConfig
    from frictionsim2d.potentials.manager import PotentialManager
    pm = PotentialManager(GlobalSettings())
    pm.set_interlayer('ilp')
    sheet = SheetConfig(mat="h-MoS2", pot_type="sw", x=4.0, y=4.0, layers=[1],
                        cif_path="examples/materials/h-MoS2.cif",
                        pot_path="examples/potentials/sw/MoS2_wen.sw")
    pm.register_component("sheet_1", sheet)
    pm.register_component("tip_1", sheet)
    assert pm.components["sheet_1"]["split_chalcogens"] == {"S"}
    assert pm.components["tip_1"]["split_chalcogens"] == set()
    assert len(pm.types.ids_by_component("sheet_1")) == 3
