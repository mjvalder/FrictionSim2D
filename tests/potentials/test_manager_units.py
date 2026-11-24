"""Unit tests for PotentialManager pieces the larger suite only reached through builds."""

import logging
from pathlib import Path

import pytest

from frictionsim2d.core.config import SheetConfig
from frictionsim2d.potentials import manager as manager_module
from frictionsim2d.potentials.manager import PotentialManager, TypeRegistry

ROOT = Path(__file__).resolve().parents[2]
REAXFF = ROOT / 'examples' / 'potentials' / 'reaxff' / 'ffield_Wang.reax.MoS'


def _reaxff_pm(mock_settings, tmp_path, virtual=False, cross=False):
    config = SheetConfig(mat='h-MoS2', pot_type='reaxff', pot_path=str(REAXFF),
                         cif_path=str(ROOT / 'examples' / 'materials' / 'h-MoS2.cif'),
                         x=50.0, y=50.0, layers=[1])
    pm = PotentialManager(mock_settings, use_langevin=False, potentials_dir=tmp_path / 'pots')
    pm.register_component('sheet', config)
    pm.add_self_interaction('sheet')
    if virtual:
        pm.register_virtual_atom()
    if cross:
        pm.register_component('other', config)
        pm.add_self_interaction('other')
        pm.add_cross_interaction('sheet', 'other')
    return pm


def _write(pm, tmp_path):
    out = tmp_path / 'system.in.settings'
    pm.write_file(out)
    return out.read_text(encoding='utf-8').splitlines()


def test_lone_reaxff_writes_the_null_safezone_mincap_style(mock_settings, tmp_path):
    lines = _write(_reaxff_pm(mock_settings, tmp_path), tmp_path)
    safezone = mock_settings.potential.reaxff_safezone
    mincap = mock_settings.potential.reaxff_mincap
    assert f'pair_style reaxff NULL safezone {safezone} mincap {mincap}' in lines
    coeff = next(l for l in lines if l.startswith('pair_coeff * *'))
    assert coeff.split('#')[0].split() == ['pair_coeff', '*', '*', coeff.split()[3], 'Mo', 'S']
    assert 'reaxff' not in coeff.split('#')[0].split()


def test_reaxff_in_hybrid_keeps_the_same_null_token(mock_settings, tmp_path):
    lines = _write(_reaxff_pm(mock_settings, tmp_path, virtual=True), tmp_path)
    safezone = mock_settings.potential.reaxff_safezone
    mincap = mock_settings.potential.reaxff_mincap
    style = next(l for l in lines if l.startswith('pair_style'))
    assert style == (f'pair_style hybrid reaxff NULL safezone {safezone} '
                     f'mincap {mincap} lj/cut {mock_settings.potential.lj_cutoff}')
    assert next(l for l in lines if l.startswith('pair_coeff * *')).startswith(
        'pair_coeff * * reaxff ')


def test_contiguous_runs_split_non_contiguous_types():
    assert TypeRegistry.contiguous_runs([]) == []
    assert TypeRegistry.contiguous_runs([3, 1]) == [[1], [3]]
    assert TypeRegistry.contiguous_runs([1, 2, 3, 5]) == [[1, 2, 3], [5]]


def test_mass_lines_split_an_element_across_components(mock_settings):
    reg = TypeRegistry()
    for component, element in (('a', 'Mo'), ('b', 'S'), ('c', 'Mo'), ('d', 'S')):
        reg.register(component, element, element)
    pm = PotentialManager(mock_settings, use_langevin=False)
    pm.types = reg
    masses = [l.split('#')[0].split()[1] for l in pm.get_masses_string().splitlines()]
    assert masses == ['1', '3', '2', '4']

    contiguous = TypeRegistry()
    contiguous.register('a', 'Mo', 'Mo')
    contiguous.register('a', 'Mo', 'Mo')
    pm.types = contiguous
    assert pm.get_masses_string().startswith('mass 1*2 ')


def test_unknown_element_mass_falls_back_with_a_warning(mock_settings, caplog):
    assert manager_module._element_mass('Mo') == pytest.approx(95.95)
    with caplog.at_level(logging.WARNING):
        assert manager_module._element_mass('Xx') is None
    assert "Could not find mass for element 'Xx'" in caplog.text
    reg = TypeRegistry()
    reg.register('a', 'Xx', 'Xx')
    pm = PotentialManager(mock_settings, use_langevin=False)
    pm.types = reg
    assert pm.get_masses_string() == 'mass 1 1.0 #Xx (unknown)'


def test_padded_maps_pad_with_null_and_narrow_by_layer():
    reg = TypeRegistry()
    reg.register('sheet', 'Mo', 'Mo1', layer=0)
    reg.register('sheet', 'Mo', 'Mo2', layer=1)
    reg.register('tip', 'Si', 'Si1')
    assert reg.build_null_map('sheet') == ['Mo1', 'Mo2', 'NULL']
    assert reg.build_null_map('sheet', layer=1) == ['NULL', 'Mo2', 'NULL']
    assert reg.build_element_map(['sheet', 'tip']) == ['Mo', 'Mo', 'Si']
    assert reg.build_element_map(['tip']) == ['NULL', 'NULL', 'Si']


def test_cross_interaction_orders_the_pair_by_first_type(mock_settings, tmp_path):
    pm = _reaxff_pm(mock_settings, tmp_path, cross=True)
    for line in pm.cross_interaction_commands:
        t1, t2 = line.split()[1:3]
        assert int(t1.split('*')[0]) <= int(t2.split('*')[0])
    pm2 = PotentialManager(mock_settings, use_langevin=False)
    pm2.types.register('b', 'S', 'S')
    pm2.types.register('a', 'Mo', 'Mo')
    pm2.components = {n: {'config': SimpleConfig()} for n in ('a', 'b')}
    pm2.add_cross_interaction('a', 'b')
    assert pm2.cross_interaction_commands[0].startswith('pair_coeff 1 2 lj/cut')


class SimpleConfig:
    pot_type = 'sw'


def test_lj_g_provenance_marks_an_explicit_override(mock_settings, mos2_sheet_config, tmp_path):
    def provenance(**over):
        pm = PotentialManager(mock_settings, use_langevin=False, potentials_dir=tmp_path)
        config = mos2_sheet_config.model_copy(update=over)
        pm.register_component('sheet', config, n_layers=2)
        pm.set_interlayer('lj_g')
        pm.add_interlayer(['sheet'])
        return pm.interlayer_provenance()['g_meV']['sheet']

    assert provenance(interlayer_g=123.0) == {'value': 123.0, 'source': 'override'}
    assert provenance()['source'] == 'table'
