"""interlayer_style sweep and il- directory token (interlayer spec, Task 2)."""
from pathlib import Path

import pytest

from frictionsim2d.run.expand import expand_config_sweeps
from frictionsim2d.run.naming import (
    LOOPABILITY, interlayer_dir_token, output_dir_for, parse_run_path,
)
from tests.run.test_potential_sweep import _minimal_afm_dict

ROOT = Path('frictionsim2d/data/potentials')
POT = 'sw/sw_lammps/h-MoS2.sw'


def test_lj_path_unchanged():
    d = output_dir_for(_minimal_afm_dict(POT), model='afm')
    assert d.as_posix() == (
        'afm/h-MoS2/100x_100y/sub_amorph-Si_tip_Si_r25/K300/pot_sw_h-MoS2')


def test_loopability():
    assert LOOPABILITY[('general', 'interlayer_style')] == 'sweep'


def test_token_values():
    assert interlayer_dir_token('lj', False) is None
    assert interlayer_dir_token('lj_g', False) == 'il-ljg'
    assert interlayer_dir_token('lj_g', True) == 'il-ljg-cmp'
    assert interlayer_dir_token('lj', True) == 'il-lj-cmp'
    assert interlayer_dir_token('ilp', False) == 'il-ilp'


def test_interlayer_sweep_dirs():
    base = _minimal_afm_dict(POT)
    base['general']['interlayer_style'] = ['lj', 'lj_g', 'ilp']
    configs = expand_config_sweeps(base, base_dir=ROOT)
    dirs = [output_dir_for(c, model='afm').as_posix() for c in configs]
    assert len(dirs) == 3
    assert dirs[0].endswith('pot_sw_h-MoS2')
    assert dirs[1].endswith('pot_sw_h-MoS2/il-ljg')
    assert dirs[2].endswith('pot_sw_h-MoS2/il-ilp')


def test_segment_without_pot_path():
    cfg = {'general': {'temp': 300, 'interlayer_style': 'kc'},
           '2D': {'mat': 'h-MoS2', 'x': 100, 'y': 100}}
    assert output_dir_for(cfg, model='sheetonsheet').as_posix().endswith(
        'K300/il-kc')


def test_comparison_token():
    cfg = _minimal_afm_dict(POT)
    cfg['general']['interlayer_comparison'] = True
    assert output_dir_for(cfg, model='afm').as_posix().endswith(
        'pot_sw_h-MoS2/il-lj-cmp')


def test_duplicate_style_rejected():
    base = _minimal_afm_dict(POT)
    base['general']['interlayer_style'] = ['ilp', 'ilp']
    with pytest.raises(ValueError, match='interlayer_style'):
        expand_config_sweeps(base, base_dir=ROOT)


def test_parse_roundtrip():
    p = ('afm/h-MoS2/100x_100y/sub_amorph-Si_tip_Si_r25/K300/'
         'pot_sw_x/il-ilp')
    assert parse_run_path(p).interlayer_token == 'il-ilp'
    assert parse_run_path(p + '/L2').layers == 2
    old = 'afm/h-MoS2/100x_100y/sub_amorph-Si_tip_Si_r25/K300/pot_sw_x'
    assert parse_run_path(old).interlayer_token is None
    no_pot = 'sheetonsheet/h-MoS2/100x_100y/K300/il-ljg-cmp'
    assert parse_run_path(no_pot).interlayer_token == 'il-ljg-cmp'
