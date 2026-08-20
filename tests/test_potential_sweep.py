"""Potential sweep: type resolution, naming, scope guard (spec §13, Task 12)."""
from pathlib import Path

import pytest

from src.core.potential_resolve import (
    PotentialTypeError, pot_dir_token, resolve_pot_type,
)
from src.core.run import expand_config_sweeps, output_dir_for

ROOT = Path('examples/potentials')


def test_directory_inference():
    r = resolve_pot_type('sw/sw_lammps/h-MoS2.sw', ROOT)
    assert r.pot_type == 'sw' and r.route == 'directory'


def test_explicit_overrides_directory():
    r = resolve_pot_type({'path': 'sw/sw_lammps/h-MoS2.sw',
                          'pot_type': 'sw/mod'}, ROOT)
    assert r.pot_type == 'sw/mod' and r.route == 'explicit'


def test_variant_never_inferred():
    # a .sw file can serve sw OR sw/mod; inference must yield only 'sw'
    r = resolve_pot_type('somewhere/else/h-MoS2.sw', ROOT)
    assert r.route == 'extension' and '/' not in r.pot_type


def test_reaxff_directory_infers_base_family():
    # 'reaxff' has no '/': directory inference is legitimate for it
    r = resolve_pot_type('reaxff/ffield_ost.reax.MoS', ROOT)
    assert r.pot_type == 'reaxff' and r.route == 'directory'


def test_reaxff_outside_typed_dir_requires_explicit_type():
    # .reax.* serves both reaxff and legacy reax/c — extension never decides
    with pytest.raises(PotentialTypeError, match='explicitly'):
        resolve_pot_type('potentials/ffield_ost.reax.MoS', ROOT)


def test_unknown_type_fails_build():
    with pytest.raises(PotentialTypeError):
        resolve_pot_type('mystery/thing.xyz', ROOT)


def test_explicit_unknown_type_fails():
    with pytest.raises(PotentialTypeError):
        resolve_pot_type({'path': 'sw/x.sw', 'pot_type': 'quantum-magic'}, ROOT)


def test_dir_token_slug():
    assert pot_dir_token('sw/mod', 'MoS2_sevik.sw') == 'pot_sw-mod_MoS2_sevik'
    assert pot_dir_token('reax/c', 'ffield_ost.reax.MoS') == \
        'pot_reax-c_ffield_ost.reax.MoS'


def _minimal_afm_dict(pot_path):
    return {
        '2D': {'mat': 'h-MoS2', 'pot_path': pot_path, 'pot_type': 'sw',
               'cif_path': 'examples/materials/h-MoS2.cif',
               'x': 100, 'y': 100, 'layers': [2]},
        'tip': {'mat': 'Si', 'pot_type': 'sw', 'pot_path': 'Si.sw', 'r': 25},
        'sub': {'mat': 'Si', 'pot_type': 'sw', 'pot_path': 'Si.sw',
                'amorph': 'a'},
        'general': {'temp': 300.0, 'force': 30.0, 'scan_speed': 2.0},
    }


def test_sweep_expansion_creates_distinct_dirs():
    base = _minimal_afm_dict(['sw/sw_lammps/h-MoS2.sw',
                              'sw/mos2_jiang2.sw'])
    configs = expand_config_sweeps(base, base_dir=ROOT)
    assert len(configs) == 2
    types = {c['2D']['pot_type'] for c in configs}
    assert types == {'sw'}
    dirs = {output_dir_for(c, model='afm') for c in configs}
    assert len(dirs) == 2   # no collision — the LOOPABILITY rule honoured


def test_pot_segment_always_emitted():
    base = _minimal_afm_dict('sw/sw_lammps/h-MoS2.sw')
    d = output_dir_for(base, model='afm')
    assert d.name == 'pot_sw_h-MoS2'
    assert d.parent.name == 'K300'


def test_hetero_sweep_rejected():
    base = _minimal_afm_dict(['sw/a.sw', 'sw/b.sw'])
    base['2D-2'] = dict(base['2D'])
    with pytest.raises(ValueError, match='heterostructure'):
        expand_config_sweeps(base, base_dir=ROOT)


def test_flake_sweep_rejected():
    base = _minimal_afm_dict(['sw/a.sw', 'sw/b.sw'])
    base['flake'] = {'mat': 'h-MoS2'}
    with pytest.raises(ValueError, match='flake'):
        expand_config_sweeps(base, base_dir=ROOT)
