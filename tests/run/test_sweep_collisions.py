"""A sweep whose members would build into one directory must fail at expansion."""
from pathlib import Path

import pytest

from frictionsim2d.run.expand import count_expansion, expand_config_sweeps
from frictionsim2d.run.naming import output_dir_for

ROOT = Path('examples/potentials')


def _afm(**general):
    return {
        '2D': {'mat': 'h-MoS2', 'pot_path': 'sw/sw_lammps/h-MoS2.sw', 'pot_type': 'sw',
               'cif_path': 'examples/materials/h-MoS2.cif', 'x': 100, 'y': 100, 'layers': [1]},
        'tip': {'mat': 'Si', 'pot_type': 'sw', 'pot_path': 'Si.sw', 'r': 25},
        'sub': {'mat': 'Si', 'pot_type': 'sw', 'pot_path': 'Si.sw', 'amorph': 'a'},
        'general': {'temp': 300.0, 'force': 30.0, 'scan_speed': 2.0, **general},
    }


def test_temperatures_that_differ_only_in_decimals_collide():
    base = _afm(temp=[300.0, 300.5])

    with pytest.raises(ValueError, match=r"\[general\] temp.*300\.0.*300\.5"):
        expand_config_sweeps(base, base_dir=ROOT, model='afm')


def test_a_repeated_temperature_collides():
    with pytest.raises(ValueError, match=r"\[general\] temp"):
        expand_config_sweeps(_afm(temp=[300.0, 300.0]), base_dir=ROOT, model='afm')


def test_whole_kelvin_temperatures_expand_to_distinct_directories():
    configs = expand_config_sweeps(_afm(temp=[300.0, 350.0]), base_dir=ROOT, model='afm')

    assert len({output_dir_for(c, 'afm') for c in configs}) == 2


def test_count_expansion_reports_the_collision():
    with pytest.raises(ValueError, match=r"\[general\] temp"):
        count_expansion(_afm(temp=[300.0, 300.5]), 'afm', ROOT)


def test_without_a_model_the_collision_is_not_checked():
    assert len(expand_config_sweeps(_afm(temp=[300.0, 300.5]), base_dir=ROOT)) == 2


def _with_material_list(tmp_path, section_overrides):
    (tmp_path / 'mats.txt').write_text('h-MoS2\nh-WS2\n', encoding='utf-8')
    base = _afm()
    section = {**base['2D'], 'materials_list': 'mats.txt', **section_overrides}
    base['2D'] = {k: v for k, v in section.items() if v is not None}
    return base


def test_single_material_sweep_without_a_marker_is_rejected(tmp_path):
    base = _with_material_list(tmp_path, {'pot_path': None})

    with pytest.raises(ValueError, match=r"\[2D\] section holds no material template"):
        expand_config_sweeps(base, base_dir=tmp_path, model='afm')


def test_single_material_sweep_without_a_marker_is_rejected_without_a_model(tmp_path):
    base = _with_material_list(tmp_path, {'pot_path': None})

    with pytest.raises(ValueError, match="materials_list"):
        expand_config_sweeps(base, base_dir=tmp_path)


def test_single_material_sweep_that_keeps_mat_literal_collides(tmp_path):
    """The marker is only in a path, so every member is still named for the literal ``mat``."""
    (tmp_path / 'h-MoS2.cif').write_text('x', encoding='utf-8')
    (tmp_path / 'h-WS2.cif').write_text('x', encoding='utf-8')
    base = _with_material_list(tmp_path, {'cif_path': '{mat}.cif', 'pot_path': None})

    with pytest.raises(ValueError, match=r"\[2D\] materials_list.*h-MoS2.*h-WS2"):
        expand_config_sweeps(base, base_dir=tmp_path, model='afm')


def test_single_material_sweep_with_a_marker_still_expands(tmp_path):
    (tmp_path / 'h-MoS2.cif').write_text('x', encoding='utf-8')
    (tmp_path / 'h-WS2.cif').write_text('x', encoding='utf-8')
    base = _with_material_list(tmp_path, {'mat': '{mat}', 'cif_path': '{mat}.cif',
                                          'pot_path': None})

    configs = expand_config_sweeps(base, base_dir=tmp_path, model='afm')

    assert [c['2D']['mat'] for c in configs] == ['h-MoS2', 'h-WS2']


def test_two_pot_paths_with_the_same_type_and_stem_collide():
    base = _afm()
    base['2D']['pot_path'] = ['sw/a/x.sw', 'sw/b/x.sw']

    with pytest.raises(ValueError, match=r"\[2D\] pot_path.*sw/a/x\.sw.*sw/b/x\.sw"):
        expand_config_sweeps(base, base_dir=ROOT, model='afm')


def test_a_repeated_pot_path_collides():
    base = _afm()
    base['2D']['pot_path'] = ['sw/a/x.sw', 'sw/a/x.sw']

    with pytest.raises(ValueError, match=r"\[2D\] pot_path"):
        expand_config_sweeps(base, base_dir=ROOT, model='afm')


def test_pot_paths_with_distinct_stems_expand():
    base = _afm()
    base['2D']['pot_path'] = ['sw/a/x.sw', 'sw/b/y.sw']

    assert len(expand_config_sweeps(base, base_dir=ROOT, model='afm')) == 2


def test_a_numbered_primary_section_names_its_own_key_in_the_error():
    base = _afm(temp=[300.0, 300.4])
    base['2D-1'] = base.pop('2D')

    with pytest.raises(ValueError, match=r"\[general\] temp"):
        expand_config_sweeps(base, base_dir=ROOT, model='afm')


def test_a_blank_temperature_is_not_a_collision_check_failure():
    base = _afm(temp=None)
    base['2D']['pot_path'] = ['sw/a/x.sw', 'sw/b/y.sw']

    assert len(expand_config_sweeps(base, base_dir=ROOT, model='afm')) == 2
