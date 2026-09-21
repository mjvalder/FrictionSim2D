"""parse_run_path is the exact inverse of output_dir_for (spec §3 F2)."""
import dataclasses

import pytest

from src.run.naming import RunTokens, output_dir_for, parse_run_path


def _cfg(**over):
    d = {
        'general': {'temp': 300},
        '2D': {'mat': 'h-MoS2', 'x': 100, 'y': 100},
        'tip': {'mat': 'Si', 'amorph': 'c', 'r': 25},
        'sub': {'mat': 'Si', 'amorph': 'a'},
    }
    for key, val in over.items():
        if key in d and isinstance(val, dict):
            d[key] = {**d[key], **val}
        else:
            d[key] = val
    return d


MATRIX = [
    ('afm', _cfg(), None),
    ('afm', _cfg(**{'2D': {'pot_path': '/p/MoS.sw', 'pot_type': 'sw'}}), None),
    ('afm', _cfg(flake={'mat': 'h-WS2'}), None),
    ('afm', _cfg(**{'general': {'temp': 77}}), 'L3'),
    ('indent', _cfg(), 'L1'),
    ('pes_tip', _cfg(**{'tip': {'amorph': 'a'}}), None),
    ('sheetonsheet', _cfg(), None),
    ('sheetonsheet', {'general': {'temp': 300},
                      '2D-1': {'mat': 'h-MoS2', 'x': 100, 'y': 50},
                      '2D-2': {'mat': 'h-WS2', 'x': 100, 'y': 50}}, None),
    ('sheetonsheet', _cfg(Flake={'mat': 'h-WS2'}), None),
    ('pes_sheet', _cfg(**{'2D': {'x': 12.5, 'y': 12.5}}), None),
]


@pytest.mark.parametrize('model, cfg, layer', MATRIX)
def test_round_trip(model, cfg, layer):
    rel = output_dir_for(cfg, model).as_posix()
    if layer:
        rel = f'{rel}/{layer}'

    tokens = parse_run_path(rel)

    assert tokens.rel_path == rel
    assert tokens.model == model
    assert tokens.family == model
    sheet = cfg.get('2D') or cfg['2D-1']
    assert tokens.size_x == float(sheet['x'])
    assert tokens.size_y == float(sheet['y'])
    assert tokens.temperature == float(cfg['general']['temp'])
    assert tokens.layers == (int(layer[1:]) if layer else None)


def test_hetero_and_flake_materials_split():
    tokens = parse_run_path(
        'sheetonsheet/h-MoS2+h-WS2+flake-h-WSe2/100x_100y/K300')

    assert tokens.materials == ('h-MoS2', 'h-WS2')
    assert tokens.flake_material == 'h-WSe2'


@pytest.mark.parametrize('mat, amorph, expected_amorphous', [
    ('h-MoS2', 'a', True),
    ('h-MoS2', 'c', False),
    ('h_MoS2', 'a', True),
    ('h_MoS2', 'c', False),
])
def test_prefixed_substrate_material_amorphous_roundtrip(mat, amorph, expected_amorphous):
    """A substrate material carrying its own structure prefix (h-/h_) must
    not be misread: only the leading 'a' output_dir_for adds for
    amorph == 'a' marks it amorphous, never the material name's own 'h'."""
    cfg = _cfg(sub={'mat': mat, 'amorph': amorph})
    rel = output_dir_for(cfg, 'afm').as_posix()

    tokens = parse_run_path(rel)

    assert tokens.rel_path == rel
    assert tokens.substrate_amorphous is expected_amorphous


def test_amorphous_substrate_and_tip_tokens():
    tokens = parse_run_path('afm/h-MoS2/100x_100y/sub_aSi_tip_aSi_r25/K300/L2')

    assert tokens.substrate == 'aSi'
    assert tokens.substrate_amorphous is True
    assert tokens.tip_material == 'aSi'
    assert tokens.tip_radius == 25.0
    assert tokens.layers == 2


def test_pot_segment_is_captured_whole():
    tokens = parse_run_path(
        'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/pot_sw_MoS/L1')

    assert tokens.pot_token == 'pot_sw_MoS'


def test_unknown_model_is_refused():
    with pytest.raises(ValueError, match='unknown model'):
        parse_run_path('poke/h-MoS2/100x_100y/K300')


def test_family_can_be_replaced_without_touching_the_path():
    tokens = parse_run_path('sheetonsheet/h-MoS2+h-WS2/100x_100y/K300')

    swapped = dataclasses.replace(tokens, family='hetero')

    assert isinstance(swapped, RunTokens)
    assert swapped.family == 'hetero'
    assert swapped.rel_path == tokens.rel_path
