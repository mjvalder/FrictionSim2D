"""Output-directory naming for MLIP runs (merge decision 1, "option B").

An MLIP run must be loopable in a sweep alongside its classical counterpart,
so its flavour is encoded in the path. It rides in the existing
``pot_<type-slug>_<stem>`` segment that ``output_dir_for`` emits whenever the
sheet has a ``pot_path`` (always true for an MLIP sheet): ``pot_mace_<stem>``
for a bare model, ``pot_mace-d3_<stem>`` with ``dispersion = d3`` (same for
deepmd). The ``K<temp>`` leaf carries no suffix, so every upstream segment --
the ones ``parse_run_path`` and the postprocessing filters read -- is
identical to the classical path.
"""

from pathlib import Path

import pytest

from frictionsim2d.run.naming import (
    mlip_flavor_from_pot_token,
    output_dir_for,
    parse_run_path,
)


def _sheet(**overrides):
    sheet = {'mat': 'h-MoS2', 'pot_type': 'sw', 'x': 100, 'y': 100}
    sheet.update(overrides)
    return sheet


def _mlip_sheet(pot_type='deepmd', model='dpa3.pth', **overrides):
    return _sheet(pot_type=pot_type, pot_path=f'models/{model}', lat_c=6.15,
                  **overrides)


def _afm_dict(sheet=None):
    return {
        'general': {'temp': 300},
        'tip': {'mat': 'Si', 'amorph': 'c', 'r': 25},
        'sub': {'mat': 'Si', 'amorph': 'a'},
        '2D': sheet if sheet is not None else _sheet(),
    }


def _sheet_dict(sheet=None, temp=0):
    return {'general': {'temp': temp},
            '2D': sheet if sheet is not None else _sheet()}


# ---------------------------------------------------------------------------
# Regression: classical paths carry no MLIP token
# ---------------------------------------------------------------------------

def test_classical_afm_path_has_no_pot_segment():
    assert output_dir_for(_afm_dict(), 'afm') == Path(
        'afm/h-MoS2/100x_100y/sub_amorph-Si_tip_Si_r25/K300')


def test_classical_sheet_path_has_no_pot_segment():
    assert output_dir_for(_sheet_dict(), 'sheetonsheet') == Path(
        'sheetonsheet/h-MoS2/100x_100y/K0')


def test_classical_path_ignores_mlip_defaults_written_out_explicitly():
    """An SW sheet that happens to carry the default keys is still classical."""
    assert output_dir_for(_sheet_dict(_sheet(dispersion='none')),
                          'sheetonsheet') == Path('sheetonsheet/h-MoS2/100x_100y/K0')


def test_classical_pot_path_d3_is_not_encoded_in_the_slug():
    """``dispersion`` only rides in the slug for an MLIP pot_type."""
    path = output_dir_for(
        _sheet_dict(_sheet(pot_path='pots/MoS.sw', dispersion='d3')), 'sheetonsheet')
    assert path.name == 'pot_sw_MoS'


# ---------------------------------------------------------------------------
# MLIP flavour in the pot_ segment
# ---------------------------------------------------------------------------

def test_deepmd_without_d3_gets_a_bare_slug_and_unsuffixed_leaf():
    path = output_dir_for(_sheet_dict(_mlip_sheet()), 'sheetonsheet')
    assert path == Path('sheetonsheet/h-MoS2/100x_100y/K0/pot_deepmd_dpa3')


def test_mace_with_d3_gets_the_d3_slug():
    path = output_dir_for(
        _sheet_dict(_mlip_sheet('mace', 'mace-mp.json', dispersion='d3'), temp=300),
        'sheetonsheet')
    assert path == Path('sheetonsheet/h-MoS2/100x_100y/K300/pot_mace-d3_mace-mp')


@pytest.mark.parametrize('model,stem', [
    ('m.json', 'm'), ('m.pth', 'm'), ('m.pb', 'm'),
])
def test_model_extensions_are_stripped_from_the_stem(model, stem):
    pot_type = 'mace' if model.endswith('.json') else 'deepmd'
    path = output_dir_for(_sheet_dict(_mlip_sheet(pot_type, model)), 'sheetonsheet')
    assert path.name == f'pot_{pot_type}_{stem}'


def test_d3_and_bare_runs_of_one_model_never_share_a_directory():
    bare = output_dir_for(_sheet_dict(_mlip_sheet()), 'sheetonsheet')
    d3 = output_dir_for(_sheet_dict(_mlip_sheet(dispersion='d3')), 'sheetonsheet')
    assert bare != d3
    assert bare.parent == d3.parent


def test_afm_mlip_leaves_upstream_segments_untouched():
    path = output_dir_for(_afm_dict(_mlip_sheet()), 'afm')
    assert path == Path(
        'afm/h-MoS2/100x_100y/sub_amorph-Si_tip_Si_r25/K300/pot_deepmd_dpa3')
    # Only the extra pot_ segment differs from the classical path.
    assert path.parent == output_dir_for(_afm_dict(), 'afm')


@pytest.mark.parametrize('model', ['pes_tip', 'indent'])
def test_token_applies_to_every_afm_like_model(model):
    assert output_dir_for(_afm_dict(_mlip_sheet()), model).name == 'pot_deepmd_dpa3'


# ---------------------------------------------------------------------------
# Decoding: parse_run_path / mlip_flavor_from_pot_token
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('token,expected', [
    ('pot_mace_mace-mp', ('mace', 'none')),
    ('pot_mace-d3_mace-mp', ('mace', 'd3')),
    ('pot_deepmd_dpa3', ('deepmd', 'none')),
    ('pot_deepmd-d3_dpa3', ('deepmd', 'd3')),
])
def test_mlip_flavor_decodes_every_emitted_slug(token, expected):
    assert mlip_flavor_from_pot_token(token) == expected


@pytest.mark.parametrize('token', [
    'pot_sw_MoS', 'pot_tersoff-d3_x', 'pot_sw-mod_x', 'K300', 'L1', '', None,
])
def test_classical_or_non_pot_segment_decodes_to_none(token):
    assert mlip_flavor_from_pot_token(token) is None


def test_parse_run_path_round_trips_an_mlip_afm_path():
    path = output_dir_for(_afm_dict(_mlip_sheet(dispersion='d3')), 'afm')
    tokens = parse_run_path(str(path / 'L1'))

    assert tokens.mlip_framework == 'deepmd'
    assert tokens.dispersion == 'd3'
    assert tokens.pot_token == 'pot_deepmd-d3_dpa3'
    # Upstream tokens parse exactly as for the classical path.
    assert tokens.materials == ('h-MoS2',)
    assert (tokens.size_x, tokens.size_y) == (100.0, 100.0)
    assert tokens.temperature == 300.0
    assert (tokens.tip_material, tokens.tip_radius) == ('Si', 25.0)


def test_parse_run_path_round_trips_an_mlip_sheet_path():
    path = output_dir_for(
        _sheet_dict(_mlip_sheet('mace', 'mace-mp.json'), temp=300), 'sheetonsheet')
    tokens = parse_run_path(str(path))

    assert (tokens.mlip_framework, tokens.dispersion) == ('mace', 'none')
    assert tokens.materials == ('h-MoS2',)
    assert tokens.temperature == 300.0


def test_parse_run_path_of_a_classical_path_has_no_flavor():
    tokens = parse_run_path(str(output_dir_for(_afm_dict(), 'afm')))
    assert tokens.mlip_framework is None
    assert tokens.dispersion is None
