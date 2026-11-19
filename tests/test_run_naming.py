"""Output-directory naming, including the MLIP flavor token (spec §1.3).

An MLIP run must be loopable in a sweep alongside its classical counterpart, so
its flavor is encoded in the path. The token goes on the **innermost
run-parameter segment** (the ``K<temp>`` leaf, which already varies with
temperature) and nowhere else: the upstream segments — material and size — are
what ``src/postprocessing/read_data.py``'s filters parse, and they must stay
untouched.
"""

from pathlib import Path

import pytest

from src.core.run import output_dir_for
from src.postprocessing.read_data import DataReader


def _sheet(**overrides):
    sheet = {'mat': 'h-MoS2', 'pot_type': 'sw', 'x': 100, 'y': 100}
    sheet.update(overrides)
    return sheet


def _afm_dict(**sheet_overrides):
    return {
        'general': {'temp': 300},
        'tip': {'mat': 'Si', 'amorph': 'c', 'r': 25},
        'sub': {'mat': 'Si', 'amorph': 'a'},
        '2D': _sheet(**sheet_overrides),
    }


def _sheet_dict(**sheet_overrides):
    return {'general': {'temp': 0}, '2D': _sheet(**sheet_overrides)}


# ---------------------------------------------------------------------------
# Regression: classical paths are byte-identical
# ---------------------------------------------------------------------------

def test_classical_afm_path_is_unchanged():
    """Golden captured before the flavor token existed."""
    assert output_dir_for(_afm_dict(), 'afm') == Path(
        'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300')


def test_classical_sheet_path_is_unchanged():
    assert output_dir_for(_sheet_dict(), 'sheetonsheet') == Path(
        'sheetonsheet/h-MoS2/100x_100y/K0')


def test_classical_path_ignores_mlip_defaults_written_out_explicitly():
    """An SW sheet that happens to carry the default keys is still classical."""
    run = _sheet_dict(dispersion='none')
    assert output_dir_for(run, 'sheetonsheet') == Path(
        'sheetonsheet/h-MoS2/100x_100y/K0')


# ---------------------------------------------------------------------------
# MLIP flavor token
# ---------------------------------------------------------------------------

def test_deepmd_base_token_on_the_leaf_segment():
    run = _sheet_dict(pot_type='deepmd', lat_c=6.15)
    assert output_dir_for(run, 'sheetonsheet') == Path(
        'sheetonsheet/h-MoS2/100x_100y/K0_deepmd-native-base')


def test_mace_native_d3_token_on_the_leaf_segment():
    run = _sheet_dict(pot_type='mace', dispersion='d3', lat_c=6.15)
    run['general']['temp'] = 300
    assert output_dir_for(run, 'sheetonsheet') == Path(
        'sheetonsheet/h-MoS2/100x_100y/K300_mace-native-d3')


def test_afm_token_leaves_upstream_segments_untouched():
    run = _afm_dict(pot_type='deepmd', lat_c=6.15)
    path = output_dir_for(run, 'afm')
    assert path == Path(
        'afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300_deepmd-native-base')
    # Only the leaf differs from the classical path.
    classical = output_dir_for(_afm_dict(), 'afm')
    assert path.parent == classical.parent


def test_defaults_fill_in_when_the_keys_are_absent():
    """A raw dict need not spell out the SheetConfig defaults.

    The coverage segment is constant: full-cell is the only topology (v4 §3),
    so it is stamped whether or not the dict says anything about coverage.
    """
    run = _sheet_dict(pot_type='deepmd', lat_c=6.15)
    assert output_dir_for(run, 'sheetonsheet').name == 'K0_deepmd-native-base'


@pytest.mark.parametrize('model', ['pes_tip', 'indent'])
def test_token_applies_to_every_afm_like_model(model):
    run = _afm_dict(pot_type='deepmd', lat_c=6.15)
    assert output_dir_for(run, model).name == 'K300_deepmd-native-base'


# ---------------------------------------------------------------------------
# Postprocessing filters still parse a tokened path (spec §1.3, §6)
# ---------------------------------------------------------------------------

def test_postprocessing_still_extracts_material_and_size(tmp_path):
    reader = DataReader(results_dir=str(tmp_path))

    tip_root = str(tmp_path / 'afm' / 'h_MoS2' / '100x_100y' /
                   'sub_aSi_tip_cSi_r25' / 'K300_deepmd-lj-base' / 'L1' /
                   'results')
    sim_type, material, size, substrate, tip_material, tip_radius = \
        reader._resolve_path_info(tip_root)
    assert (sim_type, material, size) == ('tip', 'h_MoS2', '100x_100y')
    assert (substrate, tip_material, tip_radius) == ('aSi', 'cSi', 'r25')

    sheet_root = str(tmp_path / 'sheetonsheet' / 'h_MoS2' / '100x_100y' /
                     'K300_mace-native-d3' / 'results')
    sim_type, material, size, _sub, _tip, _r = \
        reader._resolve_path_info(sheet_root)
    assert (sim_type, material, size) == ('sheet', 'h_MoS2', '100x_100y')
