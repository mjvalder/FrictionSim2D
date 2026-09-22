"""The synthetic tree is itself pinned.

Every reader test in this directory is only as good as the tree it walks, so
the helper round-trips through the real `discover_runs`/`parse_run_path`
rather than through a second copy of the layout rules.
"""

from __future__ import annotations

import json

import pytest

from src.core.observables import FAMILIES
from src.run.naming import parse_run_path
from src.run.tree import discover_runs
from tests.helpers.results_tree import (
    FAMILY_LAYOUT, MODEL_FOR_FAMILY, make_family_run, write_final_data,
    write_indent_curve, write_pes_grid,
)


def test_every_family_has_a_layout():
    assert set(FAMILY_LAYOUT) == set(FAMILIES)
    assert set(MODEL_FOR_FAMILY) == set(FAMILIES)


@pytest.mark.parametrize('family', FAMILIES)
def test_a_run_is_discoverable_and_parses(tmp_path, family):
    run_dir = make_family_run(tmp_path / family, family)

    locators = discover_runs(tmp_path / family)

    assert len(locators) == 1
    loc = locators[0]
    assert loc.run_dir == run_dir
    assert loc.tokens.family == family
    assert loc.tokens.model == MODEL_FOR_FAMILY[family]
    assert parse_run_path(loc.tokens.rel_path).model == MODEL_FOR_FAMILY[family]


def test_a_potential_and_temperature_sweep_makes_four_distinct_runs(tmp_path):
    for pot in ('sw_a', 'sw_b'):
        for temp in (77, 300):
            make_family_run(tmp_path, 'afm', pot=pot, temp=temp)

    rel = [loc.tokens.rel_path for loc in discover_runs(tmp_path)]

    assert len(rel) == 4
    assert len(set(rel)) == 4
    assert all('pot_sw_' in r for r in rel)


def test_five_layers_and_three_tiers(tmp_path):
    run_dir = make_family_run(tmp_path, 'afm', layers=5,
                              tiers=('window', 'raw', 'peratom'))

    names = sorted(p.name for p in (run_dir / 'results').iterdir())

    assert names == ['friction_f30_a0_s2_layer5.txt',
                     'peratom_f30_a0_s2_layer5.txt',
                     'raw_f30_a0_s2_layer5.txt']
    assert run_dir.name == 'L5'


def test_hetero_and_flake_material_tokens(tmp_path):
    hetero = make_family_run(tmp_path / 'h', 'hetero', material='h-MoS2',
                             second_material='h-WS2')
    flake = make_family_run(tmp_path / 'f', 'flake', material='h-MoS2',
                            flake_material='h-WSe2')

    h_tokens = parse_run_path(
        hetero.relative_to(tmp_path / 'h').as_posix())
    f_tokens = parse_run_path(
        flake.relative_to(tmp_path / 'f').as_posix())

    assert h_tokens.materials == ('h-MoS2', 'h-WS2')
    assert f_tokens.flake_material == 'h-WSe2'


def test_provenance_is_a_real_schema_two_record(tmp_path):
    run_dir = make_family_run(tmp_path, 'sheetonsheet')

    record = json.loads((run_dir / 'provenance.json').read_text(encoding='utf-8'))

    assert record['schema_version'] == '2.0.0'
    assert record['family'] == 'sheetonsheet'
    assert record['model'] == 'sheetonsheet'
    assert [m['role'] for m in record['materials']] == ['sheet']


def test_pes_indent_and_final_data_artefacts(tmp_path):
    pes = make_family_run(tmp_path / 'p', 'pes_sheet')
    grid = write_pes_grid(pes, 'pes_sheet')
    indent = make_family_run(tmp_path / 'i', 'indent', layers=1)
    curve = write_indent_curve(indent, load=30.0, layers=1)
    afm = make_family_run(tmp_path / 'a', 'afm')
    final = write_final_data(afm, 'afm')

    assert grid.name == 'pes_scan.csv'
    assert grid.read_text(encoding='utf-8').splitlines()[1] == 'dx,dy,energy_eV'
    # make_family_run already wrote one; the explicit call just overwrites it
    assert not list((pes / 'results').glob('friction_*'))
    assert curve.name == 'indent_f30N_l1.csv'
    assert final.name == 'final_f30N_a0.data'
    assert 'Atoms' in final.read_text(encoding='utf-8')


def test_the_tip_grid_uses_the_tip_template_columns(tmp_path):
    """afm/pes_scan.lmp:82 prints `x,y,energy_eV,fx,fy,fz`; the sheet scan
    prints `dx,dy,energy_eV`. The tip-only descriptors are computed only when
    those force columns are present."""
    import pandas as pd

    from src.postprocessing.pes_scan import grid_descriptors

    run_dir = make_family_run(tmp_path / 't', 'pes_tip')
    grid = run_dir / 'results' / 'pes_scan.csv'

    assert grid.read_text(encoding='utf-8').splitlines()[1] == \
        'x,y,energy_eV,fx,fy,fz'
    descriptors = grid_descriptors(pd.read_csv(grid, comment='#'), 'tip')
    assert {'pes_fmax', 'pes_flat_rms', 'pes_load_mean',
            'pes_load_std'} <= set(descriptors)
