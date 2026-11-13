"""Post-stream channels (P6): derived here, never emitted by LAMMPS.

The registry declares gap_min/deflect_max (afm, indent) and
gap_min_il/corrugation_max (sheet families) with stream='post' and no
lammps_compute. Nothing has ever written them; this is their producer.
"""

from __future__ import annotations

import json

import pytest

from src.data.provenance import read_provenance
from src.postprocessing.post_channels import (final_data_path, post_channels,
                                              post_channels_for, read_atom_z)
from src.postprocessing.read_data import DataReader
from tests.helpers.results_tree import make_family_run, write_final_data


def test_read_atom_z_groups_by_type(tmp_path):
    run_dir = make_family_run(tmp_path, 'afm')
    path = write_final_data(run_dir, 'afm')

    by_type = read_atom_z(path)

    assert sorted(by_type) == [1, 2, 3]
    assert by_type[1] == [0.0, -0.4]
    assert by_type[3] == [9.6, 10.0]


def test_final_data_path_matches_on_value_not_spelling(tmp_path):
    run_dir = make_family_run(tmp_path, 'afm')
    written = write_final_data(run_dir, 'afm', load=30.0, angle=0.0)

    assert final_data_path(run_dir / 'data', 'force', 30.0, 0.0) == written
    assert final_data_path(run_dir / 'data', 'force', 30.0, 90.0) is None
    assert final_data_path(run_dir / 'data', 'pressure', 30.0, 0.0) is None
    assert final_data_path(run_dir / 'missing', 'force', 30.0, 0.0) is None


def test_afm_channels(tmp_path):
    run_dir = make_family_run(tmp_path, 'afm')
    path = write_final_data(run_dir, 'afm')
    record = read_provenance(run_dir / 'provenance.json',
                             family='afm', model='afm')

    out = post_channels(path, record, 'afm')

    # tip = types outside layer_type_map lying above the sheet; its lowest
    # atom is 9.6, the sheet's highest is 6.5
    assert out['gap_min'] == pytest.approx(3.1)
    # unloaded_reference_z['layer_2'] is 6.5; the layer's lowest atom is 6.1
    assert out['deflect_max'] == pytest.approx(0.4)


def test_sheet_channels(tmp_path):
    run_dir = make_family_run(tmp_path, 'sheetonsheet')
    path = write_final_data(run_dir, 'sheetonsheet', load_kind='pressure',
                            load=30.0, angle=0.0)
    record = read_provenance(run_dir / 'provenance.json',
                             family='sheetonsheet', model='sheetonsheet')

    out = post_channels(path, record, 'sheetonsheet')

    assert out['gap_min_il'] == pytest.approx(6.1)     # 6.1 - 0.0
    assert out['corrugation_max'] == pytest.approx(0.4)  # 6.5 - 6.1
    assert 'gap_min' not in out


def test_an_empty_layer_type_map_yields_nothing(tmp_path):
    run_dir = make_family_run(tmp_path, 'flake')
    path = write_final_data(run_dir, 'flake', load_kind='pressure')
    prov = run_dir / 'provenance.json'
    record = json.loads(prov.read_text(encoding='utf-8'))
    record['layer_type_map'] = {}
    prov.write_text(json.dumps(record), encoding='utf-8')

    out = post_channels(path, read_provenance(prov, family='flake',
                                              model='sheetonsheet'), 'flake')

    assert out == {}


def test_the_reader_fills_run_frames_post(tmp_path):
    make_family_run(tmp_path, 'afm', final_data=True)

    reader = DataReader(results_dir=str(tmp_path))

    run, = reader.runs
    assert run.post['gap_min'] == pytest.approx(3.1)
    assert run.post['deflect_max'] == pytest.approx(0.4)


def test_no_final_data_means_an_empty_post_dict(tmp_path):
    make_family_run(tmp_path, 'afm', final_data=False)

    reader = DataReader(results_dir=str(tmp_path))

    assert reader.runs[0].post == {}


def test_post_channels_reach_run_summary_extras(tmp_path):
    """P6 puts these in RunFrames.post AND RunSummary.extras. Only the first
    half is this module's; the second is what carries them to the local store
    and the central DB, and without it they would stop at the CSV."""
    from src.data.models import summarise_run

    make_family_run(tmp_path, 'afm', final_data=True)
    run, = DataReader(results_dir=str(tmp_path)).runs

    summary = summarise_run(run.frames_by_tier(), 'afm', post=run.post)

    assert summary.extras['gap_min'] == pytest.approx(3.1)
    assert summary.extras['deflect_max'] == pytest.approx(0.4)
    assert 'gap_min' in summary.as_dict()


def test_summarise_run_without_post_is_unchanged(tmp_path):
    from src.data.models import summarise_run

    make_family_run(tmp_path, 'afm', final_data=True)
    run, = DataReader(results_dir=str(tmp_path)).runs

    assert 'gap_min' not in summarise_run(run.frames_by_tier(), 'afm').extras


def test_post_channels_for_reads_the_locator(tmp_path):
    from src.run.tree import discover_runs
    make_family_run(tmp_path, 'afm', final_data=True)
    locator, = discover_runs(tmp_path)

    out = post_channels_for(locator, 'force', 30.0, 0.0)

    assert set(out) == {'gap_min', 'deflect_max'}


def test_final_data_path_finds_the_pes_end_snapshot(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    assert final_data_path(data, 'none', 0.0, 0.0) is None
    (data / 'final_pes.data').write_text('x', encoding='utf-8')
    assert final_data_path(data, 'none', 0.0, 0.0) == data / 'final_pes.data'
    assert final_data_path(data, 'none', 5.0, 45.0) == data / 'final_pes.data'
    assert final_data_path(data, 'force', 0.0, 0.0) is None
