"""The store's consumers against a real store (plan Task 12): the GUI's
`db_worker` (list, detail, export, import) and the plot source
`aiida://dataset/<name>`.

`tests/gui/test_gui_db_worker.py` drives the same dispatch against fakes;
this file is what proves the uuids the pane hands over load real nodes.
"""
import json
import math

import pytest

pytest.importorskip('aiida')
pytest.importorskip('scipy')
pytestmark = pytest.mark.aiida

from aiida import orm  # noqa: E402

from src.aiida.block_schema import BLOCK_KINDS  # noqa: E402
from src.aiida.ingest import import_root  # noqa: E402
from src.aiida.steps import build_dataset, postprocess_group  # noqa: E402
from src.gui import db_worker  # noqa: E402
from src.postprocessing.plot_data import PlotDataError, Plotter  # noqa: E402
from tests.aiida.test_steps_integrity import _tree  # noqa: E402


@pytest.fixture
def no_profile_switch(monkeypatch):
    import src.aiida as aiida_pkg
    monkeypatch.setattr(aiida_pkg, 'load_aiida_profile', lambda profile=None: None)


@pytest.fixture
def store(aiida_profile_clean, no_profile_switch, tmp_path):
    """One AFM run at 10/30/50 N imported as set `camp` (10 N fails its
    integrity check), summarised, with dataset `paper` built over it."""
    _tree(tmp_path / 'tree', eject=(10.0,))
    report = import_root(tmp_path / 'tree', label='camp')
    postprocess_group('camp')
    build_dataset('paper', ['camp'])
    return report


def _list():
    result = db_worker.dispatch({'kind': 'list', 'source': 'local', 'filters': {}})
    assert result['ok'] is True, result.get('error')
    return result


def _row(load):
    (row,) = [r for r in _list()['records'] if r['load'] == load]
    return row


# --- list -----------------------------------------------------------------------

def test_list_rows_carry_the_verdict_and_the_point_identity(store):
    result = _list()

    assert result['total_count'] == 3
    assert {'status', 'reasons', 'point_uuid', 'node_uuid'} <= set(result['columns'])
    by_load = {r['load']: r for r in result['records']}
    assert by_load[10.0]['status'] == 'failed_integrity'
    assert by_load[10.0]['reasons']
    assert by_load[30.0]['status'] == 'complete'
    assert by_load[30.0]['mean_cof'] is not None
    assert result['set_sizes'] == {'camp': 1}
    json.dumps(result)


# --- detail ---------------------------------------------------------------------

def test_detail_loads_the_point_its_verdict_summary_and_related_counts(store):
    row = _row(30.0)

    detail = db_worker.dispatch({'kind': 'detail', 'source': 'local',
                                 'node_uuid': row['node_uuid'],
                                 'point_uuid': row['point_uuid']})

    assert detail['ok'] is True, detail.get('error')
    assert detail['point']['uuid'] == row['node_uuid']
    assert detail['point']['attributes']['point_uuid'] == row['point_uuid']
    assert detail['verdict']['attributes']['status'] == 'complete'
    assert detail['summary']['attributes']['mean_cof'] == pytest.approx(row['mean_cof'])
    assert set(detail['related']) == set(BLOCK_KINDS) | {'load', 'speed'}
    assert 'realization' in detail['related']
    # The other two loads of the same run differ only along `load`.
    assert detail['related']['load'] == 2
    assert detail['related']['speed'] == 0
    assert detail['files'] and all(f['node'] == 'point' for f in detail['files'])
    assert detail['total_bytes'] == sum(f['bytes'] for f in detail['files'])
    json.dumps(detail)


def test_detail_of_a_failed_point_has_its_verdict_and_no_summary(store):
    row = _row(10.0)

    detail = db_worker.dispatch({'kind': 'detail', 'source': 'local',
                                 'node_uuid': row['node_uuid']})

    assert detail['ok'] is True
    assert detail['verdict']['attributes']['status'] == 'failed_integrity'
    assert detail['summary'] is None, 'a failed point is never summarised'


# --- export ---------------------------------------------------------------------

def _selection(*loads):
    return [{key: _row(load)[key] for key in ('node_uuid', 'point_uuid', 'set_label')}
            for load in loads]


def test_export_results_copies_the_point_files_under_its_rel_path(store, tmp_path):
    row = _row(30.0)

    result = db_worker.dispatch({'kind': 'export', 'source': 'local', 'what': 'results',
                                 'dest': str(tmp_path / 'out'),
                                 'selection': _selection(30.0)})

    assert result['ok'] is True, result.get('error')
    assert result['files']
    run_dir = tmp_path / 'out' / row['rel_path']
    assert all(str(f).startswith(str(run_dir)) for f in result['files'])
    assert any('results' in f for f in result['files'])


def test_export_provenance_copies_the_bundle_once_per_run(store, tmp_path):
    row = _row(30.0)

    result = db_worker.dispatch({'kind': 'export', 'source': 'local',
                                 'what': 'provenance', 'dest': str(tmp_path / 'out'),
                                 'selection': _selection(30.0, 50.0)})

    assert result['ok'] is True, result.get('error')
    names = sorted(f.split(row['rel_path'] + '/', 1)[1] for f in result['files'])
    assert 'provenance.json' in names
    assert len(names) == len(set(names)), 'two points of one run: one bundle'


def test_export_run_tree_writes_the_whole_set(store, tmp_path):
    result = db_worker.dispatch({'kind': 'export', 'source': 'local',
                                 'what': 'run_tree', 'dest': str(tmp_path / 'out'),
                                 'selection': _selection(30.0)})

    assert result['ok'] is True, result.get('error')
    assert result['files'] and result['warnings']


# --- import ---------------------------------------------------------------------

def test_import_maps_the_real_report_fields(aiida_profile_clean, no_profile_switch,
                                            tmp_path):
    _tree(tmp_path / 'tree')

    result = db_worker.dispatch({'kind': 'import', 'source': 'local',
                                 'root': str(tmp_path / 'tree'), 'label': 'camp'})

    assert result['ok'] is True, result.get('error')
    assert result['ingest_ok'] is True
    group = orm.Group.collection.get(label='friction2d/camp')
    assert result['group_uuid'] == group.uuid
    assert result['n_points'] == 3
    assert result['verdict_counts'] == {'complete': 3}
    assert result['no_version'] == [] and result['failures'] == []
    assert 'set_uuid' not in result and 'empty_runs' not in result


# --- the plot source ------------------------------------------------------------

def test_a_dataset_plot_source_loads_the_dataset_rows(store, tmp_path):
    plotter = Plotter(data_dirs=['aiida://dataset/paper'], labels=['paper'],
                      output_dir=str(tmp_path / 'plots'))

    frame = plotter._get_summary_data_df()  # pylint: disable=protected-access

    assert sorted(frame['force']) == [30.0, 50.0], 'the failed point is not in it'
    assert set(frame['dataset_label']) == {'paper'}
    assert set(frame['id']) == {'h_MoS2'}
    assert set(frame['size']) == {'100x100y'}
    assert set(frame['family']) == {'afm'}
    assert all(math.isnan(p) for p in frame['pressure'])
    assert frame['cof'].notna().all()
    assert (frame['cof'] == frame['mean_cof']).all()
    assert plotter.dataset_units['paper']['afm']['mean_nf'] == 'nN'
    assert set(frame['load_unit']) == {'nN'}


def test_an_unknown_dataset_is_refused(store, tmp_path):
    with pytest.raises(PlotDataError, match='nope'):
        Plotter(data_dirs=['aiida://dataset/nope'], labels=['x'],
                output_dir=str(tmp_path / 'plots'))


def test_dataset_timeseries_equals_export(store, tmp_path):
    """A time-series plot of `aiida://dataset/paper` sees the same frames as one
    of the set exported by hand -- minus the point the dataset leaves out."""
    import pandas as pd  # pylint: disable=import-outside-toplevel

    from src.aiida.export import export_results  # pylint: disable=import-outside-toplevel
    from src.postprocessing.read_data import DataReader  # pylint: disable=import-outside-toplevel

    export_results('camp', tmp_path / 'by_hand')
    DataReader(results_dir=str(tmp_path / 'by_hand')).export_full_data_to_json()
    by_hand = Plotter(data_dirs=[str(tmp_path / 'by_hand')], labels=['x'],
                      output_dir=str(tmp_path / 'p1'))
    stored = Plotter(data_dirs=['aiida://dataset/paper'], labels=['paper'],
                     output_dir=str(tmp_path / 'p2'))

    (key,) = by_hand.full_data_files['x']
    expected = {r['load']: r['df'] for r in by_hand._extract_all_runs('x', key)}
    got = {r['load']: r['df'] for r in stored._extract_all_runs('paper', key)}

    assert sorted(got) == [30.0, 50.0], 'the failed 10 N point is not in the dataset'
    for load, frame in got.items():
        pd.testing.assert_frame_equal(frame, expected[load])
    assert not (tmp_path / 'p2').exists() or not any((tmp_path / 'p2').rglob('*.json'))


def test_dataset_timeseries_reads_only_the_points_the_plot_selects(store, tmp_path,
                                                                   monkeypatch):
    from src.aiida import query  # pylint: disable=import-outside-toplevel

    read = []
    real = query.point_frames
    monkeypatch.setattr(query, 'point_frames',
                        lambda calc, point: read.append(point) or real(calc, point))
    stored = Plotter(data_dirs=['aiida://dataset/paper'], labels=['paper'],
                     output_dir=str(tmp_path / 'p'))
    runs = list(stored._extract_all_runs('paper', '100x100y',
                                         plot_config={'filter_forces': [30]}))
    assert [r['load'] for r in runs] == [30.0]
    assert len(read) == 1
