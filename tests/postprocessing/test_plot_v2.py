"""The plot stack reads the v2 export and refuses v1 (Q1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.postprocessing.plot_data import PlotDataError, Plotter
from src.postprocessing.read_data import DataReader
from tests.helpers.results_tree import make_family_run


def _dataset(tmp_path: Path, family: str = 'afm', **kw) -> Path:
    """A workspace directory with outputs/output_full_*.json + summary csv."""
    root = tmp_path / f'ds_{family}'
    make_family_run(root, family, **kw)
    reader = DataReader(results_dir=str(root))
    reader.export_full_data_to_json()
    reader.export_summary_csv()
    return root


def test_extract_all_runs_yields_one_dict_per_run(tmp_path):
    root = _dataset(tmp_path, 'afm', loads=(10.0, 30.0), angles=(0.0, 90.0),
                    pot='sw_a')
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))

    runs = list(plotter._extract_all_runs('ds', '100x100y'))  # pylint: disable=protected-access

    assert len(runs) == 4
    one = runs[0]
    assert one['id'] == 'h_MoS2'
    assert one['family'] == 'afm'
    assert one['model'] == 'afm'
    assert one['potential'] == 'pot_sw_a'
    assert one['temperature'] == 300.0
    assert one['load_kind'] == 'force'
    assert one['force'] == one['load']
    assert one['pressure'] is None
    assert one['layer'] == 2
    assert one['speed'] == 2.0
    assert 'lf' in one['df'].columns
    assert 'time' in one['df'].columns


def test_a_sweep_arm_is_not_lost(tmp_path):
    """Two runs differing only in the potential collapsed to one leaf under
    v1; the v2 runs list cannot collapse."""
    root = tmp_path / 'ds'
    make_family_run(root, 'afm', pot='sw_a')
    make_family_run(root, 'afm', pot='sw_b')
    reader = DataReader(results_dir=str(root))
    reader.export_full_data_to_json()
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))

    runs = list(plotter._extract_all_runs('ds', '100x100y'))

    assert sorted(r['potential'] for r in runs) == ['pot_sw_a', 'pot_sw_b']


def test_a_pressure_run_reports_pressure_not_force(tmp_path):
    root = _dataset(tmp_path, 'sheetonsheet')
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))

    run, = plotter._extract_all_runs('ds', '100x100y')

    assert run['load_kind'] == 'pressure'
    assert run['pressure'] == run['load']
    assert run['force'] is None


def test_a_dataset_directory_that_is_not_there_is_refused(tmp_path):
    """`_discover_data_files` used to log an error and carry on, so a typo in
    a data_dir produced an empty summary frame, an early return in
    `_generate_summary_plot` and a silent no-op -- a plot that never appeared
    and never said why. This is also what makes the worker's and the GUI's
    `failed` path reachable at all."""
    with pytest.raises(PlotDataError, match='no output_full'):
        Plotter(data_dirs=[str(tmp_path / 'typo')], labels=['ds'],
                output_dir=str(tmp_path / 'out'))


def test_an_empty_dataset_list_is_still_fine(tmp_path):
    """`Plotter(data_dirs=[], labels=[])` is how several unit tests reach the
    mixins' helpers; refusing an ABSENT directory must not refuse NO
    directories."""
    plotter = Plotter(data_dirs=[], labels=[], output_dir=str(tmp_path))

    assert plotter.full_data_files == {}


def test_a_db_uri_is_not_checked_for_files(tmp_path):
    """The `db://` branch returns before the file check, so a DB-backed
    dataset is never refused for having no output_full_*.json. Its own query
    failure is caught and logged by `_load_db_labels`; Task 22 rewires that
    to the v3 client."""
    plotter = Plotter(data_dirs=['db://central?family=afm'], labels=['db'],
                      output_dir=str(tmp_path / 'out'))

    assert plotter.full_data_files == {'db': {}}


def test_a_v1_document_is_refused(tmp_path):
    outputs = tmp_path / 'legacy' / 'outputs'
    outputs.mkdir(parents=True)
    (outputs / 'output_full_100x100y.json').write_text(json.dumps({
        'metadata': {}, 'results': {'MoS2': {'f10.0': {'a0': {
            'columns': ['lfx', 'lfy', 'nf'], 'data': [[1.0, 0.0, 10.0]]}}}},
    }), encoding='utf-8')

    with pytest.raises(PlotDataError, match='schema'):
        Plotter(data_dirs=[str(tmp_path / 'legacy')], labels=['old'],
                output_dir=str(tmp_path / 'out'))


def test_the_refusal_says_how_to_fix_it(tmp_path):
    outputs = tmp_path / 'legacy' / 'outputs'
    outputs.mkdir(parents=True)
    (outputs / 'output_full_100x100y.json').write_text(
        json.dumps({'metadata': {}, 'results': {}}), encoding='utf-8')

    with pytest.raises(PlotDataError, match=r'postprocess read <root> --export'):
        Plotter(data_dirs=[str(tmp_path / 'legacy')], labels=['old'],
                output_dir=str(tmp_path / 'out'))


def test_metadata_merges_across_datasets(tmp_path):
    a = _dataset(tmp_path, 'afm', material='h-MoS2')
    b = tmp_path / 'ds_b'
    make_family_run(b, 'afm', material='h-WS2')
    DataReader(results_dir=str(b)).export_full_data_to_json()

    plotter = Plotter(data_dirs=[str(a), str(b)], labels=['a', 'b'],
                      output_dir=str(tmp_path / 'out'))

    assert sorted(plotter.metadata['materials']) == ['h_MoS2', 'h_WS2']
    assert plotter.material_type_map['h_MoS2'] == 'h_type'


def test_the_tier_field_selects_the_frame(tmp_path):
    root = _dataset(tmp_path, 'afm', tiers=('window', 'raw'))
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'],
                      output_dir=str(tmp_path / 'out'))

    window, = plotter._extract_all_runs('ds', '100x100y')
    raw, = plotter._extract_all_runs('ds', '100x100y', tier='raw')

    assert 'nf' in window['df'].columns
    assert 'raw_nf' in raw['df'].columns


def test_written_starts_empty_and_records_what_save_plot_writes(tmp_path):
    """Asserted through `_save_plot` directly, NOT through a whole summary
    plot: at this point in the plan the summary path still runs the old
    `df.mean()` `_calculate_summary_statistics`, and a test that quietly
    depended on that tolerating the new run records would be pinning the wrong
    thing. Task 19 rewrites that path and re-asserts `written` end to end."""
    import matplotlib.pyplot as plt

    root = _dataset(tmp_path, 'afm', loads=(10.0, 20.0, 30.0))
    out = tmp_path / 'out'
    plotter = Plotter(data_dirs=[str(root)], labels=['ds'], output_dir=str(out))
    assert plotter.written == []

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    plotter._save_plot(fig, 'lf_vs_load.png')  # pylint: disable=protected-access

    assert plotter.written == [out / 'lf_vs_load.png']
