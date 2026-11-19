"""Stored analysis results against a real store (plan Task 1).

The store is the one of `tests/aiida/test_store_consumers.py`: one AFM run at
10/30/50 nN imported as set `camp` (10 nN fails its integrity check, so the
dataset `paper` holds 30 and 50 nN).
"""
import math

import pandas as pd
import pytest
from pydantic import BaseModel

pytest.importorskip('aiida')
pytest.importorskip('scipy')
pytestmark = pytest.mark.aiida

from aiida import orm  # noqa: E402

from src.aiida.ingest import import_root  # noqa: E402
from src.aiida.steps import build_dataset, postprocess_group  # noqa: E402
from src.analysis import base  # noqa: E402
from src.analysis.base import Analysis, register  # noqa: E402
from src.analysis.source import RunSource  # noqa: E402
from src.analysis.store import (ANALYSIS_GROUP_PREFIX, find_result,  # noqa: E402
                                run_analysis, store_result)
from tests.aiida.test_steps_integrity import _tree  # noqa: E402

#: Floats a lossy CSV would bend.
AWKWARD = [0.1 + 0.2, 1.0 / 3.0, 1e-300, 123456789.12345678, -2.5e-17, math.pi]


@pytest.fixture
def no_profile_switch(monkeypatch):
    import src.aiida as aiida_pkg
    monkeypatch.setattr(aiida_pkg, 'load_aiida_profile', lambda profile=None: None)


def _write_extras(run_dir):
    """A settings include and a built structure, which `_tree` does not write."""
    (run_dir / 'lammps' / 'system.in.settings').write_text(
        'pair_style sw\n', encoding='utf-8')
    for deck in (run_dir / 'lammps').glob('*.in'):
        text = deck.read_text(encoding='utf-8')
        deck.write_text('include lammps/system.in.settings\n' + text, encoding='utf-8')
    build = run_dir / 'build'
    build.mkdir(exist_ok=True)
    # The run is L2 (`n_layers` 2): the stacked sheet is `<mat>_2.lmp`
    # (`construction/components.py:2636`); the others are decoys.
    (build / 'h-MoS2_2.lmp').write_text('# built sheet\n', encoding='utf-8')
    (build / 'h-MoS2_1.lmp').write_text('# one layer\n', encoding='utf-8')
    (build / 'tip.lmp').write_text('# tip\n', encoding='utf-8')


@pytest.fixture
def store(aiida_profile_clean, no_profile_switch, tmp_path):
    """One AFM run at 10/30/50 N imported as set `camp` (10 N fails its
    integrity check), summarised, with dataset `paper` built over it."""
    run_dir = _tree(tmp_path / 'tree', eject=(10.0,))
    _write_extras(run_dir)
    report = import_root(tmp_path / 'tree', label='camp')
    postprocess_group('camp')
    build_dataset('paper', ['camp'])
    return report


@pytest.fixture
def probe(monkeypatch):
    """A trivial registered analysis: one row per pass point."""
    monkeypatch.setattr(base, 'ANALYSES', dict(base.ANALYSES))
    calls = []

    @register
    class _Probe(Analysis):
        name = 'probe'
        version = '1'
        level = 'run'
        families = frozenset({'afm'})

        class Params(BaseModel):
            """The one knob."""
            scale: float = 1.0

        def compute(self, source, params):
            calls.append(params)
            rows = source.rows.reset_index(drop=True)
            return pd.DataFrame({
                'run_uuid': rows.run_uuid, 'material': 'h-MoS2',
                'layer': rows.layers, 'load': rows.load, 'load_kind': rows.load_kind,
                'angle': rows.angle,
                'value': [params.scale * AWKWARD[i % len(AWKWARD)]
                          for i in range(len(rows))],
            })

    _Probe.calls = calls
    return _Probe


def _results(dataset='paper'):
    group = orm.load_group(f'{ANALYSIS_GROUP_PREFIX}{dataset}')
    return [n for n in group.nodes if isinstance(n, orm.SinglefileData)]


def test_identical_rerun_reuses_the_result(store, probe):
    first = run_analysis('probe', 'paper')
    second = run_analysis('probe', 'paper', {'scale': 1.0})

    assert len(probe.calls) == 1
    pd.testing.assert_frame_equal(first, second)
    (node,) = _results()
    extras = node.base.extras.all
    assert extras['analysis'] == 'probe' and extras['version'] == '1'
    assert extras['dataset'] == 'paper'
    assert extras['n_rows'] == 2 and extras['n_dropped'] == 0
    func = node.creator
    assert func.process_label == 'record_analysis'
    assert func.inputs.dataset.base.extras.get('friction2d_dataset') == 'paper'
    assert func.inputs.params.get_dict()['params'] == {'scale': 1.0}


def test_forced_rerun_computes_again(store, probe):
    run_analysis('probe', 'paper')
    run_analysis('probe', 'paper', force=True)

    assert len(probe.calls) == 2
    assert len(_results()) == 2


def test_new_params_make_a_new_result(store, probe):
    one = run_analysis('probe', 'paper')
    two = run_analysis('probe', 'paper', {'scale': 2.0})

    assert len(probe.calls) == 2
    assert len(_results()) == 2
    assert list(two.value) == [2.0 * v for v in one.value]
    hashes = {n.base.extras.get('params_hash') for n in _results()}
    assert len(hashes) == 2
    for digest in hashes:
        assert find_result('probe', 'paper', params_hash=digest) is not None
    assert find_result('probe', 'paper', version='2') is None
    assert find_result('other', 'paper') is None


def test_result_round_trips_floats_exactly(store, probe):
    table = pd.DataFrame({'run_uuid': ['r'] * len(AWKWARD), 'value': AWKWARD,
                          'n': list(range(len(AWKWARD)))})

    uuid = store_result(probe, 'paper', {'scale': 3.0}, table,
                        dropped=[('p9', 'failed_integrity')])

    back = find_result('probe', 'paper')
    assert list(back.value) == AWKWARD            # exact, not approx
    pd.testing.assert_frame_equal(back, table)
    node = orm.load_node(uuid)
    assert node.base.extras.get('n_dropped') == 1
    assert node.base.attributes.get('dropped') == [['p9', 'failed_integrity']]


def test_store_source_reads_frames_snapshot_settings_and_structure(store):
    source = RunSource.from_dataset('paper')

    assert source.family == 'afm'
    assert sorted(source.rows.load) == [30.0, 50.0]
    row = next(source.rows.itertuples())
    assert 'window' in source.frames(row)
    final = source.snapshot(row, 'final')
    assert final is not None and final.read_text().strip()
    assert final.name == f'final_f{row.load:g}N_a{row.angle:g}.data'
    loaded = source.snapshot(row, 'load')
    assert loaded is not None and loaded.name == f'load_{row.load:g}N.data'
    assert loaded.read_text().strip()
    assert source.settings_text(row) == 'pair_style sw\n'
    assert source.structure_text(row) == '# built sheet\n'
