"""`tests.helpers.synthetic_results` must build a PES grid a real reader can
use, not a degenerate tiered-friction file (fix round 1, Task 9 review).

`write_results` builds `friction_*/raw_*/peratom_*` files from
`src.core.observables.rename_map(family, tier)`, but neither PES family has
any `Observable` registered (`rename_map` returns `{}` for both), so it must
refuse them instead of silently writing a header with no columns.
`make_run` must route PES families to `write_pes_grid` instead, producing the
single `results/pes_scan.csv` `src.postprocessing.pes_scan` actually reads.
"""
import pandas as pd
import pytest

from src.postprocessing.pes_scan import KIND_BY_TOPDIR, grid_descriptors
from tests.helpers.synthetic_results import make_run, write_pes_grid, write_results

_PES_RUNS = {
    'pes_tip': 'pes_tip/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300',
    'pes_sheet': 'pes_sheet/h-MoS2/100x_100y/K300',
}
_EXPECTED_COLUMNS = {
    'pes_tip': {'x', 'y', 'energy_eV', 'fx', 'fy', 'fz'},
    'pes_sheet': {'dx', 'dy', 'energy_eV'},
}


@pytest.mark.parametrize('family', ['pes_tip', 'pes_sheet'])
def test_write_results_refuses_pes_families(tmp_path, family):
    with pytest.raises(ValueError, match='pes_scan.csv'):
        write_results(tmp_path / 'run', family)


@pytest.mark.parametrize('family', ['pes_tip', 'pes_sheet'])
def test_write_pes_grid_refuses_non_pes_families(tmp_path, family):
    other = 'sheetonsheet' if family == 'pes_tip' else 'afm'
    with pytest.raises(ValueError, match='write_pes_grid is for'):
        write_pes_grid(tmp_path / 'run', other)


@pytest.mark.parametrize('family', ['pes_tip', 'pes_sheet'])
def test_make_run_writes_a_grid_pes_scan_reads(tmp_path, family):
    run_dir = make_run(tmp_path, _PES_RUNS[family], family)

    csv_path = run_dir / 'results' / 'pes_scan.csv'
    assert csv_path.is_file()
    # no tiered friction file leaks in alongside the grid
    assert list((run_dir / 'results').glob('friction_*')) == []
    assert csv_path.read_text(encoding='utf-8').splitlines()[0] == f'# family={family}'

    df = pd.read_csv(csv_path, comment='#')
    assert _EXPECTED_COLUMNS[family] <= set(df.columns)
    assert len(df) >= 4

    # this is exactly what src.postprocessing.pes_scan.extract_all does per row
    feats = grid_descriptors(df, KIND_BY_TOPDIR[family])
    assert 'pes_barrier_U' in feats
    if family == 'pes_tip':
        assert 'pes_fmax' in feats
