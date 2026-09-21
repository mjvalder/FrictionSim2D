"""Every PES grid states its own family in the CSV header (spec §3 F1)."""
from pathlib import Path

import pandas as pd
import pytest

from src.core.observables import FAMILIES
from src.postprocessing.pes_scan import grid_descriptors

TEMPLATES = Path(__file__).resolve().parents[2] / 'src' / 'templates'
CASES = [
    ('afm/pes_scan.lmp', 'pes_tip'),
    ('afm_freestanding/pes_scan.lmp', 'pes_tip'),
    ('sheetonsheet/pes_scan.lmp', 'pes_sheet'),
    ('hetero/pes_scan.lmp', 'pes_sheet'),
]


@pytest.mark.parametrize('rel, family', CASES)
def test_template_writes_the_family_line_first(rel, family):
    text = (TEMPLATES / rel).read_text(encoding='utf-8')
    assert family in FAMILIES
    assert f'print           "# family={family}" file {{{{ results_csv }}}} screen no' in text
    # the column header must now APPEND, or it truncates the family line away
    assert 'file {{ results_csv }} screen no\nprint' in text
    assert text.count('file {{ results_csv }} screen no') == 1


def test_reader_skips_the_family_line(tmp_path):
    csv = tmp_path / 'pes_scan.csv'
    csv.write_text(
        '# family=pes_sheet\ndx,dy,energy_eV\n'
        '0,0,-1.0\n1,0,-0.5\n0,1,-0.6\n1,1,-0.2\n',
        encoding='utf-8')

    df = pd.read_csv(csv, comment='#')
    out = grid_descriptors(df, 'sheet')

    assert list(df.columns) == ['dx', 'dy', 'energy_eV']
    assert out['pes_barrier_U'] == pytest.approx(0.8)
