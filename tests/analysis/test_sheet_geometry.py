"""`sheet_geometry` (plan Task 3): the lattice period along the scan."""
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip('scipy')

import src.analysis.catalogue  # noqa: E402,F401  (registers the analyses)
from src.analysis.base import get_analysis  # noqa: E402
from src.analysis.source import RunSource  # noqa: E402
from src.analysis.sheet_geometry import read_lmp  # noqa: E402

#: The settings of a run whose sheet types are the structure's + 6 (s42:85).
SETTINGS = """\
mass 1*6 28.085000 #Si
mass 7 12.011 #C
mass 8 12.011 #C
group layer_1 type 7
group layer_2 type 8
pair_coeff 4*6 7*8 lj/cut 0.0044 3.4 # Si(tip)-C(sheet)
"""


def _lmp(box, atoms):
    lines = ['LAMMPS data file', '', f'{len(atoms)} atoms', '2 atom types', '',
             f'0 {box[0]} xlo xhi', f'0 {box[1]} ylo yhi', '-5 20 zlo zhi', '',
             'Masses', '', '1 12.011', '2 12.011', '', 'Atoms # atomic', '']
    lines += [f'{i} {t} {x} {y} {z} 0 0 0' for i, (t, x, y, z) in enumerate(atoms, 1)]
    return '\n'.join(lines) + '\n'


def _bilayer(points, box, gap=3.4):
    """The 2D points as layer 1 (z = 0, type 1) and layer 2 (z = gap, type 2)."""
    atoms = [(1, x, y, 0.0) for x, y in points] + [(2, x, y, gap) for x, y in points]
    return _lmp(box, atoms)


def _square(a, n=8):
    return [(i * a, j * a) for i in range(n) for j in range(n)], (n * a, n * a)


def _hexagonal(a, nx=8, ny=5):
    """Honeycomb with the zigzag rows along x, in its rectangular cell."""
    box = (nx * a, ny * math.sqrt(3) * a)
    pts = set()
    for m in range(2 * ny):
        for i in range(-m, nx + 1):
            for b in ((0, 0), (a / 2, a / (2 * math.sqrt(3)))):
                x = i * a + m * a / 2 + b[0]
                y = m * math.sqrt(3) * a / 2 + b[1]
                pts.add((round(x % box[0], 6) % box[0], round(y % box[1], 6) % box[1]))
    return sorted(pts), box


class _Fetch:
    def __init__(self, text):
        self.text = text

    def structure_text(self, row):
        return self.text

    def settings_text(self, row):
        return SETTINGS

    def close(self):
        pass


def _geometry(text):
    rows = pd.DataFrame([{'point_uuid': 'p1', 'verdict_status': 'complete', 'family': 'afm',
                          'materials': ['h-X'], 'layers': 2}])
    source = RunSource(rows, 'afm', fetch=_Fetch(text))
    return get_analysis('sheet_geometry')().compute(source, get_analysis('sheet_geometry').Params())


def test_registered_for_afm_and_sheet_families():
    cls = get_analysis('sheet_geometry')

    assert cls.level == 'material'
    assert cls.families == {'afm', 'sheetonsheet'}


def test_period_of_a_square_lattice():
    pts, box = _square(3.0)

    row = _geometry(_bilayer(pts, box)).iloc[0]

    assert row.period_x_top == pytest.approx(3.0, abs=0.002)
    assert row.d_nn_top == pytest.approx(3.0)
    assert row.r_hollow == pytest.approx(3.0 / math.sqrt(2), abs=0.01)
    assert row.d_interlayer == pytest.approx(3.4)
    assert row.top_element == 'C'
    assert row.top_sigma == pytest.approx(3.4)
    assert bool(row.x_is_nn)
    assert (row.material, row.layer) == ('h_X', 2)


def test_hexagonal_period_along_x():
    pts, box = _hexagonal(2.46)

    row = _geometry(_bilayer(pts, box)).iloc[0]

    assert row.period_x_top == pytest.approx(2.46, abs=0.002)
    assert row.d_nn_top == pytest.approx(2.46 / math.sqrt(3), abs=1e-3)
    assert row.rho_top == pytest.approx(4 / (2.46 * math.sqrt(3) * 2.46), rel=1e-3)


def test_one_row_per_material_and_layer_and_missing_structure_dropped():
    pts, box = _square(3.0)
    rows = pd.DataFrame([
        {'point_uuid': f'p{i}', 'verdict_status': 'complete', 'family': 'afm',
         'materials': ['M'], 'layers': 2} for i in range(3)])
    texts = iter([None, _bilayer(pts, box), _bilayer(pts, box)])

    class _Some(_Fetch):
        def structure_text(self, row):
            return next(texts)

    source = RunSource(rows, 'afm', fetch=_Some(None))
    table = get_analysis('sheet_geometry')().compute(
        source, get_analysis('sheet_geometry').Params())

    assert len(table) == 1  # the first run has no structure; the second is read
    assert source.dropped == []


def test_material_without_any_structure_is_dropped():
    rows = pd.DataFrame([{'point_uuid': 'p1', 'verdict_status': 'complete',
                          'family': 'afm', 'materials': ['M'], 'layers': 1}])
    source = RunSource(rows, 'afm', fetch=_Fetch(None))

    table = get_analysis('sheet_geometry')().compute(
        source, get_analysis('sheet_geometry').Params())

    assert table.empty
    assert source.dropped == [('p1', 'no_structure')]


_PUB_SCRIPT = """
import sys
import pandas as pd
from src.aiida import load_aiida_profile
load_aiida_profile('friction2d_pub')
from src.aiida.query import dataset_rows
from src.analysis.catalogue import sheet_geometry
from src.analysis.source import RunSource
rows, _ = dataset_rows('260909-afm')
with RunSource(pd.DataFrame(rows), 'afm') as source:
    cls = sheet_geometry.SheetGeometry
    cls().compute(source, cls.Params()).to_csv(sys.argv[1], index=False)
"""


def _table_from_pub_store(tmp_path):
    """The analysis over the read-only `friction2d_pub` store, in a subprocess:
    the aiida pytest plugin points this process's AIIDA_PATH at a temporary
    config, so the real profile is invisible here."""
    out = tmp_path / 'geometry.csv'
    env = {k: v for k, v in os.environ.items() if k != 'AIIDA_PATH'}
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[2])
    done = subprocess.run([sys.executable, '-c', _PUB_SCRIPT, str(out)], env=env,
                          capture_output=True, text=True, check=False)
    if done.returncode != 0:
        pytest.skip(f'friction2d_pub / dataset 260909-afm unavailable: {done.stderr[-300:]}')
    return pd.read_csv(out, float_precision='round_trip')


@pytest.mark.publication
def test_validates_against_frozen_ranking(tmp_path):
    frozen = Path('/home/matteo/mnt/PhD/Sync/Publication-data/publication_export/data/'
                  '260909_afm_v4/analysis/cof_ranking_v2.csv')
    if not frozen.exists():
        pytest.skip('frozen cof_ranking_v2.csv absent')
    pytest.importorskip('aiida')
    table = _table_from_pub_store(tmp_path)

    ranking = pd.read_csv(frozen, float_precision='round_trip').set_index('material')['a_x_A']
    bilayer = table[table.layer == 2].set_index('material')['period_x_top']
    common = sorted(set(ranking.index) & set(bilayer.index))
    bad = {m: (float(bilayer[m]), float(ranking[m])) for m in common
           if not (abs(bilayer[m] - ranking[m]) <= 1e-3)}
    print(f'{len(common) - len(bad)} / {len(common)} within 1e-3 A; '
          f'missing from table: {sorted(set(ranking.index) - set(bilayer.index))}; mismatches: {bad}')

    assert common
    assert not bad


def test_reads_every_point_whatever_its_verdict():
    # The built structure does not depend on the run's verdict.
    pts, box = _square(3.0)
    rows = pd.DataFrame([{'point_uuid': 'p1', 'verdict_status': 'failed_integrity',
                          'family': 'afm', 'materials': ['M'], 'layers': 2}])
    source = RunSource(rows, 'afm', fetch=_Fetch(_bilayer(pts, box)))

    table = get_analysis('sheet_geometry')().compute(
        source, get_analysis('sheet_geometry').Params())

    assert len(table) == 1 and table.iloc[0].period_x_top == pytest.approx(3.0, abs=0.002)
    assert source.dropped == [('p1', 'failed_integrity')]   # no second drop


def test_sheet_rows_without_layers_take_the_settings_layering():
    pts, box = _square(3.0)
    rows = pd.DataFrame([{'point_uuid': 'p1', 'verdict_status': 'complete',
                          'family': 'sheetonsheet', 'materials': ['M'], 'layers': None}])
    source = RunSource(rows, 'sheetonsheet', fetch=_Fetch(_bilayer(pts, box)))

    table = get_analysis('sheet_geometry')().compute(
        source, get_analysis('sheet_geometry').Params())

    assert table.iloc[0].layer == 2        # layer_1, layer_2 in SETTINGS
    assert table.iloc[0].period_x_top == pytest.approx(3.0, abs=0.002)


@pytest.mark.parametrize('style, fmt', [
    ('atomic', '{i} {t} {x} {y} {z}'),
    ('molecular', '{i} 0 {t} {x} {y} {z} 0 0 0'),   # the sheet family's build/<mat>_4.lmp
    ('full', '{i} 1 {t} 0.0 {x} {y} {z}'),
    ('charge', '{i} {t} 0.5 {x} {y} {z}'),
])
def test_read_lmp_follows_the_atom_style(style, fmt):
    text = ('LAMMPS data file\n\n2 atoms\n\n0 10 xlo xhi\n0 20 ylo yhi\n\n'
            f'Atoms # {style}\n\n'
            + '\n'.join(fmt.format(i=i, t=t, x=x, y=y, z=z) for i, (t, x, y, z)
                        in enumerate([(3, 1.0, 2.0, 3.0), (4, 4.0, 5.0, 6.0)], 1)) + '\n')

    box, typ, xyz = read_lmp(text)

    assert box == {'x': 10.0, 'y': 20.0}
    assert list(typ) == [3, 4]
    assert xyz.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
