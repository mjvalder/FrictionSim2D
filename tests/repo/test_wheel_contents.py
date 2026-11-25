"""What the wheel ships: the library and the one amorphous block, under 20 MB.

Builds the real wheel with hatchling's WheelBuilder (offline, no isolation, not
installed) and lists it; skipped where hatchling is not installed.
"""
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MAX_WHEEL_BYTES = 20 * 1024 * 1024
SHIPPED_BLOCK = 'frictionsim2d/data/materials/amor_Si__021d81ab.lmp'


def _wheel(tmp_path: Path) -> Path:
    wheel = pytest.importorskip('hatchling.builders.wheel')
    return Path(next(wheel.WheelBuilder(str(REPO)).build(directory=str(tmp_path),
                                                         versions=['standard'])))


@pytest.fixture(scope='module', name='wheel')
def _built(tmp_path_factory):
    path = _wheel(tmp_path_factory.mktemp('wheel'))
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
    return path, names


def test_wheel_contents(wheel):
    path, names = wheel
    amorphous = sorted(n for n in names if Path(n).name.startswith('amor_'))
    assert amorphous == [SHIPPED_BLOCK, SHIPPED_BLOCK + '.json']
    assert path.stat().st_size < MAX_WHEEL_BYTES, f'{path.stat().st_size} bytes'


def test_wheel_carries_the_library_not_the_examples(wheel):
    _, names = wheel
    assert 'frictionsim2d/data/materials/h-MoS2.cif' in names
    assert 'frictionsim2d/data/potentials/sw/MoS2_wen.sw' in names
    assert 'frictionsim2d/data/potentials/ilp/TMD.ILP' in names
    assert 'frictionsim2d/postprocessing/presets/publication.json' in names
    assert not [n for n in names if n.startswith('examples/')]
