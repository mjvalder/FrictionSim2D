"""What the wheel ships: the library and the one amorphous block, under 20 MB.

With hatchling importable the real wheel is built (offline, no isolation) and
listed. Without it -- the dev env does not carry it -- the wheel is emulated:
hatchling's ``packages = ["frictionsim2d"]`` takes every git-tracked file under
``frictionsim2d/``, so those files are deflated into a zip the same way. No
network either way.
"""
import importlib.util
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import List

import pytest

REPO = Path(__file__).resolve().parents[2]
MAX_WHEEL_BYTES = 20 * 1024 * 1024
SHIPPED_BLOCK = 'frictionsim2d/data/materials/amor_Si__021d81ab.lmp'


def _tracked_package_files() -> List[str]:
    if shutil.which('git') is None or not (REPO / '.git').exists():
        pytest.skip('needs a git checkout to know what hatchling packs')
    out = subprocess.run(['git', 'ls-files', '-z', '--', 'frictionsim2d'], cwd=REPO,
                         capture_output=True, text=True, check=True).stdout
    return [name for name in out.split('\0') if name and (REPO / name).is_file()]


def _wheel(tmp_path: Path) -> Path:
    if importlib.util.find_spec('hatchling') is not None:
        from hatchling.builders.wheel import WheelBuilder  # pylint: disable=import-outside-toplevel
        return Path(next(WheelBuilder(str(REPO)).build(directory=str(tmp_path),
                                                       versions=['standard'])))
    path = tmp_path / 'emulated.whl'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in _tracked_package_files():
            archive.write(REPO / name, name)
    return path


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
