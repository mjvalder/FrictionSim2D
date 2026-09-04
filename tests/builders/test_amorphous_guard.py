"""Render must never be able to start a melt-quench.

Generating an amorphous structure shells out to ``mpiexec -np N lmp``, takes
minutes, and writes ``amor_<mat>.lmp`` into the installed package data
directory ``src/data/materials`` -- which, in an editable install, is the git
working tree. An interactive caller triggering that by accident is the worst
failure mode in the GUI, so the refusal lives in the builder rather than in a
GUI-side pre-check, and it has to happen before anything is written.

A cache *hit* is not the same as "no generation": ``make_amorphous`` also
compares the cached box against the requested size and regenerates when it is
too small.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import pytest

from src.builders.construction.components import (
    AmorphousNotCachedError,
    get_amorphous_path,
    make_amorphous,
)
from src.core.config import GlobalSettings
from src.builders.utils.lammps_data import get_model_dimensions

MATERIALS_DIR = Path(__file__).resolve().parents[2] / 'src' / 'data' / 'materials'

#: The one material with a structure shipped in the repo.
CACHED_MATERIAL = 'Si'
#: Canonical shipped-material dir; the only copy that survives install.
PACKAGE_MATERIALS = Path(str(resources.files('src.data'))) / 'materials'
SI_STRUCTURE = PACKAGE_MATERIALS / 'Si.lmp'


def _cached_extent(mat: str) -> tuple:
    path = get_amorphous_path(mat)
    assert path is not None, f"expected a cached amorphous {mat}"
    dims = get_model_dimensions(path)
    return (dims['xhi'] - dims['xlo'],
            dims['yhi'] - dims['ylo'],
            dims['zhi'] - dims['zlo'])


class TestMakeAmorphousRefusesToGenerate:

    @pytest.fixture(name='guard')
    def _guard(self, tmp_path):
        """Call make_amorphous with generation forbidden, for a too-large box."""
        x, y, z = _cached_extent(CACHED_MATERIAL)
        cif = SI_STRUCTURE

        def call(output_dir: Path = None):
            return make_amorphous(
                CACHED_MATERIAL, cif,
                target_x=x * 2, target_y=y, target_z=z,
                pot_path=Path('examples/potentials/sw/Si.sw'),
                pot_type='sw',
                output_dir=output_dir or tmp_path,
                settings=GlobalSettings(),
                allow_generate=False,
            )
        return call

    def test_it_raises(self, guard):
        with pytest.raises(AmorphousNotCachedError, match='allow_generate=False'):
            guard()

    def test_it_writes_nothing_to_the_output_directory(self, guard, tmp_path):
        out = tmp_path / 'build'
        out.mkdir()
        with pytest.raises(AmorphousNotCachedError):
            guard(out)
        assert list(out.iterdir()) == [], "refusal must happen before any file is written"

    def test_it_does_not_touch_the_package_materials_directory(self, guard):
        before = {p.name: p.stat().st_mtime_ns for p in MATERIALS_DIR.iterdir()}
        with pytest.raises(AmorphousNotCachedError):
            guard()
        after = {p.name: p.stat().st_mtime_ns for p in MATERIALS_DIR.iterdir()}
        assert after == before

    def test_a_usable_cache_still_returns_without_generating(self, tmp_path):
        x, y, z = _cached_extent(CACHED_MATERIAL)
        result = make_amorphous(
            CACHED_MATERIAL, SI_STRUCTURE,
            target_x=x / 2, target_y=y / 2, target_z=z / 2,
            pot_path=Path('examples/potentials/sw/Si.sw'), pot_type='sw',
            output_dir=tmp_path, settings=GlobalSettings(),
            allow_generate=False,
        )
        assert result == get_amorphous_path(CACHED_MATERIAL)


def test_allow_generate_defaults_to_true_so_the_cli_is_unchanged():
    import inspect

    from src.builders import components

    for func in (components.make_amorphous, components.build_tip,
                 components.build_substrate, components._create_base_slab):
        param = inspect.signature(func).parameters['allow_generate']
        assert param.default is True, f"{func.__name__} changed CLI behaviour"
