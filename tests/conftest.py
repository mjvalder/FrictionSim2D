"""Pytest fixtures and configuration for FrictionSim2D tests."""

import os
import pytest
from pathlib import Path
from importlib import resources
from unittest.mock import MagicMock

from src.core.config import (
    GlobalSettings,
    GeometrySettings,
    ThermostatSettings,
    SimulationSettings,
    QuenchSettings,
    OutputSettings,
    PotentialSettings,
    AiidaSettings,
    ComponentConfig,
    TipConfig,
    SubstrateConfig,
    SheetConfig,
)


@pytest.fixture(scope="session", autouse=True)
def isolated_xdg_config_home(tmp_path_factory):
    """Point ``$XDG_CONFIG_HOME`` at a session temp dir.

    ``load_settings()`` falls back to ``$XDG_CONFIG_HOME/FrictionSim2D/settings.yaml``
    (``src/core/config.py::_global_settings_path``). Without this the suite would
    read the developer's real settings file when one exists. Session scope so
    ``monkeypatch``-based tests can still override per test.
    """
    xdg = tmp_path_factory.mktemp("xdg_config_home")
    previous = os.environ.get("XDG_CONFIG_HOME")
    os.environ["XDG_CONFIG_HOME"] = str(xdg)
    yield xdg
    if previous is None:
        os.environ.pop("XDG_CONFIG_HOME", None)
    else:
        os.environ["XDG_CONFIG_HOME"] = previous


@pytest.fixture
def mock_settings() -> GlobalSettings:
    """Creates mock GlobalSettings for testing without loading files."""
    return GlobalSettings(
        geometry=GeometrySettings(
            tip_reduction_factor=2.0,
            rigid_tip=True,
            lat_c_default=6.0,
        ),
        thermostat=ThermostatSettings(
            type='langevin',
            time_int_type='nvt',
            langevin_boundaries={},
        ),
        simulation=SimulationSettings(
            timestep=0.001,
            thermo=100,
            min_style='cg',
            minimization_command='minimize 1e-6 1e-6 1000 10000',
            neighbor_list=2.0,
            neigh_modify_command='neigh_modify every 1 delay 0 check yes',
            slide_run_steps=100000,
            drive_method='smd',
        ),
        quench=QuenchSettings(
            run_local=True,
            n_procs=4,
            quench_slab_dims=[50, 50, 50],
            quench_rate=10.0,
            quench_melt_temp=3500.0,
            quench_target_temp=300.0,
            timestep=0.001,
            melt_steps=50000,
            equilibrate_steps=20000,
        ),
        output=OutputSettings(
            dump={'atom': True, 'force': True},
            dump_frequency={'slide': 1000, 'minimize': 100},
            results_frequency=100,
        ),
        potential=PotentialSettings(
            LJ_cutoff=11.0,
            LJ_type='lj/cut',
        ),
        aiida=AiidaSettings(
            lammps_code_label='lammps@localhost',
            postprocess_code_label='python@localhost',
            postprocess_script_path='/path/to/script',
            num_cpus=4,
            walltime_seconds=3600,
        ),
    )


@pytest.fixture
def mock_settings_no_langevin(mock_settings) -> GlobalSettings:
    """GlobalSettings with Nose-Hoover thermostat (no Langevin)."""
    mock_settings.thermostat.type = 'nose-hoover'
    return mock_settings


@pytest.fixture
def data_path() -> Path:
    """Returns the path to package data directory."""
    return Path(str(resources.files('src.data')))


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Legacy fixture alias used by older tests."""
    return tmp_path


@pytest.fixture
def package_materials_path() -> Path:
    """Canonical home of shipped material data (``src/data/materials``).

    This is the only copy that survives installation -- the sdist ships
    ``src/**`` only -- and it is what ``get_amorphous_path`` resolves via
    ``importlib.resources``. Use this for ``.lmp`` structures; use
    ``materials_path`` for the ``.cif`` example library.
    """
    return Path(str(resources.files('src.data'))) / 'materials'


@pytest.fixture
def materials_path() -> Path:
    """Returns the path to examples materials directory."""
    return Path(__file__).resolve().parents[1] / 'examples' / 'materials'


@pytest.fixture
def potentials_path() -> Path:
    """Returns the path to examples potentials directory."""
    return Path(__file__).resolve().parents[1] / 'examples' / 'potentials'


# =========================================================================
# Component Config Fixtures
# =========================================================================

@pytest.fixture
def si_tip_config(materials_path, potentials_path) -> TipConfig:
    """Silicon tip configuration using SW potential."""
    return TipConfig(
        mat='Si',
        pot_type='sw',
        pot_path=str(potentials_path / 'sw' / 'Si.sw'),
        cif_path=str(materials_path / 'Si.cif'),
        r=20.0,
        amorph='c',
        dspring=0.1,
        s=1.0,
    )


@pytest.fixture
def si_sub_config(materials_path, potentials_path) -> SubstrateConfig:
    """Silicon substrate configuration using SW potential."""
    return SubstrateConfig(
        mat='Si',
        pot_type='sw',
        pot_path=str(potentials_path / 'sw' / 'Si.sw'),
        cif_path=str(materials_path / 'Si.cif'),
        thickness=10.0,
        amorph='c',
    )


@pytest.fixture
def mos2_sheet_config(materials_path, potentials_path) -> SheetConfig:
    """MoS2 sheet configuration using SW potential."""
    return SheetConfig(
        mat='h-MoS2',
        pot_type='sw',
        pot_path=str(potentials_path / 'sw' / 'MoS2_wen.sw'),
        cif_path=str(materials_path / 'h-MoS2.cif'),
        x=50.0,
        y=50.0,
        layers=[1],
    )


@pytest.fixture  
def mos2_multilayer_config(materials_path, potentials_path) -> SheetConfig:
    """MoS2 4-layer sheet configuration for sheet-on-sheet."""
    return SheetConfig(
        mat='h-MoS2',
        pot_type='sw',
        pot_path=str(potentials_path / 'sw' / 'MoS2_wen.sw'),
        cif_path=str(materials_path / 'h-MoS2.cif'),
        x=50.0,
        y=50.0,
        layers=[1, 2, 3, 4],
    )


@pytest.fixture
def silicene_sheet_config(materials_path, potentials_path) -> SheetConfig:
    """Silicene sheet configuration using SW potential."""
    return SheetConfig(
        mat='silicene',
        pot_type='sw',
        pot_path=str(potentials_path / 'sw' / 'sw_lammps' / 'silicene.sw'),
        cif_path=str(materials_path / 'silicene.cif'),
        x=50.0,
        y=50.0,
        layers=[1],
    )


# =========================================================================
# Mock CIF Data Fixtures (to avoid file I/O in unit tests)
# =========================================================================

@pytest.fixture
def mock_cifread_si(monkeypatch):
    """Mocks cifread to return Silicon data."""
    def mock_cif(*args, **kwargs):
        return {'elements': ['Si']}
    monkeypatch.setattr('src.potentials.manager.cifread', mock_cif)


@pytest.fixture
def mock_cifread_mos2(monkeypatch):
    """Mocks cifread to return MoS2 data."""
    def mock_cif(*args, **kwargs):
        return {'elements': ['Mo', 'S']}
    monkeypatch.setattr('src.potentials.manager.cifread', mock_cif)


@pytest.fixture
def mock_count_atomtypes_si(monkeypatch):
    """Mocks count_atomtypes for Silicon (1 type per element)."""
    def mock_count(*args, **kwargs):
        return {'Si': 1}
    monkeypatch.setattr('src.potentials.manager.count_atomtypes', mock_count)


@pytest.fixture
def mock_count_atomtypes_mos2(monkeypatch):
    """Mocks count_atomtypes for MoS2 (SW has 6 types: Mo1-6, S1-6)."""
    def mock_count(*args, **kwargs):
        return {'Mo': 6, 'S': 6}
    monkeypatch.setattr('src.potentials.manager.count_atomtypes', mock_count)


@pytest.fixture
def mock_count_atomtypes_mos2_simple(monkeypatch):
    """Mocks count_atomtypes for MoS2 with simple 1 type per element."""
    def mock_count(*args, **kwargs):
        return {'Mo': 1, 'S': 1}
    monkeypatch.setattr('src.potentials.manager.count_atomtypes', mock_count)


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Tag every ``lmp_serial``-guarded test with the ``lammps`` marker.

    The guards (42 sites, 12 files) predate the marker registered in
    pyproject.toml. Deriving the marker from the guard keeps the two from
    drifting apart, instead of hand-editing every site.
    """
    for item in items:
        for mark in item.iter_markers(name="skipif"):
            if "lmp_serial" in str(mark.kwargs.get("reason", "")):
                item.add_marker(pytest.mark.lammps)
                break
