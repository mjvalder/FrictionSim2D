"""Pytest fixtures and configuration for FrictionSim2D tests."""

import os
import pytest
from pathlib import Path
from importlib import resources

from src.core.config import (
    GlobalSettings,
    GeometrySettings,
    ThermostatSettings,
    SimulationSettings,
    QuenchSettings,
    OutputSettings,
    PotentialSettings,
    AiidaSettings,
    TipConfig,
    SubstrateConfig,
    SheetConfig,
)


@pytest.fixture(scope="session", autouse=True)
def isolated_xdg_config_home(tmp_path_factory):
    """Point ``$XDG_CONFIG_HOME`` at a session temp dir.

    ``load_settings()`` falls back to ``$XDG_CONFIG_HOME/FrictionSim2D/settings.yaml``
    (``src/core/config.py::_global_settings_path``). Without this the suite would
    read the developer's real settings file when one exists. Session scope: one temp
    dir for the whole run; per-test monkeypatch of the variable still works on top.
    """
    xdg = tmp_path_factory.mktemp("xdg_config_home")
    previous = os.environ.get("XDG_CONFIG_HOME")
    os.environ["XDG_CONFIG_HOME"] = str(xdg)
    yield xdg
    if previous is None:
        os.environ.pop("XDG_CONFIG_HOME", None)
    else:
        os.environ["XDG_CONFIG_HOME"] = previous


@pytest.fixture(autouse=True)
def fixed_lammps_pair_styles(monkeypatch):
    """Keep pot_type validation independent of the host's ``lmp`` binary.

    ``validate_pot_type`` asks ``src.interfaces.lammps.lammps_pair_styles`` (looked
    up through the module, so this patch reaches it) for custom styles. Without a
    fixed answer the result depends on whether ``lmp`` is on PATH. Tests needing
    ``None`` or the real parse override it.
    """
    from src.potentials.utils import KNOWN_TYPES
    monkeypatch.setattr("src.interfaces.lammps.lammps_pair_styles",
                        lambda: frozenset(KNOWN_TYPES | {"mlip", "ilp/tmd"}))


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
        aiida=AiidaSettings(),
    )


@pytest.fixture
def data_path() -> Path:
    """Returns the path to package data directory."""
    return Path(str(resources.files('src.data')))


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Legacy fixture alias used by older tests."""
    return tmp_path


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
