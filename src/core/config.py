"""Pydantic-based configuration models for FrictionSim2D.

This module provides robust, type-safe data schemas for all simulation
parameters. By leveraging Pydantic, it ensures that all configurations—from
low-level engine settings to high-level experimental parameters—are validated
for correctness before a simulation is executed.
"""
from importlib import resources
import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Literal
import yaml
from pydantic import BaseModel, Field, field_validator, ValidationInfo

from src.core.utils import read_config, get_potential_path, get_material_path

# --- Internal Settings ---

class GeometrySettings(BaseModel):
    """Geometry settings for tip and substrate positioning."""
    tip_reduction_factor: float = 2.25
    rigid_tip: bool = False
    tip_base_z: float = 55.0
    lat_c_default: float = 6.0

class ThermostatSettings(BaseModel):
    """Thermostat and time integration settings."""
    type: Literal['langevin', 'nose-hoover'] = 'langevin'
    time_integration: Literal['verlet', 'respa', 'nvt', 'nve'] = Field(
        default='nvt', alias='time_int_type')
    langevin_boundaries: Dict[str, Dict[str, List[float]]] = Field(
        default_factory=lambda: {
            'tip': {'fix': [3.0, 0.0], 'thermo': [6.0, 3.0]},
            'sub': {'fix': [0.0, 0.3], 'thermo': [0.3, 0.6]}
        })
    model_config = {'validate_by_name': True}

class SimulationSettings(BaseModel):
    """Simulation run parameters."""
    timestep: float = 0.001
    thermo: int = 100000
    min_style: str = 'cg'
    minimization_command: str = 'minimize 1e-4 1e-8 1000000 1000000'
    neighbor_list: float = 0.3
    neigh_modify_command: str = 'neigh_modify every 1 delay 0 check yes'
    slide_run_steps: int = 200000
    drive_method: Literal['smd', 'fix_move', 'virtual_atom'] = 'virtual_atom'

class QuenchSettings(BaseModel):
    """Quenching parameters for amorphous material generation."""
    run_local: bool = True
    n_procs: int = 16
    quench_slab_dims: List[int] = Field(default_factory=lambda: [200, 200, 50])
    quench_rate: float = 1e12
    melt_temp: float = Field(default=2500.0, alias='quench_melt_temp')
    quench_temp: float = Field(default=300.0, alias='quench_target_temp')
    timestep: float = 0.002
    melt_steps: int = 50000
    quench_steps: int = 100000
    equilibrate_steps: int = 20000
    model_config = {'validate_by_name': True}

class OutputSettings(BaseModel):
    """Output and dump settings."""
    dump: Dict[str, bool] = Field(
        default_factory=lambda: {'system_init': True, 'slide': True})
    dump_frequency: Dict[str, int] = Field(
        default_factory=lambda: {'system_init': 10000, 'slide': 10000})
    results_frequency: int = 1000

class PotentialSettings(BaseModel):
    """Settings for interatomic potentials."""
    lj_cutoff: float = Field(default=8.0, alias='LJ_cutoff')
    lj_type: str = Field(default='LJ_base', alias='LJ_type')
    model_config = {'validate_by_name': True}

class AiidaSettings(BaseModel):
    """AiiDA and HPC workflow settings."""
    # Code labels (for online AiiDA workflow)
    lammps_code_label: str = 'lammps@my_hpc'
    postprocess_code_label: str = 'python@my_hpc'
    postprocess_script_path: str = ''

    # Resource allocation
    num_cpus: int = 32
    memory_gb: int = 8
    walltime_seconds: int = 72000

    # Scheduler settings (for offline HPC workflow)
    scheduler_type: Literal['pbs', 'slurm'] = 'pbs'
    queue: str = 'normal'
    account: str = ''
    max_array_size: int = 300
    # Environment modules to load on HPC
    modules: List[str] = Field(default_factory=lambda: ['lammps/2023'])

class GlobalSettings(BaseModel):
    """Represents the full structure of settings.yaml with hardcoded defaults."""
    geometry: GeometrySettings = Field(default_factory=GeometrySettings)
    thermostat: ThermostatSettings = Field(default_factory=ThermostatSettings)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    quench: QuenchSettings = Field(default_factory=QuenchSettings)
    output: OutputSettings = Field(default_factory=OutputSettings)
    potential: PotentialSettings = Field(default_factory=PotentialSettings)
    aiida: AiidaSettings = Field(default_factory=AiidaSettings)

# --- User Input Settings (From .ini files) ---

class ComponentConfig(BaseModel):
    """Base configuration for any material component.

    Attributes:
        mat: Material identifier.
        pot_type: Potential type (e.g., 'sw', 'tersoff').
        pot_path: Path to potential file.
        cif_path: Path to CIF structure file.
    """
    mat: str
    pot_type: str
    pot_path: str
    cif_path: str

    @field_validator('pot_path', 'cif_path', mode='after')
    @classmethod
    def resolve_path(cls, v: str, info: ValidationInfo) -> str:
        """Resolve filesystem paths from config values.

        Args:
            v: Path value from config.
            info: Pydantic validation info including field name.

        Returns:
            Resolved absolute path as string.
        """
        path = Path(v)
        if path.exists():
            return str(path)
        # Dispatch specific lookup logic
        if info.field_name == 'pot_path':
            resolved = get_potential_path(v)
        else:
            resolved = get_material_path(v, 'cif')

        return str(resolved) if resolved.exists() else v

class TipConfig(ComponentConfig):
    """Tip configuration parameters."""
    r: float = Field(..., description="Tip radius in Angstroms")
    amorph: Literal['c', 'a'] = Field('c', description="'c' for crystalline, 'a' for amorphous")
    dspring: float = Field(0.0, description="Damping constant")
    s: float = Field(..., description="Sliding speed (m/s)")
    
    @field_validator('amorph', mode='before')
    @classmethod
    def handle_empty_amorph(cls, v):
        """Convert None to default 'c' value."""
        return 'c' if v is None else v

class SubstrateConfig(ComponentConfig):
    """Substrate configuration parameters."""
    thickness: float
    amorph: Literal['c', 'a'] = 'c'
    @field_validator('amorph', mode='before')
    @classmethod
    def handle_empty_amorph(cls, v):
        """Convert None to default 'c' value."""
        return 'c' if v is None else v

class SheetConfig(ComponentConfig):
    """Sheet configuration parameters."""
    x: Union[float, List[float]]
    y: Union[float, List[float]]
    layers: List[int]
    stack_type: str = 'AA'
    lat_c: Optional[float] = None

class GeneralConfig(BaseModel):
    """General simulation parameters."""
    temp: float
    force: Optional[Union[float, List[float]]] = None
    pressure: Optional[Union[float, List[float]]] = None
    scan_angle: Optional[Union[float, List[float]]] = 0.0
    scan_speed: Optional[Union[float, List[float]]] = None
    bond_spring: Optional[float] = Field(5.0, description="Spring constant for harmonically bonded sheets")
    driving_spring: Optional[float] = Field(50, description="Driving spring constant")


class AFMSimulationConfig(BaseModel):
    """Master configuration object for an AFM simulation run."""
    general: GeneralConfig
    tip: TipConfig
    sub: SubstrateConfig
    sheet: SheetConfig = Field(..., alias='2D')  # Map [2D] section to .sheet
    settings: GlobalSettings

class SheetOnSheetSimulationConfig(BaseModel):
    """Master configuration object for a Sheet-on-Sheet simulation run."""
    general: GeneralConfig
    sheet: SheetConfig = Field(..., alias='2D')
    settings: GlobalSettings

# --- Helper Functions ---

def load_settings() -> GlobalSettings:
    """Load settings from package settings.yaml, or use hardcoded defaults.

    Checks for settings.yaml in the package data/settings folder. If it exists
    and is populated, loads those settings. Otherwise returns hardcoded defaults.

    Returns:
        GlobalSettings with values from settings.yaml if present, else defaults.
    """
    try:
        settings_resource = resources.files('src.data.settings').joinpath('settings.yaml')
        if settings_resource.is_file():
            with settings_resource.open('r') as f:
                user_settings = yaml.safe_load(f) or {}
                if user_settings:  # Only use if file has content
                    return GlobalSettings(**user_settings)
    except (FileNotFoundError, AttributeError):
        pass

    # Return defaults if no settings file or empty file
    return GlobalSettings()


def parse_config(config_source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Parses configuration from various sources into a dictionary suitable for Pydantic.

    This function acts as a unified entry point for configuration loading. It can
    handle:
    1. File paths (str or Path) pointing to .ini, .yaml/.yml, or .json files.
    2. Dictionaries (e.g., from a CLI arg parser or UI form).

    Args:
        config_source (Union[str, Path, Dict]): The configuration source.

    Returns:
        Dict[str, Any]: A standardized dictionary ready for validation.

    Raises:
        ValueError: If the file extension is not supported.
        TypeError: If the input type is not supported.
    """
    if isinstance(config_source, (str, Path)):
        path = Path(config_source)
        ext = path.suffix.lower()

        if ext == '.ini':
            return read_config(path)

        if ext in ('.yaml', '.yml'):
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}

        if ext == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)

        else:
            raise ValueError(f"Unsupported configuration file format: {ext}. Supported formats: .ini, .yaml, .yml, .json")

    elif isinstance(config_source, dict):
        return config_source

    else:
        raise TypeError(f"Unsupported configuration source type: {type(config_source)}")
