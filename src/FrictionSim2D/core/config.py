"""Pydantic-based configuration models for FrictionSim2D.

This module provides robust, type-safe data schemas for all simulation
parameters. By leveraging Pydantic, it ensures that all configurations—from
low-level engine settings to high-level experimental parameters—are validated
for correctness before a simulation is executed.
"""
import json
from pathlib import Path
from importlib import resources
from typing import List, Optional, Union, Dict, Any, Literal
import yaml
from pydantic import BaseModel, Field

from FrictionSim2D.core.utils import read_config

# --- Internal Settings ---

class GeometrySettings(BaseModel):
    tip_reduction_factor: float
    rigid_tip: bool
    tip_base_z: float
    lat_c_default: float

class ThermostatSettings(BaseModel):
    type: Literal['langevin', 'nose-hoover']
    time_integration: Literal['verlet', 'respa','nvt','nve']
    langevin_boundaries: Dict[str, Dict[str, List[float]]]

class SimulationSettings(BaseModel):
    timestep: float
    thermo: int
    min_style: str
    minimization_command: str
    neighbor_list: float
    neigh_modify_command: str
    slide_run_steps: int
    drive_method: Literal['smd', 'fix_move', 'virtual_atom']

class QuenchSettings(BaseModel):
    run_local: bool
    n_procs: int
    quench_slab_dims: List[int]
    quench_rate: float
    quench_melt_temp: float
    timestep: float

class OutputSettings(BaseModel):
    dump: Dict[str, bool]
    dump_frequency: Dict[str, int]
    results_frequency: int
    
class PotentialSettings(BaseModel):
    lj_cutoff: float
    lj_type: str

class AiidaSettings(BaseModel):
    lammps_code_label: str 
    postprocess_code_label: str
    postprocess_script_path: str
    num_cpus: int
    walltime_seconds: int
    
class GlobalSettings(BaseModel):
    """Represents the full structure of settings.yaml / settings_default.yaml."""
    geometry: GeometrySettings
    thermostat: ThermostatSettings
    simulation: SimulationSettings
    quench: QuenchSettings
    output: OutputSettings
    potential: PotentialSettings
    aiida: AiidaSettings

# --- User Input Models (From .ini files) ---

class ComponentConfig(BaseModel):
    """Base configuration for any material component (Tip, Substrate, Sheet)."""
    mat: str
    pot_type: str
    pot_path: str
    cif_path: str

class TipConfig(ComponentConfig):
    r: float = Field(..., description="Tip radius in Angstroms")
    amorph: Literal['c', 'a'] = Field('c', description="'c' for crystalline, 'a' for amorphous")
    cspring: float = Field(..., description="Spring constant")
    dspring: float = Field(0.0, description="Damping constant")
    s: float = Field(..., description="Sliding speed (m/s)")

class SubstrateConfig(ComponentConfig):
    thickness: float
    amorph: Literal['c', 'a'] = 'c'

class SheetConfig(ComponentConfig):
    x: Union[float, List[float]]
    y: Union[float, List[float]]
    layers: List[int]
    stack_type: str = 'AA'
    lat_c: Optional[float] = None

class GeneralConfig(BaseModel):
    temp: float
    force: Optional[Union[float, List[float]]] = None
    pressure: Optional[Union[float, List[float]]] = None
    scan_angle: Optional[Union[float, List[float]]] = 0.0
    scan_speed: Optional[Union[float, List[float]]] = None
    bond_spring: Optional[float] = Field(None, description="Spring constant for harmonically bonded sheets")
    driving_spring: Optional[float] = Field(None, description="Driving spring constant")


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

def get_settings_path(filename: str = 'settings.yaml') -> Path:
    """Returns the path to the installed settings file (if mutable) or package resource."""
    # Check local directory first
    local_settings = Path("settings.yaml")
    if local_settings.exists():
        return local_settings.resolve()

    # Otherwise return the package resource context (read-only usually)
    return resources.files('FrictionSim2D.data.settings').joinpath(filename)

def load_default_settings() -> GlobalSettings:
    """Loads settings from 'settings.yaml', falling back to 'settings_default.yaml'.

    Logic:
    1. Attempt to load 'settings.yaml'. If it exists and is not empty, it is
        considered the primary, complete configuration.
    2. If 'settings.yaml' does not exist, 'settings_default.yaml' is loaded
        as the fallback configuration.
    """
    settings_dir = resources.files('FrictionSim2D.data.settings')
    settings_path = settings_dir / 'settings.yaml'
    default_settings_path = settings_dir / 'settings_default.yaml'
    final_settings = {}

    # Try to load the primary settings.yaml first
    try:
        with resources.as_file(settings_path) as p_user:
            if p_user.exists():
                with open(p_user, 'r', encoding='utf-8') as f:
                    final_settings = yaml.safe_load(f) or {}
    except (FileNotFoundError, ImportError):
        final_settings = {}

    # If the primary settings file was not found or was empty, load the default
    if not final_settings:
        print("Info: 'settings.yaml' not found or is empty. Loading 'settings_default.yaml' as fallback.")
        with resources.as_file(default_settings_path) as p_def:
            with open(p_def, 'r', encoding='utf-8') as f:
                final_settings = yaml.safe_load(f) or {}

    return GlobalSettings(**final_settings)

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
        # It's already a dictionary (e.g., from CLI args), pass it through
        return config_source

    else:
        raise TypeError(f"Unsupported configuration source type: {type(config_source)}")
