"""Configuration models for FrictionSim2D simulations.

This module defines strict data schemas using Pydantic. It replaces the
legacy dictionary-based parameter handling with validated objects, ensuring
types (integers, floats, lists) are correct before simulation begins.
"""
import json
from pathlib import Path
from importlib import resources
from typing import List, Optional, Union, Dict, Any, Literal
import yaml
from pydantic import BaseModel, Field

# Import the utility for reading configurations
from FrictionSim2D.core.utils import read_config

# --- Internal Settings Models (Matching your YAML structure) ---

class GeometrySettings(BaseModel):
    tip_reduction_factor: float
    rigid_tip: bool
    tip_base_z: float

class ThermostatSettings(BaseModel):
    type: Literal['langevin', 'nose-hoover']
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

class GlobalSettings(BaseModel):
    """Represents the full structure of settings.yaml / settings_default.yaml."""
    geometry: GeometrySettings
    thermostat: ThermostatSettings
    simulation: SimulationSettings
    quench: QuenchSettings
    output: OutputSettings

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
    s: float = Field(..., description="Sliding speed (m/s or similar units)")

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
    # Support single value or list [start, end, step] for sweeping
    force: Optional[Union[float, List[float]]] = None
    pressure: Optional[Union[float, List[float]]] = None
    scan_angle: Optional[Union[float, List[float]]] = 0.0
    scan_speed: Optional[float] = None

class AFMSimulationConfig(BaseModel):
    """Master configuration object for an AFM simulation run."""
    general: GeneralConfig
    tip: TipConfig
    sub: SubstrateConfig
    sheet: SheetConfig = Field(..., alias='2D')  # Map [2D] section to .sheet
    settings: GlobalSettings

    class Config:
        populate_by_name = True 

# --- Helper Functions ---

def _recursive_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """Recursively updates a dictionary."""
    for k, v in update_dict.items():
        if isinstance(v, dict) and k in base_dict:
            base_dict[k] = _recursive_update(base_dict[k], v)
        else:
            base_dict[k] = v
    return base_dict

def load_default_settings() -> GlobalSettings:
    """Loads settings by merging default and user-specific YAML files.

    Logic:
    1. Load 'settings_default.yaml' (The immutable base)
    2. Load 'settings.yaml' (The user overrides)
    3. Merge them, letting 'settings.yaml' win.
    """
    settings_dir = resources.files('FrictionSim2D.data.settings')

    # 1. Load Defaults
    with resources.as_file(settings_dir / 'settings_default.yaml') as p_def:
        with open(p_def, 'r', encoding='utf-8') as f:
            combined_settings = yaml.safe_load(f) or {}

    # 2. Load Overrides (if they exist)
    try:
        with resources.as_file(settings_dir / 'settings.yaml') as p_user:
            if p_user.exists():
                with open(p_user, 'r', encoding='utf-8') as f:
                    user_settings = yaml.safe_load(f) or {}
                combined_settings = _recursive_update(combined_settings, user_settings)
    except (FileNotFoundError, ImportError):
        # It's okay if the override file is missing
        pass

    return GlobalSettings(**combined_settings)

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