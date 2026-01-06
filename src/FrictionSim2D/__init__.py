"""FrictionSim2D - A modular framework for 2D material friction simulations.

This package provides tools for setting up and running LAMMPS-based 
molecular dynamics simulations of friction in 2D materials, including:
- AFM tip-on-sheet simulations
- Sheet-on-sheet friction simulations
- Automatic potential generation with hybrid pair styles
- Integration with AiiDA for HPC workflows
"""

__version__ = "0.2.0"

from pathlib import Path
import logging

from FrictionSim2D.builders.afm import AFMSimulation
from FrictionSim2D.builders.sheetonsheet import SheetOnSheetSimulation
from FrictionSim2D.core.potential_manager import PotentialManager
from FrictionSim2D.core.config import (
    AFMSimulationConfig,
    SheetOnSheetSimulationConfig,
    load_default_settings,
    parse_config,
)

logger = logging.getLogger(__name__)


def afm(config_file: str = "afm_config.ini"):
    """Run AFM simulations from a config file.
    
    Simple interface that mimics the old tribo_2D behavior:
        from FrictionSim2D import afm
        afm("afm_config.ini")
    
    This function:
    1. Reads the config file
    2. Expands material lists and parameter sweeps
    3. Creates the folder structure matching the old tribo_2D format:
       afm/{mat}/{x}x_{y}y/sub_{sub}_tip_{tip}_r{r}/K{temp}/l_{layer}/
    4. Builds and writes all simulation files
    
    Args:
        config_file: Path to the .ini configuration file.
    """
    _run_all(config_file, model="afm")


def sheetonsheet(config_file: str = "sheet_config.ini"):
    """Run SheetOnSheet simulations from a config file.
    
    Args:
        config_file: Path to the .ini configuration file.
    """
    _run_all(config_file, model="sheetonsheet")


def _run_all(config_file: str, model: str = "afm"):
    """Core function that runs all simulations defined in a config file.
    
    Handles:
    - Material list expansion ({mat} template)
    - Parameter sweeps (force, layers, temp)
    - Folder structure creation matching old tribo_2D format
    """
    from FrictionSim2D.cli import expand_config_sweeps
    
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load base config and settings
    base_dict = parse_config(config_path)
    defaults = load_default_settings()

    # Expand sweeps
    configs_to_run = expand_config_sweeps(base_dict)
    
    print(f"Found {len(configs_to_run)} simulation configurations to run.")

    for i, run_dict in enumerate(configs_to_run):
        # Inject settings
        run_dict['settings'] = defaults.dict()

        # Extract parameters for folder naming
        mat = run_dict['2D'].get('mat', 'unknown')
        x = run_dict['2D'].get('x', 100)
        y = run_dict['2D'].get('y', 100)
        layers = run_dict['2D'].get('layers', [1])
        n_layers = layers[0] if isinstance(layers, list) else layers
        
        temp = run_dict['general'].get('temp', 300)
        
        # Tip/Sub info for folder name
        tip_mat = run_dict.get('tip', {}).get('mat', 'Si')
        tip_amorph = run_dict.get('tip', {}).get('amorph', 'c')
        tip_r = run_dict.get('tip', {}).get('r', 25)
        sub_mat = run_dict.get('sub', {}).get('mat', 'Si')
        sub_amorph = run_dict.get('sub', {}).get('amorph', 'a')
        
        # Build folder path matching old tribo_2D structure:
        # afm/{mat}/{x}x_{y}y/sub_{amorph}{sub}_tip_{amorph}{tip}_r{r}/K{temp}/l_{layer}/
        if model == "afm":
            sub_str = f"{sub_amorph}{sub_mat}" if sub_amorph == 'a' else sub_mat
            tip_str = f"{tip_amorph}{tip_mat}" if tip_amorph == 'a' else tip_mat
            
            output_dir = (
                Path("afm") / mat / f"{x}x_{y}y" / 
                f"sub_{sub_str}_tip_{tip_str}_r{int(tip_r)}" / 
                f"K{int(temp)}" / f"l_{n_layers}"
            )
        else:
            # sheetonsheet
            output_dir = (
                Path("sheetonsheet") / mat / f"{x}x_{y}y" / 
                f"K{int(temp)}" / f"l_{n_layers}"
            )

        print(f"--- Run {i+1}/{len(configs_to_run)}: {output_dir} ---")

        try:
            if model == 'afm':
                config_obj = AFMSimulationConfig(**run_dict)
                builder = AFMSimulation(config_obj, output_dir)
            else:
                config_obj = SheetOnSheetSimulationConfig(**run_dict)
                builder = SheetOnSheetSimulation(config_obj, output_dir)

            builder.build()
            print(f"  -> Completed: {output_dir}")

        except Exception as e:
            logger.error(f"Run {i+1} failed: {e}")
            print(f"  -> Failed: {e}")
            continue


__all__ = [
    # Simple interface (like old tribo_2D)
    "afm",
    "sheetonsheet",
    # Classes for programmatic use
    "AFMSimulation",
    "SheetOnSheetSimulation",
    "PotentialManager",
    "AFMSimulationConfig",
    "SheetOnSheetSimulationConfig",
    "load_default_settings",
    "parse_config",
]