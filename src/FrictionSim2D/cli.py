"""Command Line Interface for FrictionSim2D.

Handles simulation execution, parameter sweeps, and settings management.
"""

import argparse
import logging
import sys
import shutil
import yaml
import itertools
from pathlib import Path
from copy import deepcopy
from typing import List, Dict, Any

from FrictionSim2D.core.config import (
    parse_config, AFMSimulationConfig, SheetVsSheetConfig, 
    load_default_settings, GlobalSettings
)
from FrictionSim2D.builders.afm import AFMSimulation
from FrictionSim2D.builders.sheet_vs_sheet import SheetVsSheetSimulation
from FrictionSim2D.builders import components
from FrictionSim2D.interfaces.atomsk import AtomskWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def expand_config_sweeps(base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expands a config dictionary into a list of single-run configurations.
    
    Handles:
    1. Lists in 'general' (force, temp, etc.)
    2. File paths in 'mat' (reading material lists)
    """
    # 1. Identify Sweepable Parameters
    sweep_params = {}
    
    # Helper to extract material list from file
    def resolve_material(mat_value):
        if isinstance(mat_value, str) and mat_value.endswith(".txt"):
            path = Path(mat_value)
            if path.exists():
                with open(path, 'r') as f:
                    return [line.strip() for line in f if line.strip()]
            logger.warning(f"Material list file {path} not found, using as string literal.")
        return [mat_value] if not isinstance(mat_value, list) else mat_value

    # Check 2D/Sheet material
    if '2D' in base_config and 'mat' in base_config['2D']:
         sweep_params[('2D', 'mat')] = resolve_material(base_config['2D']['mat'])
         
    # Check General params (temp, force, etc)
    if 'general' in base_config:
        for key, val in base_config['general'].items():
            if isinstance(val, list):
                sweep_params[('general', key)] = val

    # 2. Generate Permutations
    if not sweep_params:
        return [base_config]

    keys, values = zip(*sweep_params.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    expanded_configs = []
    for perm in permutations:
        # Create deep copy of base
        new_conf = deepcopy(base_config)
        # Apply this permutation's values
        for (section, key), val in perm.items():
            new_conf[section][key] = val
        expanded_configs.append(new_conf)
        
    return expanded_configs

def handle_settings(args):
    """Handler for 'settings' subcommand."""
    if args.action == 'show':
        defaults = load_default_settings()
        print(yaml.dump(defaults.dict(), default_flow_style=False))
        
    elif args.action == 'reset':
        local_settings = Path("settings.yaml")
        if local_settings.exists():
            local_settings.unlink()
            logger.info("Removed local settings.yaml. Using package defaults.")
        else:
            logger.info("No local settings found to reset.")
            
    elif args.action == 'init':
        # Copy defaults to local dir for editing
        from importlib import resources
        with resources.as_file(resources.files('FrictionSim2D.data.settings') / 'settings_default.yaml') as p:
            shutil.copy(p, "settings.yaml")
        logger.info("Created mutable 'settings.yaml' in current directory.")

def handle_build(args):
    """Handler for 'build' subcommand (individual components)."""
    # ... Logic to instantiate Config -> components.build_X ...
    pass

def handle_run(args):
    """Handler for 'run' subcommand."""
    config_path = Path(args.config_file)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    # 1. Load Base Config & Settings
    base_dict = parse_config(config_path)
    defaults = load_default_settings()
    
    # 2. Expand Sweeps (Looping Logic)
    configs_to_run = expand_config_sweeps(base_dict)
    
    logger.info(f"Found {len(configs_to_run)} simulation configurations to run.")
    
    for i, run_dict in enumerate(configs_to_run):
        logger.info(f"--- Starting Run {i+1}/{len(configs_to_run)} ---")
        
        # Inject settings
        run_dict['settings'] = defaults.dict()
        
        # Define output for this specific run (e.g. append parameters to dir name)
        # Simple logic: separate folders or same folder with different filenames?
        # Usually separate folders is cleaner for LAMMPS.
        
        # Generate dynamic output dir based on run params (material, temp)
        run_name = f"{run_dict['2D']['mat']}_{run_dict['general'].get('temp', 0)}K"
        run_output_dir = Path(args.output_dir) / run_name
        
        # Dispatch
        try:
            if args.model == 'afm':
                # Validate
                config_obj = AFMSimulationConfig(**run_dict)
                builder = AFMSimulation(config_obj, run_output_dir)
            elif args.model == 'sheet_vs_sheet':
                config_obj = SheetVsSheetConfig(**run_dict)
                builder = SheetVsSheetSimulation(config_obj, run_output_dir)
            
            builder.build()
            builder.write_inputs()
            
        except Exception as e:
            logger.error(f"Run {i+1} failed: {e}")
            continue # Move to next run in sweep

def main():
    parser = argparse.ArgumentParser(description="FrictionSim2D CLI")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- RUN Command ---
    run_parser = subparsers.add_parser('run', help='Run simulations')
    run_parser.add_argument("config_file", type=str)
    run_parser.add_argument("--model", "-m", choices=['afm', 'sheet_vs_sheet'], default='afm')
    run_parser.add_argument("--output-dir", "-o", default="simulation_output")
    run_parser.set_defaults(func=handle_run)

    # --- SETTINGS Command ---
    settings_parser = subparsers.add_parser('settings', help='Manage settings')
    settings_parser.add_argument("action", choices=['show', 'init', 'reset'])
    settings_parser.set_defaults(func=handle_settings)
    
    # --- BUILD Command ---
    build_parser = subparsers.add_parser('build', help='Build individual component')
    build_parser.add_argument("component", choices=['tip', 'sheet', 'sub'])
    # ... add args for config ...
    build_parser.set_defaults(func=handle_build)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()