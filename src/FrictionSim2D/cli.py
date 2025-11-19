"""Command-line interface for FrictionSim2D.

This module defines the entry point for the application. It handles:
1. Parsing command-line arguments.
2. Loading and validating configuration files.
3. Dispatching the appropriate Simulation Builder.
4. Managing internal application settings.
"""

import argparse
import logging
import sys
import shutil
import yaml
from pathlib import Path
from importlib import resources

from FrictionSim2D.core.config import (
    AFMSimulationConfig, 
    parse_config, 
    load_default_settings
)
from FrictionSim2D.builders.afm import AFMSimulation
from FrictionSim2D.builders.sheetvsheet import SheetVsSheetSimulation

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FrictionSim2D")

def get_settings_path() -> Path:
    """Helper to get the path to the mutable settings file."""
    # We need the actual file path on disk to write to it.
    # importlib.resources.files returns a Traversable, which can be cast to Path.
    return Path(str(resources.files('FrictionSim2D.data.settings') / 'settings.yaml'))

def get_defaults_path() -> Path:
    """Helper to get the path to the immutable defaults file."""
    return Path(str(resources.files('FrictionSim2D.data.settings') / 'settings_default.yaml'))

def run_simulation(args):
    """Handles the 'run' command."""
    input_path = Path(args.input)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    try:
        # 1. Parse Config (INI/YAML/JSON -> Dict)
        logger.info(f"Parsing configuration from {input_path}...")
        raw_config = parse_config(input_path)
        
        # 2. Load Global Settings
        settings = load_default_settings()
        
        # 3. Create Config Object (Validation happens here)
        # Note: AFMSimulationConfig requires 'settings' in the dict.
        # We inject it here if not present in the user file.
        if 'settings' not in raw_config:
            raw_config['settings'] = settings.dict()
        
        # Check simulation type
        # The config object is currently specific to AFM. 
        # Ideally, we'd have a 'type' field or detect structure.
        # For now, we assume AFMSimulationConfig covers both AFM and Sheet-vs-Sheet
        # (since they share Tip/Sub/Sheet structure, SvS just ignores Tip/Sub)
        # OR we look for a flag. 
        # Let's assume the user specifies mode in CLI or config.
        # If 'sheetvsheet' is in config or args, use that builder.
        
        is_sheet_vs_sheet = args.mode == 'sheet_vs_sheet'
        
        config = AFMSimulationConfig(**raw_config)
        
        # 4. Instantiate Builder
        if is_sheet_vs_sheet:
            logger.info("Initializing Sheet-vs-Sheet Simulation...")
            builder = SheetVsSheetSimulation(config, output_dir=Path.cwd())
        else:
            logger.info("Initializing AFM Simulation...")
            builder = AFMSimulation(config, output_dir=Path.cwd())

        # 5. Build
        builder.build()
        logger.info("Simulation setup complete.")

    except Exception as e:
        logger.exception(f"Simulation failed: {e}")
        sys.exit(1)

def manage_settings(args):
    """Handles the 'settings' command group."""
    settings_path = get_settings_path()
    
    if args.action == 'show':
        # Load and print current settings
        settings = load_default_settings()
        print(yaml.dump(settings.dict(), default_flow_style=False))
        
    elif args.action == 'set':
        if not args.key or args.value is None:
            logger.error("Must provide --key and --value (e.g. --key simulation.timestep --value 0.002)")
            return

        # Load current (or default if clean)
        if settings_path.exists():
            with open(settings_path, 'r') as f:
                current_data = yaml.safe_load(f) or {}
        else:
            current_data = {}
            
        # Update logic: simulation.timestep -> dict['simulation']['timestep']
        keys = args.key.split('.')
        target = current_data
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        
        # Try to cast value
        val = args.value
        if val.replace('.', '', 1).isdigit():
            val = float(val) if '.' in val else int(val)
        
        target[keys[-1]] = val
        
        # Write back
        with open(settings_path, 'w') as f:
            yaml.dump(current_data, f)
        logger.info(f"Updated setting '{args.key}' to '{val}'")
        
    elif args.action == 'reset':
        if settings_path.exists():
            settings_path.unlink()
        logger.info("Settings reset to defaults.")

def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description="FrictionSim2D Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run a simulation setup")
    run_parser.add_argument("-i", "--input", required=True, help="Path to configuration file (.ini, .yaml, .json)")
    run_parser.add_argument("--mode", choices=['afm', 'sheet_vs_sheet'], default='afm', help="Simulation mode")
    run_parser.set_defaults(func=run_simulation)

    # Command: settings
    settings_parser = subparsers.add_parser("settings", help="Manage internal settings")
    settings_sub = settings_parser.add_subparsers(dest="action", required=True)
    
    # settings show
    settings_sub.add_parser("show", help="Show current effective settings")
    
    # settings reset
    settings_sub.add_parser("reset", help="Reset settings to defaults")
    
    # settings set
    set_parser = settings_sub.add_parser("set", help="Modify a setting")
    set_parser.add_argument("--key", required=True, help="Dot-separated key (e.g. thermostat.type)")
    set_parser.add_argument("--value", required=True, help="New value")
    
    settings_parser.set_defaults(func=manage_settings)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()