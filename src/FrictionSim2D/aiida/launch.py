"""
This script provides a simple, single-entry point for launching Tribo_2D AiiDA workflows.

Usage:
    python -m tribo_2D.aiida.launch /path/to/your/afm_config.ini
"""
import sys
import os

try:
    from aiida.engine import run
    from aiida.orm import SinglefileData
    from .workflows import Tribo2DWorkChain
except ImportError:
    sys.exit("AiiDA not found. Please make sure you have loaded the AiiDA environment (e.g., by running `verdi shell`).")


def main():
    """
    Parses command-line arguments and launches the Tribo2DWorkChain.
    """
    if len(sys.argv) != 2:
        print("Usage: python -m tribo_2D.aiida.launch <path_to_config_file>")
        sys.exit(1)

    config_file_path = sys.argv[1]
    if not os.path.exists(config_file_path):
        print(f"Error: Configuration file not found at: {config_file_path}")
        sys.exit(1)

    # The only input the workflow now needs is the config file itself.
    inputs = {
        'config_file': SinglefileData(file=os.path.abspath(config_file_path))
    }

    print(f"--- Launching AiiDA workflow for config: {config_file_path} ---")
    run(Tribo2DWorkChain, **inputs)
    print("\n--- Workflow submitted to the AiiDA daemon. ---")
    print("You can monitor its progress with `verdi process list`.")

if __name__ == "__main__":
    main()
