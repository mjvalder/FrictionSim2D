"""LAMMPS interface wrapper for FrictionSim2D.

This module provides utilities for interacting with LAMMPS via its Python interface.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def run_lammps_commands(commands: List[str]) -> None:
    """Runs a list of LAMMPS commands using the Python interface.
    
    Args:
        commands: List of LAMMPS command strings to execute.
        
    Raises:
        Exception: If any LAMMPS command fails during execution.
    """
    from lammps import lammps  # pylint: disable=import-outside-toplevel

    lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        for cmd in commands:
            lmp.command(cmd)
    except Exception as e:
        logger.error("LAMMPS execution failed: %s", e)
        raise
    finally:
        lmp.close()
