"""LAMMPS interface wrapper for FrictionSim2D.

This module provides utilities for interacting with LAMMPS via its Python interface.
"""

import importlib
import logging
import re
import sys
import time
from typing import List

logger = logging.getLogger(__name__)


def _compat_strptime(value: str, fmt: str):
    """Accept LAMMPS version strings with an extra numeric suffix.

    Some packaged LAMMPS Python builds expose version strings like
    ``2022.06.23.2.0`` while their import code still parses using
    ``%Y.%m.%d``. This trims the trailing suffix only for that specific
    format during import.
    """
    if fmt == "%Y.%m.%d":
        match = re.match(r"^(\d{4}\.\d{1,2}\.\d{1,2})(?:\.\d+(?:\.\d+)*)?$", value)
        if match:
            value = match.group(1)
    return _ORIGINAL_STRPTIME(value, fmt)


def _import_lammps_module():
    """Import the lammps module with a narrow compatibility fallback."""
    module = sys.modules.get("lammps")
    if module is not None:
        return module

    try:
        return importlib.import_module("lammps")
    except ValueError as exc:
        if "unconverted data remains" not in str(exc):
            raise

        sys.modules.pop("lammps", None)
        time.strptime = _compat_strptime
        try:
            return importlib.import_module("lammps")
        finally:
            time.strptime = _ORIGINAL_STRPTIME


_ORIGINAL_STRPTIME = time.strptime

# Preload the module so later direct imports elsewhere in the codebase reuse
# the already-imported module and avoid repeating the failing import path.
try:
    _import_lammps_module()
except (ImportError, OSError, ValueError):
    logger.debug("LAMMPS preload skipped", exc_info=True)


def run_lammps_commands(commands: List[str]) -> None:
    """Runs a list of LAMMPS commands using the Python interface.
    
    Args:
        commands: List of LAMMPS command strings to execute.
        
    Raises:
        Exception: If any LAMMPS command fails during execution.
    """
    lammps = _import_lammps_module().lammps

    lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    try:
        for cmd in commands:
            lmp.command(cmd)
    except Exception as e:
        logger.error("LAMMPS execution failed: %s", e)
        raise
    finally:
        lmp.close()
