"""LAMMPS Python-binding helpers."""

import functools
import importlib
import logging
import subprocess
import threading
from pathlib import Path
from typing import FrozenSet, Optional

logger = logging.getLogger(__name__)
_IMPORT_LOCK = threading.Lock()


def _import_lammps_module():
    """Import and return the ``lammps`` module."""
    with _IMPORT_LOCK:
        return importlib.import_module("lammps")


def create_lammps_instance():
    """Start a headless LAMMPS instance (no log, no screen output)."""
    return _import_lammps_module().lammps(
        cmdargs=["-log", "none", "-screen", "none", "-nocite"])


def run_lammps_commands(commands: list[str]) -> None:
    """Runs a list of LAMMPS commands using the Python interface.
    
    Args:
        commands: List of LAMMPS command strings to execute.
        
    Raises:
        Exception: If any LAMMPS command fails during execution.
    """
    lmp = create_lammps_instance()
    try:
        for cmd in commands:
            lmp.command(cmd)
    except Exception as e:
        logger.error("LAMMPS execution failed: %s", e)
        raise
    finally:
        lmp.close()


def run_on_data_file(
    input_path: Path,
    atom_style: str,
    commands: list[str],
    output_path: Path,
) -> None:
    """Read ``input_path`` in metal units, run ``commands``, write ``output_path``.

    Args:
        input_path: LAMMPS data file to read.
        atom_style: ``atom_style`` used to read it.
        commands: LAMMPS commands run between ``read_data`` and ``write_data``.
        output_path: Data file to write.
    """
    run_lammps_commands([
        "clear",
        "units metal",
        f"atom_style {atom_style}",
        f'read_data "{input_path}"',
        *commands,
        f'write_data "{output_path}"',
    ])


def replicate_data_file(input_path: Path, output_path: Path, nx: int, ny: int, nz: int) -> None:
    """Replicate a LAMMPS data file ``nx x ny x nz`` with native ``replicate``.

    Atom type IDs are preserved. An existing ``output_path`` is overwritten.
    """
    if output_path.exists():
        output_path.unlink()
    run_on_data_file(input_path, "atomic", [f"replicate {nx} {ny} {nz}"], output_path)


@functools.lru_cache(maxsize=1)
def lammps_pair_styles() -> Optional[FrozenSet[str]]:
    """Pair styles of the installed serial LAMMPS (``lmp -h``), lowercased.

    Returns:
        The styles from the "Pair styles:" block, or None when the binary is
        missing or the call fails.
    """
    try:
        out = subprocess.run(["lmp", "-h"], capture_output=True, text=True,
                             timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    styles = []
    in_block = False
    for line in out.splitlines():
        if "styles:" in line:
            if in_block:
                break
            in_block = "Pair styles:" in line
            continue
        if in_block:
            styles.extend(tok.lower() for tok in line.split())
    return frozenset(styles) if styles else None
