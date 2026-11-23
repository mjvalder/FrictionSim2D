"""External tool and system interfaces for FrictionSim2D.

This package provides wrappers around external tools and libraries used
in simulation preparation and execution:
    - AtomskWrapper: Interface to the Atomsk structure generation tool
    - PackageLoader: Custom Jinja2 template loader for package resources
    - run_lammps_commands: Interface to LAMMPS via Python bindings
"""

from .atomsk import AtomskWrapper, AtomskError
from .jinja import PackageLoader
from .lammps import run_lammps_commands

__all__ = [
    'AtomskWrapper',
    'AtomskError',
    'PackageLoader',
    'run_lammps_commands',
]
