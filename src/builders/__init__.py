"""Simulation builders for FrictionSim2D.

This package contains the high-level builders that orchestrate
the construction of complete simulation setups:
- AFMSimulation: AFM tip-on-sheet friction simulations
- SheetOnSheetSimulation: Sheet-on-sheet friction simulations
- Component builders (tip, sheet, substrate)
"""

from . import components
from .afm import AFMSimulation
from .sheetonsheet import SheetOnSheetSimulation

__all__ = [
    "AFMSimulation",
    "SheetOnSheetSimulation",
    "components",
]
