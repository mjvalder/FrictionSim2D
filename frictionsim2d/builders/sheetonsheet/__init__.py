"""Sheet-family builders: sheet-on-sheet sliding, with or without a flake.

``SheetOnSheetSimulation`` is re-exported so ``from frictionsim2d.builders.sheetonsheet
import SheetOnSheetSimulation`` keeps working. See ``afm/__init__.py`` for why
module-level names are deliberately not re-exported.
"""
from .base import SheetOnSheetSimulation

__all__ = ['SheetOnSheetSimulation']
