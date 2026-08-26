"""Simulation builders for FrictionSim2D.

This package exposes high-level builders and component helpers via lazy
attribute loading to avoid circular imports during startup.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "AFMSimulation",
    "SheetOnSheetSimulation",
    "PESSheetSimulation",
    "PESTipSimulation",
    "components",
]


def __getattr__(name: str) -> Any:
    if name == "AFMSimulation":
        return import_module(".afm", __name__).AFMSimulation
    if name == "SheetOnSheetSimulation":
        return import_module(".sheetonsheet", __name__).SheetOnSheetSimulation
    if name == "PESSheetSimulation":
        return import_module(".sheetonsheet.pes_sheet", __name__).PESSheetSimulation
    if name == "PESTipSimulation":
        return import_module(".afm.pes_tip", __name__).PESTipSimulation
    if name == "components":
        return import_module(".construction.components", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
