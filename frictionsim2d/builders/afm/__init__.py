"""AFM-family builders: a tip on a sheet, with or without a substrate.

``AFMSimulation`` is re-exported so ``from frictionsim2d.builders.afm import
AFMSimulation`` keeps working -- that is this package's public name, not a
deprecation shim, and it is what 17 call sites already use.

Deliberately NOT re-exported: module-level names such as
``get_model_dimensions``. ``base.py`` imports those into its own namespace, so
a re-export here would be a *third* binding, and patching it would silently
miss the one the code reads. Patch ``frictionsim2d.builders.afm.base.<name>`` instead.
"""
from .base import AFMSimulation, K_CONTACT

__all__ = ['AFMSimulation', 'K_CONTACT']
