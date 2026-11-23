"""Potential handling: parsing, resolution, UFF parameters, the manager.

Intentionally empty. An eager re-export here deadlocks: frictionsim2d/core/utils.py
imports frictionsim2d.potentials.uff, so importing core.utils would trigger this
__init__, which would import .manager, which imports core.utils while it is
still half-initialised. Import the submodules directly.
"""
