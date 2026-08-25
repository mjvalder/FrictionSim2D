"""Potential handling: parsing, resolution, UFF parameters, the manager.

Intentionally empty. An eager re-export here deadlocks: src/core/utils.py
imports src.potentials.uff, so importing core.utils would trigger this
__init__, which would import .manager, which imports core.utils while it is
still half-initialised. Import the submodules directly.
"""
