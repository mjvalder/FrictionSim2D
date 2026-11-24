"""Potential handling: parsing, resolution, UFF parameters, the manager.

Intentionally empty. An eager re-export here would import .manager (and with it
core and ASE) whenever any potentials submodule is imported; tests/repo/test_import_weight.py
pins the import cost. Import the submodules directly.
"""
