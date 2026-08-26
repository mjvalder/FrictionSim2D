"""Core modules for FrictionSim2D.

Exports are resolved lazily to keep import-time side effects low and to avoid
builder/run/hpc circular imports: ``SimulationBase`` now lives in
``src.builders.simulation_base``, and resolving it eagerly here would pull the builders --
and through them ``src.hpc`` -- into every ``import src.core``.

This package holds the shared contracts and vocabulary that everything else
imports:
    - Configuration models (Pydantic) and the .ini read/write pair
    - ``observables``: what the LAMMPS templates are expected to emit
    - ``utils/``: path resolution, token formatting, LAMMPS data-file surgery,
      structure reading -- split by import weight, see tests/test_import_weight.py
"""

from importlib import import_module
from typing import Any

__all__ = [
    "ComponentConfig",
    "TipConfig",
    "SubstrateConfig",
    "SheetConfig",
    "GeneralConfig",
    "AFMSimulationConfig",
    "SheetOnSheetSimulationConfig",
    "GlobalSettings",
    "load_settings",
    "parse_config",
    "PotentialManager",
    "SimulationBase",
    "cifread",
    "count_atomtypes",
    "lj_params",
    "get_material_path",
    "get_potential_path",
    "read_config",
    "get_model_dimensions",
    "get_num_atom_types",
    "charge2atom",
    "atomic2molecular",
    "get_potential_element_order",
    "check_potential_cif_compatibility",
    "expand_config_sweeps",
    "collect_hpc_simulation_paths",
    "generate_hpc_scripts_for_root",
    "run_simulations",
]


_EXPORT_MAP = {
    "ComponentConfig": (".config", "ComponentConfig"),
    "TipConfig": (".config", "TipConfig"),
    "SubstrateConfig": (".config", "SubstrateConfig"),
    "SheetConfig": (".config", "SheetConfig"),
    "GeneralConfig": (".config", "GeneralConfig"),
    "AFMSimulationConfig": (".config", "AFMSimulationConfig"),
    "SheetOnSheetSimulationConfig": (".config", "SheetOnSheetSimulationConfig"),
    "GlobalSettings": (".config", "GlobalSettings"),
    "load_settings": (".config", "load_settings"),
    "parse_config": (".config", "parse_config"),
    "PotentialManager": ("..potentials.manager", "PotentialManager"),
    "SimulationBase": ("..builders.simulation_base", "SimulationBase"),
    "cifread": (".utils.structures", "cifread"),
    "count_atomtypes": (".utils.structures", "count_atomtypes"),
    "lj_params": ("..potentials.utils", "lj_params"),
    "get_material_path": (".utils.paths", "get_material_path"),
    "get_potential_path": (".utils.paths", "get_potential_path"),
    "read_config": (".config", "read_config"),
    "get_model_dimensions": ("..builders.utils.lammps_data", "get_model_dimensions"),
    "get_num_atom_types": ("..builders.utils.lammps_data", "get_num_atom_types"),
    "charge2atom": ("..builders.utils.lammps_data", "charge2atom"),
    "atomic2molecular": ("..builders.utils.lammps_data", "atomic2molecular"),
    "get_potential_element_order": ("..potentials.utils", "get_potential_element_order"),
    "check_potential_cif_compatibility": ("..potentials.utils", "check_potential_cif_compatibility"),
    "expand_config_sweeps": ("..run.expand", "expand_config_sweeps"),
    "collect_hpc_simulation_paths": ("..run.hpc_dispatch", "collect_hpc_simulation_paths"),
    "generate_hpc_scripts_for_root": ("..run.hpc_dispatch", "generate_hpc_scripts_for_root"),
    "run_simulations": ("..run.simulate", "run_simulations"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)
