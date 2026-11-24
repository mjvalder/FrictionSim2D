"""Shared contracts and vocabulary: config models, observables, comparison, utils.

Exports resolve lazily so ``import frictionsim2d.core`` does not pull in the
builders and, through them, ``frictionsim2d.hpc``.
"""

from importlib import import_module
from typing import Any

# pylint: disable=undefined-all-variable
# resolved by __getattr__; pinned by tests/repo/test_import_surface.py
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
    "MLIPSpec",
    "MLIP_REGISTRY",
    "is_mlip",
    "get_mlip",
    "cifread",
    "count_atomtypes",
    "lj_params",
    "get_material_path",
    "get_potential_path",
    "read_config",
    "get_potential_element_order",
    "check_potential_cif_compatibility",
]
# pylint: enable=undefined-all-variable


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
    "MLIPSpec": (".mlip", "MLIPSpec"),
    "MLIP_REGISTRY": (".mlip", "MLIP_REGISTRY"),
    "is_mlip": (".mlip", "is_mlip"),
    "get_mlip": (".mlip", "get_mlip"),
    "cifread": (".utils.structures", "cifread"),
    "count_atomtypes": (".utils.structures", "count_atomtypes"),
    "lj_params": ("..potentials.utils", "lj_params"),
    "get_material_path": (".utils.paths", "get_material_path"),
    "get_potential_path": (".utils.paths", "get_potential_path"),
    "read_config": (".config", "read_config"),
    "get_potential_element_order": ("..potentials.utils", "get_potential_element_order"),
    "check_potential_cif_compatibility": (
        "..potentials.utils", "check_potential_cif_compatibility"
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name, __name__)
    return getattr(module, attr_name)
