"""AiiDA integration for FrictionSim2D.

This package intentionally keeps a thin root module and lazy-loads
advanced symbols to reduce import-time coupling across AiiDA plugins.
"""

from importlib import import_module
from typing import Any, Dict, Tuple

try:
    import aiida  # noqa: F401
    AIIDA_AVAILABLE = True
except ImportError:
    AIIDA_AVAILABLE = False

if AIIDA_AVAILABLE:
    from aiida.manage.configuration import load_profile as LOAD_PROFILE
else:
    LOAD_PROFILE = None

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    'FrictionSimulationData': (f'{__name__}.data', 'FrictionSimulationData'),
    'FrictionSimulationSetData': (f'{__name__}.data', 'FrictionSimulationSetData'),
    'FrictionResultsData': (f'{__name__}.data', 'FrictionResultsData'),
    'FrictionProvenanceData': (f'{__name__}.data', 'FrictionProvenanceData'),
    'Friction2DDB': (f'{__name__}.query', 'Friction2DDB'),
    'register_simulation_batch': (f'{__name__}.integration', 'register_simulation_batch'),
    'register_single_simulation': (f'{__name__}.integration', 'register_single_simulation'),
    'import_results_to_aiida': (f'{__name__}.integration', 'import_results_to_aiida'),
    'import_simulation_set': (f'{__name__}.integration', 'import_simulation_set'),
    'list_simulation_sets': (f'{__name__}.integration', 'list_simulation_sets'),
    'dump_results_to_json': (f'{__name__}.integration', 'dump_results_to_json'),
    'rebuild_simulation_set': (f'{__name__}.integration', 'rebuild_simulation_set'),
    'clear_all_nodes': (f'{__name__}.integration', 'clear_all_nodes'),
    'export_archive': (f'{__name__}.integration', 'export_archive'),
    'import_archive': (f'{__name__}.integration', 'import_archive'),
    'LammpsFrictionCalcJob': (f'{__name__}.calcjob', 'LammpsFrictionCalcJob'),
    'FrictionWorkChain': (f'{__name__}.workchain', 'FrictionWorkChain'),
    'run_with_aiida': (f'{__name__}.submit', 'run_with_aiida'),
    'smart_submit': (f'{__name__}.submit', 'smart_submit'),
    'full_setup': (f'{__name__}.setup', 'full_setup'),
}

__all__ = ['AIIDA_AVAILABLE', 'load_aiida_profile']


def load_aiida_profile(profile_name=None):
    """Load an AiiDA profile, required before using any AiiDA functionality.

    Should be called once at application startup (e.g. from the CLI) before
    any AiiDA nodes are created or queried.

    Args:
        profile_name: Name of the AiiDA profile to load. If ``None``,
            the default profile is loaded.

    Returns:
        The loaded ``aiida.manage.configuration.Profile`` instance.

    Raises:
        RuntimeError: If AiiDA is not installed.
        aiida.common.exceptions.ProfileConfigurationError: If profile not found.
    """
    if not AIIDA_AVAILABLE:
        raise RuntimeError(
            "AiiDA is not installed. Install with: pip install 'FrictionSim2D[aiida]'"
        )
    if LOAD_PROFILE is None:
        raise RuntimeError("AiiDA profile loader is unavailable")
    return LOAD_PROFILE(profile_name)


def __getattr__(name: str) -> Any:
    """Lazy-load optional AiiDA symbols exported at package root."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    if not AIIDA_AVAILABLE:
        raise AttributeError(
            f"AiiDA is unavailable; cannot access '{name}'. "
            "Install with: pip install 'FrictionSim2D[aiida]'"
        )

    module_name, symbol_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, symbol_name)
    globals()[name] = value
    return value
