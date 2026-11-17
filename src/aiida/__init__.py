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
    'Friction2DDB': (f'{__name__}.query', 'Friction2DDB'),
    'import_root': (f'{__name__}.ingest', 'import_root'),
    'export_results': (f'{__name__}.export', 'export_results'),
    'rebuild_set': (f'{__name__}.export', 'rebuild_set'),
    'export_archive': (f'{__name__}.export', 'export_archive'),
    'import_archive': (f'{__name__}.export', 'import_archive'),
    'list_simulation_sets': (f'{__name__}.integration', 'list_simulation_sets'),
    'delete_simulation_set': (f'{__name__}.integration', 'delete_simulation_set'),
    'clear_all_nodes': (f'{__name__}.integration', 'clear_all_nodes'),
    'LammpsFrictionCalcJob': (f'{__name__}.calcjob', 'LammpsFrictionCalcJob'),
    'full_setup': (f'{__name__}.setup', 'full_setup'),
}

__all__ = ['AIIDA_AVAILABLE', 'load_aiida_profile']


def load_aiida_profile(profile=None):
    """Load an AiiDA profile, required before using any AiiDA functionality.

    Should be called once at application startup (e.g. from the CLI) before
    any AiiDA nodes are created or queried.

    ``profile=None`` means "the configured one": `settings.aiida.profile` if it
    is set, otherwise AiiDA's own default. That is the only reader of the
    `AiidaSettings.profile` field, and it is why every command's `--profile`
    can be omitted on a machine whose settings.yaml already names one.

    Args:
        profile: Name of the AiiDA profile to load. If ``None``, the
            profile configured in `settings.yaml` is used, falling back to
            AiiDA's own default profile if that is not set either.

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
    if profile is None:
        try:
            from ..core.config import load_settings  # pylint: disable=import-outside-toplevel
            profile = load_settings().aiida.profile or None
        except (OSError, ValueError, AttributeError):
            profile = None
    return LOAD_PROFILE(profile)


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
