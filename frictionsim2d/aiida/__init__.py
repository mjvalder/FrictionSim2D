"""AiiDA integration for FrictionSim2D.

The root module stays thin; import the submodule you need.
"""

from typing import Any

try:
    import aiida  # noqa: F401
    AIIDA_AVAILABLE = True
except ImportError:
    AIIDA_AVAILABLE = False

if AIIDA_AVAILABLE:
    from aiida.manage.configuration import load_profile as LOAD_PROFILE
else:
    LOAD_PROFILE = None

__all__ = ['AIIDA_AVAILABLE', 'aiida_setting', 'load_aiida_profile']


def aiida_setting(name: str, default: Any = None, settings_file: Any = None) -> Any:
    """``settings.aiida.<name>``, or ``default`` when it is unset or unreadable."""
    from ..core.config import load_settings  # pylint: disable=import-outside-toplevel
    try:
        value = getattr(load_settings(settings_file=settings_file).aiida, name)
    except (OSError, ValueError, AttributeError):
        return default
    return default if value is None else value


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
        profile = aiida_setting('profile', None) or None
    return LOAD_PROFILE(profile)
