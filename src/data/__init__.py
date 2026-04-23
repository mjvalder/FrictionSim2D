"""Data models, validation, and database access for FrictionSim2D."""

import os

from .models import ResultRecord, compute_friction_stats

__all__ = ['ResultRecord', 'compute_friction_stats']


def get_client(mode: str = 'direct', **kwargs):
    """Create a database client in the specified mode.

    Args:
        mode: ``'direct'`` for a direct PostgreSQL connection via
            :class:`~src.data.database.FrictionDB`, or ``'api'`` for an
            HTTP client via :class:`~src.api.client.FrictionHTTPClient`.
        **kwargs: Passed to the underlying client constructor.
            For ``'api'`` mode: ``api_url``, ``api_key``, ``timeout``.
            For ``'direct'`` mode: ``host``, ``port``, ``dbname``, ``user``,
            ``password``, ``auto_create``.

    Returns:
        A :class:`~src.data.database.FrictionDB` or
        :class:`~src.api.client.FrictionHTTPClient` instance.

    Raises:
        ValueError: If *mode* is not ``'direct'`` or ``'api'``.
        ImportError: If the required backend library is not installed.
    """
    if mode == 'api':
        from ..core.config import load_settings  # pylint: disable=import-outside-toplevel
        from ..api.client import FrictionHTTPClient  # pylint: disable=import-outside-toplevel

        settings_file = kwargs.pop('settings_file', None)
        settings = load_settings(settings_file=settings_file)
        if not kwargs.get('api_url'):
            kwargs['api_url'] = settings.database.api_url
        if kwargs.get('api_key') is None:
            kwargs['api_key'] = (
                os.environ.get('FRICTION_DB_API_KEY')
                or settings.database.central.api_key
                or None
            )

        return FrictionHTTPClient(**kwargs)
    if mode == 'direct':
        from .database import FrictionDB  # pylint: disable=import-outside-toplevel
        return FrictionDB(**kwargs)
    raise ValueError(f"Unknown mode {mode!r}. Use 'direct' or 'api'.")
