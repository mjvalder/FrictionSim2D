"""Data models, validation, and database access for FrictionSim2D."""

from src.data.models import ResultRecord, compute_friction_stats
from src.data.database import db_from_profile

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
		from src.api.client import FrictionHTTPClient  # noqa: PLC0415
		return FrictionHTTPClient(**kwargs)
	if mode == 'direct':
		from src.data.database import FrictionDB  # noqa: PLC0415
		return FrictionDB(**kwargs)
	raise ValueError(f"Unknown mode {mode!r}. Use 'direct' or 'api'.")
