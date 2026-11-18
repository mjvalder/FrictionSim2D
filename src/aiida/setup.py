"""AiiDA profile setup for FrictionSim2D (``FrictionSim2D aiida setup``).

Only the profile is created here. The store is import-only: the machine
``Computer`` and the LAMMPS ``Code`` of each run are created when it is
imported (:mod:`src.aiida.store`), and no daemon is ever needed.
"""

import logging
import subprocess

logger = logging.getLogger(__name__)


def setup_profile(
    profile_name: str = 'friction2d',
    use_postgres: bool = False,
) -> bool:
    """Create and configure an AiiDA profile with ``verdi presto``.

    The import-only store works on either storage backend, with no broker
    and no daemon: SQLite is enough, and the default. ``use_postgres``
    (``aiida setup --postgres``) tries PostgreSQL first and falls back to
    SQLite when it is not available.

    Args:
        profile_name: Name for the AiiDA profile.
        use_postgres: Try a PostgreSQL profile first (needs a running
            PostgreSQL server and psycopg2). Off by default.

    Returns:
        ``True`` if profile was created or already exists.
    """
    from aiida.manage.configuration import (  # pylint: disable=import-outside-toplevel
        get_config,
        load_profile,
        reset_config,
    )
    from aiida.manage import get_manager  # pylint: disable=import-outside-toplevel

    config = get_config()

    # Check if profile already exists
    if profile_name in [p.name for p in config.profiles]:
        logger.info("Profile '%s' already exists", profile_name)
        load_profile(profile_name)
        return True

    # Use verdi presto for quick setup
    cmd = ['verdi', 'presto', '--profile-name', profile_name]
    if use_postgres:
        cmd.append('--use-postgres')

    logger.info("Creating AiiDA profile '%s'...", profile_name)
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode != 0 and use_postgres:
        logger.warning(
            "PostgreSQL profile creation failed (%s), retrying with SQLite ...",
            result.stderr.strip().splitlines()[-1] if result.stderr.strip() else 'unknown error',
        )
        cmd_sqlite = ['verdi', 'presto', '--profile-name', profile_name]
        result = subprocess.run(cmd_sqlite, capture_output=True, text=True, check=False)

    # Treat "profile already exists" as a success (created in a previous run)
    already_exists = result.returncode != 0 and 'already exists' in (result.stderr + result.stdout)

    if result.returncode != 0 and not already_exists:
        logger.error("Profile creation failed: %s", result.stderr)
        return False

    # verdi presto ran in a subprocess so the in-process config cache is stale.
    # Reset the loaded config/profile through the public AiiDA APIs so the
    # next load_profile() re-reads from disk.
    manager = get_manager()
    reset_config()
    manager.reset_profile()

    load_profile(profile_name)
    logger.info("Profile '%s' created and loaded", profile_name)
    return True


def full_setup(profile_name: str = 'friction2d', *, use_postgres: bool = False) -> dict:
    """First-time AiiDA setup for FrictionSim2D: the profile only.

    Computers and Codes are created at import (one per machine and LAMMPS
    version, :func:`src.aiida.store.machine_computer` and
    :func:`~src.aiida.store.lammps_code`), so nothing else is registered.

    Args:
        profile_name: AiiDA profile name to CREATE (or load, if it exists).
        use_postgres: Try PostgreSQL first (:func:`setup_profile`).

    Returns:
        ``{'profile': True}``.

    Raises:
        RuntimeError: The profile could not be created.
    """
    if not setup_profile(profile_name, use_postgres=use_postgres):
        raise RuntimeError("AiiDA profile setup failed — cannot continue")
    return {'profile': True}
