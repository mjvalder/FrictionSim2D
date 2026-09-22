"""AiiDA environment setup utilities for FrictionSim2D.

Provides helper functions for configuring AiiDA on an HPC cluster,
including profile creation, computer/code registration, and
daemon management. These are used by the CLI ``FrictionSim2D aiida setup``
commands.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from aiida import orm

logger = logging.getLogger(__name__)

_LOOKUP_EXCEPTIONS = (LookupError, ValueError, RuntimeError)


def setup_profile(
    profile_name: str = 'friction2d',
    use_postgres: bool = True,
) -> bool:
    """Create and configure an AiiDA profile.

    On HPC, this typically uses ``verdi presto`` for a quick setup with
    either PostgreSQL (recommended) or SQLite.

    Args:
        profile_name: Name for the AiiDA profile.
        use_postgres: Whether to use PostgreSQL (requires psycopg2).

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


def setup_localhost_computer(
    label: str = 'localhost',
    workdir: Optional[str] = None,
) -> 'orm.Computer':
    """Register the local machine as an AiiDA computer.

    Needed when running AiiDA directly on the HPC (the HPC is ``localhost``
    from AiiDA's perspective).

    Args:
        label: Label for the computer in AiiDA.
        workdir: Working directory for calculations. Defaults to
            ``~/aiida_workdir``.

    Returns:
        The configured ``Computer`` node.
    """
    from aiida import orm  # pylint: disable=import-outside-toplevel

    workdir = workdir or str(Path.home() / 'aiida_workdir')

    try:
        computer = orm.Computer.collection.get(label=label)
        logger.info("Computer '%s' already exists (PK=%d)", label, computer.pk)
        return computer
    except _LOOKUP_EXCEPTIONS:
        pass

    computer = orm.Computer(
        label=label,
        hostname='localhost',
        description='Local machine (HPC login/compute node)',
        transport_type='core.local',
        scheduler_type='core.direct',
        workdir=workdir,
    )
    computer.store()
    computer.configure()

    logger.info("Registered computer '%s' → %s", label, workdir)
    return computer


def setup_lammps_code(
    label: str = 'lammps',
    computer_label: str = 'localhost',
    executable: Optional[str] = None,
) -> 'orm.Code':
    """Register LAMMPS as an AiiDA code.

    Args:
        label: Label for the code (used as ``code_label`` elsewhere).
        computer_label: Label of the computer where LAMMPS is installed.
        executable: Path to the LAMMPS executable. Auto-detected if ``None``.

    Returns:
        The configured ``InstalledCode`` node.
    """
    from aiida import orm  # pylint: disable=import-outside-toplevel
    from aiida.common.exceptions import NotExistent  # pylint: disable=import-outside-toplevel

    full_label = f'{label}@{computer_label}'

    try:
        code = orm.load_code(full_label)
        logger.info("Code '%s' already exists (PK=%d)", full_label, code.pk)
        return code
    except (*_LOOKUP_EXCEPTIONS, NotExistent):
        pass

    # Auto-detect LAMMPS
    if executable is None:
        executable = _find_lammps_executable()

    computer = orm.Computer.collection.get(label=computer_label)

    code = orm.InstalledCode(
        label=label,
        computer=computer,
        filepath_executable=executable,
        description='LAMMPS for FrictionSim2D simulations',
        default_calc_job_plugin='friction2d.lammps',
    )
    code.store()

    logger.info("Registered code '%s' → %s", full_label, executable)
    return code


def _find_lammps_executable() -> str:
    """Attempt to locate the LAMMPS executable on the system.

    Checks common names: ``lmp``, ``lmp_mpi``, ``lmp_serial``, ``lammps``.

    Returns:
        Path to the LAMMPS executable.

    Raises:
        FileNotFoundError: If no LAMMPS executable is found.
    """
    for name in ('lmp', 'lmp_mpi', 'lmp_serial', 'lammps'):
        path = shutil.which(name)
        if path:
            return path

    raise FileNotFoundError(
        "Could not find LAMMPS executable. "
        "Specify the path explicitly with --executable."
    )


def check_daemon_status() -> dict:
    """Check the status of the AiiDA daemon.

    Returns:
        Dict with ``running`` (bool), ``workers`` (int), ``pid`` (int or None).
    """
    result = subprocess.run(
        ['verdi', 'daemon', 'status'],
        capture_output=True, text=True, check=False,
    )
    output = result.stdout.lower()

    return {
        'running': 'running' in output and 'not running' not in output,
        'output': result.stdout.strip(),
    }


def start_daemon(n_workers: int = 1) -> bool:
    """Start the AiiDA daemon.

    Args:
        n_workers: Number of daemon workers.

    Returns:
        ``True`` if the daemon was started successfully.
    """
    result = subprocess.run(
        ['verdi', 'daemon', 'start', str(n_workers)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode == 0:
        logger.info("AiiDA daemon started with %d worker(s)", n_workers)
        return True

    logger.error("Failed to start daemon: %s", result.stderr)
    return False


def full_setup(
    profile_name: str = 'friction2d',
    computer_label: str = 'localhost',
    lammps_label: str = 'lammps',
    lammps_executable: Optional[str] = None,
    start_daemon_workers: int = 0,
) -> dict:
    """First-time AiiDA setup for FrictionSim2D.

    localhost only. The SSH-connected HPC mode is gone (spec U5): the HPC
    blocks direct connections, so the two supported remote workflows are
    remote-manual (rsync + `aiida import`) and hpc-direct (AiiDA in the HPC
    conda env, `aiida export` there, `import-archive` at home). No RabbitMQ
    is started -- `aiida submit` runs through the engine unless `--daemon`
    is passed.

    Args:
        profile_name: AiiDA profile name to CREATE.
        computer_label: Computer label to register.
        lammps_label: LAMMPS code label.
        lammps_executable: Path to LAMMPS (auto-detected if ``None``).
        start_daemon_workers: Number of daemon workers to start; ``0`` (the
            default) starts none.

    Returns:
        Dict with setup results (``profile``, ``computer``, ``code``,
        ``daemon_workers``, ``daemon``).
    """
    results = {}
    results['profile'] = setup_profile(profile_name)
    if not results['profile']:
        raise RuntimeError("AiiDA profile setup failed — cannot continue")
    results['computer'] = setup_localhost_computer(computer_label)
    results['code'] = setup_lammps_code(
        lammps_label, computer_label, lammps_executable
    )
    results['daemon_workers'] = start_daemon_workers
    results['daemon'] = (start_daemon(start_daemon_workers)
                         if start_daemon_workers else False)
    return results
