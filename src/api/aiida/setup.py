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
from typing import Optional, TYPE_CHECKING

from ..core.config import HPCSettings, AiidaSettings

if TYPE_CHECKING:
    from aiida import orm

logger = logging.getLogger(__name__)

_LOOKUP_EXCEPTIONS = (LookupError, ValueError, RuntimeError)

_SCHEDULER_MAP = {
    'pbs': 'core.pbspro',
    'pbspro': 'core.pbspro',
    'slurm': 'core.slurm',
}

_TRANSPORT_MAP = {
    'ssh': 'core.ssh',
    'local': 'core.local',
}


def start_rabbitmq() -> bool:
    """Start the RabbitMQ broker in detached mode.

    AiiDA requires RabbitMQ for daemon communication. The broker is
    provided by the ``aiida-core.services`` conda package.

    Returns:
        ``True`` if RabbitMQ is already running or was started successfully.
    """
    # Check if already running
    probe = subprocess.run(
        ['rabbitmqctl', 'status'],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode == 0:
        logger.info("RabbitMQ is already running")
        return True

    logger.info("Starting RabbitMQ broker ...")
    result = subprocess.run(
        ['rabbitmq-server', '-detached'],
        capture_output=True, text=True, check=False,
    )
    if result.returncode == 0:
        logger.info("RabbitMQ started in detached mode")
        return True

    logger.error(
        "Failed to start RabbitMQ: %s\n"
        "Install via: conda install -c conda-forge aiida-core.services",
        result.stderr,
    )
    return False


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
    )

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

    if result.returncode != 0:
        logger.error("Profile creation failed: %s", result.stderr)
        return False

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

    full_label = f'{label}@{computer_label}'

    try:
        code = orm.load_code(full_label)
        logger.info("Code '%s' already exists (PK=%d)", full_label, code.pk)
        return code
    except _LOOKUP_EXCEPTIONS:
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


def setup_computer_from_hpc_settings(
    hpc_settings: 'HPCSettings',
    aiida_settings: 'AiidaSettings',
) -> 'orm.Computer':
    """Register an AiiDA computer from settings.

    Args:
        hpc_settings: HPCSettings from GlobalSettings.hpc
        aiida_settings: AiidaSettings from GlobalSettings.aiida

    Returns:
        Configured AiiDA Computer instance
    """
    from aiida import orm  # pylint: disable=import-outside-toplevel

    label = aiida_settings.computer_label or 'hpc'
    scheduler = hpc_settings.scheduler_type
    transport = aiida_settings.transport

    scheduler_type = _SCHEDULER_MAP.get(scheduler, scheduler) or 'core.pbspro'
    transport_type = _TRANSPORT_MAP.get(transport, transport) or 'core.local'

    try:
        computer = orm.Computer.collection.get(label=label)
        logger.info("Computer '%s' already exists (PK=%d)", label, computer.pk)
        return computer
    except _LOOKUP_EXCEPTIONS:
        pass

    computer = orm.Computer(
        label=label,
        hostname=aiida_settings.hostname or 'localhost',
        description=f"HPC computer ({scheduler})",
        transport_type=transport_type,
        scheduler_type=scheduler_type,
        workdir=aiida_settings.workdir or str(Path.home() / 'aiida_workdir'),
    )
    computer.store()

    if transport_type == 'core.ssh':
        computer.configure(
            username=aiida_settings.username or '',
            port=aiida_settings.ssh_port,
            key_filename=aiida_settings.key_filename or '',
            safe_interval=30.0
        )

    computer.set_default_mpiprocs_per_machine(hpc_settings.num_cpus)
    logger.info("Registered computer '%s' (PK=%d)", label, computer.pk)
    return computer


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
    start_daemon_workers: int = 1,
    hpc_settings: Optional['HPCSettings'] = None,
    aiida_settings: Optional['AiidaSettings'] = None,
) -> dict:
    """Complete AiiDA setup for FrictionSim2D on HPC.

    Performs all first-time setup steps in order:

    1. Start RabbitMQ broker (``rabbitmq-server -detached``)
    2. Create AiiDA profile (``verdi presto --use-postgres``)
    3. Register localhost or remote HPC as an AiiDA computer
    4. Register the LAMMPS code
    5. Start the AiiDA daemon

    Prerequisites (conda)::

        conda install -c conda-forge aiida-core aiida-core.services
        pip install --upgrade typing_extensions

    Args:
        profile_name: AiiDA profile name.
        computer_label: Computer label (used if no HPC settings provided).
        lammps_label: LAMMPS code label.
        lammps_executable: Path to LAMMPS (auto-detected if ``None``).
        start_daemon_workers: Number of daemon workers to start.
        hpc_settings: HPCSettings from settings.yaml (preferred method).
        aiida_settings: AiidaSettings from settings.yaml (preferred method).

    Returns:
        Dict with setup results (``rabbitmq``, ``profile``, ``computer``,
        ``code``, ``daemon``).
    """
    results = {}

    results['rabbitmq'] = start_rabbitmq()
    results['profile'] = setup_profile(profile_name)

    if hpc_settings or aiida_settings:
        resolved_hpc = hpc_settings or HPCSettings()
        resolved_aiida = aiida_settings or AiidaSettings()
        results['computer'] = setup_computer_from_hpc_settings(
            resolved_hpc, resolved_aiida
        )
        computer_label = results['computer'].label
    else:
        results['computer'] = setup_localhost_computer(computer_label)

    results['code'] = setup_lammps_code(
        lammps_label, computer_label, lammps_executable
    )
    results['daemon'] = start_daemon(start_daemon_workers)

    return results
