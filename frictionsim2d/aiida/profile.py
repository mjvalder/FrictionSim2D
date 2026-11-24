"""The AiiDA profile: creating it, and checking it is usable.

Two parts, both importable without aiida-core (every AiiDA import is
deferred into the function that needs it):

* **Setup** (``FrictionSim2D aiida setup``): :func:`setup_profile` and
  :func:`full_setup`. Only the profile is created here. The store is
  import-only: the machine ``Computer`` and the LAMMPS ``Code`` of each run
  are created when it is imported (:mod:`frictionsim2d.aiida.store`), and no daemon
  is ever needed.
* **Pre-flight** (``FrictionSim2D aiida preflight``):
  :func:`preflight` and :func:`main`. Distinguishes "no profile initialised
  here" from "profile broken" -- the generation machine may legitimately
  have no profile (simulations run elsewhere under the two-tier setup), and
  the campaign generator must fail loudly with the right message rather
  than a stack trace. Never raises: every check is wrapped and reported in
  ``messages``.
"""

import importlib.util
import logging
import subprocess
from dataclasses import dataclass, field
from typing import List

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
    version, :func:`frictionsim2d.aiida.store.machine_computer` and
    :func:`~frictionsim2d.aiida.store.lammps_code`), so nothing else is registered.

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


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

@dataclass
class PreflightReport:
    """What `preflight()` found, and what to print about it."""

    profile_found: bool = False
    profile_ok: bool = False
    entry_points_ok: bool = False
    messages: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (self.profile_found and self.profile_ok
                and self.entry_points_ok)


#: The plugins the store needs: ``(entry-point group, name)``.
REQUIRED_ENTRY_POINTS = (('aiida.calculations', 'friction2d.lammps'),
                         ('aiida.parsers', 'friction2d.lammps'))


def _check_entry_points(messages: List[str]) -> bool:
    """Whether the CalcJob and parser entry points are registered and no
    ``friction2d.*`` ``aiida.data`` one is (the custom Data nodes are gone;
    a stale one means the package metadata predates their removal)."""
    from importlib.metadata import entry_points  # pylint: disable=import-outside-toplevel

    ok = True
    for group, name in REQUIRED_ENTRY_POINTS:
        if any(ep.name == name for ep in entry_points(group=group)):
            messages.append(f"entry point {group}:{name} registered")
        else:
            ok = False
            messages.append(f"entry point {group}:{name} NOT registered — "
                            "reinstall the package (pip install -e .)")
    stale = sorted(ep.name for ep in entry_points(group='aiida.data')
                   if ep.name.startswith('friction2d'))
    if stale:
        ok = False
        messages.append(f"stale aiida.data entry points {stale} — the friction2d "
                        "Data types are gone; reinstall the package (pip install -e .)")
    return ok


def preflight() -> PreflightReport:
    # This module -- and this function most of all -- must stay importable
    # and callable without aiida-core installed (a generation-only machine is
    # a supported setup, see the module docstring), so every AiiDA import
    # below is deferred to the branch that actually needs it.
    # pylint: disable=import-outside-toplevel
    rep = PreflightReport()

    if importlib.util.find_spec('aiida') is None:
        rep.messages.append(
            "aiida-core is not installed in this environment "
            "(pip install -e '.[aiida]')")
        return rep

    # profile discovery vs. profile health
    try:
        from aiida.manage.configuration import get_config
        config = get_config()
        profiles = list(config.profiles)
        if not profiles:
            rep.messages.append(
                "no AiiDA profile initialised on this machine — run "
                "`FrictionSim2D aiida setup` on the run host; generation-only "
                "machines can proceed without one")
        else:
            rep.profile_found = True
            rep.messages.append(
                f"profiles: {[p.name for p in profiles]}")
    except Exception as exc:  # pylint: disable=broad-except
        rep.messages.append(f"could not read AiiDA config: {exc}")
        return rep

    if rep.profile_found:
        try:
            from aiida import load_profile
            from aiida.orm import QueryBuilder, Computer
            load_profile()
            QueryBuilder().append(Computer).count()
            rep.profile_ok = True
            rep.messages.append("profile loads and the database answers")
        except Exception as exc:  # pylint: disable=broad-except
            rep.messages.append(
                f"profile exists but is BROKEN (database unreachable?): {exc}")

    try:
        rep.entry_points_ok = _check_entry_points(rep.messages)
    except Exception as exc:  # pylint: disable=broad-except
        rep.messages.append(f"entry-point scan failed: {exc}")

    return rep


def main() -> int:
    rep = preflight()
    for msg in rep.messages:
        print(f"  - {msg}")
    print(f"profile_found={rep.profile_found} profile_ok={rep.profile_ok} "
          f"entry_points_ok={rep.entry_points_ok}")
    return 0 if rep.ok else 1
