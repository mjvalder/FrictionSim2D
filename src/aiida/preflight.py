"""AiiDA pre-flight verification (spec §8).

Distinguishes "no profile initialised here" from "profile broken" — the
generation machine may legitimately have no profile (simulations run
elsewhere under the two-tier setup), and the campaign generator must fail
loudly with the right message rather than a stack trace.

Never raises: every check is wrapped and reported in ``messages``.
"""
import importlib.util
from dataclasses import dataclass, field
from typing import List


@dataclass
class PreflightReport:
    """What `preflight()` found, and what to print about it."""

    profile_found: bool = False
    profile_ok: bool = False
    sync_configured: bool = False
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
                "frictionsim2d-start-aiida on the run host; generation-only "
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
            from aiida.orm import Computer as _C
            from aiida import orm
            remotes = orm.QueryBuilder().append(
                _C, project=['label']).all(flat=True)
            rep.sync_configured = bool(remotes)
            rep.messages.append(f"computers configured: {list(remotes)}")
        except Exception as exc:  # pylint: disable=broad-except
            rep.messages.append(f"computer listing failed: {exc}")

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
          f"sync_configured={rep.sync_configured} "
          f"entry_points_ok={rep.entry_points_ok}")
    return 0 if rep.ok else 1
