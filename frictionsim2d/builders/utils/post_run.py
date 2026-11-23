"""Post-run enrichment of a build-time ``provenance.json`` from LAMMPS logs."""
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _configured_loads(output_root: Path) -> Optional[list]:
    """``general.force`` from the run's ``provenance/config.json``, as a list."""
    for base in (output_root, output_root.parent):
        cfg_path = base / 'provenance' / 'config.json'
        if not cfg_path.is_file():
            continue
        try:
            force = json.loads(cfg_path.read_text(encoding='utf-8'))['general']['force']
        except (ValueError, OSError, KeyError, TypeError) as exc:
            logger.warning("Could not read loads from %s (%s).", cfg_path, exc)
            return None
        if force is None:
            return None
        return [float(x) for x in (force if isinstance(force, list) else [force])]
    return None


def enrich_provenance_post_run(output_root: Path,
                               log_path: Path) -> Optional[Path]:
    """Enrich a build-time provenance.json with post-run log fields.

    Parses ``log_path`` (the system.in log, source of the
    ``RAMP_STATUS``/``UNLOADED_REF`` lines) via ``provenance.parse_log_extras``
    and merges the result into the ``provenance.json`` written at build time:
    ``ramp_*``, ``unloaded_reference_z`` and ``lammps_version``. Reached from
    ``FrictionSim2D hpc enrich-provenance`` and the generated job scripts.

    Repeated calls with different logs from the same run are safe: the merge
    only sets the keys a log supplied. When the two phases ran against
    different LAMMPS builds the last log wins ``lammps_version``.

    Returns the provenance.json path, or ``None`` if the record or the log is
    missing, or the record on disk is unreadable (logged, not raised: a run's
    real output must never be hidden behind a failed enrichment).
    """
    # pylint: disable-next=import-outside-toplevel
    from ...data.provenance import (
        parse_hold_files, parse_log_extras, read_provenance, write_provenance)

    output_root = Path(output_root)
    log_path = Path(log_path)
    prov_path = output_root / 'provenance.json'
    if not prov_path.is_file():
        logger.warning("No provenance.json in %s to enrich.", output_root)
        return None
    if not log_path.is_file():
        logger.warning("No log at %s to enrich provenance from.", log_path)
        return None

    try:
        record = read_provenance(prov_path)
    # ValidationError (a ValueError subclass) covers both a truncated/invalid
    # JSON body and a well-formed document missing required fields; OSError
    # covers an unreadable/undecodable file. A killed job leaving a half-written
    # provenance.json is exactly the case this function exists to report on, so
    # it must not be the one case that raises.
    except (ValueError, OSError) as exc:
        logger.warning(
            "Could not read provenance.json at %s (%s); leaving it "
            "unenriched.", prov_path, exc)
        return None

    loads = _configured_loads(output_root)
    updates = parse_log_extras(log_path, loads)
    achieved = parse_hold_files(output_root / 'data')
    if achieved:
        updates['load_achieved_by_load'] = achieved
        # The last configured load when known, else the largest held load:
        # `achieved` is in filename order, where hold_5N sorts after hold_10N.
        last = f'{float(loads[-1]):g}' if loads else None
        updates['load_achieved'] = achieved.get(
            last, achieved[max(achieved, key=float)])
    if not updates:
        logger.warning(
            "No RAMP_STATUS/UNLOADED_REF/LAMMPS-version lines found in "
            "%s; provenance.json left unenriched.", log_path)
        return prov_path

    record = record.model_copy(update=updates)
    if record.ramp_converged is False:
        logger.warning(
            "Settle loop did not converge for %s (r=%s after %s "
            "iterations) -- tip z was still oscillating past the "
            "iteration budget when it hit ramp_max_iterations. See "
            "provenance.json's ramp_r/ramp_iterations.",
            output_root, record.ramp_r, record.ramp_iterations)

    write_provenance(output_root, record)
    return prov_path
