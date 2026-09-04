"""Driving HPC script generation over a finished output tree.

Split out of src/core/run.py. Named hpc_dispatch rather than hpc so it does not
read as a second src/hpc/ -- that package generates the scripts, this module
decides when and over which directories to call it.
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from ..hpc import HPCScriptGenerator, HPCConfig
from .naming import layer_aware_path_sort_key

logger = logging.getLogger(__name__)


def count_generated_jobs(simulation_root: Path) -> int:
    """Exact job count for an already-generated tree.

    Uses the same discovery as the HPC manifest: every ``*slide*.in`` in a
    simulation directory, falling back to ``*system*.in`` when there are none.
    """
    paths = collect_hpc_simulation_paths(simulation_root)
    entries, _ = _build_hpc_manifest_entries(simulation_root, paths, [])
    return len(entries)


def collect_hpc_simulation_paths(
    simulation_root: Path,
    lammps_scripts: Optional[List[str]] = None,
) -> List[str]:
    """Collect simulation paths relative to a shared simulation root."""
    scripts = ['system.in', 'slide.in'] if lammps_scripts is None else lammps_scripts

    def script_exists(lammps_dir: Path, script_name: str) -> bool:
        if '*' in script_name or '?' in script_name:
            return any(lammps_dir.glob(script_name))
        if script_name == 'system.in':
            return (
                (lammps_dir / 'system.in').exists()
                or any(lammps_dir.glob('*system*.in'))
            )
        if script_name == 'slide.in':
            return (
                (lammps_dir / 'slide.in').exists()
                or any(lammps_dir.glob('*slide*.in'))
            )
        return (lammps_dir / script_name).exists()

    simulation_paths = []
    for lammps_dir in simulation_root.rglob('lammps'):
        if not lammps_dir.is_dir():
            continue

        if scripts:
            if not any(script_exists(lammps_dir, script) for script in scripts):
                continue
        elif not list(lammps_dir.glob('*.in')):
            continue

        rel_path = lammps_dir.parent.relative_to(simulation_root)
        simulation_paths.append(str(rel_path))
    return sorted(set(simulation_paths), key=layer_aware_path_sort_key)


def _build_hpc_manifest_entries(
    simulation_root: Path,
    simulation_paths: List[str],
    configured_scripts: List[str],
) -> Tuple[List[str], List[str]]:
    """Build manifest rows as full script paths discovered from built outputs.

    Each row is a full script path relative to ``simulation_root`` such as
    ``sheetonsheet/.../lammps/slide_p1gpa.in``.

    Discovery strategy per simulation directory:
        - If any ``*slide*.in`` files exist, add all of them (array-per-slide).
        - Otherwise add all ``*system*.in`` files.
    """
    manifest_entries: List[str] = []

    for sim_rel in simulation_paths:
        lammps_dir = simulation_root / sim_rel / 'lammps'
        if not lammps_dir.exists():
            continue

        system_scripts = sorted(
            p.relative_to(simulation_root).as_posix()
            for p in lammps_dir.glob('*.in')
            if 'system' in p.name
        )
        slide_scripts = sorted(
            p.relative_to(simulation_root).as_posix()
            for p in lammps_dir.glob('*.in')
            if 'slide' in p.name
        )

        if slide_scripts:
            manifest_entries.extend(slide_scripts)
        elif system_scripts:
            manifest_entries.extend(system_scripts)

    if manifest_entries:
        return manifest_entries, ['__MANIFEST_PATH__']

    return simulation_paths, configured_scripts


def dispatch_hpc_generation(
    generator: 'HPCScriptGenerator',
    simulation_root: Path,
    output_dir: Path,
    scheduler: str,
    base_dir: str,
    manifest_entries: List[str],
):
    """Generate HPC scripts, choosing combined, two-phase, or single-phase layout.

    AFM simulations have one system.in and one slide.in per simulation
    directory. When both phases are present in equal numbers (1:1 per sim dir)
    we emit a single combined job that runs system then slide sequentially —
    no PBS dependency needed. When counts differ (unusual) we fall back to two
    dependent array jobs. Sheet-on-sheet simulations have only slide scripts
    and use a single array job.

    Returns whatever the chosen generator method returns.
    """
    from ..hpc.manifest import JobManifest  # pylint: disable=import-outside-toplevel
    # deferred: hpc.manifest is only needed by this combined-job path, not every dispatch call

    manifest = JobManifest.from_simulation_directory(simulation_root)
    if manifest.has_system_jobs():
        system_jobs = manifest.get_system_jobs()
        slide_jobs = manifest.get_slide_jobs()
        if len(system_jobs) == len(slide_jobs):
            logger.info(
                "System+slide jobs detected (1:1) — generating combined HPC script."
            )
            return generator.generate_combined_scripts(
                manifest, output_dir, scheduler=scheduler, base_dir=base_dir
            )
        logger.info("System/slide counts differ — generating two-phase HPC scripts.")
        return generator.generate_two_phase_scripts(
            manifest, output_dir, scheduler=scheduler, base_dir=base_dir
        )
    return generator.generate_scripts(
        simulation_paths=manifest_entries,
        output_dir=output_dir,
        scheduler=scheduler,
        base_dir=base_dir,
    )


def hpc_generation_blocker(settings) -> Optional[str]:
    """Why HPC script generation would be skipped, or None if it would run.

    Generation has a precondition that is easy to miss: ``settings.hpc.modules``
    defaults to unset, and without it nothing is written. Exposing the reason
    lets a caller say so up front instead of appearing to succeed while
    producing no scripts.
    """
    if not getattr(settings.hpc, 'modules', None):
        return (
            "The HPC 'Modules' setting is empty. It is the only one that gates "
            "job-script generation: scheduler, nodes, CPUs and walltime are all "
            "written into the scripts but none of them substitute for it. "
            "Set it to the environment modules your cluster loads before "
            "running, e.g. 'lammps, openmpi'."
        )
    return None


def generate_hpc_scripts_for_root(simulation_root: Path, settings) -> Optional[str]:
    """Generate shared HPC scripts for a simulation root directory.

    Returns:
        None when scripts were generated, otherwise the reason they were not.
    """

    blocker = hpc_generation_blocker(settings)
    if blocker is not None:
        logger.warning("Skipping HPC script generation: %s", blocker)
        return blocker

    hpc_config = HPCConfig.from_settings(settings.hpc, job_name=simulation_root.name)

    simulation_paths = collect_hpc_simulation_paths(
        simulation_root,
        lammps_scripts=hpc_config.lammps_scripts,
    )
    if not simulation_paths:
        reason = "No simulations were found to write job scripts for."
        logger.warning("%s", reason)
        return reason

    manifest_entries, resolved_scripts = _build_hpc_manifest_entries(
        simulation_root,
        simulation_paths,
        hpc_config.lammps_scripts,
    )
    hpc_config.lammps_scripts = resolved_scripts

    hpc_dir = simulation_root / 'hpc'
    hpc_dir.mkdir(parents=True, exist_ok=True)

    if hpc_config.hpc_home:
        base_dir = hpc_config.hpc_home.rstrip('/') + '/' + simulation_root.name
    elif hpc_config.scheduler_type == 'pbs':
        base_dir = '$PBS_O_WORKDIR'
    else:
        base_dir = '$SLURM_SUBMIT_DIR'
    generator = HPCScriptGenerator(hpc_config)
    dispatch_hpc_generation(
        generator,
        simulation_root,
        hpc_dir,
        hpc_config.scheduler_type,
        base_dir,
        manifest_entries,
    )
    return None
