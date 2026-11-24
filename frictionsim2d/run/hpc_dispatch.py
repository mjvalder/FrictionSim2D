"""Driving HPC script generation over a finished output tree.

``frictionsim2d/hpc/`` generates the scripts; this module decides when and over
which directories to call it.
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
    """Collect simulation paths relative to a shared simulation root.

    A ``lammps/`` holding a ``quench*.in`` (a member of a ``run quench`` root)
    always counts: it is a single-phase deck whatever ``lammps_scripts`` says.
    """
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

        if any(lammps_dir.glob('quench*.in')):
            pass  # a quench member: always a job
        elif scripts and not any(script_exists(lammps_dir, script) for script in scripts):
            continue
        elif not scripts and not list(lammps_dir.glob('*.in')):
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
        - Otherwise add all ``quench*.in`` files (one melt-quench per member).
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
        else:
            manifest_entries.extend(sorted(
                p.relative_to(simulation_root).as_posix()
                for p in lammps_dir.glob('quench*.in')))

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
    preflight: bool = False,
):
    """Generate HPC scripts, choosing combined, two-phase, or single-phase layout.

    AFM simulations have one system.in and one slide.in per simulation
    directory. When both phases are present in equal numbers (1:1 per sim dir)
    we emit a single combined job that runs system then slide sequentially —
    no PBS dependency needed. When counts differ (unusual) we fall back to two
    dependent array jobs. Sheet-on-sheet simulations have only slide scripts
    and use a single array job.

    With ``preflight`` it also writes ``preflight.sh`` and its scheduler's
    submit wrapper (``preflight.pbs`` / ``preflight.slurm``) next to the job
    scripts.

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
            result = generator.generate_combined_scripts(
                manifest, output_dir, scheduler=scheduler, base_dir=base_dir,
                simulation_root=simulation_root,
            )
        else:
            logger.info("System/slide counts differ — generating two-phase HPC scripts.")
            result = generator.generate_two_phase_scripts(
                manifest, output_dir, scheduler=scheduler, base_dir=base_dir,
                simulation_root=simulation_root,
            )
    else:
        result = generator.generate_scripts(
            simulation_paths=manifest_entries,
            output_dir=output_dir,
            scheduler=scheduler,
            base_dir=base_dir,
            simulation_root=simulation_root,
        )

    if preflight:
        generator.generate_preflight(output_dir)
    return result


def generate_scripts_for_tree(
    simulation_root: Path,
    hpc_config: 'HPCConfig',
    output_dir: Path,
    simulation_paths: List[str],
    base_dir: str,
    preflight: Optional[bool] = None,
):
    """Write the job scripts for ``simulation_paths`` found under ``simulation_root``.

    ``preflight`` defaults to ``hpc_config.preflight``. Returns whatever
    :func:`dispatch_hpc_generation` returns.
    """
    manifest_entries, resolved_scripts = _build_hpc_manifest_entries(
        simulation_root, simulation_paths, hpc_config.lammps_scripts,
    )
    hpc_config.lammps_scripts = resolved_scripts
    return dispatch_hpc_generation(
        HPCScriptGenerator(hpc_config),
        simulation_root,
        output_dir,
        hpc_config.scheduler_type,
        base_dir,
        manifest_entries,
        preflight=hpc_config.preflight if preflight is None else preflight,
    )


def hpc_generation_blocker(settings) -> Optional[str]:
    """Why HPC script generation would be skipped, or None if it would run.

    Nothing is written unless one of ``hpc.modules``, ``hpc.conda_env`` or
    ``hpc.env_lines`` is set.
    """
    hpc = settings.hpc
    if not hpc.modules and not hpc.conda_env and not hpc.env_lines:
        return (
            "The HPC 'Modules', 'Conda environment' and 'env_lines' settings are all "
            "empty. Set one of them so the scripts know where LAMMPS comes from: "
            "modules, e.g. 'lammps, openmpi'; a conda prefix holding lmp, e.g. "
            "'$HOME/envs/lammps'; or explicit env_lines."
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

    hpc_dir = simulation_root / 'hpc'
    hpc_dir.mkdir(parents=True, exist_ok=True)
    generate_scripts_for_tree(
        simulation_root, hpc_config, hpc_dir, simulation_paths,
        hpc_config.base_dir(simulation_root.name),
    )
    return None
