"""Job manifest for offline HPC workflow tracking."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from ..run.naming import MODELS, mlip_flavor_from_pot_token


def parse_mlip_flavor(path_str: str) -> Optional[Tuple[str, str]]:
    """Recover ``(framework, dispersion)`` from a run path.

    ``hpc generate`` builds its manifest by discovering a *built tree*
    (:meth:`JobManifest.from_simulation_directory`), never by reading the
    configs, so the run directory is the only surviving record of which
    framework a job needs. It lives in the ``pot_<slug>_<stem>`` segment
    :func:`frictionsim2d.run.naming.output_dir_for` always emits for an MLIP sheet:
    ``pot_mace_<stem>`` / ``pot_mace-d3_<stem>`` (same for deepmd), decoded by
    :func:`frictionsim2d.run.naming.mlip_flavor_from_pot_token`.

    Every segment is checked, not just the last, because AFM nests a per-layer
    ``L<N>`` directory (and an ``il-*`` level may follow) *below* the pot
    segment, and the path may carry the simulation root in front.

    Returns:
        ``(framework, 'none' | 'd3')``, or ``None`` for a classical path.
    """
    for segment in reversed(Path(path_str).parts):
        flavor = mlip_flavor_from_pot_token(segment)
        if flavor is not None:
            return flavor
    return None


def _mlip_job_fields(path_str: str) -> Dict[str, Any]:
    """Resource fields for a discovered job, inferred from its path.

    Classical jobs get today's defaults verbatim, so a non-MLIP campaign still
    renders as a single CPU array exactly as before.
    """
    flavor = parse_mlip_flavor(path_str)
    if flavor is None:
        return {}
    framework = flavor[0]
    return {
        'needs_gpu': True,
        'framework': framework,
    }


@dataclass
class JobEntry:
    """Single simulation job.
    
    Attributes:
        job_id: Unique identifier for the job.
        simulation_path: Relative path to simulation directory.
        lammps_script: Optional LAMMPS script name (e.g., 'slide_20ms.in').
        material: Material identifier.
        layers: Number of layers in the material.
        force: Applied force in nN.
        angle: Scan angle in degrees.
        speed: Sliding speed in m/s.
        needs_gpu: Whether this job must run on a GPU node.
        framework: MLIP framework driving the job ('deepmd' | 'mace'), or None
            for a classical potential. Together with ``needs_gpu`` this is the
            key ``HPCScriptGenerator`` groups on: MACE and DeePMD are two
            mutually incompatible LAMMPS builds, so they cannot share one array
            job even when their resource request is identical.
    """
    job_id: str
    simulation_path: str
    lammps_script: Optional[str] = None
    material: str = ""
    layers: int = 1
    force: float = 0.0
    angle: float = 0.0
    speed: float = 2.0
    # MLIP resource fields; the defaults describe a plain CPU job.
    needs_gpu: bool = False
    framework: Optional[str] = None


@dataclass
class JobManifest:
    """The LAMMPS scripts discovered in a built simulation tree.

    Attributes:
        name: Manifest identifier.
        source_directory: Path to simulation root directory.
        jobs: List of job entries.
    """
    name: str = "friction2d_manifest"
    source_directory: Optional[str] = None
    jobs: List[JobEntry] = field(default_factory=list)

    def __post_init__(self):
        if self.jobs is None:
            self.jobs = []

    @property
    def n_jobs(self) -> int:
        """Total number of jobs in manifest."""
        return len(self.jobs)

    def add_job(self, entry: JobEntry) -> None:
        """Add job entry to manifest.
        
        Args:
            entry: JobEntry to add.
        """
        self.jobs.append(entry)

    def get_system_jobs(self) -> List[JobEntry]:
        """Get all system initialization jobs.
        
        Returns:
            List of jobs with system*.in scripts.
        """
        return [j for j in self.jobs
                if j.lammps_script and j.lammps_script.startswith('system')]

    def get_slide_jobs(self) -> List[JobEntry]:
        """Get all slide simulation jobs.
        
        Returns:
            List of jobs with slide*.in scripts.
        """
        return [j for j in self.jobs
                if j.lammps_script and j.lammps_script.startswith('slide')]

    def has_system_jobs(self) -> bool:
        """Check if manifest contains system initialization jobs.
        
        Returns:
            True if system jobs exist.
        """
        return bool(self.get_system_jobs())

    @classmethod
    def from_simulation_directory(cls, sim_dir: Path,
                                    name: Optional[str] = None) -> 'JobManifest':
        """Create manifest by discovering jobs in simulation directory.
        
        Scans directory tree for lammps/ subdirectories and creates job entries
        for all discovered LAMMPS scripts. Processes system*.in scripts first,
        then slide*.in scripts.
        
        Args:
            sim_dir: Directory containing simulation outputs.
            name: Optional manifest name (defaults to directory name).
            
        Returns:
            New JobManifest with entries for all discovered simulations.
        """
        sim_dir = Path(sim_dir)
        manifest = cls(name=name or sim_dir.name, source_directory=str(sim_dir))
        layer_pattern = re.compile(r'l[_]?(\d+)', re.IGNORECASE)

        sim_directories = []
        for lammps_dir in sim_dir.rglob('lammps'):
            if not lammps_dir.is_dir():
                continue
            sim_path = lammps_dir.parent
            rel_path = sim_path.relative_to(sim_dir)
            path_str = rel_path.as_posix()
            parts = rel_path.parts
            material = (
                parts[1]
                if len(parts) >= 2 and parts[0] in MODELS
                else (parts[0] if parts else "")
            )
            layer_match = layer_pattern.search(path_str)
            layers = int(layer_match.group(1)) if layer_match else 1
            sim_directories.append({
                'lammps_dir': lammps_dir,
                'rel_path': rel_path,
                'path_str': path_str,
                'material': material,
                'layers': layers,
                # GPU/framework facts survive only in the run-dir flavour token.
                'mlip': _mlip_job_fields(path_str),
            })

        for sim_info in sim_directories:
            lammps_dir = sim_info['lammps_dir']
            for script_name in sorted(p.name for p in lammps_dir.glob('system*.in')):
                script_tag = Path(script_name).stem.replace('-', '_')
                job_id = f"{sim_info['path_str']}_{script_tag}".replace('/', '_').replace('-', '_')
                manifest.add_job(JobEntry(
                    job_id=job_id,
                    simulation_path=str(sim_info['rel_path']),
                    lammps_script=script_name,
                    material=sim_info['material'],
                    layers=sim_info['layers'],
                    **sim_info['mlip'],
                ))

        for sim_info in sim_directories:
            lammps_dir = sim_info['lammps_dir']
            slide_scripts = sorted(p.name for p in lammps_dir.glob('slide*.in'))
            if not slide_scripts:
                if not list(lammps_dir.glob('system*.in')):
                    job_id = sim_info['path_str'].replace('/', '_').replace('-', '_')
                    manifest.add_job(JobEntry(
                        job_id=job_id,
                        simulation_path=str(sim_info['rel_path']),
                        material=sim_info['material'],
                        layers=sim_info['layers'],
                        **sim_info['mlip'],
                    ))
                continue

            for script_name in slide_scripts:
                stem = Path(script_name).stem
                script_tag = stem.replace('-', '_')
                job_id = f"{sim_info['path_str']}_{script_tag}".replace('/', '_').replace('-', '_')
                speed = 2.0
                speed_match = re.match(r"slide_([0-9]+(?:p[0-9]+)?)ms", stem)
                if speed_match:
                    speed = float(speed_match.group(1).replace('p', '.'))
                # Per-load slide scripts (AFM outer_loop=force): slide_f<load>N.in
                force = 0.0
                force_match = re.match(r"slide_f([0-9]+(?:p[0-9]+)?)N$", stem)
                if force_match:
                    force = float(force_match.group(1).replace('p', '.'))
                # Per-angle slide scripts (outer_loop=scan_angle): slide_a<token>.in,
                # token from format_numeric_token: 'm' negative prefix, 'p' decimal.
                angle = 0.0
                angle_match = re.match(r"slide_a(m?[0-9]+(?:p[0-9]+)?)$", stem)
                if angle_match:
                    token = angle_match.group(1)
                    sign = -1.0 if token.startswith('m') else 1.0
                    digits = token[1:] if token.startswith('m') else token
                    angle = sign * float(digits.replace('p', '.'))
                manifest.add_job(JobEntry(
                    job_id=job_id,
                    simulation_path=str(sim_info['rel_path']),
                    lammps_script=script_name,
                    material=sim_info['material'],
                    layers=sim_info['layers'],
                    force=force,
                    angle=angle,
                    speed=speed,
                    **sim_info['mlip'],
                ))
        return manifest

    def __repr__(self) -> str:
        return (f"<JobManifest: {self.name} "
                f"({self.n_jobs} jobs)>")
