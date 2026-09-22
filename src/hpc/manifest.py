"""Job manifest for offline HPC workflow tracking."""

import json
import logging
import re
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, fields, asdict

from ..core.observables import FAMILIES

logger = logging.getLogger(__name__)


def _known_fields(data_cls: type, data: Dict[str, Any], kind: str) -> Dict[str, Any]:
    """Drop dict keys that are not fields of *data_cls*, logging what's dropped.

    Tolerates a manifest.json written by an older version of the dataclass --
    e.g. the AiiDA-uuid fields and status methods dropped from ``JobEntry`` --
    so an existing file still loads instead of raising ``TypeError`` on the
    unexpected keyword.
    """
    known = {f.name for f in fields(data_cls)}
    legacy_keys = data.keys() - known
    if legacy_keys:
        logger.debug("Dropping legacy %s keys: %s", kind, sorted(legacy_keys))
    return {k: v for k, v in data.items() if k in known}


class JobStatus(Enum):
    """Simulation job lifecycle states."""
    PREPARED = "prepared"
    EXPORTED = "exported"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    IMPORTED = "imported"


@dataclass
class JobEntry:
    """Single simulation job with status tracking.
    
    Attributes:
        job_id: Unique identifier for the job.
        simulation_path: Relative path to simulation directory.
        lammps_script: Optional LAMMPS script name (e.g., 'slide_20ms.in').
        material: Material identifier.
        layers: Number of layers in the material.
        force: Applied force in nN.
        angle: Scan angle in degrees.
        speed: Sliding speed in m/s.
        status: Current job status.
        hpc_job_id: PBS/SLURM job ID when submitted.
        created_at: ISO timestamp of creation.
        submitted_at: ISO timestamp of submission.
        completed_at: ISO timestamp of completion.
        error_message: Error description if failed.
    """
    job_id: str
    simulation_path: str
    lammps_script: Optional[str] = None
    material: str = ""
    layers: int = 1
    force: float = 0.0
    angle: float = 0.0
    speed: float = 2.0
    status: str = JobStatus.PREPARED.value
    hpc_job_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    submitted_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobEntry':
        """Create JobEntry from dictionary.

        Tolerates legacy keys no longer in the dataclass (see `_known_fields`).

        Args:
            data: Dictionary with job entry fields.

        Returns:
            New JobEntry instance.
        """
        return cls(**_known_fields(cls, data, 'JobEntry'))


@dataclass
class JobManifest:
    """Batch job tracker for offline HPC workflows with AiiDA integration.
    
    Central tracking system for simulations in disconnected HPC workflows where
    AiiDA cannot directly monitor jobs. Stores job metadata, status, and enables
    manual HPC job management while maintaining provenance.
    
    Attributes:
        name: Manifest identifier.
        created_at: ISO timestamp of creation.
        last_updated: ISO timestamp of last modification.
        config_file: Path to source config file.
        source_directory: Path to simulation root directory.
        jobs: List of job entries.
        scheduler: HPC scheduler type ('pbs' or 'slurm').
        package_directory: Path to HPC package if exported.
    """
    name: str = "friction2d_manifest"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    config_file: Optional[str] = None
    source_directory: Optional[str] = None
    jobs: List[JobEntry] = field(default_factory=list)
    scheduler: str = "pbs"
    package_directory: Optional[str] = None

    #: Top-directory vocabulary for `from_simulation_directory`'s material
    #: parse. An alias of `core.observables.FAMILIES` -- see the constraint
    #: in that module's docstring on why `hpc/` imports FAMILIES rather than
    #: `run.naming.MODELS` (importing `run` here would close a cycle).
    _TOP_DIRS = FAMILIES

    def __post_init__(self):
        if self.jobs is None:
            self.jobs = []

    @property
    def n_jobs(self) -> int:
        """Total number of jobs in manifest."""
        return len(self.jobs)

    def _count_status(self, *statuses: JobStatus) -> int:
        status_values = {s.value for s in statuses}
        return sum(1 for j in self.jobs if j.status in status_values)

    @property
    def n_prepared(self) -> int:
        """Number of prepared jobs."""
        return self._count_status(JobStatus.PREPARED)

    @property
    def n_submitted(self) -> int:
        """Number of submitted or running jobs."""
        return self._count_status(JobStatus.SUBMITTED, JobStatus.RUNNING)

    @property
    def n_completed(self) -> int:
        """Number of completed jobs."""
        return self._count_status(JobStatus.COMPLETED)

    @property
    def n_failed(self) -> int:
        """Number of failed jobs."""
        return self._count_status(JobStatus.FAILED)

    @property
    def n_imported(self) -> int:
        """Number of jobs with imported results."""
        return self._count_status(JobStatus.IMPORTED)

    def add_job(self, entry: JobEntry) -> None:
        """Add job entry to manifest.
        
        Args:
            entry: JobEntry to add.
        """
        self.jobs.append(entry)
        self._update_timestamp()

    def get_job(self, job_id: str) -> Optional[JobEntry]:
        """Get job by ID.
        
        Args:
            job_id: Job identifier.
            
        Returns:
            JobEntry if found, None otherwise.
        """
        return next((j for j in self.jobs if j.job_id == job_id), None)

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

    def _update_timestamp(self) -> None:
        self.last_updated = datetime.now().isoformat()

    def save(self, filepath: Path) -> None:
        """Save manifest to JSON file.
        
        Args:
            filepath: Path to save manifest.
        """
        data = {
            'name': self.name,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'config_file': self.config_file,
            'source_directory': self.source_directory,
            'scheduler': self.scheduler,
            'package_directory': self.package_directory,
            'jobs': [job.to_dict() for job in self.jobs],
        }
        Path(filepath).write_text(json.dumps(data, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, filepath: Path) -> 'JobManifest':
        """Load manifest from JSON file.

        Tolerates legacy keys no longer in the dataclass (see `_known_fields`).

        Args:
            filepath: Path to manifest file.

        Returns:
            Loaded JobManifest instance.
        """
        data = json.loads(Path(filepath).read_text(encoding='utf-8'))
        jobs = [JobEntry.from_dict(j) for j in data.pop('jobs', [])]
        manifest = cls(**_known_fields(cls, data, 'JobManifest'))
        manifest.jobs = jobs
        return manifest

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
                if len(parts) >= 2 and parts[0] in cls._TOP_DIRS
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
                ))
        return manifest

    def __repr__(self) -> str:
        return (f"<JobManifest: {self.name} "
                f"({self.n_jobs} jobs, {self.n_completed} completed, "
                f"{self.n_failed} failed)>")
