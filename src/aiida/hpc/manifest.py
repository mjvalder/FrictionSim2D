"""Job Manifest for tracking offline HPC workflow.

This module provides a manifest-based system for tracking simulations
in disconnected HPC workflows where AiiDA cannot directly monitor jobs.
"""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict


class JobStatus(Enum):
    """Status of a simulation job."""
    PREPARED = "prepared"        # Files generated, ready for submission
    EXPORTED = "exported"        # Included in HPC package
    SUBMITTED = "submitted"      # Submitted to HPC (manually marked)
    RUNNING = "running"          # Running on HPC (manually marked)
    COMPLETED = "completed"      # Finished successfully
    FAILED = "failed"            # Failed during execution
    IMPORTED = "imported"        # Results imported into AiiDA


@dataclass
class JobEntry:
    """A single job entry in the manifest."""
    
    # Identification
    job_id: str                  # Unique identifier (e.g., "afm_h-MoS2_L1_F5_A0")
    simulation_path: str         # Relative path to simulation directory
    
    # Simulation parameters (for quick reference)
    material: str = ""
    layers: int = 1
    force: float = 0.0
    angle: float = 0.0
    speed: float = 2.0
    
    # Status tracking
    status: str = JobStatus.PREPARED.value
    hpc_job_id: Optional[str] = None  # PBS/SLURM job ID when submitted
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    submitted_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # AiiDA linking
    simulation_node_uuid: Optional[str] = None
    results_node_uuid: Optional[str] = None
    
    # Error tracking
    error_message: Optional[str] = None
    
    def update_status(self, new_status: JobStatus, **kwargs):
        """Update the job status and related timestamps.
        
        Args:
            new_status: New status to set
            **kwargs: Additional fields to update
        """
        self.status = new_status.value
        
        if new_status == JobStatus.SUBMITTED:
            self.submitted_at = datetime.now().isoformat()
        elif new_status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.IMPORTED):
            self.completed_at = datetime.now().isoformat()
        
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'JobEntry':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class JobManifest:
    """Manifest tracking all jobs in a simulation batch.
    
    This manifest is the central tracking mechanism for offline HPC workflows.
    It stores information about all simulations and their current status,
    enabling the user to manually manage HPC jobs while maintaining AiiDA
    integration for provenance.
    """
    
    # Manifest metadata
    name: str = "friction2d_manifest"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Source information
    config_file: Optional[str] = None
    source_directory: Optional[str] = None
    
    # AiiDA linking
    config_node_uuid: Optional[str] = None
    provenance_node_uuid: Optional[str] = None
    
    # Jobs
    jobs: List[JobEntry] = field(default_factory=list)
    
    # HPC info
    scheduler: str = "pbs"
    package_directory: Optional[str] = None
    
    def __post_init__(self):
        """Ensure jobs is a list."""
        if self.jobs is None:
            self.jobs = []
    
    @property
    def n_jobs(self) -> int:
        """Total number of jobs."""
        return len(self.jobs)
    
    @property
    def n_prepared(self) -> int:
        """Number of jobs in prepared state."""
        return sum(1 for j in self.jobs if j.status == JobStatus.PREPARED.value)
    
    @property
    def n_submitted(self) -> int:
        """Number of jobs submitted or running."""
        statuses = {JobStatus.SUBMITTED.value, JobStatus.RUNNING.value}
        return sum(1 for j in self.jobs if j.status in statuses)
    
    @property
    def n_completed(self) -> int:
        """Number of jobs completed successfully."""
        return sum(1 for j in self.jobs if j.status == JobStatus.COMPLETED.value)
    
    @property
    def n_failed(self) -> int:
        """Number of failed jobs."""
        return sum(1 for j in self.jobs if j.status == JobStatus.FAILED.value)
    
    @property
    def n_imported(self) -> int:
        """Number of jobs with results imported."""
        return sum(1 for j in self.jobs if j.status == JobStatus.IMPORTED.value)
    
    def add_job(self, entry: JobEntry) -> None:
        """Add a job entry to the manifest.
        
        Args:
            entry: JobEntry to add
        """
        self.jobs.append(entry)
        self._update_timestamp()
    
    def get_job(self, job_id: str) -> Optional[JobEntry]:
        """Get a job by its ID.
        
        Args:
            job_id: Job identifier
            
        Returns:
            JobEntry if found, None otherwise
        """
        for job in self.jobs:
            if job.job_id == job_id:
                return job
        return None
    
    def get_jobs_by_status(self, status: JobStatus) -> List[JobEntry]:
        """Get all jobs with a specific status.
        
        Args:
            status: Status to filter by
            
        Returns:
            List of matching JobEntry objects
        """
        return [j for j in self.jobs if j.status == status.value]
    
    def get_jobs_by_material(self, material: str) -> List[JobEntry]:
        """Get all jobs for a specific material.
        
        Args:
            material: Material name
            
        Returns:
            List of matching JobEntry objects
        """
        return [j for j in self.jobs if j.material == material]
    
    def update_job_status(self, 
                         job_id: str, 
                         new_status: JobStatus,
                         **kwargs) -> bool:
        """Update status of a specific job.
        
        Args:
            job_id: Job identifier
            new_status: New status to set
            **kwargs: Additional fields to update
            
        Returns:
            True if job was found and updated, False otherwise
        """
        job = self.get_job(job_id)
        if job:
            job.update_status(new_status, **kwargs)
            self._update_timestamp()
            return True
        return False
    
    def mark_all_submitted(self, hpc_job_prefix: Optional[str] = None) -> int:
        """Mark all prepared jobs as submitted.
        
        Args:
            hpc_job_prefix: Optional prefix for HPC job IDs
            
        Returns:
            Number of jobs updated
        """
        count = 0
        for job in self.jobs:
            if job.status == JobStatus.PREPARED.value:
                hpc_id = f"{hpc_job_prefix}_{count}" if hpc_job_prefix else None
                job.update_status(JobStatus.SUBMITTED, hpc_job_id=hpc_id)
                count += 1
        
        if count > 0:
            self._update_timestamp()
        return count
    
    def mark_completed_from_results(self, results_dir: Path) -> Dict[str, str]:
        """Mark jobs as completed based on presence of result files.
        
        Args:
            results_dir: Directory containing simulation results
            
        Returns:
            Dictionary of job_id -> status for updated jobs
        """
        results_dir = Path(results_dir)
        updated = {}
        
        for job in self.jobs:
            if job.status in (JobStatus.SUBMITTED.value, JobStatus.RUNNING.value):
                # Check if results exist
                sim_path = results_dir / job.simulation_path / 'results'
                
                if sim_path.exists() and any(sim_path.iterdir()):
                    job.update_status(JobStatus.COMPLETED)
                    updated[job.job_id] = JobStatus.COMPLETED.value
        
        if updated:
            self._update_timestamp()
        
        return updated
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of manifest status.
        
        Returns:
            Dictionary with status counts and metadata
        """
        return {
            'name': self.name,
            'total_jobs': self.n_jobs,
            'prepared': self.n_prepared,
            'submitted': self.n_submitted,
            'completed': self.n_completed,
            'failed': self.n_failed,
            'imported': self.n_imported,
            'scheduler': self.scheduler,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
        }
    
    def _update_timestamp(self) -> None:
        """Update the last_updated timestamp."""
        self.last_updated = datetime.now().isoformat()
    
    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    
    def save(self, filepath: Path) -> None:
        """Save manifest to JSON file.
        
        Args:
            filepath: Path to save to
        """
        filepath = Path(filepath)
        data = {
            'name': self.name,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'config_file': self.config_file,
            'source_directory': self.source_directory,
            'config_node_uuid': self.config_node_uuid,
            'provenance_node_uuid': self.provenance_node_uuid,
            'scheduler': self.scheduler,
            'package_directory': self.package_directory,
            'jobs': [job.to_dict() for job in self.jobs],
        }
        filepath.write_text(json.dumps(data, indent=2))
    
    @classmethod
    def load(cls, filepath: Path) -> 'JobManifest':
        """Load manifest from JSON file.
        
        Args:
            filepath: Path to load from
            
        Returns:
            Loaded JobManifest instance
        """
        filepath = Path(filepath)
        data = json.loads(filepath.read_text())
        
        jobs = [JobEntry.from_dict(j) for j in data.pop('jobs', [])]
        
        manifest = cls(**data)
        manifest.jobs = jobs
        
        return manifest
    
    @classmethod
    def from_simulation_directory(cls,
                                   sim_dir: Path,
                                   name: Optional[str] = None) -> 'JobManifest':
        """Create a manifest from a FrictionSim2D output directory.
        
        Args:
            sim_dir: Directory containing simulation outputs
            name: Optional manifest name
            
        Returns:
            New JobManifest with entries for all simulations
        """
        import re
        
        sim_dir = Path(sim_dir)
        manifest = cls(
            name=name or sim_dir.name,
            source_directory=str(sim_dir)
        )
        
        # Pattern to extract simulation parameters from path
        # Example: afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/l_1/
        layer_pattern = re.compile(r'l[_]?(\d+)')
        
        # Find all simulation directories
        for lammps_dir in sim_dir.rglob('lammps'):
            if not lammps_dir.is_dir():
                continue
            
            sim_path = lammps_dir.parent
            rel_path = sim_path.relative_to(sim_dir)
            
            # Try to extract parameters from path
            path_str = str(rel_path)
            parts = path_str.split('/')
            
            # Initialize with defaults
            material = ""
            layers = 1
            
            # Extract material (usually second part after afm/)
            if len(parts) >= 2:
                material = parts[1] if parts[0] in ('afm', 'sheetonsheet') else parts[0]
            
            # Extract layers
            layer_match = layer_pattern.search(path_str)
            if layer_match:
                layers = int(layer_match.group(1))
            
            # Create job ID
            job_id = path_str.replace('/', '_').replace('-', '_')
            
            entry = JobEntry(
                job_id=job_id,
                simulation_path=str(rel_path),
                material=material,
                layers=layers,
            )
            
            manifest.add_job(entry)
        
        return manifest
    
    def __repr__(self) -> str:
        return (
            f"<JobManifest: {self.name} "
            f"({self.n_jobs} jobs, "
            f"{self.n_completed} completed, "
            f"{self.n_failed} failed)>"
        )


def create_manifest_from_package(package_dir: Path) -> JobManifest:
    """Create a manifest from an exported HPC package.
    
    Args:
        package_dir: Path to the HPC package directory
        
    Returns:
        JobManifest instance
    """
    package_dir = Path(package_dir)
    
    # Load package info
    info_file = package_dir / 'package_info.json'
    if info_file.exists():
        info = json.loads(info_file.read_text())
    else:
        info = {}
    
    # Create manifest from simulations directory
    sim_dir = package_dir / 'simulations'
    if sim_dir.exists():
        manifest = JobManifest.from_simulation_directory(sim_dir, name=package_dir.name)
    else:
        manifest = JobManifest(name=package_dir.name)
    
    manifest.scheduler = info.get('scheduler', 'pbs')
    manifest.package_directory = str(package_dir)
    
    # Mark all as exported
    for job in manifest.jobs:
        job.update_status(JobStatus.EXPORTED)
    
    return manifest
