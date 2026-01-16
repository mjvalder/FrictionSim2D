"""Main workflow orchestrator for FrictionSim2D.

This module provides the top-level workflow class that coordinates
all phases of the friction simulation pipeline.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class WorkflowPhase(Enum):
    """Phases of the friction simulation workflow."""
    PREPARATION = "preparation"
    EXPORT = "export"
    EXECUTION = "execution"  # Manual HPC step
    IMPORT = "import"
    POSTPROCESS = "postprocess"
    COMPLETE = "complete"


@dataclass
class WorkflowState:
    """Current state of the workflow."""
    
    phase: WorkflowPhase = WorkflowPhase.PREPARATION
    config_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    package_dir: Optional[Path] = None
    results_dir: Optional[Path] = None
    manifest_path: Optional[Path] = None
    
    config_node_uuid: Optional[str] = None
    provenance_node_uuid: Optional[str] = None
    n_simulations: int = 0
    n_completed: int = 0
    n_imported: int = 0
    
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            'phase': self.phase.value,
            'config_path': str(self.config_path) if self.config_path else None,
            'output_dir': str(self.output_dir) if self.output_dir else None,
            'package_dir': str(self.package_dir) if self.package_dir else None,
            'results_dir': str(self.results_dir) if self.results_dir else None,
            'manifest_path': str(self.manifest_path) if self.manifest_path else None,
            'config_node_uuid': self.config_node_uuid,
            'provenance_node_uuid': self.provenance_node_uuid,
            'n_simulations': self.n_simulations,
            'n_completed': self.n_completed,
            'n_imported': self.n_imported,
            'errors': self.errors,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowState':
        """Create state from dictionary."""
        state = cls()
        state.phase = WorkflowPhase(data.get('phase', 'preparation'))
        
        if data.get('config_path'):
            state.config_path = Path(data['config_path'])
        if data.get('output_dir'):
            state.output_dir = Path(data['output_dir'])
        if data.get('package_dir'):
            state.package_dir = Path(data['package_dir'])
        if data.get('results_dir'):
            state.results_dir = Path(data['results_dir'])
        if data.get('manifest_path'):
            state.manifest_path = Path(data['manifest_path'])
        
        state.config_node_uuid = data.get('config_node_uuid')
        state.provenance_node_uuid = data.get('provenance_node_uuid')
        state.n_simulations = data.get('n_simulations', 0)
        state.n_completed = data.get('n_completed', 0)
        state.n_imported = data.get('n_imported', 0)
        state.errors = data.get('errors', [])
        
        return state


class FrictionSimWorkflow:
    """Main workflow orchestrator for friction simulations.
    
    This class coordinates the complete friction simulation pipeline:
    
    1. **Preparation**: Generate LAMMPS files from config
    2. **Export**: Create HPC-ready package
    3. **Execution**: (Manual) User runs on HPC
    4. **Import**: Import results back
    5. **Postprocess**: Convert to JSON and store in AiiDA
    
    The workflow maintains state and can be resumed at any phase.
    
    Example:
        >>> workflow = FrictionSimWorkflow(
        ...     config_path='afm_config.ini',
        ...     output_dir='./friction_sim'
        ... )
        >>> 
        >>> # Phase 1: Prepare
        >>> workflow.prepare()
        >>> 
        >>> # Phase 2: Export for HPC
        >>> workflow.export(scheduler='pbs')
        >>> 
        >>> # ... User transfers to HPC, runs jobs, transfers back ...
        >>> 
        >>> # Phase 3: Import and process results
        >>> workflow.import_results('./returned_results')
        >>> workflow.postprocess()
    """
    
    def __init__(self,
                 config_path: Optional[Union[str, Path]] = None,
                 output_dir: Optional[Union[str, Path]] = None,
                 simulation_type: str = 'afm',
                 state: Optional[WorkflowState] = None):
        """Initialize the workflow.
        
        Args:
            config_path: Path to config.ini file
            output_dir: Base output directory
            simulation_type: Type of simulation ('afm' or 'sheetonsheet')
            state: Optional existing state to resume from
        """
        self.simulation_type = simulation_type
        
        if state:
            self.state = state
        else:
            self.state = WorkflowState()
            if config_path:
                self.state.config_path = Path(config_path).resolve()
            if output_dir:
                self.state.output_dir = Path(output_dir).resolve()
    
    @property
    def current_phase(self) -> WorkflowPhase:
        """Get the current workflow phase."""
        return self.state.phase
    
    def save_state(self, filepath: Optional[Path] = None) -> Path:
        """Save workflow state to JSON file.
        
        Args:
            filepath: Optional output path (default: output_dir/workflow_state.json)
            
        Returns:
            Path to saved state file
        """
        import json
        
        if filepath is None:
            if self.state.output_dir:
                filepath = self.state.output_dir / 'workflow_state.json'
            else:
                filepath = Path.cwd() / 'workflow_state.json'
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(json.dumps(self.state.to_dict(), indent=2, default=str))
        
        return filepath
    
    @classmethod
    def load_state(cls, filepath: Path) -> 'FrictionSimWorkflow':
        """Load workflow from saved state.
        
        Args:
            filepath: Path to state file
            
        Returns:
            FrictionSimWorkflow instance
        """
        import json
        
        data = json.loads(Path(filepath).read_text())
        state = WorkflowState.from_dict(data)
        
        return cls(state=state)
    
    def prepare(self) -> bool:
        """Execute the preparation phase.
        
        Generates all LAMMPS input files and creates AiiDA nodes.
        
        Returns:
            True if successful
        """
        from .preparation import PreparationWorkflow
        
        if not self.state.config_path:
            self.state.errors.append("No config path specified")
            return False
        
        logger.info("Starting preparation phase...")
        
        workflow = PreparationWorkflow(
            config_path=self.state.config_path,
            output_dir=self.state.output_dir,
            simulation_type=self.simulation_type,
        )
        
        result = workflow.run()
        
        # Update state
        self.state.output_dir = result.output_dir
        self.state.n_simulations = result.n_simulations
        self.state.manifest_path = result.manifest_path
        self.state.config_node_uuid = result.config_node_uuid
        self.state.provenance_node_uuid = result.provenance_node_uuid
        self.state.errors.extend(result.errors)
        
        if result.success:
            self.state.phase = WorkflowPhase.EXPORT
            logger.info(f"Preparation complete: {result.n_simulations} simulations")
        else:
            logger.error(f"Preparation failed: {result.errors}")
        
        self.save_state()
        return result.success
    
    def export(self, 
               scheduler: str = 'pbs',
               package_dir: Optional[Path] = None,
               **hpc_config) -> Optional[Path]:
        """Export simulations as HPC package.
        
        Args:
            scheduler: HPC scheduler type ('pbs' or 'slurm')
            package_dir: Output directory for package
            **hpc_config: Additional HPC configuration options
            
        Returns:
            Path to package directory if successful
        """
        from src.aiida.hpc import HPCScriptGenerator, HPCConfig
        from src.aiida.hpc.scripts import create_hpc_package
        
        if self.state.phase.value not in ('preparation', 'export'):
            logger.warning("Export should follow preparation")
        
        if not self.state.output_dir:
            self.state.errors.append("No output directory available")
            return None
        
        if package_dir is None:
            package_dir = self.state.output_dir.parent / f"{self.state.output_dir.name}_hpc"
        
        logger.info(f"Exporting HPC package to {package_dir}...")
        
        try:
            config = HPCConfig(**hpc_config) if hpc_config else None
            
            package_path = create_hpc_package(
                self.state.output_dir,
                package_dir,
                scheduler=scheduler,
                config=config
            )
            
            self.state.package_dir = package_path
            self.state.phase = WorkflowPhase.EXECUTION
            
            logger.info(f"Package created: {package_path}")
            self.save_state()
            
            return package_path
            
        except Exception as e:
            self.state.errors.append(f"Export failed: {e}")
            logger.error(f"Export failed: {e}")
            return None
    
    def import_results(self, results_dir: Union[str, Path]) -> bool:
        """Import results from HPC.
        
        Args:
            results_dir: Directory containing returned results
            
        Returns:
            True if successful
        """
        from src.aiida.hpc import JobManifest, JobStatus
        
        results_dir = Path(results_dir).resolve()
        self.state.results_dir = results_dir
        
        logger.info(f"Importing results from {results_dir}...")
        
        # Load manifest
        manifest_path = results_dir / 'manifest.json'
        if not manifest_path.exists() and self.state.manifest_path:
            manifest_path = self.state.manifest_path
        
        if manifest_path.exists():
            manifest = JobManifest.load(manifest_path)
            completed = manifest.mark_completed_from_results(results_dir)
            self.state.n_completed = len(completed)
            
            # Copy UUIDs if not already set
            if manifest.config_node_uuid and not self.state.config_node_uuid:
                self.state.config_node_uuid = manifest.config_node_uuid
            if manifest.provenance_node_uuid and not self.state.provenance_node_uuid:
                self.state.provenance_node_uuid = manifest.provenance_node_uuid
            
            manifest.save(results_dir / 'manifest_imported.json')
            self.state.manifest_path = results_dir / 'manifest_imported.json'
        
        self.state.phase = WorkflowPhase.POSTPROCESS
        logger.info(f"Imported {self.state.n_completed} completed simulations")
        
        self.save_state()
        return True
    
    def postprocess(self) -> bool:
        """Execute postprocessing phase.
        
        Converts results to JSON and stores in AiiDA.
        
        Returns:
            True if successful
        """
        from .postprocess import PostProcessWorkflow
        
        if not self.state.results_dir:
            # Use output_dir if results_dir not set
            if self.state.output_dir:
                self.state.results_dir = self.state.output_dir
            else:
                self.state.errors.append("No results directory specified")
                return False
        
        logger.info("Starting postprocessing phase...")
        
        workflow = PostProcessWorkflow(
            results_dir=self.state.results_dir,
            manifest_path=self.state.manifest_path,
            store_in_aiida=True,
        )
        
        result = workflow.run()
        
        self.state.n_imported = result.n_processed
        self.state.errors.extend(result.errors)
        
        if result.success:
            self.state.phase = WorkflowPhase.COMPLETE
            logger.info(f"Postprocessing complete: {result.n_processed} stored")
        else:
            logger.error(f"Postprocessing had errors: {result.errors}")
        
        self.save_state()
        return result.success
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the workflow state.
        
        Returns:
            Dictionary with workflow summary
        """
        return {
            'current_phase': self.state.phase.value,
            'simulation_type': self.simulation_type,
            'config_path': str(self.state.config_path) if self.state.config_path else None,
            'n_simulations': self.state.n_simulations,
            'n_completed': self.state.n_completed,
            'n_imported': self.state.n_imported,
            'has_errors': len(self.state.errors) > 0,
            'n_errors': len(self.state.errors),
        }
    
    def print_status(self) -> None:
        """Print current workflow status."""
        summary = self.get_summary()
        
        print("\n" + "=" * 50)
        print("FrictionSim2D Workflow Status")
        print("=" * 50)
        print(f"Phase:        {summary['current_phase'].upper()}")
        print(f"Type:         {summary['simulation_type']}")
        print(f"Simulations:  {summary['n_simulations']}")
        print(f"Completed:    {summary['n_completed']}")
        print(f"Imported:     {summary['n_imported']}")
        
        if summary['has_errors']:
            print(f"\n⚠️  {summary['n_errors']} errors occurred:")
            for error in self.state.errors[-5:]:  # Show last 5 errors
                print(f"   - {error}")
        
        print("=" * 50)
        
        # Next steps guidance
        print("\nNext steps:")
        if self.state.phase == WorkflowPhase.PREPARATION:
            print("  1. Run workflow.prepare()")
        elif self.state.phase == WorkflowPhase.EXPORT:
            print("  1. Run workflow.export(scheduler='pbs')")
        elif self.state.phase == WorkflowPhase.EXECUTION:
            print("  1. Transfer package to HPC")
            print("  2. Submit jobs")
            print("  3. Transfer results back")
            print("  4. Run workflow.import_results('./results')")
        elif self.state.phase == WorkflowPhase.POSTPROCESS:
            print("  1. Run workflow.postprocess()")
        elif self.state.phase == WorkflowPhase.COMPLETE:
            print("  ✅ Workflow complete!")
            print("  Use Friction2DDB to query your results")
