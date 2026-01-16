"""Preparation workflow for FrictionSim2D simulations.

This workflow handles the first phase: taking a config file and generating
all necessary LAMMPS input files using FrictionSim2D.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PreparationResult:
    """Result of simulation preparation."""
    
    output_dir: Path
    n_simulations: int
    simulation_paths: List[str]
    manifest_path: Optional[Path] = None
    provenance_node_uuid: Optional[str] = None
    materials: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        """Whether preparation was successful."""
        return self.n_simulations > 0 and len(self.errors) == 0


class PreparationWorkflow:
    """Workflow for preparing friction simulations.
    
    This workflow:
    1. Reads the config.ini and settings.yaml
    2. Identifies all materials and parameter combinations
    3. Uses FrictionSim2D builders to generate LAMMPS files
    4. Creates AiiDA nodes for config and provenance
    5. Creates a JobManifest for HPC tracking
    
    Example:
        >>> workflow = PreparationWorkflow(
        ...     config_path='afm_config.ini',
        ...     output_dir='./afm_output'
        ... )
        >>> result = workflow.run()
        >>> print(f"Generated {result.n_simulations} simulations")
    """
    
    def __init__(self,
                 config_path: Union[str, Path],
                 output_dir: Optional[Union[str, Path]] = None,
                 simulation_type: str = 'afm',
                 settings_path: Optional[Union[str, Path]] = None,
                 register_aiida: bool = True):
        """Initialize the preparation workflow.
        
        Args:
            config_path: Path to the config.ini file
            output_dir: Directory for output files (default: auto-generated)
            simulation_type: Type of simulation ('afm' or 'sheetonsheet')
            settings_path: Optional path to settings.yaml
            register_aiida: Whether to register nodes in AiiDA
        """
        self.config_path = Path(config_path).resolve()
        self.simulation_type = simulation_type
        self.register_aiida = register_aiida
        
        if output_dir:
            self.output_dir = Path(output_dir).resolve()
        else:
            self.output_dir = self.config_path.parent / f"{simulation_type}_output"
        
        if settings_path:
            self.settings_path = Path(settings_path).resolve()
        else:
            # Try to find settings.yaml in same directory
            self.settings_path = self.config_path.parent / 'settings.yaml'
            if not self.settings_path.exists():
                self.settings_path = None
        
        self._config_dict: Optional[Dict[str, Any]] = None
        self._materials: List[str] = []
    
    def _read_config(self) -> Dict[str, Any]:
        """Read and parse the configuration file."""
        from src.core.utils import read_config
        
        if self._config_dict is None:
            self._config_dict = read_config(self.config_path)
        
        return self._config_dict
    
    def _get_materials_list(self) -> List[str]:
        """Get list of materials to process."""
        if self._materials:
            return self._materials
        
        config = self._read_config()
        
        # Check for materials list file
        if '2D' in config:
            sheet_config = config['2D']
            
            if 'materials_list' in sheet_config and sheet_config['materials_list']:
                list_path = self.config_path.parent / sheet_config['materials_list']
                if list_path.exists():
                    self._materials = [
                        line.strip()
                        for line in list_path.read_text().splitlines()
                        if line.strip() and not line.startswith('#')
                    ]
                    return self._materials
            
            # Single material
            mat = sheet_config.get('mat', '')
            if mat and '{mat}' not in mat:
                self._materials = [mat]
        
        return self._materials
    
    def _generate_simulations(self) -> List[str]:
        """Generate simulation files using FrictionSim2D.
        
        Returns:
            List of relative paths to simulation directories
        """
        import os
        
        logger.info(f"Generating {self.simulation_type} simulations...")
        
        # FrictionSim2D generates files relative to cwd, so we temporarily
        # change to the output directory's parent
        original_cwd = os.getcwd()
        
        try:
            # Change to output dir parent so generated files go where we want
            os.chdir(self.output_dir.parent)
            
            # Call the appropriate builder
            # Import here to avoid circular imports
            from src import afm, sheetonsheet
            
            if self.simulation_type == 'afm':
                afm(str(self.config_path))
            elif self.simulation_type == 'sheetonsheet':
                sheetonsheet(str(self.config_path))
            else:
                raise ValueError(f"Unknown simulation type: {self.simulation_type}")
        finally:
            os.chdir(original_cwd)
        
        # Find all generated simulation directories
        simulation_paths = []
        for lammps_dir in self.output_dir.rglob('lammps'):
            if lammps_dir.is_dir():
                rel_path = lammps_dir.parent.relative_to(self.output_dir)
                simulation_paths.append(str(rel_path))
        
        return simulation_paths
    
    def _create_manifest(self, simulation_paths: List[str]) -> Path:
        """Create a job manifest for HPC tracking.
        
        Args:
            simulation_paths: List of relative simulation paths
            
        Returns:
            Path to the saved manifest
        """
        from src.aiida.hpc import JobManifest, JobEntry
        
        manifest = JobManifest(
            name=self.output_dir.name,
            source_directory=str(self.output_dir),
            config_file=str(self.config_path),
        )
        
        for sim_path in simulation_paths:
            # Parse simulation info from path
            job_id = sim_path.replace('/', '_').replace('-', '_')
            
            entry = JobEntry(
                job_id=job_id,
                simulation_path=sim_path,
            )
            
            # Try to extract material and layers from path
            parts = sim_path.split('/')
            if len(parts) >= 2:
                if parts[0] in ('afm', 'sheetonsheet'):
                    entry.material = parts[1]
                else:
                    entry.material = parts[0]
            
            import re
            layer_match = re.search(r'l[_]?(\d+)', sim_path)
            if layer_match:
                entry.layers = int(layer_match.group(1))
            
            manifest.add_job(entry)
        
        manifest_path = self.output_dir / 'manifest.json'
        manifest.save(manifest_path)
        
        return manifest_path
    
    def _register_in_aiida(self) -> str:
        """Register provenance in AiiDA.
        
        Returns:
            Provenance node UUID, or None if failed
        """
        try:
            from src.aiida.data import FrictionProvenanceData
            
            # Look for provenance folder in output
            provenance_dir = self.output_dir / 'provenance'
            
            if not provenance_dir.exists():
                # Look in first simulation folder
                for sim_dir in self.output_dir.rglob('provenance'):
                    if sim_dir.is_dir():
                        provenance_dir = sim_dir
                        break
            
            if provenance_dir.exists():
                prov_node = FrictionProvenanceData.from_provenance_folder(
                    provenance_dir,
                    simulation_type=self.simulation_type
                )
                prov_node.store()
                logger.info(f"Registered provenance node: {prov_node.uuid}")
                return str(prov_node.uuid)
            else:
                logger.warning("No provenance folder found")
                return None
            
        except ImportError:
            logger.warning("AiiDA not available, skipping registration")
            return None
        except Exception as e:
            logger.error(f"Failed to register in AiiDA: {e}")
            return None
    
    def run(self) -> PreparationResult:
        """Execute the preparation workflow.
        
        Returns:
            PreparationResult with all outputs and metadata
        """
        errors = []
        prov_uuid = None
        
        # Read configuration
        try:
            config = self._read_config()
            materials = self._get_materials_list()
        except Exception as e:
            return PreparationResult(
                output_dir=self.output_dir,
                n_simulations=0,
                simulation_paths=[],
                errors=[f"Failed to read config: {e}"]
            )
        
        # Generate simulations
        try:
            simulation_paths = self._generate_simulations()
        except Exception as e:
            return PreparationResult(
                output_dir=self.output_dir,
                n_simulations=0,
                simulation_paths=[],
                materials=materials,
                errors=[f"Failed to generate simulations: {e}"]
            )
        
        # Create manifest
        try:
            manifest_path = self._create_manifest(simulation_paths)
        except Exception as e:
            errors.append(f"Failed to create manifest: {e}")
            manifest_path = None
        
        # Register in AiiDA
        if self.register_aiida:
            prov_uuid = self._register_in_aiida()
            
            # Update manifest with UUID if created
            if manifest_path and prov_uuid:
                from src.aiida.hpc import JobManifest
                manifest = JobManifest.load(manifest_path)
                manifest.provenance_node_uuid = prov_uuid
                manifest.save(manifest_path)
        
        return PreparationResult(
            output_dir=self.output_dir,
            n_simulations=len(simulation_paths),
            simulation_paths=simulation_paths,
            manifest_path=manifest_path,
            provenance_node_uuid=prov_uuid,
            materials=materials,
            errors=errors,
        )
