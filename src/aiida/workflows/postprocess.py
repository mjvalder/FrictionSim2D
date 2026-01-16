"""Postprocessing workflow for FrictionSim2D simulations.

This workflow handles the final phase: taking completed simulation results,
processing them, and storing them in the AiiDA database.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class PostProcessResult:
    """Result of postprocessing workflow."""
    
    results_dir: Path
    n_processed: int
    n_incomplete: int
    json_files: List[Path] = field(default_factory=list)
    stored_uuids: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        """Whether postprocessing was successful."""
        return self.n_processed > 0


class PostProcessWorkflow:
    """Workflow for postprocessing friction simulation results.
    
    This workflow:
    1. Reads raw LAMMPS output files
    2. Converts them to structured JSON using read_data.py
    3. Creates AiiDA FrictionResultsData nodes
    4. Links results to simulation and config nodes
    5. Updates the job manifest
    
    Example:
        >>> workflow = PostProcessWorkflow(
        ...     results_dir='./returned_results',
        ...     manifest_path='./manifest.json'
        ... )
        >>> result = workflow.run()
        >>> print(f"Processed {result.n_processed} simulations")
    """
    
    def __init__(self,
                 results_dir: Union[str, Path],
                 manifest_path: Optional[Union[str, Path]] = None,
                 store_in_aiida: bool = True,
                 skip_fraction: float = 0.2):
        """Initialize the postprocessing workflow.
        
        Args:
            results_dir: Directory containing simulation results
            manifest_path: Optional path to job manifest
            store_in_aiida: Whether to store results in AiiDA
            skip_fraction: Fraction of initial data to skip for statistics
        """
        self.results_dir = Path(results_dir).resolve()
        self.store_in_aiida = store_in_aiida
        self.skip_fraction = skip_fraction
        
        if manifest_path:
            self.manifest_path = Path(manifest_path).resolve()
        else:
            # Try to find manifest
            self.manifest_path = self.results_dir / 'manifest.json'
            if not self.manifest_path.exists():
                self.manifest_path = None
        
        self._manifest = None
    
    def _load_manifest(self):
        """Load the job manifest if available."""
        if self.manifest_path and self.manifest_path.exists():
            from src.aiida.hpc import JobManifest
            self._manifest = JobManifest.load(self.manifest_path)
            logger.info(f"Loaded manifest with {self._manifest.n_jobs} jobs")
    
    def _run_data_reader(self) -> List[Path]:
        """Run the DataReader to convert raw results to JSON.
        
        Returns:
            List of generated JSON file paths
        """
        from src.postprocessing.read_data import DataReader
        
        logger.info(f"Processing results in {self.results_dir}")
        
        reader = DataReader(results_dir=str(self.results_dir))
        reader.export_full_data_to_json()
        reader.export_issue_reports()
        
        # Find generated JSON files
        output_dir = self.results_dir / 'outputs'
        json_files = list(output_dir.glob('output_*.json'))
        
        return json_files
    
    def _parse_json_and_store(self, json_files: List[Path]) -> List[str]:
        """Parse JSON files and store results in AiiDA.
        
        Args:
            json_files: List of JSON file paths
            
        Returns:
            List of stored node UUIDs
        """
        if not self.store_in_aiida:
            return []
        
        try:
            from src.aiida.data import FrictionSimulationData, FrictionResultsData
            from src.aiida.hpc import JobStatus
        except ImportError:
            logger.warning("AiiDA not available, skipping storage")
            return []
        
        stored_uuids = []
        
        for json_path in json_files:
            try:
                with open(json_path) as f:
                    data = json.load(f)
                
                metadata = data.get('metadata', {})
                results = data.get('results', {})
                
                # Process each material in the JSON
                for material, mat_data in results.items():
                    self._store_material_results(
                        material, mat_data, metadata, stored_uuids
                    )
                    
            except Exception as e:
                logger.error(f"Failed to process {json_path}: {e}")
        
        return stored_uuids
    
    def _store_material_results(self,
                                 material: str,
                                 mat_data: Dict,
                                 metadata: Dict,
                                 stored_uuids: List[str]) -> None:
        """Store results for a single material.
        
        Args:
            material: Material name
            mat_data: Nested data for this material
            metadata: Global metadata from JSON
            stored_uuids: List to append stored UUIDs to
        """
        from src.aiida.data import FrictionSimulationData, FrictionResultsData
        from src.aiida.hpc import JobStatus
        import pandas as pd
        
        time_series = metadata.get('time_series', [])
        
        # Navigate the nested structure
        # material -> size -> substrate -> tip_material -> tip_radius -> layer -> speed -> force -> angle -> data
        
        def process_nested(data: Dict, path: Dict):
            """Recursively process nested data structure."""
            for key, value in data.items():
                new_path = path.copy()
                
                # Parse the key to determine what level we're at
                if key.startswith('l'):  # layer
                    new_path['layers'] = int(key[1:])
                elif key.startswith('s'):  # speed
                    new_path['speed'] = int(key[1:])
                elif key.startswith('f'):  # force
                    new_path['force'] = float(key[1:])
                elif key.startswith('a'):  # angle
                    new_path['angle'] = float(key[1:])
                elif key.startswith('r'):  # tip radius
                    new_path['tip_radius'] = key
                elif 'x' in key:  # size
                    new_path['size'] = key
                else:
                    # Could be substrate, tip_material, etc.
                    if 'substrate' not in new_path:
                        new_path['substrate'] = key
                    elif 'tip_material' not in new_path:
                        new_path['tip_material'] = key
                
                # Check if this is actual data (has 'columns' and 'data' keys)
                if isinstance(value, dict) and 'columns' in value and 'data' in value:
                    # This is a DataFrame - create nodes
                    try:
                        df = pd.DataFrame(value['data'], columns=value['columns'])
                        
                        # Add time column if available
                        if time_series and len(time_series) == len(df):
                            df.insert(0, 'time', time_series)
                        
                        # Create results node
                        results_node = FrictionResultsData.from_dataframe(df, {
                            'material': material,
                            'layers': new_path.get('layers', 1),
                            'force': new_path.get('force', 0),
                            'angle': new_path.get('angle', 0),
                            'speed': new_path.get('speed', 2),
                            'size': new_path.get('size', ''),
                        })
                        results_node.store()
                        
                        # Create simulation node
                        sim_node = FrictionSimulationData()
                        sim_node.material = material.replace('_', '-')
                        sim_node.layers = new_path.get('layers', 1)
                        sim_node.force = new_path.get('force', 0)
                        sim_node.scan_angle = new_path.get('angle', 0)
                        sim_node.scan_speed = new_path.get('speed', 2)
                        sim_node.substrate_material = new_path.get('substrate', '')
                        sim_node.tip_material = new_path.get('tip_material', '')
                        sim_node.status = 'imported'
                        sim_node.results_uuid = str(results_node.uuid)
                        
                        # Link to config/provenance if manifest available
                        if self._manifest:
                            if self._manifest.config_node_uuid:
                                sim_node.config_uuid = self._manifest.config_node_uuid
                            if self._manifest.provenance_node_uuid:
                                sim_node.provenance_uuid = self._manifest.provenance_node_uuid
                        
                        sim_node.store()
                        stored_uuids.append(str(sim_node.uuid))
                        
                    except Exception as e:
                        logger.error(f"Failed to store {material} {new_path}: {e}")
                
                elif isinstance(value, dict):
                    # Continue recursing
                    process_nested(value, new_path)
        
        process_nested(mat_data, {'material': material})
    
    def _update_manifest(self, n_processed: int) -> None:
        """Update the job manifest with completion status."""
        if not self._manifest:
            return
        
        from src.aiida.hpc import JobStatus
        
        # Mark completed jobs as imported
        updated = self._manifest.mark_completed_from_results(self.results_dir)
        
        for job in self._manifest.jobs:
            if job.status == JobStatus.COMPLETED.value:
                job.update_status(JobStatus.IMPORTED)
        
        # Save updated manifest
        output_path = self.results_dir / 'manifest_processed.json'
        self._manifest.save(output_path)
        logger.info(f"Updated manifest saved to {output_path}")
    
    def run(self) -> PostProcessResult:
        """Execute the postprocessing workflow.
        
        Returns:
            PostProcessResult with all outputs and metadata
        """
        errors = []
        
        # Load manifest if available
        try:
            self._load_manifest()
        except Exception as e:
            errors.append(f"Failed to load manifest: {e}")
        
        # Run data reader
        try:
            json_files = self._run_data_reader()
            logger.info(f"Generated {len(json_files)} JSON files")
        except Exception as e:
            return PostProcessResult(
                results_dir=self.results_dir,
                n_processed=0,
                n_incomplete=0,
                errors=[f"Failed to run data reader: {e}"]
            )
        
        # Store in AiiDA
        stored_uuids = []
        if self.store_in_aiida:
            try:
                stored_uuids = self._parse_json_and_store(json_files)
                logger.info(f"Stored {len(stored_uuids)} simulation nodes")
            except Exception as e:
                errors.append(f"Failed to store in AiiDA: {e}")
        
        # Update manifest
        try:
            self._update_manifest(len(stored_uuids))
        except Exception as e:
            errors.append(f"Failed to update manifest: {e}")
        
        # Count incomplete files
        n_incomplete = 0
        incomplete_files = list(self.results_dir.glob('outputs/incomplete_*.txt'))
        for f in incomplete_files:
            n_incomplete += len(f.read_text().splitlines())
        
        return PostProcessResult(
            results_dir=self.results_dir,
            n_processed=len(stored_uuids),
            n_incomplete=n_incomplete,
            json_files=json_files,
            stored_uuids=stored_uuids,
            errors=errors,
        )
