"""Friction2DDB - Query interface for friction simulation data.

This module provides a unified interface for querying stored friction
simulation data, supporting FAIR principles for scientific data management.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass
import json

try:
    from aiida.orm import QueryBuilder, load_node
    from aiida.common.exceptions import NotExistent
    AIIDA_AVAILABLE = True
except ImportError:
    AIIDA_AVAILABLE = False
    QueryBuilder = None


@dataclass
class QueryResult:
    """Container for query results with metadata."""
    
    simulations: List[Any]  # List of FrictionSimulationData nodes
    total_count: int
    query_params: Dict[str, Any]
    
    def to_dataframe(self):
        """Convert results to a pandas DataFrame for analysis.
        
        Returns:
            pandas DataFrame with simulation parameters and key results
        """
        import pandas as pd
        
        records = []
        for sim in self.simulations:
            record = sim.to_dict()
            
            # Try to get summary statistics from results
            results = sim.get_results()
            if results:
                try:
                    stats = results.get_summary_statistics()
                    record.update({
                        'mean_nf': stats.get('nf', {}).get('mean'),
                        'mean_lfx': stats.get('lfx', {}).get('mean'),
                        'mean_lfy': stats.get('lfy', {}).get('mean'),
                        'friction_coefficient': stats.get('friction_coefficient'),
                    })
                except Exception:
                    pass
            
            records.append(record)
        
        return pd.DataFrame(records)
    
    def export_csv(self, filepath: Path) -> Path:
        """Export results to CSV file.
        
        Args:
            filepath: Output file path
            
        Returns:
            Path to written file
        """
        df = self.to_dataframe()
        filepath = Path(filepath)
        df.to_csv(filepath, index=False)
        return filepath
    
    def export_json(self, filepath: Path) -> Path:
        """Export results to JSON file.
        
        Args:
            filepath: Output file path
            
        Returns:
            Path to written file
        """
        data = {
            'query_params': self.query_params,
            'total_count': self.total_count,
            'simulations': [sim.to_dict() for sim in self.simulations]
        }
        filepath = Path(filepath)
        filepath.write_text(json.dumps(data, indent=2, default=str))
        return filepath


class Friction2DDB:
    """Query interface for the Friction2D database.
    
    This class provides a user-friendly API for querying stored friction
    simulation data. It wraps AiiDA's QueryBuilder to provide domain-specific
    query methods.
    
    Example:
        >>> db = Friction2DDB()
        >>> results = db.query_by_material('h-MoS2')
        >>> for sim in results.simulations:
        ...     print(f"{sim.material} L{sim.layers}: μ = {sim.get_friction_coefficient()}")
        
        >>> # Complex query
        >>> results = db.query(
        ...     materials=['h-MoS2', 'h-WS2'],
        ...     force_range=(5.0, 15.0),
        ...     layers=[1, 2, 3]
        ... )
    """
    
    def __init__(self):
        """Initialize the database interface."""
        if not AIIDA_AVAILABLE:
            raise RuntimeError(
                "AiiDA is not available. Please install aiida-core and load the profile."
            )
        
        # Import data types
        from src.aiida.data import (
            FrictionSimulationData,
            FrictionConfigData,
            FrictionResultsData,
            FrictionProvenanceData,
        )
        
        self.SimulationData = FrictionSimulationData
        self.ConfigData = FrictionConfigData
        self.ResultsData = FrictionResultsData
        self.ProvenanceData = FrictionProvenanceData
    
    # -------------------------------------------------------------------------
    # Simple Query Methods
    # -------------------------------------------------------------------------
    
    def query_by_material(self, 
                          material: str, 
                          status: Optional[str] = None) -> QueryResult:
        """Find all simulations for a specific material.
        
        Args:
            material: Material name (e.g., 'h-MoS2', 'graphene')
            status: Optional status filter ('completed', 'failed', etc.)
            
        Returns:
            QueryResult containing matching simulations
        """
        qb = QueryBuilder()
        qb.append(self.SimulationData, filters={'attributes.material': material})
        
        if status:
            qb.add_filter(self.SimulationData, {'attributes.status': status})
        
        simulations = [x[0] for x in qb.all()]
        
        return QueryResult(
            simulations=simulations,
            total_count=len(simulations),
            query_params={'material': material, 'status': status}
        )
    
    def query_by_conditions(self,
                            force: Optional[float] = None,
                            layers: Optional[int] = None,
                            temperature: Optional[float] = None,
                            angle: Optional[float] = None,
                            speed: Optional[float] = None) -> QueryResult:
        """Query simulations by experimental conditions.
        
        Args:
            force: Normal force in nN
            layers: Number of layers
            temperature: Temperature in K
            angle: Scan angle in degrees
            speed: Scan speed in m/s
            
        Returns:
            QueryResult containing matching simulations
        """
        qb = QueryBuilder()
        qb.append(self.SimulationData)
        
        filters = {}
        if force is not None:
            filters['attributes.force'] = force
        if layers is not None:
            filters['attributes.layers'] = layers
        if temperature is not None:
            filters['attributes.temperature'] = temperature
        if angle is not None:
            filters['attributes.scan_angle'] = angle
        if speed is not None:
            filters['attributes.scan_speed'] = speed
        
        if filters:
            for key, value in filters.items():
                qb.add_filter(self.SimulationData, {key: value})
        
        simulations = [x[0] for x in qb.all()]
        
        return QueryResult(
            simulations=simulations,
            total_count=len(simulations),
            query_params={
                'force': force, 'layers': layers, 'temperature': temperature,
                'angle': angle, 'speed': speed
            }
        )
    
    def query_by_tip(self,
                     tip_material: Optional[str] = None,
                     tip_radius: Optional[float] = None) -> QueryResult:
        """Query simulations by tip properties.
        
        Args:
            tip_material: Tip material (e.g., 'Si')
            tip_radius: Tip radius in Angstroms
            
        Returns:
            QueryResult containing matching simulations
        """
        qb = QueryBuilder()
        qb.append(self.SimulationData)
        
        if tip_material:
            qb.add_filter(self.SimulationData, {'attributes.tip_material': tip_material})
        if tip_radius is not None:
            qb.add_filter(self.SimulationData, {'attributes.tip_radius': tip_radius})
        
        simulations = [x[0] for x in qb.all()]
        
        return QueryResult(
            simulations=simulations,
            total_count=len(simulations),
            query_params={'tip_material': tip_material, 'tip_radius': tip_radius}
        )
    
    def query_by_substrate(self,
                           substrate_material: Optional[str] = None,
                           amorphous: Optional[bool] = None) -> QueryResult:
        """Query simulations by substrate properties.
        
        Args:
            substrate_material: Substrate material (e.g., 'Si', 'SiO2')
            amorphous: Whether substrate is amorphous
            
        Returns:
            QueryResult containing matching simulations
        """
        qb = QueryBuilder()
        qb.append(self.SimulationData)
        
        if substrate_material:
            qb.add_filter(self.SimulationData, 
                         {'attributes.substrate_material': substrate_material})
        if amorphous is not None:
            qb.add_filter(self.SimulationData, 
                         {'attributes.substrate_amorphous': amorphous})
        
        simulations = [x[0] for x in qb.all()]
        
        return QueryResult(
            simulations=simulations,
            total_count=len(simulations),
            query_params={'substrate_material': substrate_material, 'amorphous': amorphous}
        )
    
    # -------------------------------------------------------------------------
    # Advanced Query Methods
    # -------------------------------------------------------------------------
    
    def query(self,
              materials: Optional[List[str]] = None,
              simulation_type: Optional[str] = None,
              force_range: Optional[Tuple[float, float]] = None,
              layers: Optional[Union[int, List[int]]] = None,
              temperature_range: Optional[Tuple[float, float]] = None,
              angle_range: Optional[Tuple[float, float]] = None,
              status: Optional[Union[str, List[str]]] = None,
              potential_type: Optional[str] = None,
              limit: Optional[int] = None,
              order_by: Optional[str] = None) -> QueryResult:
        """Advanced query with multiple filter options.
        
        Args:
            materials: List of material names to include
            simulation_type: 'afm' or 'sheetonsheet'
            force_range: (min, max) force range in nN
            layers: Single layer count or list of layer counts
            temperature_range: (min, max) temperature range in K
            angle_range: (min, max) angle range in degrees
            status: Status string or list of statuses
            potential_type: Potential type (e.g., 'sw', 'tersoff')
            limit: Maximum number of results
            order_by: Attribute to order by (prefix with '-' for descending)
            
        Returns:
            QueryResult containing matching simulations
        """
        qb = QueryBuilder()
        qb.append(self.SimulationData, tag='sim')
        
        # Material filter
        if materials:
            qb.add_filter('sim', {'attributes.material': {'in': materials}})
        
        # Simulation type filter
        if simulation_type:
            qb.add_filter('sim', {'attributes.simulation_type': simulation_type})
        
        # Force range filter
        if force_range:
            qb.add_filter('sim', {'attributes.force': {'>=': force_range[0]}})
            qb.add_filter('sim', {'attributes.force': {'<=': force_range[1]}})
        
        # Layers filter
        if layers is not None:
            if isinstance(layers, int):
                qb.add_filter('sim', {'attributes.layers': layers})
            else:
                qb.add_filter('sim', {'attributes.layers': {'in': layers}})
        
        # Temperature range filter
        if temperature_range:
            qb.add_filter('sim', {'attributes.temperature': {'>=': temperature_range[0]}})
            qb.add_filter('sim', {'attributes.temperature': {'<=': temperature_range[1]}})
        
        # Angle range filter
        if angle_range:
            qb.add_filter('sim', {'attributes.scan_angle': {'>=': angle_range[0]}})
            qb.add_filter('sim', {'attributes.scan_angle': {'<=': angle_range[1]}})
        
        # Status filter
        if status:
            if isinstance(status, str):
                qb.add_filter('sim', {'attributes.status': status})
            else:
                qb.add_filter('sim', {'attributes.status': {'in': status}})
        
        # Potential type filter
        if potential_type:
            qb.add_filter('sim', {'attributes.potential_type': potential_type})
        
        # Ordering
        if order_by:
            if order_by.startswith('-'):
                qb.order_by({'sim': [{f'attributes.{order_by[1:]}': 'desc'}]})
            else:
                qb.order_by({'sim': [{f'attributes.{order_by}': 'asc'}]})
        
        # Limit
        if limit:
            qb.limit(limit)
        
        simulations = [x[0] for x in qb.all()]
        
        return QueryResult(
            simulations=simulations,
            total_count=len(simulations),
            query_params={
                'materials': materials,
                'simulation_type': simulation_type,
                'force_range': force_range,
                'layers': layers,
                'temperature_range': temperature_range,
                'angle_range': angle_range,
                'status': status,
                'potential_type': potential_type,
            }
        )
    
    # -------------------------------------------------------------------------
    # Aggregation and Analysis Methods
    # -------------------------------------------------------------------------
    
    def get_available_materials(self) -> List[str]:
        """Get list of all unique materials in the database.
        
        Returns:
            Sorted list of material names
        """
        qb = QueryBuilder()
        qb.append(self.SimulationData, project=['attributes.material'])
        
        materials = set(x[0] for x in qb.all() if x[0])
        return sorted(materials)
    
    def get_available_conditions(self) -> Dict[str, List[Any]]:
        """Get all unique experimental conditions in the database.
        
        Returns:
            Dictionary with keys: forces, layers, angles, speeds, temperatures
        """
        qb = QueryBuilder()
        qb.append(self.SimulationData, project=[
            'attributes.force',
            'attributes.layers', 
            'attributes.scan_angle',
            'attributes.scan_speed',
            'attributes.temperature',
        ])
        
        forces = set()
        layers = set()
        angles = set()
        speeds = set()
        temperatures = set()
        
        for row in qb.all():
            if row[0] is not None:
                forces.add(row[0])
            if row[1] is not None:
                layers.add(row[1])
            if row[2] is not None:
                angles.add(row[2])
            if row[3] is not None:
                speeds.add(row[3])
            if row[4] is not None:
                temperatures.add(row[4])
        
        return {
            'forces': sorted(forces),
            'layers': sorted(layers),
            'angles': sorted(angles),
            'speeds': sorted(speeds),
            'temperatures': sorted(temperatures),
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dictionary with counts by status, type, material, etc.
        """
        stats = {
            'total_simulations': 0,
            'by_status': {},
            'by_type': {},
            'by_material': {},
            'n_materials': 0,
        }
        
        qb = QueryBuilder()
        qb.append(self.SimulationData, project=[
            'attributes.status',
            'attributes.simulation_type',
            'attributes.material',
        ])
        
        for row in qb.all():
            stats['total_simulations'] += 1
            
            status = row[0] or 'unknown'
            sim_type = row[1] or 'unknown'
            material = row[2] or 'unknown'
            
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
            stats['by_type'][sim_type] = stats['by_type'].get(sim_type, 0) + 1
            stats['by_material'][material] = stats['by_material'].get(material, 0) + 1
        
        stats['n_materials'] = len(stats['by_material'])
        
        return stats
    
    def compare_materials(self,
                         materials: List[str],
                         conditions: Optional[Dict[str, Any]] = None):
        """Compare friction coefficients across materials.
        
        Args:
            materials: List of materials to compare
            conditions: Optional dict of conditions to filter by
                       (force, layers, angle, speed)
            
        Returns:
            DataFrame with material comparison
        Returns:
            DataFrame with material comparison
        """
        import pandas as pd
        
        # Build query
        query_params = {'materials': materials}
        if conditions:
            if 'force' in conditions:
                query_params['force_range'] = (conditions['force'], conditions['force'])
            if 'layers' in conditions:
                query_params['layers'] = conditions['layers']
            if 'angle' in conditions:
                query_params['angle_range'] = (conditions['angle'], conditions['angle'])
        
        result = self.query(**query_params, status='completed')
        
        # Build comparison data
        records = []
        for sim in result.simulations:
            results = sim.get_results()
            if results:
                try:
                    mu = results.get_friction_coefficient()
                    records.append({
                        'material': sim.material,
                        'layers': sim.layers,
                        'force': sim.force,
                        'angle': sim.scan_angle,
                        'friction_coefficient': mu,
                        'mean_lf': results.compute_mean('lfx'),
                    })
                except Exception:
                    pass
        
        return pd.DataFrame(records)
    
    # -------------------------------------------------------------------------
    # Provenance and Reproducibility Methods
    # -------------------------------------------------------------------------
    
    def get_provenance(self, simulation: Any) -> Dict[str, Any]:
        """Get full provenance information for a simulation.
        
        Args:
            simulation: FrictionSimulationData node or UUID string
            
        Returns:
            Dictionary containing all provenance information
        """
        if isinstance(simulation, str):
            simulation = load_node(simulation)
        
        provenance = {
            'simulation': simulation.to_dict(),
            'config': None,
            'files': None,
        }
        
        config = simulation.get_config()
        if config:
            provenance['config'] = config.to_dict()
        
        prov_node = simulation.get_provenance()
        if prov_node:
            provenance['files'] = prov_node.to_dict()
        
        return provenance
    
    def export_for_reproduction(self,
                                simulation: Any,
                                output_dir: Path) -> Path:
        """Export all files needed to reproduce a simulation.
        
        Args:
            simulation: FrictionSimulationData node or UUID string
            output_dir: Directory to export to
            
        Returns:
            Path to export directory
        """
        if isinstance(simulation, str):
            simulation = load_node(simulation)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export config
        config = simulation.get_config()
        if config:
            (output_dir / config.config_filename).write_text(config.config_content)
            if config.settings_content:
                (output_dir / 'settings.yaml').write_text(config.settings_content)
        
        # Export provenance files
        prov = simulation.get_provenance()
        if prov:
            prov.export_to_directory(output_dir)
        
        # Write reproduction info
        info = {
            'simulation_uuid': str(simulation.uuid),
            'material': simulation.material,
            'parameters': simulation.to_dict(),
        }
        (output_dir / 'reproduction_info.json').write_text(
            json.dumps(info, indent=2, default=str)
        )
        
        return output_dir
    
    def find_similar(self,
                     simulation: Any,
                     tolerance: Dict[str, float] = None) -> QueryResult:
        """Find simulations with similar parameters.
        
        Args:
            simulation: Reference simulation node
            tolerance: Dict of parameter tolerances (e.g., {'force': 1.0, 'angle': 5.0})
            
        Returns:
            QueryResult with similar simulations
        """
        if isinstance(simulation, str):
            simulation = load_node(simulation)
        
        tolerance = tolerance or {'force': 1.0, 'angle': 5.0, 'temperature': 10.0}
        
        force = simulation.force
        angle = simulation.scan_angle
        temp = simulation.temperature
        
        return self.query(
            materials=[simulation.material],
            layers=simulation.layers,
            force_range=(force - tolerance.get('force', 1), 
                        force + tolerance.get('force', 1)),
            angle_range=(angle - tolerance.get('angle', 5),
                        angle + tolerance.get('angle', 5)),
            temperature_range=(temp - tolerance.get('temperature', 10),
                              temp + tolerance.get('temperature', 10)),
        )
