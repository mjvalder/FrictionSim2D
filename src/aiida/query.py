"""Database query interface for FrictionSim2D.

Provides :class:`Friction2DDB`, a domain-specific wrapper around AiiDA's
:class:`~aiida.orm.QueryBuilder` for querying stored friction simulation
data following FAIR principles (Findable, Accessible, Interoperable, Reusable).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from aiida.orm import QueryBuilder as _QueryBuilder
    from aiida.orm import load_node as _load_node
    AIIDA_AVAILABLE = True
except ImportError:
    AIIDA_AVAILABLE = False
    _QueryBuilder = None  # type: ignore[assignment]
    _load_node = None  # type: ignore[assignment]


@dataclass
class QueryResult:
    """Container for query results with metadata.

    Attributes:
        simulations: List of ``FrictionSimulationData`` nodes.
        total_count: Number of matching simulations.
        query_params: Parameters used for the query.
    """

    simulations: List[Any]
    total_count: int
    query_params: Dict[str, Any]

    def to_dataframe(self):
        """Convert results to a pandas DataFrame.

        Returns:
            DataFrame with simulation parameters and (if available) summary
            statistics from linked result nodes.
        """
        import pandas as pd  # pylint: disable=import-outside-toplevel

        def _fallback_record(sim) -> Dict[str, Any]:
            attrs = sim.base.attributes
            return {
                'uuid': str(getattr(sim, 'uuid', '')),
                'pk': getattr(sim, 'pk', None),
                'simulation_type': attrs.get('simulation_type', ''),
                'material': attrs.get('material', ''),
                'substrate_material': attrs.get('substrate_material', ''),
                'substrate_amorphous': attrs.get('substrate_amorphous', False),
                'tip_material': attrs.get('tip_material', ''),
                'tip_radius': attrs.get('tip_radius', 0.0),
                'layers': attrs.get('layers', 1),
                'force': attrs.get('force', 0.0),
                'pressure': attrs.get('pressure', None),
                'scan_angle': attrs.get('scan_angle', 0.0),
                'scan_speed': attrs.get('scan_speed', 0.0),
                'temperature': attrs.get('temperature', 300.0),
                'size_x': attrs.get('size_x', None),
                'size_y': attrs.get('size_y', None),
                'stack_type': attrs.get('stack_type', ''),
                'potential_type': attrs.get('potential_type', ''),
                'status': attrs.get('status', ''),
                'simulation_path': attrs.get('simulation_path', ''),
            }

        records = []
        for sim in self.simulations:
            try:
                record = sim.to_dict()
            except (AttributeError, KeyError, TypeError):
                record = _fallback_record(sim)
            try:
                results = sim.get_results()
            except (AttributeError, KeyError, TypeError):
                results = None
            if results:
                try:
                    stats = results.get_summary_statistics()
                    record.update({
                        'mean_nf': stats.get('nf', {}).get('mean'),
                        'mean_lfx': stats.get('lfx', {}).get('mean'),
                        'mean_lfy': stats.get('lfy', {}).get('mean'),
                        'friction_coefficient': stats.get('friction_coefficient'),
                    })
                except (KeyError, AttributeError):
                    pass
            records.append(record)

        return pd.DataFrame(records)

    def export_csv(self, filepath: Path) -> Path:
        """Export results to a CSV file.

        Args:
            filepath: Output file path.

        Returns:
            Path to the written file.
        """
        df = self.to_dataframe()
        filepath = Path(filepath)
        df.to_csv(filepath, index=False)
        return filepath

    def export_json(self, filepath: Path) -> Path:
        """Export results to a JSON file.

        Args:
            filepath: Output file path.

        Returns:
            Path to the written file.
        """
        data = {
            'query_params': self.query_params,
            'total_count': self.total_count,
            'simulations': [sim.to_dict() for sim in self.simulations],
        }
        filepath = Path(filepath)
        filepath.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')
        return filepath


class Friction2DDB:
    """Query interface for the Friction2D simulation database.

    Wraps AiiDA's ``QueryBuilder`` with domain-specific methods for querying
    friction simulation data.

    Example::

        db = Friction2DDB()
        results = db.query_by_material('h-MoS2')
        for sim in results.simulations:
            print(f"{sim.material} L{sim.layers}: μ = {sim.friction_coefficient}")

        # Advanced query
        results = db.query(
            materials=['h-MoS2', 'h-WS2'],
            force_range=(5.0, 15.0),
            layers=[1, 2, 3],
        )

    Raises:
        RuntimeError: If AiiDA is not available or no profile is loaded.
    """

    def __init__(self):
        if not AIIDA_AVAILABLE:
            raise RuntimeError(
                "AiiDA is not available. Install with: pip install 'FrictionSim2D[aiida]'"
            )
        if _QueryBuilder is None or _load_node is None:
            raise RuntimeError("AiiDA ORM is not available")
        self._query_builder_cls = _QueryBuilder
        self._load_node = _load_node
        # Lazy import to avoid circular dependencies
        from .data import (  # pylint: disable=import-outside-toplevel
            FrictionSimulationData,
            FrictionResultsData,
            FrictionProvenanceData,
        )
        self._simulation_data_cls = FrictionSimulationData
        self._results_data_cls = FrictionResultsData
        self._provenance_data_cls = FrictionProvenanceData
        from .data import FrictionSimulationSetData  # pylint: disable=import-outside-toplevel
        self._simulation_set_cls = FrictionSimulationSetData

    # -- Simulation set queries -----------------------------------------------

    def list_sets(self) -> List[Any]:
        """Return all ``FrictionSimulationSetData`` nodes, newest first.

        Returns:
            List of set nodes ordered by creation time (descending).
        """
        qb = self._query_builder_cls()
        qb.append(self._simulation_set_cls, project=['*'])
        qb.order_by({self._simulation_set_cls: [{'ctime': 'desc'}]})
        return [row[0] for row in qb.all()]

    def query_by_set(self, set_uuid: str) -> QueryResult:
        """Return all ``FrictionSimulationData`` nodes belonging to a set.

        Args:
            set_uuid: UUID of the ``FrictionSimulationSetData`` node.

        Returns:
            Matching simulation nodes.
        """
        qb = self._query_builder_cls()
        qb.append(self._simulation_data_cls, filters={'attributes.set_uuid': set_uuid})
        simulations = [row[0] for row in qb.all()]
        return QueryResult(
            simulations=simulations,
            total_count=len(simulations),
            query_params={'set_uuid': set_uuid},
        )

    def get_set_results(self, set_uuid: str) -> List[Any]:
        """Return all ``FrictionResultsData`` nodes belonging to a set.

        Args:
            set_uuid: UUID of the ``FrictionSimulationSetData`` node.

        Returns:
            List of results nodes.
        """
        qb = self._query_builder_cls()
        qb.append(self._results_data_cls, filters={'attributes.set_uuid': set_uuid})
        return [row[0] for row in qb.all()]

    # -- Simple queries -------------------------------------------------------

    def query_by_material(self, material: str,
                          status: Optional[str] = None) -> QueryResult:
        """Find all simulations for a specific material.

        Args:
            material: Material name (e.g. ``'h-MoS2'``).
            status: Optional status filter.

        Returns:
            Matching simulations.
        """
        qb = self._query_builder_cls()
        qb.append(self._simulation_data_cls, filters={'attributes.material': material})
        if status:
            qb.add_filter(self._simulation_data_cls, {'attributes.status': status})

        simulations = [row[0] for row in qb.all()]
        return QueryResult(
            simulations=simulations,
            total_count=len(simulations),
            query_params={'material': material, 'status': status},
        )

    def query_by_conditions(
        self,
        force: Optional[float] = None,
        layers: Optional[int] = None,
        temperature: Optional[float] = None,
        angle: Optional[float] = None,
        speed: Optional[float] = None,
    ) -> QueryResult:
        """Query simulations by experimental conditions.

        Args:
            force: Normal force (nN).
            layers: Number of layers.
            temperature: Temperature (K).
            angle: Scan angle (degrees).
            speed: Scan speed (m/s).

        Returns:
            Matching simulations.
        """
        qb = self._query_builder_cls()
        qb.append(self._simulation_data_cls, tag='sim')

        attr_map = {
            'force': force,
            'layers': layers,
            'temperature': temperature,
            'scan_angle': angle,
            'scan_speed': speed,
        }
        for attr, value in attr_map.items():
            if value is not None:
                qb.add_filter('sim', {f'attributes.{attr}': value})

        simulations = [row[0] for row in qb.all()]
        return QueryResult(
            simulations=simulations,
            total_count=len(simulations),
            query_params=attr_map,
        )

    def query_by_tip(self, tip_material: Optional[str] = None,
                     tip_radius: Optional[float] = None) -> QueryResult:
        """Query simulations by tip properties.

        Args:
            tip_material: Tip material (e.g. ``'Si'``).
            tip_radius: Tip radius (Angstroms).

        Returns:
            Matching simulations.
        """
        qb = self._query_builder_cls()
        qb.append(self._simulation_data_cls, tag='sim')
        if tip_material:
            qb.add_filter('sim', {'attributes.tip_material': tip_material})
        if tip_radius is not None:
            qb.add_filter('sim', {'attributes.tip_radius': tip_radius})

        simulations = [row[0] for row in qb.all()]
        return QueryResult(
            simulations=simulations,
            total_count=len(simulations),
            query_params={'tip_material': tip_material, 'tip_radius': tip_radius},
        )

    def query_by_substrate(self, substrate_material: Optional[str] = None,
                           amorphous: Optional[bool] = None) -> QueryResult:
        """Query simulations by substrate properties.

        Args:
            substrate_material: Substrate material.
            amorphous: Whether substrate is amorphous.

        Returns:
            Matching simulations.
        """
        qb = self._query_builder_cls()
        qb.append(self._simulation_data_cls, tag='sim')
        if substrate_material:
            qb.add_filter('sim', {'attributes.substrate_material': substrate_material})
        if amorphous is not None:
            qb.add_filter('sim', {'attributes.substrate_amorphous': amorphous})

        simulations = [row[0] for row in qb.all()]
        return QueryResult(
            simulations=simulations,
            total_count=len(simulations),
            query_params={'substrate_material': substrate_material, 'amorphous': amorphous},
        )

    # -- Advanced queries -----------------------------------------------------

    def query(  # pylint: disable=too-many-arguments,too-many-branches
        self,
        materials: Optional[List[str]] = None,
        simulation_type: Optional[str] = None,
        force_range: Optional[Tuple[float, float]] = None,
        layers: Optional[Union[int, List[int]]] = None,
        temperature_range: Optional[Tuple[float, float]] = None,
        angle_range: Optional[Tuple[float, float]] = None,
        status: Optional[Union[str, List[str]]] = None,
        potential_type: Optional[str] = None,
        limit: Optional[int] = None,
        order_by: Optional[str] = None,
    ) -> QueryResult:
        """Advanced query with multiple filter options.

        Args:
            materials: Material names to include.
            simulation_type: ``'afm'`` or ``'sheetonsheet'``.
            force_range: ``(min, max)`` force range (nN).
            layers: Single value or list of layer counts.
            temperature_range: ``(min, max)`` temperature range (K).
            angle_range: ``(min, max)`` angle range (degrees).
            status: Status string or list of statuses.
            potential_type: Potential type (e.g. ``'sw'``).
            limit: Maximum number of results.
            order_by: Attribute to sort by (prefix ``'-'`` for descending).

        Returns:
            Matching simulations.
        """
        qb = self._query_builder_cls()
        qb.append(self._simulation_data_cls, tag='sim')

        if materials:
            qb.add_filter('sim', {'attributes.material': {'in': materials}})
        if simulation_type:
            qb.add_filter('sim', {'attributes.simulation_type': simulation_type})
        if force_range:
            qb.add_filter('sim', {'attributes.force': {'>=': force_range[0]}})
            qb.add_filter('sim', {'attributes.force': {'<=': force_range[1]}})
        if layers is not None:
            if isinstance(layers, int):
                qb.add_filter('sim', {'attributes.layers': layers})
            else:
                qb.add_filter('sim', {'attributes.layers': {'in': layers}})
        if temperature_range:
            qb.add_filter('sim', {'attributes.temperature': {'>=': temperature_range[0]}})
            qb.add_filter('sim', {'attributes.temperature': {'<=': temperature_range[1]}})
        if angle_range:
            qb.add_filter('sim', {'attributes.scan_angle': {'>=': angle_range[0]}})
            qb.add_filter('sim', {'attributes.scan_angle': {'<=': angle_range[1]}})
        if status:
            if isinstance(status, str):
                qb.add_filter('sim', {'attributes.status': status})
            else:
                qb.add_filter('sim', {'attributes.status': {'in': status}})
        if potential_type:
            qb.add_filter('sim', {'attributes.potential_type': potential_type})
        if order_by:
            if order_by.startswith('-'):
                qb.order_by({'sim': [{f'attributes.{order_by[1:]}': 'desc'}]})
            else:
                qb.order_by({'sim': [{f'attributes.{order_by}': 'asc'}]})
        if limit:
            qb.limit(limit)

        simulations = [row[0] for row in qb.all()]
        return QueryResult(
            simulations=simulations,
            total_count=len(simulations),
            query_params={
                'materials': materials, 'simulation_type': simulation_type,
                'force_range': force_range, 'layers': layers,
                'temperature_range': temperature_range, 'angle_range': angle_range,
                'status': status, 'potential_type': potential_type,
            },
        )

    # -- Aggregation ----------------------------------------------------------

    def get_available_materials(self) -> List[str]:
        """Get all unique material names in the database.

        Returns:
            Sorted list of material names.
        """
        qb = self._query_builder_cls()
        qb.append(self._simulation_data_cls, project=['attributes.material'])
        return sorted({row[0] for row in qb.all() if row[0]})

    def get_available_conditions(self) -> Dict[str, List[Any]]:
        """Get all unique experimental conditions in the database.

        Returns:
            Dict with keys ``forces``, ``layers``, ``angles``, ``speeds``,
            ``temperatures``.
        """
        qb = self._query_builder_cls()
        qb.append(self._simulation_data_cls, project=[
            'attributes.force', 'attributes.layers',
            'attributes.scan_angle', 'attributes.scan_speed',
            'attributes.temperature',
        ])

        forces, layers, angles, speeds, temps = set(), set(), set(), set(), set()
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
                temps.add(row[4])

        return {
            'forces': sorted(forces),
            'layers': sorted(layers),
            'angles': sorted(angles),
            'speeds': sorted(speeds),
            'temperatures': sorted(temps),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dict with ``total_simulations``, ``by_status``, ``by_type``,
            ``by_material``, ``n_materials``.
        """
        stats: Dict[str, Any] = {
            'total_simulations': 0,
            'by_status': {},
            'by_type': {},
            'by_material': {},
            'n_materials': 0,
        }
        qb = self._query_builder_cls()
        qb.append(self._simulation_data_cls, project=[
            'attributes.status', 'attributes.simulation_type', 'attributes.material',
        ])
        for row in qb.all():
            stats['total_simulations'] += 1
            for idx, bucket in enumerate(['by_status', 'by_type', 'by_material']):
                key = row[idx] or 'unknown'
                stats[bucket][key] = stats[bucket].get(key, 0) + 1

        stats['n_materials'] = len(stats['by_material'])
        return stats

    # -- Comparison -----------------------------------------------------------

    def compare_materials(
        self,
        materials: List[str],
        conditions: Optional[Dict[str, Any]] = None,
    ):
        """Compare friction coefficients across materials.

        Args:
            materials: Materials to compare.
            conditions: Optional filter dict (``force``, ``layers``, ``angle``).

        Returns:
            pandas DataFrame with comparison data.
        """
        import pandas as pd  # pylint: disable=import-outside-toplevel

        query_params: Dict[str, Any] = {'materials': materials}
        if conditions:
            if 'force' in conditions:
                query_params['force_range'] = (conditions['force'], conditions['force'])
            if 'layers' in conditions:
                query_params['layers'] = conditions['layers']
            if 'angle' in conditions:
                query_params['angle_range'] = (conditions['angle'], conditions['angle'])

        result = self.query(**query_params, status='completed')

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
                except (KeyError, AttributeError):
                    pass

        return pd.DataFrame(records)

    # -- Provenance and reproducibility ---------------------------------------

    def get_provenance(self, simulation) -> Dict[str, Any]:
        """Get full provenance information for a simulation.

        Args:
            simulation: ``FrictionSimulationData`` node or UUID string.

        Returns:
            Dict with ``simulation``, ``files`` provenance data.
        """
        if isinstance(simulation, str):
            simulation = self._load_node(simulation)

        provenance = {
            'simulation': simulation.to_dict(),
            'files': None,
        }
        prov_node = simulation.get_provenance()
        if prov_node:
            provenance['files'] = prov_node.to_dict()

        return provenance

    def export_for_reproduction(self, simulation, output_dir: Path) -> Path:
        """Export all files needed to reproduce a simulation.

        Args:
            simulation: ``FrictionSimulationData`` node or UUID string.
            output_dir: Directory to export to.

        Returns:
            Path to the export directory.
        """
        if isinstance(simulation, str):
            simulation = self._load_node(simulation)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        prov = simulation.get_provenance()
        if prov:
            prov.export_to_directory(output_dir)

        info = {
            'simulation_uuid': str(simulation.uuid),
            'material': simulation.material,
            'parameters': simulation.to_dict(),
        }
        (output_dir / 'reproduction_info.json').write_text(
            json.dumps(info, indent=2, default=str), encoding='utf-8'
        )
        return output_dir

    def find_similar(self, simulation,
                     tolerance: Optional[Dict[str, float]] = None) -> QueryResult:
        """Find simulations with similar parameters.

        Args:
            simulation: Reference simulation node or UUID string.
            tolerance: Parameter tolerances, e.g.
                ``{'force': 1.0, 'angle': 5.0, 'temperature': 10.0}``.

        Returns:
            Matching simulations.
        """
        if isinstance(simulation, str):
            simulation = self._load_node(simulation)

        tolerance = tolerance or {'force': 1.0, 'angle': 5.0, 'temperature': 10.0}

        return self.query(
            materials=[simulation.material],
            layers=simulation.layers,
            force_range=(simulation.force - tolerance.get('force', 1),
                         simulation.force + tolerance.get('force', 1)),
            angle_range=(simulation.scan_angle - tolerance.get('angle', 5),
                         simulation.scan_angle + tolerance.get('angle', 5)),
            temperature_range=(simulation.temperature - tolerance.get('temperature', 10),
                               simulation.temperature + tolerance.get('temperature', 10)),
        )
