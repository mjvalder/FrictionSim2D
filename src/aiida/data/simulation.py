"""FrictionSimulationData - Main queryable node for friction simulations.

This node stores all simulation metadata in a structured, queryable format.
It serves as the central node linking configs, results, and provenance.
"""

from typing import Optional, List, Dict, Any, Union
from aiida.orm import Data, QueryBuilder, load_node
from aiida.common.exceptions import NotExistent


class FrictionSimulationData(Data):
    """AiiDA Data node representing a friction simulation.
    
    This node stores simulation metadata in a way that enables efficient
    querying across the Friction2DDB. Each node represents a single
    simulation run (one specific combination of parameters).
    
    Attributes stored:
        - Simulation type (AFM, SheetOnSheet)
        - Material (2D sheet material)
        - Substrate material and properties
        - Tip material and properties (for AFM)
        - Layer count
        - Applied force/pressure
        - Scan angle
        - Scan speed
        - Temperature
        - Size (x, y dimensions)
        - Status (prepared, running, completed, failed)
    """
    
    # Define valid simulation types
    SIMULATION_TYPES = ('afm', 'sheetonsheet')
    STATUSES = ('prepared', 'submitted', 'running', 'completed', 'failed', 'imported')
    
    def __init__(self, **kwargs):
        """Initialize the simulation data node."""
        super().__init__(**kwargs)
    
    # -------------------------------------------------------------------------
    # Core Properties (stored in attributes for querying)
    # -------------------------------------------------------------------------
    
    @property
    def simulation_type(self) -> str:
        """The type of simulation (afm or sheetonsheet)."""
        return self.base.attributes.get('simulation_type', '')
    
    @simulation_type.setter
    def simulation_type(self, value: str):
        if value not in self.SIMULATION_TYPES:
            raise ValueError(f"simulation_type must be one of {self.SIMULATION_TYPES}")
        self.base.attributes.set('simulation_type', value)
    
    @property
    def material(self) -> str:
        """The 2D sheet material (e.g., 'h-MoS2', 'graphene')."""
        return self.base.attributes.get('material', '')
    
    @material.setter
    def material(self, value: str):
        self.base.attributes.set('material', value)
        # Also store normalized version for easier querying
        self.base.attributes.set('material_normalized', value.replace('-', '_').lower())
    
    @property
    def substrate_material(self) -> str:
        """The substrate material (e.g., 'Si', 'aSi')."""
        return self.base.attributes.get('substrate_material', '')
    
    @substrate_material.setter
    def substrate_material(self, value: str):
        self.base.attributes.set('substrate_material', value)
    
    @property
    def substrate_amorphous(self) -> bool:
        """Whether the substrate is amorphous."""
        return self.base.attributes.get('substrate_amorphous', False)
    
    @substrate_amorphous.setter
    def substrate_amorphous(self, value: bool):
        self.base.attributes.set('substrate_amorphous', value)
    
    @property
    def tip_material(self) -> str:
        """The tip material (AFM simulations only)."""
        return self.base.attributes.get('tip_material', '')
    
    @tip_material.setter
    def tip_material(self, value: str):
        self.base.attributes.set('tip_material', value)
    
    @property
    def tip_radius(self) -> float:
        """The tip radius in Angstroms (AFM simulations only)."""
        return self.base.attributes.get('tip_radius', 0.0)
    
    @tip_radius.setter
    def tip_radius(self, value: float):
        self.base.attributes.set('tip_radius', float(value))
    
    @property
    def layers(self) -> int:
        """Number of 2D material layers."""
        return self.base.attributes.get('layers', 1)
    
    @layers.setter
    def layers(self, value: int):
        self.base.attributes.set('layers', int(value))
    
    @property
    def force(self) -> float:
        """Applied normal force in nN."""
        return self.base.attributes.get('force', 0.0)
    
    @force.setter
    def force(self, value: float):
        self.base.attributes.set('force', float(value))
    
    @property
    def pressure(self) -> Optional[float]:
        """Applied pressure in GPa (alternative to force)."""
        return self.base.attributes.get('pressure')
    
    @pressure.setter
    def pressure(self, value: Optional[float]):
        self.base.attributes.set('pressure', float(value) if value is not None else None)
    
    @property
    def scan_angle(self) -> float:
        """Scan angle in degrees."""
        return self.base.attributes.get('scan_angle', 0.0)
    
    @scan_angle.setter
    def scan_angle(self, value: float):
        self.base.attributes.set('scan_angle', float(value))
    
    @property
    def scan_speed(self) -> float:
        """Scan speed in m/s."""
        return self.base.attributes.get('scan_speed', 2.0)
    
    @scan_speed.setter
    def scan_speed(self, value: float):
        self.base.attributes.set('scan_speed', float(value))
    
    @property
    def temperature(self) -> float:
        """Simulation temperature in Kelvin."""
        return self.base.attributes.get('temperature', 300.0)
    
    @temperature.setter
    def temperature(self, value: float):
        self.base.attributes.set('temperature', float(value))
    
    @property
    def size_x(self) -> float:
        """Sheet size in x-direction (Angstroms)."""
        return self.base.attributes.get('size_x', 0.0)
    
    @size_x.setter
    def size_x(self, value: float):
        self.base.attributes.set('size_x', float(value))
    
    @property
    def size_y(self) -> float:
        """Sheet size in y-direction (Angstroms)."""
        return self.base.attributes.get('size_y', 0.0)
    
    @size_y.setter
    def size_y(self, value: float):
        self.base.attributes.set('size_y', float(value))
    
    @property
    def stack_type(self) -> str:
        """Stacking type (AA or AB)."""
        return self.base.attributes.get('stack_type', 'AA')
    
    @stack_type.setter
    def stack_type(self, value: str):
        self.base.attributes.set('stack_type', value)
    
    @property
    def potential_type(self) -> str:
        """Type of interatomic potential used (e.g., 'sw', 'tersoff')."""
        return self.base.attributes.get('potential_type', '')
    
    @potential_type.setter
    def potential_type(self, value: str):
        self.base.attributes.set('potential_type', value)
    
    # -------------------------------------------------------------------------
    # Status and Tracking
    # -------------------------------------------------------------------------
    
    @property
    def status(self) -> str:
        """Current status of the simulation."""
        return self.base.attributes.get('status', 'prepared')
    
    @status.setter
    def status(self, value: str):
        if value not in self.STATUSES:
            raise ValueError(f"status must be one of {self.STATUSES}")
        self.base.attributes.set('status', value)
    
    @property
    def simulation_path(self) -> str:
        """Relative path to the simulation directory within the package."""
        return self.base.attributes.get('simulation_path', '')
    
    @simulation_path.setter
    def simulation_path(self, value: str):
        self.base.attributes.set('simulation_path', value)
    
    @property
    def job_id(self) -> Optional[str]:
        """HPC job ID if submitted."""
        return self.base.attributes.get('job_id')
    
    @job_id.setter
    def job_id(self, value: Optional[str]):
        self.base.attributes.set('job_id', value)
    
    # -------------------------------------------------------------------------
    # Linked Nodes (stored as UUIDs)
    # -------------------------------------------------------------------------
    
    @property
    def config_uuid(self) -> Optional[str]:
        """UUID of the linked FrictionConfigData node."""
        return self.base.attributes.get('config_uuid')
    
    @config_uuid.setter
    def config_uuid(self, value: str):
        self.base.attributes.set('config_uuid', value)
    
    @property
    def results_uuid(self) -> Optional[str]:
        """UUID of the linked FrictionResultsData node."""
        return self.base.attributes.get('results_uuid')
    
    @results_uuid.setter
    def results_uuid(self, value: str):
        self.base.attributes.set('results_uuid', value)
    
    @property
    def provenance_uuid(self) -> Optional[str]:
        """UUID of the linked FrictionProvenanceData node."""
        return self.base.attributes.get('provenance_uuid')
    
    @provenance_uuid.setter
    def provenance_uuid(self, value: str):
        self.base.attributes.set('provenance_uuid', value)
    
    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------
    
    def get_config(self) -> Optional['FrictionConfigData']:
        """Load and return the linked config node."""
        if self.config_uuid:
            try:
                return load_node(self.config_uuid)
            except NotExistent:
                return None
        return None
    
    def get_results(self) -> Optional['FrictionResultsData']:
        """Load and return the linked results node."""
        if self.results_uuid:
            try:
                return load_node(self.results_uuid)
            except NotExistent:
                return None
        return None
    
    def get_provenance(self) -> Optional['FrictionProvenanceData']:
        """Load and return the linked provenance node."""
        if self.provenance_uuid:
            try:
                return load_node(self.provenance_uuid)
            except NotExistent:
                return None
        return None
    
    def set_from_config(self, config: Dict[str, Any], simulation_type: str = 'afm') -> None:
        """Populate attributes from a parsed configuration dictionary.
        
        Args:
            config: Dictionary containing parsed config.ini data
            simulation_type: Type of simulation ('afm' or 'sheetonsheet')
        """
        self.simulation_type = simulation_type
        
        # General parameters
        if 'general' in config:
            gen = config['general']
            if 'temp' in gen:
                self.temperature = gen['temp']
            if 'force' in gen:
                self.force = gen['force']
            if 'pressure' in gen:
                self.pressure = gen['pressure']
            if 'scan_angle' in gen:
                self.scan_angle = gen['scan_angle']
            if 'scan_speed' in gen:
                self.scan_speed = gen['scan_speed']
        
        # 2D sheet parameters
        sheet_key = '2D' if '2D' in config else 'sheet'
        if sheet_key in config:
            sheet = config[sheet_key]
            if 'mat' in sheet:
                self.material = sheet['mat']
            if 'x' in sheet:
                self.size_x = sheet['x']
            if 'y' in sheet:
                self.size_y = sheet['y']
            if 'layers' in sheet:
                self.layers = sheet['layers']
            if 'stack_type' in sheet:
                self.stack_type = sheet['stack_type']
            if 'pot_type' in sheet:
                self.potential_type = sheet['pot_type']
        
        # Substrate parameters
        if 'sub' in config:
            sub = config['sub']
            if 'mat' in sub:
                self.substrate_material = sub['mat']
            if 'amorph' in sub:
                self.substrate_amorphous = sub['amorph'] == 'a'

        # Tip parameters (AFM only)
        if 'tip' in config:
            tip = config['tip']
            if 'mat' in tip:
                self.tip_material = tip['mat']
            if 'r' in tip:
                self.tip_radius = tip['r']
            if 's' in tip:
                self.scan_speed = tip['s']

    def to_dict(self) -> Dict[str, Any]:
        return {
            'uuid': str(self.uuid),
            'pk': self.pk,
            'simulation_type': self.simulation_type,
            'material': self.material,
            'substrate_material': self.substrate_material,
            'substrate_amorphous': self.substrate_amorphous,
            'tip_material': self.tip_material,
            'tip_radius': self.tip_radius,
            'layers': self.layers,
            'force': self.force,
            'pressure': self.pressure,
            'scan_angle': self.scan_angle,
            'scan_speed': self.scan_speed,
            'temperature': self.temperature,
            'size_x': self.size_x,
            'size_y': self.size_y,
            'stack_type': self.stack_type,
            'potential_type': self.potential_type,
            'status': self.status,
            'simulation_path': self.simulation_path,
        }
    
    def __repr__(self) -> str:
        return (
            f"<FrictionSimulationData: {self.material} "
            f"L{self.layers} F{self.force}nN A{self.scan_angle}° "
            f"({self.status})>"
        )
