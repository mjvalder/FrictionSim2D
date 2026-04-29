"""AiiDA Data node representing a batch (set) of friction simulations.

A simulation set corresponds to one ``simulation_XXXXXXXX`` output folder
produced by a single invocation of ``FrictionSim2D run``.  It stores the
configuration parameters that are **shared** across every material in the
batch (temperature, pressure/force schedule, scan angles, spring constants,
simulation settings) together with a mandatory human-readable label.

Every ``FrictionSimulationData``, ``FrictionProvenanceData``, and
``FrictionResultsData`` node in the same batch carries a ``set_uuid``
attribute pointing back to this node, which makes set-level queries
trivial without duplicating the shared information.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from aiida.orm import Data


class FrictionSimulationSetData(Data):
    """AiiDA Data node grouping all simulations in one run.

    Attributes:
        label: Required human-readable identifier (e.g. ``'251125-sheetonsheet'``).
        simulation_type: ``'afm'`` or ``'sheetonsheet'``.
        description: Optional free-text description.
        run_folder: Name of the ``simulation_XXXXXXXX`` directory.
        batch_path: Absolute path to the simulation folder at import time.
        temperature: Simulation temperature in K.
        pressures: Applied pressures in GPa (sheet-on-sheet).
        forces: Applied forces in nN (AFM).
        scan_angles: Scan angles in degrees.
        scan_speed: Lateral scan speed in m/s.
        bond_spring: Bond spring constant (eV/Å² or equivalent).
        driving_spring: Driving spring constant.
        simulation_settings: Full ``settings`` block from ``config.json``.
        materials_list: Alphabetically sorted list of all material names.
        n_materials: Number of materials in the set.
    """

    SIMULATION_TYPES = ('afm', 'sheetonsheet')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # -- Required identifiers -------------------------------------------------

    @property
    def label(self) -> str:
        """Human-readable label for this simulation set (required)."""
        return self.base.attributes.get('label', '')

    @label.setter
    def label(self, value: str):
        if not value or not str(value).strip():
            raise ValueError("label is required and cannot be empty")
        cleaned = str(value).strip()
        # Store in attributes for QueryBuilder filtering (only before storing).
        if not self.is_stored:
            self.base.attributes.set('label', cleaned)
        # Always sync to the native db_dbnode.label column (mutable even when
        # stored) so it is visible in DBeaver and ``verdi node list``.
        from aiida.orm import Node as _Node  # pylint: disable=import-outside-toplevel
        _Node.label.fset(self, cleaned)

    @property
    def simulation_type(self) -> str:
        """Simulation type: ``'afm'`` or ``'sheetonsheet'`` (required)."""
        return self.base.attributes.get('simulation_type', '')

    @simulation_type.setter
    def simulation_type(self, value: str):
        if value not in self.SIMULATION_TYPES:
            raise ValueError(
                f"simulation_type must be one of {self.SIMULATION_TYPES}"
            )
        self.base.attributes.set('simulation_type', value)

    # -- Optional descriptors -------------------------------------------------

    @property
    def description(self) -> str:
        """Optional free-text description."""
        return self.base.attributes.get('description', '')

    @description.setter
    def description(self, value: str):
        text = str(value)
        if not self.is_stored:
            self.base.attributes.set('description', text)
        from aiida.orm import Node as _Node  # pylint: disable=import-outside-toplevel
        _Node.description.fset(self, text)

    @property
    def run_folder(self) -> str:
        """Name of the ``simulation_XXXXXXXX`` directory."""
        return self.base.attributes.get('run_folder', '')

    @run_folder.setter
    def run_folder(self, value: str):
        self.base.attributes.set('run_folder', str(value))

    @property
    def batch_path(self) -> str:
        """Absolute path to the simulation folder at import time."""
        return self.base.attributes.get('batch_path', '')

    @batch_path.setter
    def batch_path(self, value: str):
        self.base.attributes.set('batch_path', str(value))

    # -- Shared simulation parameters -----------------------------------------

    @property
    def temperature(self) -> float:
        """Simulation temperature in Kelvin."""
        return self.base.attributes.get('temperature', 300.0)

    @temperature.setter
    def temperature(self, value: float):
        self.base.attributes.set('temperature', float(value))

    @property
    def pressures(self) -> List[float]:
        """Applied pressures in GPa (sheet-on-sheet simulations)."""
        return self.base.attributes.get('pressures', [])

    @pressures.setter
    def pressures(self, value: List[float]):
        self.base.attributes.set('pressures', [float(p) for p in value])

    @property
    def forces(self) -> List[float]:
        """Applied forces in nN (AFM simulations)."""
        return self.base.attributes.get('forces', [])

    @forces.setter
    def forces(self, value: List[float]):
        self.base.attributes.set('forces', [float(f) for f in value])

    @property
    def scan_angles(self) -> List[float]:
        """Scan angles in degrees."""
        return self.base.attributes.get('scan_angles', [])

    @scan_angles.setter
    def scan_angles(self, value: List[float]):
        self.base.attributes.set('scan_angles', [float(a) for a in value])

    @property
    def scan_speed(self) -> float:
        """Lateral scan speed in m/s."""
        return self.base.attributes.get('scan_speed', 1.0)

    @scan_speed.setter
    def scan_speed(self, value: float):
        self.base.attributes.set('scan_speed', float(value))

    @property
    def bond_spring(self) -> Optional[float]:
        """Bond spring constant."""
        return self.base.attributes.get('bond_spring')

    @bond_spring.setter
    def bond_spring(self, value: float):
        self.base.attributes.set('bond_spring', float(value))

    @property
    def driving_spring(self) -> Optional[float]:
        """Driving spring constant."""
        return self.base.attributes.get('driving_spring')

    @driving_spring.setter
    def driving_spring(self, value: float):
        self.base.attributes.set('driving_spring', float(value))

    @property
    def simulation_settings(self) -> Dict[str, Any]:
        """Full ``settings`` block from ``config.json``."""
        return self.base.attributes.get('simulation_settings', {})

    @simulation_settings.setter
    def simulation_settings(self, value: Dict[str, Any]):
        self.base.attributes.set('simulation_settings', value)

    # -- Materials bookkeeping ------------------------------------------------

    @property
    def materials_list(self) -> List[str]:
        """Alphabetically sorted list of all material names in the set."""
        return self.base.attributes.get('materials_list', [])

    @materials_list.setter
    def materials_list(self, value: List[str]):
        sorted_mats = sorted(set(str(m) for m in value))
        self.base.attributes.set('materials_list', sorted_mats)
        self.base.attributes.set('n_materials', len(sorted_mats))

    @property
    def n_materials(self) -> int:
        """Number of materials in the set."""
        return self.base.attributes.get('n_materials', 0)

    # -- Factory / population -------------------------------------------------

    def set_from_config(self, config: Dict[str, Any]) -> None:
        """Populate shared attributes from a parsed ``config.json`` dict.

        Only the ``general`` and ``settings`` sections are read here; the
        material-specific sections (``sheet``, ``tip``, ``sub``) are handled
        by :class:`~src.aiida.data.simulation.FrictionSimulationData`.

        Args:
            config: Parsed ``config.json`` (or ``config.ini``-derived) dict.
        """
        gen = config.get('general', {})

        if gen.get('temp') is not None:
            self.temperature = gen['temp']

        raw_p = gen.get('pressure')
        if raw_p is not None:
            self.pressures = raw_p if isinstance(raw_p, list) else [raw_p]

        raw_f = gen.get('force')
        if raw_f is not None:
            self.forces = raw_f if isinstance(raw_f, list) else [raw_f]

        raw_a = gen.get('scan_angle')
        if raw_a is not None:
            self.scan_angles = raw_a if isinstance(raw_a, list) else [raw_a]

        if gen.get('scan_speed') is not None:
            self.scan_speed = gen['scan_speed']
        if gen.get('bond_spring') is not None:
            self.bond_spring = gen['bond_spring']
        if gen.get('driving_spring') is not None:
            self.driving_spring = gen['driving_spring']

        if 'settings' in config:
            self.simulation_settings = config['settings']

    # -- Serialisation --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Export metadata as a plain dictionary."""
        return {
            'uuid': str(self.uuid),
            'pk': self.pk,
            'label': self.label,
            'simulation_type': self.simulation_type,
            'description': self.description,
            'run_folder': self.run_folder,
            'batch_path': self.batch_path,
            'temperature': self.temperature,
            'pressures': self.pressures,
            'forces': self.forces,
            'scan_angles': self.scan_angles,
            'scan_speed': self.scan_speed,
            'bond_spring': self.bond_spring,
            'driving_spring': self.driving_spring,
            'n_materials': self.n_materials,
            'materials_list': self.materials_list,
        }

    def __repr__(self) -> str:
        return (
            f"<FrictionSimulationSetData: '{self.label}' "
            f"{self.simulation_type} ({self.n_materials} materials)>"
        )
