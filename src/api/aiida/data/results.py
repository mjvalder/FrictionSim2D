"""AiiDA Data node for friction simulation results.

Stores processed time-series output from LAMMPS friction simulations
(normal force, lateral force, coefficient of friction, etc.) together
with summary statistics for efficient database querying.
"""

import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
from aiida.orm import Data  # pyright: ignore[reportMissingImports]
from ...data.models import ResultRecord, compute_friction_stats, compute_time_series_hash


class FrictionResultsData(Data):
    """AiiDA Data node storing friction simulation results.

    Stores:
        - Time-series data (nf, lfx, lfy, lateral_force, cof, positions, …)
        - Summary statistics (mean, std, min, max for key fields)
        - Metadata for linking back to the simulation

    Time-series values are stored as lists (PostgreSQL-compatible).
    Summary statistics are direct attributes for efficient querying.
    """

    STANDARD_FIELDS = [
        'time', 'nf', 'lfx', 'lfy', 'lateral_force', 'cof',
        'comx', 'comy', 'comz', 'tipx', 'tipy', 'tipz',
    ]
    _SUMMARY_FIELDS = ('nf', 'lfx', 'lfy', 'lateral_force', 'cof')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # -- Time-series storage --------------------------------------------------

    @property
    def time_series(self) -> Dict[str, List[float]]:
        """Full time-series data as ``{field_name: [values…]}``."""
        return self.base.attributes.get('time_series', {})

    @time_series.setter
    def time_series(self, value: Dict[str, Union[List[float], Any]]):
        serialisable = {}
        for key, val in value.items():
            arr = np.asarray(val)
            serialisable[key] = arr.tolist()
        self.base.attributes.set('time_series', serialisable)

        # Auto-update metadata
        if serialisable:
            lengths = [len(v) for v in serialisable.values()]
            if lengths:
                self.ntimesteps = lengths[0]
                self.field_names = list(serialisable.keys())

        self._calculate_summary_statistics()

    @property
    def time(self) -> List[float]:
        """Time values for the time series."""
        return self.time_series.get('time', [])

    @property
    def ntimesteps(self) -> int:
        """Number of timesteps in the data."""
        return self.base.attributes.get('ntimesteps', 0)

    @ntimesteps.setter
    def ntimesteps(self, value: int):
        self.base.attributes.set('ntimesteps', int(value))

    @property
    def field_names(self) -> List[str]:
        """List of field names present in the data."""
        return self.base.attributes.get('fields', [])

    @field_names.setter
    def field_names(self, value: List[str]):
        self.base.attributes.set('fields', list(value))

    # -- Simulation identification --------------------------------------------

    @property
    def material(self) -> str:
        """Material this result is for."""
        return self.base.attributes.get('material', '')

    @material.setter
    def material(self, value: str):
        self.base.attributes.set('material', value)

    @property
    def layers(self) -> int:
        """Number of layers."""
        return self.base.attributes.get('layers', 1)

    @layers.setter
    def layers(self, value: int):
        self.base.attributes.set('layers', int(value))

    @property
    def force(self) -> float:
        """Applied force in nN."""
        return self.base.attributes.get('force', 0.0)

    @force.setter
    def force(self, value: float):
        self.base.attributes.set('force', float(value))

    @property
    def angle(self) -> float:
        """Scan angle in degrees."""
        return self.base.attributes.get('angle', 0.0)

    @angle.setter
    def angle(self, value: float):
        self.base.attributes.set('angle', float(value))

    @property
    def speed(self) -> float:
        """Scan speed in m/s."""
        return self.base.attributes.get('speed', 2.0)

    @speed.setter
    def speed(self, value: float):
        self.base.attributes.set('speed', float(value))

    @property
    def size(self) -> str:
        """Size identifier (e.g. ``'100x100'``)."""
        return self.base.attributes.get('size', '')

    @size.setter
    def size(self, value: str):
        self.base.attributes.set('size', value)

    @property
    def is_complete(self) -> bool:
        """Whether this result represents a complete simulation."""
        return self.base.attributes.get('is_complete', True)

    @is_complete.setter
    def is_complete(self, value: bool):
        self.base.attributes.set('is_complete', bool(value))

    # -- Summary statistics (read-only, auto-calculated) ----------------------

    @property
    def mean_nf(self) -> float:
        """Mean normal force."""
        return self.base.attributes.get('mean_nf', 0.0)

    @property
    def mean_lfx(self) -> float:
        """Mean lateral force (x-direction)."""
        return self.base.attributes.get('mean_lfx', 0.0)

    @property
    def mean_lfy(self) -> float:
        """Mean lateral force (y-direction)."""
        return self.base.attributes.get('mean_lfy', 0.0)

    @property
    def mean_lateral_force(self) -> float:
        """Mean magnitude of lateral force."""
        return self.base.attributes.get('mean_lateral_force', 0.0)

    @property
    def mean_cof(self) -> float:
        """Mean coefficient of friction."""
        return self.base.attributes.get('mean_cof', 0.0)

    @property
    def std_cof(self) -> float:
        """Standard deviation of coefficient of friction."""
        return self.base.attributes.get('std_cof', 0.0)

    @property
    def friction_coefficient(self) -> float:
        """Friction coefficient (alias for ``mean_cof``)."""
        return self.mean_cof

    def _calculate_summary_statistics(self):
        """Compute and store summary statistics from the current time-series.

        Uses :func:`~src.data.models.compute_friction_stats` as the canonical
        source for COF and force statistics (ratio-of-means, 20 % transient
        skip). Min/max values are still computed on the full series.
        """
        ts = self.time_series
        if not ts:
            return

        # ------------------------------------------------------------------
        # Canonical friction stats (with transient skip)
        # ------------------------------------------------------------------
        if all(k in ts for k in ('nf', 'lfx', 'lfy')):
            stats = compute_friction_stats(
                np.asarray(ts['nf']),
                np.asarray(ts['lfx']),
                np.asarray(ts['lfy']),
            )
            for key, val in stats.items():
                self.base.attributes.set(key, val)
            self.base.attributes.set('friction_coefficient', stats['mean_cof'])

        # ------------------------------------------------------------------
        # Per-field min/max on the full series (no skip)
        # ------------------------------------------------------------------
        for field in self._SUMMARY_FIELDS:
            if field in ts:
                arr = np.asarray(ts[field])
                self.base.attributes.set(f'min_{field}', float(np.min(arr)))
                self.base.attributes.set(f'max_{field}', float(np.max(arr)))

    def get_summary_statistics(self, skip_fraction: float = 0.2) -> Dict[str, Any]:
        """Get summary statistics for all fields.

        Args:
            skip_fraction: Fraction of initial data to skip (transient).

        Returns:
            Nested dict ``{field: {mean, std}}`` plus ``friction_coefficient``.
        """
        stats: Dict[str, Any] = {}
        for field in self.field_names:
            if field == 'time':
                continue
            try:
                stats[field] = {
                    'mean': self.compute_mean(field, skip_fraction),
                    'std': self.compute_std(field, skip_fraction),
                }
            except KeyError:
                pass

        try:
            stats['friction_coefficient'] = self.get_friction_coefficient(skip_fraction)
        except KeyError:
            pass

        return stats

    # -- Array access ---------------------------------------------------------

    def get_array(self, field: str) -> np.ndarray:
        """Get a field as a NumPy array.

        Args:
            field: Field name (e.g. ``'nf'``, ``'lfx'``).

        Returns:
            1-D NumPy array of values.

        Raises:
            KeyError: If the field does not exist.
        """
        if field not in self.time_series:
            raise KeyError(f"Field '{field}' not found. Available: {self.field_names}")
        return np.asarray(self.time_series[field])

    def get_normal_force(self) -> np.ndarray:
        """Get normal force time series."""
        return self.get_array('nf')

    def get_lateral_force_x(self) -> np.ndarray:
        """Get lateral force (x-direction) time series."""
        return self.get_array('lfx')

    def get_lateral_force_y(self) -> np.ndarray:
        """Get lateral force (y-direction) time series."""
        return self.get_array('lfy')

    def get_lateral_force_magnitude(self) -> np.ndarray:
        """Get magnitude of lateral force (computed from lfx, lfy)."""
        return np.sqrt(self.get_lateral_force_x() ** 2
                       + self.get_lateral_force_y() ** 2)

    # -- Statistical methods --------------------------------------------------

    def compute_mean(self, field: str, skip_fraction: float = 0.2) -> float:
        """Compute mean of a field, skipping the initial transient.

        Args:
            field: Field name.
            skip_fraction: Fraction of initial data to skip (default 20 %).

        Returns:
            Mean value after skipping transient.
        """
        data = self.get_array(field)
        skip_n = int(len(data) * skip_fraction)
        return float(np.mean(data[skip_n:]))

    def compute_std(self, field: str, skip_fraction: float = 0.2) -> float:
        """Compute standard deviation of a field.

        Args:
            field: Field name.
            skip_fraction: Fraction of initial data to skip (default 20 %).

        Returns:
            Standard deviation after skipping transient.
        """
        data = self.get_array(field)
        skip_n = int(len(data) * skip_fraction)
        return float(np.std(data[skip_n:]))

    def get_friction_coefficient(self, skip_fraction: float = 0.2) -> float:
        """Calculate friction coefficient μ using the canonical formula.

        Delegates to :func:`~src.data.models.compute_friction_stats`
        (ratio-of-means: ``mean(F_L) / mean(F_N)``).

        Args:
            skip_fraction: Fraction of initial data to skip.

        Returns:
            Mean friction coefficient.
        """
        stats = compute_friction_stats(
            self.get_array('nf'),
            self.get_array('lfx'),
            self.get_array('lfy'),
            skip_fraction=skip_fraction,
        )
        return stats['mean_cof']

    # -- Factory methods ------------------------------------------------------

    @classmethod
    def from_dataframe(cls, df, metadata: Optional[Dict[str, Any]] = None) -> 'FrictionResultsData':
        """Create from a pandas DataFrame.

        Args:
            df: DataFrame with columns for each field.
            metadata: Optional dict with keys ``material``, ``layers``,
                ``force``, ``angle``, ``speed``, ``size``.

        Returns:
            A new (unstored) ``FrictionResultsData`` instance.
        """
        node = cls()
        time_series = {col: df[col].tolist() for col in df.columns}
        node.time_series = time_series

        if metadata:
            for attr in ('material', 'layers', 'force', 'angle', 'speed', 'size'):
                if attr in metadata:
                    setattr(node, attr, metadata[attr])

        return node

    def to_result_record(self) -> 'ResultRecord':
        """Convert this node to a canonical :class:`ResultRecord`.

        Returns:
            A :class:`~src.data.models.ResultRecord` populated from this
            node's attributes and time-series statistics.
        """
        ts = self.time_series
        kwargs: Dict[str, Any] = {
            'material': self.material,
            'layers': self.layers,
            'force_nN': self.force if self.force else None,
            'scan_angle': self.angle if self.angle else None,
            'scan_speed': self.speed if self.speed else None,
            'ntimesteps': self.ntimesteps,
            'is_complete': self.is_complete,
            'metadata': {'aiida_uuid': str(self.uuid)},
        }

        if all(k in ts for k in ('nf', 'lfx', 'lfy')):
            nf = np.asarray(ts['nf'])
            lfx = np.asarray(ts['lfx'])
            lfy = np.asarray(ts['lfy'])
            kwargs.update(compute_friction_stats(nf, lfx, lfy))
            kwargs['time_series_hash'] = compute_time_series_hash(nf, lfx, lfy)

        return ResultRecord(**kwargs)

    @classmethod
    def from_json(cls, json_data: Union[str, Dict]) -> 'FrictionResultsData':
        """Create from JSON data (as exported by ``DataReader``).

        Args:
            json_data: JSON string or parsed dict.

        Returns:
            A new (unstored) ``FrictionResultsData`` instance.
        """
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data

        node = cls()

        if 'metadata' in data and 'time_series' in data['metadata']:
            node.time_series = {'time': data['metadata']['time_series']}

        return node

    # -- Serialisation --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Export all data as a dictionary."""
        return {
            'uuid': str(self.uuid),
            'material': self.material,
            'layers': self.layers,
            'force': self.force,
            'angle': self.angle,
            'speed': self.speed,
            'size': self.size,
            'ntimesteps': self.ntimesteps,
            'fields': self.field_names,
            'is_complete': self.is_complete,
            'time_series': self.time_series,
        }

    def __repr__(self) -> str:
        return (
            f"<FrictionResultsData: {self.material} "
            f"L{self.layers} F{self.force}nN A{self.angle}° "
            f"({self.ntimesteps} steps)>"
        )
