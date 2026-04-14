"""Canonical data models and friction statistics for FrictionSim2D.

Provides :class:`ResultRecord`, the single source-of-truth Pydantic model
for a simulation result, and :func:`compute_friction_stats`, the canonical
function for computing friction statistics from time-series data.

Every code path that computes COF or uploads results MUST use
:func:`compute_friction_stats` to guarantee consistency.
"""

from __future__ import annotations

import enum
import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Physical unit conversion constants (LAMMPS metal units)
# ---------------------------------------------------------------------------

#: eV/Å → nanoNewton (force conversion).  1 eV = 1.602176565e-19 J,
#: 1 Å = 1e-10 m  ⇒  1 eV/Å = 1.602176565 nN.
EV_A_TO_NN: float = 1.602176565

#: eV/Å³ → GigaPascal (pressure conversion).  = EV_A_TO_NN × 100.
EV_A3_TO_GPA: float = 160.2176565

#: Newton/metre → eV/Å² (spring constant conversion).  = EV_A_TO_NN × 10.
NM_TO_EV_A2: float = 16.02176565


# ---------------------------------------------------------------------------
# Canonical friction statistics
# ---------------------------------------------------------------------------


def compute_friction_stats(
    nf: np.ndarray,
    lfx: np.ndarray,
    lfy: np.ndarray,
    skip_fraction: float = 0.2,
) -> Dict[str, float]:
    """Compute canonical friction statistics from time-series force data.

    Uses **ratio-of-means** (``mean(lf) / mean(nf)``) which is the standard
    convention in tribology for the macroscopic coefficient of friction.

    The first *skip_fraction* of the data is discarded to remove the
    initial transient (acceleration / equilibration phase).

    Args:
        nf: Normal force time-series (1-D array, nN).
        lfx: Lateral force x-component time-series (1-D array, nN).
        lfy: Lateral force y-component time-series (1-D array, nN).
        skip_fraction: Fraction of initial data to skip (default 0.2 = 20 %).

    Returns:
        Dict with keys:
            ``mean_cof``, ``std_cof``,
            ``mean_lf``, ``std_lf``,
            ``mean_nf``, ``std_nf``,
            ``mean_lfx``, ``std_lfx``,
            ``mean_lfy``, ``std_lfy``.

    Raises:
        ValueError: If input arrays are empty or have mismatched lengths.
    """
    nf = np.asarray(nf, dtype=np.float64)
    lfx = np.asarray(lfx, dtype=np.float64)
    lfy = np.asarray(lfy, dtype=np.float64)

    if nf.size == 0 or lfx.size == 0 or lfy.size == 0:
        raise ValueError("Input arrays must not be empty")
    if not (nf.shape == lfx.shape == lfy.shape):
        raise ValueError(
            f"Array shape mismatch: nf={nf.shape}, lfx={lfx.shape}, lfy={lfy.shape}"
        )

    # Skip transient
    skip_n = int(len(nf) * skip_fraction)
    nf_s = nf[skip_n:]
    lfx_s = lfx[skip_n:]
    lfy_s = lfy[skip_n:]

    lateral_force = np.sqrt(lfx_s ** 2 + lfy_s ** 2)

    # Ratio-of-means: μ = mean(F_L) / mean(F_N)
    mean_nf = float(np.mean(nf_s))
    mean_lf = float(np.mean(lateral_force))

    if mean_nf > 0:
        mean_cof = mean_lf / mean_nf
    else:
        mean_cof = 0.0

    # Element-wise COF for std calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        cof_elem = np.where(nf_s > 0, lateral_force / nf_s, 0.0)
    std_cof = float(np.std(cof_elem))

    return {
        'mean_cof': mean_cof,
        'std_cof': std_cof,
        'mean_lf': mean_lf,
        'std_lf': float(np.std(lateral_force)),
        'mean_nf': mean_nf,
        'std_nf': float(np.std(nf_s)),
        'mean_lfx': float(np.mean(lfx_s)),
        'std_lfx': float(np.std(lfx_s)),
        'mean_lfy': float(np.mean(lfy_s)),
        'std_lfy': float(np.std(lfy_s)),
    }


def compute_derived_columns(
    lfx: np.ndarray,
    lfy: np.ndarray,
    nf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute lateral force magnitude and element-wise COF.

    This is the canonical function for adding derived columns to a
    time-series DataFrame. Used by :class:`DataReader` and
    :class:`FrictionResultsData`.

    Args:
        lfx: Lateral force x-component.
        lfy: Lateral force y-component.
        nf: Normal force.

    Returns:
        Tuple of ``(lateral_force, cof)`` arrays.
    """
    lfx = np.asarray(lfx, dtype=np.float64)
    lfy = np.asarray(lfy, dtype=np.float64)
    nf = np.asarray(nf, dtype=np.float64)

    lateral_force = np.sqrt(lfx ** 2 + lfy ** 2)
    cof = np.divide(
        lateral_force,
        nf,
        out=np.zeros_like(lateral_force),
        where=nf != 0,
    )
    return lateral_force, cof


def compute_time_series_hash(
    nf: np.ndarray,
    lfx: np.ndarray,
    lfy: np.ndarray,
) -> str:
    """Compute a SHA-256 hash of the raw time-series for deduplication.

    Args:
        nf: Normal force array.
        lfx: Lateral force x array.
        lfy: Lateral force y array.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    h = hashlib.sha256()
    for arr in (nf, lfx, lfy):
        h.update(np.asarray(arr, dtype=np.float64).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Status enum for staging pipeline
# ---------------------------------------------------------------------------


class ResultStatus(str, enum.Enum):
    """Status of a result in the local → central staging pipeline."""

    LOCAL = 'local'
    STAGED = 'staged'
    VALIDATED = 'validated'
    PUBLISHED = 'published'
    REJECTED = 'rejected'


# ---------------------------------------------------------------------------
# Canonical ResultRecord
# ---------------------------------------------------------------------------


class ResultRecord(BaseModel):
    """Canonical representation of a friction simulation result.

    This is the **single source of truth** for what constitutes a complete
    simulation result. All upload paths (CLI, AiiDA, API) convert to this
    model before writing to the database.

    Fields are the union of everything stored in:
    - ``database.py`` SQL schema
    - ``FrictionResultsData`` AiiDA node attributes
    - ``FrictionSimulationData`` AiiDA node attributes
    - ``DataReader`` JSON export metadata
    """

    # -- Database identity (set by DB, not by user) ---------------------------
    id: Optional[int] = None
    uploaded_at: Optional[datetime] = None

    # -- Contributor ----------------------------------------------------------
    uploader: Optional[str] = None

    # -- Material & geometry --------------------------------------------------
    material: str
    simulation_type: str = 'afm'
    layers: Optional[int] = None
    size_x: Optional[float] = None
    size_y: Optional[float] = None
    stack_type: Optional[str] = None  # 'AA' or 'AB'

    # -- Loading conditions ---------------------------------------------------
    force_nN: Optional[float] = None          # AFM: normal force (nN)
    pressure_gpa: Optional[float] = None      # Sheet-on-sheet: pressure (GPa)
    scan_angle: Optional[float] = None
    scan_speed: Optional[float] = None
    temperature: Optional[float] = None

    # -- Tip / substrate (AFM-specific) ---------------------------------------
    tip_material: Optional[str] = None
    tip_radius: Optional[float] = None
    substrate_material: Optional[str] = None
    substrate_amorphous: Optional[bool] = None

    # -- Potential ------------------------------------------------------------
    potential_type: Optional[str] = None

    # -- Results (canonical, from compute_friction_stats) ----------------------
    mean_cof: Optional[float] = None
    std_cof: Optional[float] = None
    mean_lf: Optional[float] = None
    std_lf: Optional[float] = None
    mean_nf: Optional[float] = None
    std_nf: Optional[float] = None
    mean_lfx: Optional[float] = None
    std_lfx: Optional[float] = None
    mean_lfy: Optional[float] = None
    std_lfy: Optional[float] = None

    # -- Data provenance ------------------------------------------------------
    ntimesteps: Optional[int] = None
    time_series_hash: Optional[str] = None
    is_complete: bool = True

    # -- Staging pipeline -----------------------------------------------------
    status: ResultStatus = ResultStatus.LOCAL

    # -- Free-form ------------------------------------------------------------
    notes: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    # -- File archive (federated storage) -------------------------------------
    data_url: Optional[str] = None  # URL to full time-series (Zenodo, etc.)

    model_config = {'use_enum_values': True}

    @model_validator(mode='after')
    def _check_force_or_pressure(self) -> 'ResultRecord':
        """Warn if neither force nor pressure is set."""
        # Both can be None for incomplete records; just don't set both.
        if self.force_nN is not None and self.pressure_gpa is not None:
            raise ValueError("Set force_nN (AFM) OR pressure_gpa (sheet), not both")
        return self

    def populate_stats(
        self,
        nf: np.ndarray,
        lfx: np.ndarray,
        lfy: np.ndarray,
        skip_fraction: float = 0.2,
    ) -> None:
        """Compute and set all friction statistics from raw time-series.

        Also computes the ``time_series_hash`` for deduplication and sets
        ``ntimesteps``.

        Args:
            nf: Normal force array.
            lfx: Lateral force x array.
            lfy: Lateral force y array.
            skip_fraction: Transient skip fraction.
        """
        stats = compute_friction_stats(nf, lfx, lfy, skip_fraction)
        for key, val in stats.items():
            setattr(self, key, val)
        self.ntimesteps = len(nf)
        self.time_series_hash = compute_time_series_hash(nf, lfx, lfy)

    def to_upload_dict(self) -> Dict[str, Any]:
        """Convert to a dict suitable for ``FrictionDB.upload_result()``.

        Excludes ``id``, ``uploaded_at``, and ``None`` values.
        """
        exclude = {'id', 'uploaded_at'}
        return {
            k: v for k, v in self.model_dump().items()
            if k not in exclude and v is not None
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> 'ResultRecord':
        """Deserialize from JSON string."""
        return cls.model_validate_json(json_str)

    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> 'ResultRecord':
        """Create from a database row dict."""
        return cls.model_validate(row)
