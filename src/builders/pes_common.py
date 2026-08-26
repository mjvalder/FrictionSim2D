"""Constants and helpers shared by both families' PES scans.

Split out of the former ``pes_scan.py``, which held the tip scan (AFM family)
and the sheet scan (sheet family) in one file. These three symbols are the only
ones both halves read; everything else went to whichever family owns it.

``_as_float_list`` is currently unused -- it was dead in ``pes_scan.py`` too,
and is kept here rather than deleted during a move. Removing it is a separate,
safe change.
"""
import logging
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_GRID_N = 12

PES_MINIMIZE_COMMAND = "minimize 0.0 1.0e-6 10000 100000"


def _as_float_list(value: Optional[Union[float, List[float]]]) -> List[float]:
    """Normalize a scalar/list sweep value to a list of floats."""
    if value is None:
        return []
    if isinstance(value, list):
        return [float(v) for v in value]
    return [float(value)]


def _cell_period(unit_cell: Dict[str, float], axis: str) -> float:
    """Return the orthogonalized surface-unit-cell period along ``axis`` (Å)."""
    lo = unit_cell.get(f"{axis}lo")
    hi = unit_cell.get(f"{axis}hi")
    if lo is None or hi is None:
        raise ValueError(
            "Surface unit-cell dimensions unavailable; cannot size the PES grid. "
            "Ensure build_sheet was called with unit_cell_out."
        )
    period = float(hi) - float(lo)
    if period <= 0.0:
        raise ValueError(f"Non-positive surface cell period along {axis}: {period}.")
    return period
