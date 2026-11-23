"""Constants and helpers shared by both families' PES scans.

Split out of the former ``pes_scan.py``, which held the tip scan (AFM family)
and the sheet scan (sheet family) in one file. These three symbols are the only
ones both halves read; everything else went to whichever family owns it.
"""
import logging
from typing import Dict

logger = logging.getLogger(__name__)

DEFAULT_GRID_N = 12

PES_MINIMIZE_COMMAND = "minimize 0.0 1.0e-6 10000 100000"


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
