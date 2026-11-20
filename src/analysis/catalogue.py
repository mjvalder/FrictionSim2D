"""Imports every analysis module, which registers it (see ``base.get_analysis``)."""
# pylint: disable=unused-import
from . import (cof_ranking, contact, integrity_traces, periods, pes_landscape,  # noqa: F401
               sheet_geometry, sliding_physics, stick_slip)
