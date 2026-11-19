"""Imports every analysis module, which registers it (see ``base.get_analysis``)."""
# pylint: disable=unused-import
from . import (cof_ranking, contact, integrity_traces, periods, sheet_geometry,  # noqa: F401
               stick_slip)
