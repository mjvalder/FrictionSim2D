"""AiiDA Data types for FrictionSim2D.

This module defines custom AiiDA Data node types for storing and querying
friction simulation data according to FAIR principles.
"""

from .simulation import FrictionSimulationData
from .results import FrictionResultsData
from .provenance import FrictionProvenanceData

__all__ = [
    'FrictionSimulationData',
    'FrictionResultsData',
    'FrictionProvenanceData',
]
