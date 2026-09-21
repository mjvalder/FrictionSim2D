"""AiiDA Data types for FrictionSim2D.

This module defines custom AiiDA Data node types for storing and querying
friction simulation data according to FAIR principles.
"""

from .simulation import FrictionSimulationData
from .simulation_set import FrictionSimulationSetData
from .results import FrictionResultsData, TIER_KEYS

__all__ = [
    'FrictionSimulationData',
    'FrictionSimulationSetData',
    'FrictionResultsData',
    'TIER_KEYS',
]
