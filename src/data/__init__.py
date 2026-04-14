"""Data models, validation, and database access for FrictionSim2D."""

from src.data.models import ResultRecord, compute_friction_stats
from src.data.database import db_from_profile

__all__ = ['ResultRecord', 'compute_friction_stats']
