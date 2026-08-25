"""Deprecated location. Canonical home: src.core.config.

The writer half of the config format registry was merged into config.py so it
sits beside read_config, its inverse.
"""
from .config import (  # noqa: F401
    SUPPORTED_WRITE_FORMATS,
    format_ini_value,
    write_config,
)

__all__ = ['write_config', 'format_ini_value', 'SUPPORTED_WRITE_FORMATS']
