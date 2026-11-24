"""Formatting and parsing of scalar tokens used in directory names.

Pure string/number work -- no filesystem, no external libraries.
"""
import re
from typing import Any

def is_float(value: str) -> bool:
    """True if ``value`` parses as a float, scientific notation included."""
    try:
        float(value)
        return True
    except ValueError:
        return False


def is_int_token(value: str) -> bool:
    """Return True if token represents an integer (optional sign allowed)."""
    return bool(re.fullmatch(r'[+-]?\d+', value))


def normalize_element_symbol(value: str) -> str:
    """Canonical symbol case for an element-like token (``mO`` -> ``Mo``)."""
    token = value.strip()
    if not token:
        return token
    return token[0].upper() + token[1:].lower()


def format_numeric_token(value: float) -> str:
    """Format a numeric value into a filename-safe token string.
    
    Converts numbers to strings suitable for use in filenames and directory names.
    Replaces problematic characters: minus signs become 'm', decimals become 'p'.
    
    Args:
        value: Numeric value to format (int, float, or scientific notation)
    
    Returns:
        Alphanumeric token string (e.g., '1p5' for 1.5, 'm2p3' for -2.3)
    
    Example:
        >>> format_numeric_token(1.5)
        '1p5'
        >>> format_numeric_token(-2.3)
        'm2p3'
        >>> format_numeric_token(1e-5)
        '1e-05'
    """
    token = f"{value:g}"
    token = token.replace('-', 'm').replace('.', 'p')
    return re.sub(r'[^A-Za-z0-9_]+', '_', token)


def format_dimension_token(value: Any) -> str:
    """Format a numeric dimension for directory names.

    Integer-valued floats are rendered without a trailing decimal so
    ``100.0`` becomes ``100`` and the rebuilt directory layout matches the
    original simulation tree.
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def material_id(name: str) -> str:
    """Map a directory/material name to the ML merge id (``h-MoS2`` -> ``h_MoS2``)."""
    return name.replace("-", "_").replace("/", "__")
