"""Shared filesystem naming helpers."""

from typing import Any


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