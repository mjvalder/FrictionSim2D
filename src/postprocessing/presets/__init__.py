"""Shipped plot-settings presets.

A preset is a partial `PlotSettings` dict merged onto the defaults, exactly
as `--settings` is. `publication` is vector-first (pdf/svg/png), 600 dpi,
sized for a single journal column, smaller fonts than the screen defaults,
no grid, constrained layout.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

__all__ = ['PRESETS', 'load_preset', 'preset_path']

PRESETS = ('publication',)


def preset_path(name: str) -> Path:
    """The JSON file backing a preset."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; known: {PRESETS}")
    return Path(__file__).resolve().parent / f'{name}.json'


def load_preset(name: str) -> Dict[str, Any]:
    """The preset as a settings dict, ready to merge onto the defaults."""
    return json.loads(preset_path(name).read_text(encoding='utf-8'))
