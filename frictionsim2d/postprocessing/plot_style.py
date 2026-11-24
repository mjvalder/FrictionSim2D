"""Shared plotting style defaults and lightweight plotting data classes."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from .plot_config import PlotSettings

# Extended gem24 colour palette (24 distinct colours for line plots)
# First 12: MATLAB gem12 originals; next 12: hand-picked to maximise
# hue separation while preserving legibility on white backgrounds.
GEM24_COLORS = [
    '#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE',
    '#A2142F', '#FFD60A', '#6582FD', '#FF453A', '#00A3A3', '#CB845D',
    '#FF007F', '#2EC4B6', '#E07A10', '#1B4F72', '#52B788', '#B5179E',
    '#6B4226', '#0096C7', '#C9B400', '#80B918', '#D4694A', '#7B4FBB',
]

#: The screen defaults, generated from the model so the two cannot drift.
#: `postprocess plot --preset publication` merges the shipped preset on top.
DEFAULT_SETTINGS: dict[str, Any] = PlotSettings().model_dump()


def merge_settings(base: dict, overlay: dict) -> None:
    """Merge `overlay` into `base` in place; a list REPLACES, never extends."""
    for key, value in overlay.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            merge_settings(base[key], value)
        else:
            base[key] = value


def plot_settings(preset=None, settings_path=None) -> dict:
    """`DEFAULT_SETTINGS`, then the shipped ``preset``, then the JSON file
    ``settings_path``, merged in that order (a fresh copy).

    Raises:
        KeyError: no shipped preset of that name.
    """
    # pylint: disable=import-outside-toplevel
    import copy
    import json

    from .presets import load_preset
    settings = copy.deepcopy(DEFAULT_SETTINGS)
    if preset:
        merge_settings(settings, load_preset(preset))
    if settings_path:
        with open(settings_path, 'r', encoding='utf-8') as handle:
            merge_settings(settings, json.load(handle))
    return settings


@dataclasses.dataclass
class StickSlipPSD:
    """Frequency-domain results from stick-slip FFT analysis."""

    freqs: np.ndarray
    psd: np.ndarray
    valid: np.ndarray
    peak_labels: list
    f_slip: float
    period: float
    fft_peak_rank: int
