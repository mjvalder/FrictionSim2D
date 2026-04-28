"""Shared plotting style defaults and lightweight plotting data classes."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

# Extended gem24 colour palette (24 distinct colours for line plots)
# First 12: MATLAB gem12 originals; next 12: hand-picked to maximise
# hue separation while preserving legibility on white backgrounds.
GEM24_COLORS = [
    '#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE',
    '#A2142F', '#FFD60A', '#6582FD', '#FF453A', '#00A3A3', '#CB845D',
    '#FF007F', '#2EC4B6', '#E07A10', '#1B4F72', '#52B788', '#B5179E',
    '#6B4226', '#0096C7', '#C9B400', '#80B918', '#D4694A', '#7B4FBB',
]

GEM12_COLORS = GEM24_COLORS

DEFAULT_SETTINGS: dict[str, Any] = {
    "figure": {"size": [10, 7], "dpi": 150},
    "fonts": {
        "title": 26, "axis_label": 24,
        "tick_label": 22, "legend": 16,
    },
    "colors": {"palette": "gem12"},
    "markers": {"style": "o", "size": 12},
    "lines": {"width": 1.3, "fit_style": "--", "fit_alpha": 0.8},
    "grid": {
        "show": True, "which": "both",
        "major_style": "-", "minor_style": ":",
        "major_alpha": 0.5, "minor_alpha": 0.3,
    },
    "error_bands": {"alpha": 0.2},
    "legend": {"location": "best"},
    "layout": {},
    "axes": {"use_scientific_notation": True, "scilimits": [-3, 4]},
    "export": {"formats": ["png"], "transparent": False},
}


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
