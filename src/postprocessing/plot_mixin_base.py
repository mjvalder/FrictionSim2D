"""Shared typed base for Plotter mixins.

This module provides attribute declarations used across mixins so static
analysis tools understand the facade-composed Plotter shape.
"""
# pylint: disable=too-few-public-methods

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class PlotterMixinBase:
    """Type shim for mixins that compose the Plotter facade."""

    data_dirs: list[str]
    labels: list[str]
    output_dir: Path
    settings: dict
    figure_size: tuple
    time_step_fs: float
    full_data_files: dict[str, dict]
    metadata: dict
    summary_df_cache: pd.DataFrame | None
    material_type_map: dict[str, str]
    type_display_names: dict[str, str]
    dataset_display_labels: dict[str, str]
    series_color_map: dict[str, str | None]
    next_series_color_index: int

    def __getattr__(self, name: str) -> Any:
        """Raise standard AttributeError while allowing typed dynamic lookup."""
        raise AttributeError(
            f"{type(self).__name__!s} has no attribute {name!r}",
        )
