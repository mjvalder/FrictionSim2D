"""Shared typed base for Plotter mixins.

This module provides attribute declarations used across mixins so static
analysis tools understand the facade-composed Plotter shape.
"""
# pylint: disable=too-few-public-methods

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class PlotDataError(RuntimeError):
    """An export document the plot stack cannot read.

    Raised rather than logged: a v1 file plotted as if it were v2 would draw
    an empty figure and say nothing, which is exactly how a sweep arm went
    missing for a year.

    Defined here rather than in `plot_data` -- which re-exports it, and is
    where callers import it from -- because `plot_common` raises it and
    `plot_data` imports `plot_common`, so a definition in `plot_data` is a
    cycle. This module already exists to hold what both sides share.
    """


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
    written: list[Path]
    summary_files: dict[str, dict[str, Path]]
    runs_cache: dict[tuple[str, str, str], list[dict]]
    constrained_layout: bool
    plot_family: str

    def __getattr__(self, name: str) -> Any:
        """Raise standard AttributeError while allowing typed dynamic lookup."""
        raise AttributeError(
            f"{type(self).__name__!s} has no attribute {name!r}",
        )
