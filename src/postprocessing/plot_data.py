"""Thin Plotter facade composed from dedicated plotting mixins."""
# pylint: disable=too-many-instance-attributes,too-many-ancestors
# pylint: disable=too-many-arguments,too-many-positional-arguments

from __future__ import annotations

import copy
import logging
from pathlib import Path

import pandas as pd

from .plot_common import PlotCommonMixin
from .plot_correlation import PlotCorrelationMixin
from .plot_ranking import PlotRankingMixin
from .plot_style import DEFAULT_SETTINGS
from .plot_stick_slip import PlotStickSlipMixin
from .plot_summary import PlotSummaryMixin
from .plot_taxonomy import PlotTaxonomyMixin
from .plot_timeseries import PlotTimeseriesMixin

logger = logging.getLogger(__name__)


class Plotter(
    PlotCommonMixin,
    PlotSummaryMixin,
    PlotTimeseriesMixin,
    PlotStickSlipMixin,
    PlotCorrelationMixin,
    PlotRankingMixin,
    PlotTaxonomyMixin,
):
    """Generate plots from friction simulation data."""

    def __init__(
        self,
        data_dirs: list[str],
        labels: list[str],
        output_dir: str,
        settings: dict | None = None,
        dataset_display_labels: dict[str, str] | None = None,
        series_color_map: dict[str, str | None] | None = None,
    ) -> None:
        self.data_dirs = data_dirs
        self.labels = labels
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_display_labels = dataset_display_labels or {}
        self.series_color_map = dict(series_color_map or {})

        # deepcopy, not .copy(): a shallow copy shared every sub-dict with the
        # module-level default, so one Plotter's --settings mutated the
        # defaults for the next one constructed in the same process.
        self.settings = copy.deepcopy(DEFAULT_SETTINGS)
        if settings:
            self._deep_merge_dict(self.settings, settings)

        self.figure_size = tuple(self.settings["figure"]["size"])
        self.constrained_layout = bool(
            self.settings.get("layout", {}).get("constrained_layout", False))
        self.time_step_fs = 1.0

        self.full_data_files: dict[str, dict] = {
            label: {} for label in self.labels
        }
        self.metadata: dict = {}
        self.summary_df_cache: pd.DataFrame | None = None
        self.material_type_map: dict[str, str] = {}
        self._db_summary_rows: list[dict] = []

        self.type_display_names = {
            'b_type': 'buckled',
            'h_type': 'hexagonal',
            't_type': 'trigonal',
            'p_type': 'puckered',
            'other': 'bi-buckled',
        }
        self.next_series_color_index = 0

        self._discover_data_files()
        self._load_db_labels()
        self._load_all_metadata()
        self._create_material_type_map()
        self._initialize_series_color_map()

    def new_figure(self, figsize=None, nrows=1, ncols=1, **kwargs):
        """A figure honouring `layout.constrained_layout` and `fonts.family`.

        One place, so every mixin gets both without each repeating the
        `plt.subplots` keyword or touching rcParams itself. `nrows`/`ncols`
        and the passthrough kwargs exist so the multi-panel taxonomy plots go
        through here too -- a setting that reached only the single-axes plots
        would be half a setting.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        plt.rcParams['font.family'] = self.settings['fonts']['family']
        return plt.subplots(nrows, ncols, figsize=figsize or self.figure_size,
                            constrained_layout=self.constrained_layout,
                            **kwargs)

    def generate_plot(self, plot_config: dict) -> None:
        """Dispatch to the appropriate plot generation method."""
        plot_type = plot_config.get('plot_type', 'summary')

        dispatch = {
            'summary': self._generate_summary_plot,
            'timeseries': self._generate_timeseries_plot,
            'stick_slip_analysis': self._generate_stick_slip_analysis_plot,
            'scatter_comparison': self._generate_scatter_comparison,
            'rank_friction': self.rank_friction,
            'correlation': self._generate_correlation_plots,
            'cof_histogram': self._generate_cof_histogram,
            'taxonomy_signature_heatmap': self._generate_taxonomy_signature_heatmap,
            'taxonomy_prevalence': self._generate_taxonomy_prevalence,
            'taxonomy_exemplar': self._generate_taxonomy_exemplar,
            'taxonomy_regime_map': self._generate_taxonomy_regime_map,
        }

        if plot_type in dispatch:
            dispatch[plot_type](plot_config)
        else:
            logger.error("Unknown plot type '%s'", plot_type)
