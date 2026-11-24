"""Thin Plotter facade composed from dedicated plotting mixins."""
# pylint: disable=too-many-instance-attributes,too-many-ancestors
# pylint: disable=too-many-arguments,too-many-positional-arguments

from __future__ import annotations

import copy
import logging
from pathlib import Path

import pandas as pd

from .plot_common import PlotCommonMixin
# Re-exported: callers import it from here, it is defined there to keep
# plot_common -> plot_data from being a cycle.
from .plot_mixin_base import PlotDataError
from .plot_correlation import PlotCorrelationMixin
from .plot_ranking import PlotRankingMixin
from .plot_style import DEFAULT_SETTINGS, merge_settings
from .plot_stick_slip import PlotStickSlipMixin
from .plot_summary import PlotSummaryMixin
from .plot_tables import PlotTablesMixin
from .plot_timeseries import PlotTimeseriesMixin

__all__ = ['Plotter', 'PlotDataError']

logger = logging.getLogger(__name__)


class Plotter(
    PlotCommonMixin,
    PlotSummaryMixin,
    PlotTimeseriesMixin,
    PlotStickSlipMixin,
    PlotCorrelationMixin,
    PlotRankingMixin,
    PlotTablesMixin,
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

        # deepcopy: a shallow copy lets one Plotter mutate the shared defaults.
        self.settings = copy.deepcopy(DEFAULT_SETTINGS)
        if settings:
            merge_settings(self.settings, settings)

        self.figure_size = tuple(self.settings["figure"]["size"])
        self.constrained_layout = bool(
            self.settings.get("layout", {}).get("constrained_layout", False))
        self.time_step_fs = 1.0

        self.full_data_files: dict[str, dict] = {
            label: {} for label in self.labels
        }
        self.metadata: dict = {}
        self.summary_df_cache: pd.DataFrame | None = None
        #: Every file `_save_plot` wrote, in order.
        self.written: list[Path] = []
        #: label -> file_key -> summary_<size>.csv
        self.summary_files: dict[str, dict[str, Path]] = {}
        self.runs_cache: dict[tuple[str, str, str], list[dict]] = {}
        #: The family of the runs most recently extracted, for label
        #: lookup. 'afm' until a dataset says otherwise -- label_for's
        #: own default.
        self.plot_family: str = 'afm'
        self.material_type_map: dict[str, str] = {}
        #: Summary rows of the ``aiida://dataset/<name>`` labels.
        self._db_summary_rows: list[dict] = []
        #: label -> the dataset's units (family -> key -> unit).
        self.dataset_units: dict[str, dict] = {}
        #: label -> (dataset name, rows) of every ``aiida://dataset/`` label;
        #: its time series are read from the store point by point.
        self._dataset_sources: dict[str, tuple[str, list[dict]]] = {}

        self.type_display_names = {
            'b_type': 'buckled',
            'h_type': 'hexagonal',
            't_type': 'trigonal',
            'p_type': 'puckered',
            'other': 'bi-buckled',
        }
        self.next_series_color_index = 0

        self._discover_data_files()
        self._load_dataset_labels()
        self._load_all_metadata()
        self._create_material_type_map()
        self._initialize_series_color_map()

    def new_figure(self, figsize=None, nrows=1, ncols=1, **kwargs):
        """A figure honouring `layout.constrained_layout` and `fonts.family`.

        One place, so every mixin gets both without each repeating the
        `plt.subplots` keyword or touching rcParams itself. `nrows`/`ncols`
        and the passthrough kwargs exist so multi-panel plots go
        through here too -- a setting that reached only the single-axes plots
        would be half a setting.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        plt.rcParams['font.family'] = self.settings['fonts']['family']
        return plt.subplots(nrows, ncols, figsize=figsize or self.figure_size,
                            constrained_layout=self.constrained_layout,
                            **kwargs)

    def generate_plot(self, plot_config) -> None:
        """Dispatch one plot spec, given as a model or a plain dict.

        A model is dumped with ``exclude_unset=True``: the mixins' own
        ``.get(key, default)`` defaults stay authoritative and the Pydantic
        defaults document them. Dumping defaults instead would change every
        plot in the corpus at once, silently.

        """
        from .plot_config import PlotBase  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

        if isinstance(plot_config, PlotBase):
            plot_config = plot_config.model_dump(exclude_unset=True)
            # model_dump(exclude_unset=True) omits plot_type when the Literal
            # defaulted it, and that is the one key the dispatcher needs.
            plot_config.setdefault('plot_type', 'summary')

        plot_type = plot_config.get('plot_type', 'summary')
        dispatch = {
            'summary': self._generate_summary_plot,
            'timeseries': self._generate_timeseries_plot,
            'stick_slip_analysis': self._generate_stick_slip_analysis_plot,
            'scatter_comparison': self._generate_scatter_comparison,
            'rank_friction': self.rank_friction,
            'correlation': self._generate_correlation_plots,
            'cof_histogram': self._generate_cof_histogram,
            'stickslip_period_histogram': self._generate_stickslip_period_histogram,
            'contact_cof_scatter': self._generate_contact_cof_scatter,
            'sheet_profile': self._generate_sheet_profile,
            'integrity_panel': self._generate_integrity_panel,
        }
        if plot_type not in dispatch:
            raise PlotDataError(
                f"unknown plot_type {plot_type!r}; known: {sorted(dispatch)}")
        dispatch[plot_type](plot_config)
