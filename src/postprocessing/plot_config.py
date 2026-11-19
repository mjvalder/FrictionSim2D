"""The plot-document schema.

`postprocess plot` used to hand raw dicts straight to the mixins, so a typo
in a key was a plot that silently ignored it. This is the same vocabulary,
declared: one model per `plot_type`, `extra='forbid'`, and every field
optional with `None` so a document says exactly what it sets and nothing more.

**Defaults document, they do not decide.** `postprocess plot` passes
`spec.model_dump(exclude_unset=True)`, so the mixins' own `.get(key, default)`
defaults stay authoritative and a field's default here is documentation. Do
not "fix" that by dumping defaults -- it would change every plot silently.

Also the GUI's form source: `src/gui/schema/introspect.describe_section`
reads these models, so a field added here appears in the Plots tab with no
GUI change at all.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator
from typing_extensions import Annotated

__all__ = ['PLOT_TYPES', 'ExternalSeries', 'SourceSpec', 'PlotBase',
           'SummaryPlot', 'TimeseriesPlot', 'StickSlipPlot',
           'ScatterComparisonPlot', 'RankFrictionPlot', 'CorrelationPlot',
           'CofHistogramPlot', 'TableSource', 'StickslipPeriodHistogramPlot',
           'ContactCofScatterPlot', 'SheetProfilePlot', 'IntegrityPanelPlot', 'AnyPlot', 'DatasetsHeader', 'PlotDocument',
           'load_plot_document', 'schema_markdown', 'FigureSettings',
           'FontSettings', 'ColorSettings', 'MarkerSettings', 'LineSettings',
           'GridSettings', 'ErrorBandSettings', 'LegendSettings',
           'LayoutSettings', 'AxesSettings', 'ExportSettings', 'PlotSettings']

#: Exactly the keys `Plotter.generate_plot`'s dispatch table accepts.
#: 'stick_slip_analysis', not 'stick_slip': a model under the shorter name
#: would validate documents the Plotter then logs as an unknown plot type.
PLOT_TYPES = ('summary', 'timeseries', 'stick_slip_analysis',
              'scatter_comparison', 'rank_friction', 'correlation',
              'cof_histogram', 'stickslip_period_histogram', 'contact_cof_scatter',
              'sheet_profile', 'integrity_panel')

_FORBID = ConfigDict(extra='forbid', populate_by_name=True, serialize_by_alias=True)


class ExternalSeries(BaseModel):
    """One literature/comparison curve overlaid on a summary or histogram."""
    model_config = _FORBID

    file: Optional[str] = Field(
        None, description='JSON file, relative to the document')
    path: Optional[List[str]] = Field(
        None, description='Key path into that JSON')
    x_key: Optional[str] = None
    y_key: Optional[str] = None
    label: Optional[str] = None
    plot_style: Optional[str] = Field(None, description="'line' | 'scatter'")
    line_style: Optional[str] = None
    marker_style: Optional[str] = None
    marker_face: Optional[str] = Field(None, description="'filled' | 'open'")
    x_transform: Optional[str] = Field(
        None, description='lambda source, evaluated')
    y_transform: Optional[str] = None
    fit_line_style: Optional[str] = Field(None, description="Fit line style (default '--')")
    solid: Optional[bool] = Field(
        None, description='cof_histogram: draw filled like a dataset (this work), not hollow')


class SourceSpec(BaseModel):
    """One axis of a scatter comparison: where its numbers come from."""
    model_config = _FORBID

    dataset: Optional[str] = None
    metric: Optional[str] = None
    file: Optional[str] = None
    filter_size: Optional[str] = None
    angle: Optional[float] = None
    force: Optional[float] = None
    force_range: Optional[List[float]] = None
    error_column: Optional[str] = None
    error_metric: Optional[str] = None
    material_column: Optional[str] = None
    value_column: Optional[str] = None
    load_column: Optional[Literal['force', 'pressure']] = Field(
        None, description='Column load_range applies to (default: force, else pressure)')
    load_range: Optional[List[float]] = Field(None, description='Alias of force_range')
    require_all_loads: Optional[bool] = Field(
        None, description='Drop a material missing any load the range holds')


class PlotBase(BaseModel):
    """What every plot type accepts: identity, axes, legend and filters."""
    model_config = _FORBID

    plot_type: str
    source: Optional[str] = Field(
        None, alias='_source',
        description='Documentation only: the config or script this plot was first made with')
    filename: Optional[str] = None
    title: Optional[str] = None
    datasets: Optional[List[str]] = None
    use_dataset_colors: Optional[bool] = None
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    x_label: Optional[str] = Field(None, description='Overrides label_for()')
    y_label: Optional[str] = Field(None, description='Overrides label_for()')
    x_limits: Optional[List[Optional[float]]] = None
    y_limits: Optional[List[Optional[float]]] = None
    legend_location: Optional[str] = None
    legend_outside: Union[bool, Literal['left', 'top']] = Field(
        False, description="True: right of the axes; 'left' or 'top' for those sides")
    legend_top_align: Optional[Literal['right']] = Field(
        None, description="With legend_outside='top': right-align instead of centring")
    legend_ncol: Optional[int] = Field(None, description="Columns of a 'top' legend (default 5)")
    legend_outside_left_anchor: Optional[float] = Field(
        None, description="x anchor of a 'left' legend in axes units (default -0.16)")
    legend_order: Optional[List[str]] = Field(
        None, description='These labels first, in this order; the rest keep theirs')
    figure_size: Optional[List[float]] = None
    filter_size: Optional[str] = None
    filter_layer: Optional[int] = None
    filter_speed: Optional[float] = None
    filter_tip_radius: Optional[float] = None
    filter_materials: Optional[List[str]] = None
    filter_forces: Optional[List[float]] = None
    filter_pressures: Optional[List[float]] = None
    #: The older spelling of `filter_pressures`. BOTH are read --
    #: `plot_timeseries.py` and `plot_stick_slip.py` are each
    #: `plot_config.get('pressures') or plot_config.get('filter_pressures')` --
    #: and `documentation/publication/plots_publication.json` sets `pressures`
    #: on three of its plots, so a schema without it refuses the project's own
    #: shipped document.
    pressures: Optional[List[float]] = None
    angle: Optional[float] = None
    force: Optional[float] = None
    filter_force_range: Optional[List[float]] = None
    filter_nf_range: Optional[List[float]] = None
    lf_source: Optional[Literal['as_was', 'sx_sy', 'fx_fy']] = None
    family: Optional[str] = Field(
        None, description='Keep only runs of this family (new in v2)')
    tier: Literal['window', 'raw', 'peratom'] = Field(
        'window', description='Which sampling tier to plot (new in v2)')
    use_scientific_notation: Optional[bool] = None
    average_first_n_timesteps: Optional[int] = None
    average_timestep_range: Optional[List[int]] = None

    @model_validator(mode='after')
    def _plot_type_is_always_set(self):
        """Keep `plot_type` in every `exclude_unset` dump.

        `plot_type` is a `Literal` with a default on each concrete model, so
        constructing `SummaryPlot(filename=...)` leaves it *unset* and
        `model_dump(exclude_unset=True)` drops it -- producing a document
        whose own discriminator then rejects it. Every writer of a plot
        document hit this: the CLI's `--init`, the GUI's PlotController, and
        anything that round-trips a plot. Marking it set fixes all of them at
        once, including the nested serialisation inside `PlotDocument`, which
        a `model_dump` override on this class would not reach.
        """
        self.__pydantic_fields_set__.add('plot_type')
        return self


class SummaryPlot(PlotBase):
    """Aggregate x-vs-y across runs, with fits and external series."""
    plot_type: Literal['summary'] = 'summary'

    add_linear_fit: Optional[bool] = None
    axes_margins_in: Optional[Dict[Literal['left', 'right', 'bottom'], float]] = Field(
        None, description='Space (in) between that side of the axes and the image edge, '
                          'so panels line up; never clips')
    append_slope_to_legend: Optional[bool] = None
    clip_fit_to_data: Optional[bool] = None
    extend_fit_to_origin: Optional[bool] = None
    external_series: Optional[List[ExternalSeries]] = None
    external_outline_width: Optional[float] = None
    fit_constraint: Optional[Literal['first_point', 'origin', 'none']] = None
    fit_only: Optional[bool] = None
    fit_x_range: Optional[List[float]] = None
    label_prefix: Optional[str] = None
    label_suffix: Optional[str] = None
    legend_figure_size: Optional[List[float]] = None
    legend_filename: Optional[str] = None
    legend_ncol: Optional[int] = None
    legend_separate_png: Optional[bool] = None
    marker_by_potential_type: Optional[bool] = None
    plot_by: Optional[str] = None
    plot_style: Optional[str] = None
    potential_markers: Optional[Dict[str, str]] = None
    shift_external_series_to_origin: Optional[bool] = None
    shift_series_to_origin: Optional[bool] = None
    show_dataset_in_legend: Optional[bool] = None
    show_error_bands: Optional[bool] = None
    slope_decimals: Optional[int] = None
    slope_fit_constraint: Optional[
        Literal['first_point', 'origin', 'none']] = None
    slope_label: Optional[str] = None
    use_visible_xrange_for_fit: Optional[bool] = None


class TimeseriesPlot(PlotBase):
    """One line per run against the time axis."""
    plot_type: Literal['timeseries'] = 'timeseries'

    plot_all_forces: Optional[bool] = None
    primary_line_alpha: Optional[float] = None
    primary_line_style: Optional[str] = None
    primary_line_width: Optional[float] = None
    run_label_mode: Optional[str] = None
    secondary_dash_pattern: Optional[List[float]] = None
    secondary_line_alpha: Optional[float] = None
    secondary_line_style: Optional[str] = None
    secondary_line_width: Optional[float] = None
    secondary_y_axis: Optional[str] = None
    secondary_y_label: Optional[str] = None
    secondary_y_scale: Optional[float] = None
    shift_time_to_zero: Optional[bool] = None
    time_scale: Optional[float] = None
    time_unit: Optional[str] = None
    use_run_labels: Optional[bool] = None


class StickSlipPlot(PlotBase):
    """One run's sawtooth, its extrema and its power spectrum."""
    plot_type: Literal['stick_slip_analysis'] = 'stick_slip_analysis'

    shift_time_to_zero: Optional[bool] = Field(
        None, description='Start the time axis at 0 (default) or keep the LAMMPS clock')
    apply_hann_window: Optional[bool] = None
    bottom_ylabel: Optional[str] = None
    displacement_axis: Optional[str] = None
    displacement_scale: Optional[float] = None
    displacement_unit: Optional[str] = None
    display_periods: Optional[float] = None
    display_time_start: Optional[float] = None
    #: `[start, end]`, not a duration: `plot_stick_slip` reads it as
    #: `len(...) == 2` and takes both ends, and the shipped publication
    #: document sets `[22, 260]` on two of its plots.
    display_time_window: Optional[List[float]] = None
    extrema_min_distance_points: Optional[int] = None
    extrema_min_prominence: Optional[float] = None
    extrema_smooth_window: Optional[int] = None
    #: Declared, never read: `plot_stick_slip` gates on `extrema_min_prominence`
    #: being set. Accepted because `documentation/publication/
    #: plots_publication.json` sets it, and a schema that rejects the project's
    #: own published document is one nobody adopts. Documented as inert here
    #: rather than deleted from the document, which would rewrite a published
    #: artefact to suit a schema.
    extrema_use_prominence: Optional[bool] = None
    fft_peak_min_distance_points: Optional[int] = None
    fft_peak_min_prominence: Optional[float] = None
    fft_peak_rank: Optional[int] = None
    fft_peaks_to_report: Optional[int] = None
    freq_max: Optional[float] = None
    freq_min: Optional[float] = None
    freq_plot_max: Optional[float] = None
    freq_plot_min: Optional[float] = None
    freq_xlabel: Optional[str] = None
    max_intervals_to_draw: Optional[int] = None
    results_filename: Optional[str] = None
    run_index: Optional[int] = None
    secondary_y_axis: Optional[str] = None
    secondary_y_scale: Optional[float] = None
    show_peak_trough_intervals: Optional[bool] = None
    single_panel_figure_size: Optional[List[float]] = None
    smooth_for_extrema: Optional[bool] = None
    spectrum_filename: Optional[str] = None
    time_scale: Optional[float] = None
    time_unit: Optional[str] = None
    top_ylabel: Optional[str] = None


class ScatterComparisonPlot(PlotBase):
    """One metric against another, one point per material."""
    plot_type: Literal['scatter_comparison'] = 'scatter_comparison'

    add_linear_fit: Optional[bool] = None
    color_by_material_class: Optional[bool] = None
    iterative_outlier_removal: Optional[bool] = None
    point_label_fontsize: Optional[float] = None
    show_error_bars: Optional[bool] = None
    show_identity_line: Optional[bool] = None
    show_point_labels: Optional[bool] = None
    show_r_squared: Optional[bool] = None
    x_source: Optional[SourceSpec] = None
    y_source: Optional[SourceSpec] = None


class RankFrictionPlot(PlotBase):
    """Rank materials by a metric and write the ranking JSON."""
    plot_type: Literal['rank_friction'] = 'rank_friction'

    aggregate_over_force_range: Optional[bool] = None
    fit_x_range: Optional[List[float]] = None
    force_range: Optional[List[float]] = None
    rank_by: Optional[str] = None
    rank_order: Optional[Literal['ascending', 'descending']] = None


class CorrelationPlot(PlotBase):
    """Rank-correlation heatmaps over the ranking JSONs."""
    plot_type: Literal['correlation'] = 'correlation'

    correlate_by: Optional[Literal['size', 'force', 'pairwise']] = None
    correlation_force: Optional[float] = None
    filename_prefix: Optional[str] = None
    sizes_to_compare: Optional[List[str]] = None


class CofHistogramPlot(PlotBase):
    """Bar chart of the LF-vs-NF regression slope per dataset (`cof_slope`)."""
    plot_type: Literal['cof_histogram'] = 'cof_histogram'

    annotate_bars: Optional[bool] = None
    bar_width: Optional[float] = None
    external_series: Optional[List[ExternalSeries]] = None
    fit_x_range: Optional[List[float]] = None
    slope_decimals: Optional[int] = None
    slope_fit_constraint: Optional[
        Literal['first_point', 'origin', 'none']] = None
    x_label_rotation: Optional[int] = None
    external_outline_width: Optional[float] = Field(
        None, description='Ring width (pt) of a hollow external-series bar (default 10)')
    show_adhesion_pairs: Optional[bool] = Field(
        None, description='Two bars per series: mean LF/NF (with adhesion) and the slope')
    ratio_x_range: Optional[List[float]] = Field(
        None, description='Load window of the mean LF/NF (default: the fit window)')
    with_adhesion_label: Optional[str] = None
    without_adhesion_label: Optional[str] = None
    without_adhesion_legend_first: Optional[bool] = None
    no_adhesion_hatch: Optional[str] = Field(
        None, description="Hatch of the slope bar (default '///')")


class TableSource(BaseModel):
    """Where a table plot reads its rows. A bare string is a file path.

    A CSV path, resolved like external-series files. (Analysis results by
    name come with the analysis layer.)
    """
    model_config = _FORBID

    file: Optional[Path] = Field(None, description='CSV file')

    @model_validator(mode='before')
    @classmethod
    def _bare_path(cls, value):
        return {'file': value} if isinstance(value, (str, Path)) else value

    @model_serializer(mode='wrap')
    def _as_bare_path(self, handler):
        """A file-only source dumps back to the bare path it was written as."""
        return str(self.file) if self.file is not None else handler(self)


class _TablePlot(PlotBase):
    """Shared fields of the analysis-figure plot types."""
    loads: Optional[List[float]] = Field(None, description='Loads (nN), one colour each')
    load_colors: Optional[List[str]] = None
    line_width: Optional[float] = None


class StickslipPeriodHistogramPlot(_TablePlot):
    """Lateral-force period / lattice period along the scan, one step line per load."""
    plot_type: Literal['stickslip_period_histogram'] = 'stickslip_period_histogram'

    stickslip_file: Optional[TableSource] = Field(
        None,
        description='material, load_nN, lf_period_A, a_x_A, lf_period_clear')
    ranking_file: Optional[TableSource] = Field(None, description='material, eligible')
    bins_arange: Optional[List[float]] = Field(None, description='[start, stop, step]')
    reference_line: Optional[float] = Field(1, description='x of the dashed line; null for none')


class ContactCofScatterPlot(_TablePlot):
    """COF against a tip-contact measure at one state, one colour per load."""
    plot_type: Literal['contact_cof_scatter'] = 'contact_cof_scatter'

    contact_file: Optional[TableSource] = Field(
        None, description='material, load_nN, state, <x_column>')
    ranking_file: Optional[TableSource] = Field(
        None, description='material, eligible, cof_<L>nN')
    x_column: Optional[str] = Field(None, description='Contact column on x (required)')
    state: Optional[str] = Field(None, description="'load' (default) or 'final'")
    marker_scale: Optional[float] = Field(None, description='times markers.size**2 (0.8)')
    alpha: Optional[float] = None
    y_scale: Optional[Literal['log', 'linear']] = None
    y_ticks: Optional[List[float]] = None


class SheetProfilePlot(_TablePlot):
    """Top surface and bottom layer height vs distance from the tip axis, per load."""
    plot_type: Literal['sheet_profile'] = 'sheet_profile'

    run_dir: Optional[str] = Field(None, description='AFM run directory holding <layer_dir>/')
    layer_dir: Optional[str] = None
    tip_types: Optional[List[int]] = None
    tip_apex_depth: Optional[float] = None
    far_field_radius: Optional[float] = None
    sublayer_dz: Optional[float] = None
    radial_bins_arange: Optional[List[float]] = Field(None, description='[start, stop, step]')
    top_label: Optional[str] = None
    bottom_label: Optional[str] = None


class IntegrityPanelPlot(PlotBase):
    """End-of-slide integrity panel of one material and layer, a column per load."""
    plot_type: Literal['integrity_panel'] = 'integrity_panel'

    material: Optional[str] = Field(None, description='Material (required)')
    run_dir: Optional[str] = Field(None, description='AFM run directory holding L<layer>/')
    layer: Optional[int] = None
    loads: Optional[List[float]] = None
    traces_file: Optional[TableSource] = Field(
        None,
        description='run_id, step, temp_sheet_3d, comz_L1, comz_L2')
    status_file: Optional[TableSource] = Field(None, description='run_id, force, status')
    dpi: Optional[int] = None


AnyPlot = Annotated[
    Union[SummaryPlot, TimeseriesPlot, StickSlipPlot, ScatterComparisonPlot,
          RankFrictionPlot, CorrelationPlot, CofHistogramPlot,
          StickslipPeriodHistogramPlot, ContactCofScatterPlot, SheetProfilePlot,
          IntegrityPanelPlot],
    Field(discriminator='plot_type')]


class DatasetsHeader(BaseModel):
    """The dataset half of a plot document, shared with the GUI's picker."""
    model_config = _FORBID

    data_dirs: List[str] = Field(
        default_factory=list,
        description="Directories holding outputs/, or aiida://dataset/<name> URIs")
    labels: List[str] = Field(default_factory=list,
                              description='One per data_dir')
    dataset_display_labels: Dict[str, str] = Field(default_factory=dict)
    series_color_map: Dict[str, Optional[str]] = Field(default_factory=dict)

    @model_validator(mode='after')
    def _labels_match_dirs(self):
        if len(self.labels) != len(self.data_dirs):
            raise ValueError(
                f"labels ({len(self.labels)}) must have one entry per "
                f"data_dirs ({len(self.data_dirs)})")
        return self


class PlotDocument(DatasetsHeader):
    """A whole `postprocess plot` document."""
    comment: Optional[str] = Field(None, alias='_comment',
                                   description='Documentation only')
    plots: List[AnyPlot] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Style settings (Q4)
#
# Every list field uses `default_factory` rather than a literal default, so two
# `PlotSettings()` never share a list and one Plotter's `--settings` cannot
# mutate the next one's.
# ---------------------------------------------------------------------------


class FigureSettings(BaseModel):
    """Canvas size and resolution."""
    model_config = _FORBID

    size: List[float] = Field(default_factory=lambda: [10, 7],
                              description='[width, height] in inches')
    dpi: int = 150


class FontSettings(BaseModel):
    """Type sizes, in points."""
    model_config = _FORBID

    family: str = 'sans-serif'
    title: int = 26
    axis_label: int = 24
    tick_label: int = 22
    legend: int = 16


class ColorSettings(BaseModel):
    """Which categorical palette series are drawn from."""
    model_config = _FORBID

    palette: str = 'gem12'


class MarkerSettings(BaseModel):
    """Default marker for scatter and summary series."""
    model_config = _FORBID

    style: str = 'o'
    size: float = 12


class LineSettings(BaseModel):
    """Line weight, and how a fit line is drawn."""
    model_config = _FORBID

    width: float = 1.3
    fit_style: str = '--'
    fit_alpha: float = 0.8


class GridSettings(BaseModel):
    """Grid visibility and weight."""
    model_config = _FORBID

    show: bool = True
    which: str = 'both'
    major_style: str = '-'
    minor_style: str = ':'
    major_alpha: float = 0.5
    minor_alpha: float = 0.3


class ErrorBandSettings(BaseModel):
    """Opacity of the +/- 1 sigma band."""
    model_config = _FORBID

    alpha: float = 0.2


class LegendSettings(BaseModel):
    """Where the legend sits when a plot does not override it."""
    model_config = _FORBID

    location: str = 'best'


class LayoutSettings(BaseModel):
    """How the axes are fitted to the canvas.

    `layout` existed in `DEFAULT_SETTINGS` as an empty dict -- declared and
    never read. `constrained_layout` is what a layout setting is for: it lets
    matplotlib size the axes around the labels instead of the fixed
    `subplots_adjust` margins `_save_plot` otherwise applies.
    """
    model_config = _FORBID

    constrained_layout: bool = False


class AxesSettings(BaseModel):
    """Tick formatting."""
    model_config = _FORBID

    use_scientific_notation: bool = True
    scilimits: List[int] = Field(default_factory=lambda: [-3, 4])


class ExportSettings(BaseModel):
    """Which files each plot writes."""
    model_config = _FORBID

    formats: List[str] = Field(default_factory=lambda: ['png'])
    transparent: bool = False


class PlotSettings(BaseModel):
    """Style settings for a whole Plotter run.

    Mirrors `plot_style.DEFAULT_SETTINGS` key for key: the dict was the schema
    and nothing documented it, so `--settings` was written by reading the
    source. `layout` was declared there and never read; it now carries
    `constrained_layout`, which is what a layout setting is for.
    """
    model_config = _FORBID

    figure: FigureSettings = Field(default_factory=FigureSettings)
    fonts: FontSettings = Field(default_factory=FontSettings)
    colors: ColorSettings = Field(default_factory=ColorSettings)
    markers: MarkerSettings = Field(default_factory=MarkerSettings)
    lines: LineSettings = Field(default_factory=LineSettings)
    grid: GridSettings = Field(default_factory=GridSettings)
    error_bands: ErrorBandSettings = Field(default_factory=ErrorBandSettings)
    legend: LegendSettings = Field(default_factory=LegendSettings)
    layout: LayoutSettings = Field(default_factory=LayoutSettings)
    axes: AxesSettings = Field(default_factory=AxesSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)


def load_plot_document(path: Path) -> PlotDocument:
    """Read and validate one plot document.

    The empty-`data_dirs` check lives here rather than on the model: a
    document being EDITED is legitimately incomplete -- the GUI's Plots tab
    starts with no datasets and the user adds them -- while one being LOADED
    to render has to name somewhere to read from. Without the check,
    `{"plots": []}` validated and rendered nothing while exiting 0.
    """
    document = PlotDocument.model_validate(
        json.loads(Path(path).read_text(encoding='utf-8')))
    if not document.data_dirs:
        raise ValueError(
            f"{path}: data_dirs is empty -- a plot document must name at "
            f"least one dataset directory (or aiida://dataset/<name> URI)")
    return document


def _model_rows(model) -> List[str]:
    rows = []
    for name, info in model.model_fields.items():
        annotation = str(info.annotation).replace('typing.', '')
        doc = (info.description or '').replace('|', r'\|')
        rows.append(f'| `{name}` | `{annotation}` | {doc} |')
    return rows


def schema_markdown() -> str:
    """The plot schema as a Markdown table per type, for the manual.

    Generated rather than hand-written so `documentation/postprocessing.md`
    cannot drift from the models the CLI actually validates against.
    """
    models = {'summary': SummaryPlot, 'timeseries': TimeseriesPlot,
              'stick_slip_analysis': StickSlipPlot,
              'scatter_comparison': ScatterComparisonPlot,
              'rank_friction': RankFrictionPlot,
              'correlation': CorrelationPlot,
              'cof_histogram': CofHistogramPlot,
              'stickslip_period_histogram': StickslipPeriodHistogramPlot,
              'contact_cof_scatter': ContactCofScatterPlot,
              'sheet_profile': SheetProfilePlot,
              'integrity_panel': IntegrityPanelPlot}
    out = ['## Document', '', '| field | type | meaning |', '|---|---|---|']
    out += _model_rows(PlotDocument)
    for plot_type, model in models.items():
        out += ['', f'## `{plot_type}`', '',
                '| field | type | meaning |', '|---|---|---|']
        out += _model_rows(model)
    return '\n'.join(out) + '\n'
