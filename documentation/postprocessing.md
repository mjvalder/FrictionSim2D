# Postprocessing

A simulation root goes in, `outputs/` comes out. `postprocess read` walks every run
under a root, reads its results files, and writes two machine-readable exports plus the
completeness reports. `postprocess plot` reads those exports — never the raw tree — and
draws figures from a validated plot document.

A "run" is one `provenance.json`. The walk is `frictionsim2d.run.tree.discover_runs`, and the
directory is parsed by `frictionsim2d.run.naming.parse_run_path`, so the potential, the
temperature, the twist and every other sweep axis reach the export as typed fields
rather than as directory names nobody parsed.

---

## The output tree

```
<simulation root>/
  afm/ sheetonsheet/ indent/ pes_tip/ pes_sheet/     the runs
  outputs/
    output_full_<size>.json          every run's time series, schema 3
    summary_<size>.csv               one row per load point
    incomplete_files_<size>.txt      every load point whose status is not complete,
                                     one line each with a [status] suffix
    incomplete_materials_<size>.txt  the materials those points belong to
```

A *size* is the `100x100y` token built from the run's `size_x`/`size_y`. One export file
per size, because a size is the one axis along which two runs are rarely compared
directly.

An incomplete run is **flagged, not dropped**: it stays in both exports and is listed in
the reports. Flagging a run and discarding it are different decisions, and only the
first belongs to the reader.

---

## `output_full_<size>.json` (schema 3)

```jsonc
{
  "schema": 3,                       // int, always 3
  "size": "100x100y",                // str, the export key
  "generated_by": "FrictionSim2D",   // str, constant
  "units": {                         // {column: unit}, from the observable registry
    "time": "step",                  //   'time' is always 'step'
    "nf": "nN", "lfx": "nN", "lfy": "nN",
    "comz_L3": "A",
    "lateral_force": "nN",           //   derived
    "cof": "1"                       //   derived, dimensionless
  },
  "metadata": {
    "families": ["afm"],                       // [str], sorted
    "models": ["afm"],                         // [str], sorted
    "materials": ["h_MoS2"],                   // [str], sorted, underscored
    "material_types": {                        // {str: [str]}, five fixed keys
      "b_type": [], "h_type": ["h_MoS2"], "t_type": [], "p_type": [], "other": []
    },
    "potentials": ["pot_sw_h-MoS2"],           // [str], sorted
    "temperatures": [300.0],                   // [float], sorted
    "sizes": ["100x100y"],                     // [str], sorted
    "layers": [1, 2],                          // [int], sorted; [] for sheet-like
    "speeds": [2.0],                           // [float], sorted
    "substrates": ["Si"],                      // [str], sorted; [] for sheet-like
    "tip_materials": ["Si"],                   // [str], sorted; [] for sheet-like
    "tip_radii": [25.0],                       // [float], sorted; [] for sheet-like
    "loads_by_kind": {                         // {kind: {load: [angles]}}
      "force": {"30.0": [0.0, 90.0]},
      "pressure": {}
    },
    "tiers": ["peratom", "raw", "window"],     // [str], sorted
    "ntimestep": 500,                          // int, longest window frame
    "time_series": [1000, 2000, 3000],         // [int], the first complete frame
    "summary_definition": {                    // which COF made the summary numbers
      "cof": "ratio_of_means: mean(LF)/mean(NF)",
      "skip_fraction": 0.2,
      "source": "frictionsim2d.data.models.summarise_run"
    }
  },
  "runs": [                                    // sorted by
                                               // (rel_path, load_kind, load, angle, speed)
    {
      "model": "afm",                          // str, the top directory
      "family": "afm",                         // str, from provenance.json
      "material": "h_MoS2",                    // str, underscored join key
      "materials": ["h-MoS2"],                 // [str], as written on disk
      "flake_material": null,                  // str | null
      "size": "100x100y",                      // str
      "size_x": 100.0, "size_y": 100.0,        // float
      "substrate": "Si",                       // str | null
      "substrate_amorphous": true,             // bool | null
      "tip_material": "Si",                    // str | null
      "tip_radius": 25.0,                      // float | null
      "temperature": 300.0,                    // float
      "layers": 2,                             // int | null (the L{n} path level)
      "potential": "pot_sw_h-MoS2",            // str | null
      "rel_path": "afm/h-MoS2/100x_100y/sub_aSi_tip_Si_r25/K300/pot_sw_h-MoS2/L2",
      "load_kind": "force",                    // 'force' | 'pressure' | 'none' (PES)
      "load": 30.0,                            // float
      "angle": 0.0,                            // float, SIGNED
      "speed": 2.0,                            // float
      "n_layers": 2,                           // int, the resolved layer count
      "tiers": {                               // absent tiers are absent keys
        "window": {
          "columns": ["time", "nf", "lfx", "lfy", "lateral_force", "cof"],
          "data": [[1000, 30.0, 3.0, 0.0, 3.0, 0.1]],
          "attrs": {"lf_schema": 2, "lf_corrected": false}
        },
        "raw":     {"columns": ["time", "raw_nf", "raw_lfx"],
                    "data": [[100, 30.1, 2.9]],
                    "attrs": {"lf_schema": 2, "lf_corrected": false}},
        "peratom": {"columns": ["time", "n_contact"],
                    "data": [[1000, 12.0]],
                    "attrs": {"lf_schema": 2, "lf_corrected": false}}
      },
      "post": {"gap_min": 3.1, "deflect_max": 0.4},  // {} when no data/final_*.data
      "status": "complete",          // schema 3: complete | truncated | no_output | unreadable | not_reached
      "status_detail": "",           //   why, e.g. "window 20/50 rows"
      "n_rows": {"window": 50},      //   rows actually read, per tier
      "expected_rows": {"window": 50},  // rows the config says, per tier ({} if unknown)
      "provenance": {...}            // schema 3: the run's record, physics config, geometry, this load's ramp
    }
  ]
}
```

Four rules the JSON cannot state:

1. **One file per size.** `runs` holds every load point of every run at that size.
2. **`time` is kept in every tier.** The v1 export dropped it and reconstructed one
   global axis from the metadata, so two runs of different lengths could not be drawn
   together.
3. **`lateral_force` and `cof` are derived onto the window tier only.** No other tier
   carries them.
4. **`post` is `{}`** when the run has no `data/final_*.data`, and for the two PES
   families, whose load point has no matching file.

A **PES run** has `"load_kind": "none"`, `"load": 0.0`, and exactly one `window` tier
parsed from `results/pes_scan.csv`. Calling its absent load a force of 0.0 nN would put
a fictitious point on every load axis.

### Schema 3 additions

Schema 3 is schema 2 plus new keys only; nothing from schema 2 is renamed, removed or
changes meaning. The plot stack reads schema 2 and schema 3 files, and **draws only
`complete` points**: `truncated`, `no_output`, `unreadable` and `not_reached` entries are
skipped (a schema 2 record has no `status` and counts as complete).

- **Four status keys per `runs[]` entry**: `status`, `status_detail`, `n_rows`,
  `expected_rows`. Every load point the run's config expects is exported, so a load
  that wrote nothing appears as `no_output` (or `not_reached` after an earlier failure
  of the same chained script) with empty `tiers`, instead of vanishing.
- **`provenance` block per `runs[]` entry**: the run's `provenance.json` record, the
  physics slice of `provenance/config.json`, the built geometry, and this load's
  `ramp` / `load_achieved`. An unreadable record yields only `error`,
  `provenance_sha256` and `config`.
- **`metadata.provenance`**: every provenance leaf pooled over runs (not load points),
  as `{"value": v}` when all runs agree or `{"values": {v: n_runs}}` when they differ;
  plus `config_hashes`, `seeds_unique` and `seeds_shared_across_runs`. Seeds, the file
  hash, `execution` and the per-load ramp fields are never pooled.
- **`metadata.status_counts`**: `{status: n}` over every load point of the size, zeros
  included. Without `--complete-only` that is exactly the `runs[]` entries; with it the
  counts still cover every point, including the non-complete ones left out of `runs[]`.
- **The v2 `metadata` keys** (`loads_by_kind`, `speeds`, `materials`, ...) cover only
  points that have data (`complete`/`truncated`), as before.
- `--complete-only` on `postprocess read` restricts the export and the summary CSV to
  `complete` points.
- **Legacy truncation rule.** A slide-family point whose config gives no expected row
  count (no `settings` in `provenance/config.json`) is `truncated` when its window is
  more than three rows shorter than the longest window of the same family and size.
  PES and indent points are never judged this way: their row counts follow the scan
  grid and the indent protocol.

A **v1 file has no `"schema"` key at all**, and the plot stack refuses it with a
`PlotDataError` naming the command that replaces it:

```bash
FrictionSim2D postprocess read <root> --export
```

---

## `summary_<size>.csv`

One row per load point. Column order:

1. **Identity** — `model, family, material, materials, flake_material, size, size_x,
   size_y, substrate, substrate_amorphous, tip_material, tip_radius, temperature,
   layers, potential, rel_path, load_kind, load, angle, speed, tiers, status,
   status_detail`.
   `materials` is `'+'.join(...)`, `tiers` is `';'.join(sorted(...))`, and `layers` is
   the **resolved** count — the `L{n}` path level for the AFM-like models, and
   `provenance.json`'s `layer_type_map` for the sheet-like ones, which emit no `L{n}`
   level at all.
2. **The fourteen `RunSummary` scalars** — `mean_cof, std_cof, mean_lf, std_lf, mean_nf,
   std_nf, mean_lfx, std_lfx, mean_lfy, std_lfy, skip_fraction, ntimesteps,
   time_series_hash, is_complete`. Here `is_complete` is `status == "complete"` **and**
   `summarise_run`'s own check (non-empty window, role columns resolved), so a point
   tagged `complete` can still read `False`.
3. **Family-specific extras** — `slip_*` for any run with a raw tier, `pes_*` for a PES
   run, the retract features for `indent`, the slide distance and driven-COM drift for a
   slide family. The header therefore differs between families *by design*; the frame
   does not. Two campaigns of the same family concatenate directly, and two of different
   families concatenate through `pd.concat`, which fills the missing extras with `NaN`.
4. **`mean_lf_sx_sy, mean_lf_fx_fy`** — the two alternative lateral-force definitions,
   `NaN` where the frame carries no `react_x`/`react_y` or `drive_fx`/`drive_fy`.
5. **`gap_min, deflect_max, gap_min_il, corrugation_max`** — the post-stream channels,
   `NaN` where the family has none.

**Every number in this file comes from `frictionsim2d.data.models.summarise_run`**, the single COF
on the store path — the same function the AiiDA node reads. In
particular `mean_cof` is `mean(LF)/mean(NF)`, computed after discarding the first
`skip_fraction` (0.2 by default) of the run as transient.

`cof_slope` is a **different quantity**: the slope of a linear fit of lateral force
against normal load across a load sweep. Both are legitimate; they have different names
because calling them by one name is how a single run reported two COFs. The COF
histogram plots `cof_slope`, and its axis says so.

---

## Migration

**Figures drawn before this rework used a different COF.** The plot stack averaged an
element-wise `cof` column over *all* samples (`df.mean()`); it now reads
`summary_<size>.csv`, whose `mean_cof` is `ratio_of_means` — `mean(LF)/mean(NF)` after
dropping the first `skip_fraction`. The new definition is the tribology convention and
is the one the store keeps, but the two disagree by more than
rounding wherever a run has a load ramp or a slow start.

**Any figure already in a paper or a talk must be regenerated and re-read, not assumed
unchanged.** The export records which definition produced its numbers, under
`metadata.summary_definition`, so a reader can tell from the file rather than from its
date.

Two smaller changes in the same direction:

- A **potential or temperature sweep** used to collapse: the v1 nested tree had no level
  for either, so two runs differing only in those landed on one leaf and the last one
  walked won. A v1 figure over such a sweep is missing an arm.
- **Per-layer columns beyond `L2`** (`comz_L3`, `pe_sum_L4`, …) reached v1 exports under
  their native `v_comz_L3` names and were invisible to every consumer keyed on channel
  names.

---

## Analyses

An analysis turns one AiiDA dataset (`FrictionSim2D aiida dataset`, see
[aiida.md](aiida.md#6-postprocessing-datasets-and-queries)) into one table. The code is
`frictionsim2d/analysis/`; every analysis is registered in `frictionsim2d/analysis/catalogue.py`.

- **Integrity-pass rule.** Every analysis reads only the dataset's points whose integrity
  verdict is `pass`. Each point it leaves out is recorded as `(point_uuid, reason)` in the
  result's `dropped` attribute (`n_dropped` extra). Two exceptions read every point,
  failed ones included: `run_status` and `integrity_traces`.
- **Stored and reused.** A result is a CSV (`SinglefileData`) made by the calcfunction
  `record_analysis` from the dataset's table `Dict` and a params `Dict`, in the Group
  `friction2d/analyses/<dataset>`. `analyse run` with the same name, version, params
  and dataset (and the same stored results it reads) returns the stored table instead of
  computing again; `--force` recomputes.
- **Reuse is keyed on the dataset name**, not on its content: a dataset deleted and
  rebuilt under the same name with other points reuses the old results. Rerun with
  `--force` after rebuilding a dataset. Deleting a dataset
  (`frictionsim2d.aiida.steps.delete_dataset`) deletes every analysis result computed from it,
  with a warning and their count in its return value.
- **Read results.** The period analyses (below) read a stored `sheet_geometry` result,
  and `sliding_physics` the newest stored `contact` result. Each is linked as an input of
  the result and is part of its reuse key: a new `sheet_geometry` result (or version)
  makes them compute again.
- **Archives.** `FrictionSim2D aiida export <set>` carries the stored results of every
  dataset built from that set, with their `record_analysis` nodes and the dataset table
  `Dict`s they read. On `import-archive`, an analysis Group whose label already exists in
  the profile is imported as `<label>_<n>` (aiida-core's collision rule). Its results are
  then not found under `friction2d/analyses/<dataset>`: rerun them with `analyse run`.
- **Families.** An analysis refuses a dataset of a family it does not read; no analysis
  joins across simulation types, and only `pes_landscape` has PES columns.
- **Identity columns.** `run` and `run_state` tables carry `run_uuid, material, layer,
  load, load_kind, angle`; `material` tables carry `material, layer`. Two exceptions,
  keyed by `run_id` (`<material>/l<layers>/f<load>/a<angle>`, the frozen tables' id):
  `integrity_traces` and `run_status`. `analyse run` refuses a non-empty table without
  its identity columns.

### Catalogue

| Name | Level | Families | Computes |
|---|---|---|---|
| `run_status` | run_state | afm, sheetonsheet | The verdict status of every point, failed ones included. |
| `integrity_traces` | run_state | afm, sheetonsheet | Window-tier traces and the stored integrity measures of every point with a trace, failed points included. |
| `sheet_geometry` | material | afm, sheetonsheet | Lattice period (`period_x_top`) and surface geometry of each material's built sheet. |
| `period_summary` | run | afm, sheetonsheet | Per run: COF, LF and NF over its whole lattice periods after the spring-loading transient; summary-shaped (`cof`, `lf`, `nf`), so plots read it as summaries. |
| `cof_ranking` | material | afm | Each material's whole-period COF over the load window, and its rank. |
| `contact` | run_state | afm | Tip-sheet contact, interlocking and LJ energy of the load and final snapshots. |
| `stick_slip` | run | afm | Ring period, lateral-force period, slips and energy oscillation of the tip trajectory. |
| `sliding_physics` | run | afm, sheetonsheet | Per-run scales and sliding descriptors (taxlib port), plus the shear strength joined from the newest stored `contact` result (left out, with a note, when none is stored). |
| `pes_landscape` | run | pes_sheet, pes_tip | Barrier, corrugation, curvature and Prandtl-Tomlinson eta of each PES grid. |

`FrictionSim2D analyse list` prints the name, version, level and families.

### Parameters

Pass each as `--param NAME=VALUE` (repeatable); a dict or list value is JSON text.

- **Period source** (`period_summary`, `cof_ranking`, `contact`, `stick_slip`,
  `sliding_physics`): where the lattice period `a_x` comes from.
  - `period_x` (default none): material -> `period_x_top` (Å), used as given.
  - `geometry_dataset` (default none): the dataset whose stored `sheet_geometry`
    supplies it (run or reused). With neither, the analysed dataset's own stored
    `sheet_geometry` supplies it (run or reused).
  - `lattice` (default `{}`): material -> Å, the fallback for a material with no
    geometry row. A run with neither is dropped: it has no whole periods.
- `loads` (default `(10, 20, 30, 50)` nN): the load window of `cof_ranking`, `contact`,
  `stick_slip`.
- `angle` (default `0.0`): the scan angle read by `cof_ranking`, `contact`, `stick_slip`,
  `sliding_physics`. Only 0° runs are cut into periods (`a_x` is along x).
- `layer` (default `2`): the layer `cof_ranking` ranks.
- `trib_index` (default none): `cof_ranking`'s tribology-index JSON (`materials`,
  `tribIndex`, `dev`), given as content so the params hash covers it.
- `taxlib_a` (default `{}`): material -> the lattice `a` taxlib scales by
  (`stick_slip`, `sliding_physics`); a material not in it takes `a_x`.
- `k`, `a` (no defaults): `pes_landscape`'s spring (eV/Å²) and lattice period (Å, one
  number or material -> Å) of the eta; either missing leaves `pt_eta` NaN.
- `materials`, `layer` (default none: every point): the points `integrity_traces`
  traces. `materials` is a list (JSON text or comma-separated on the command line), in
  either spelling (`h-MoS2` or `h_MoS2`). The full table of a 156-material campaign is
  about 1.4 million rows; select what a figure needs.
- `run_status`, `sheet_geometry`: no parameters.

### `analyse` commands

```bash
FrictionSim2D analyse list
FrictionSim2D analyse run NAME DATASET [--param NAME=VALUE ...] [--force] [--profile TEXT]
FrictionSim2D analyse show NAME DATASET [--rows INT] [--param NAME=VALUE ...] [--profile TEXT]
FrictionSim2D analyse export NAME DATASET OUT.csv [--param NAME=VALUE ...] [--profile TEXT]
```

- `run` prints the row count, the dropped count and the result's uuid.
- `show` prints the stored result's extras and its first `--rows` (default 10) rows.
- `export` writes the stored result's CSV.
- `show` and `export` read the newest result under exactly the `--param`s given, the
  others at their defaults (hashed as `run` hashes them). With no `--param`, that is
  the default-params result. When there is none but results under other params are
  stored, they refuse and name those params.

### Analyses as plot sources

- A table field takes `TableSource(analysis=<name>, dataset=<dataset>)` (in JSON
  `{"analysis": ..., "dataset": ..., "version": <optional>, "params": {<optional>}}`;
  default version: the registered one), or a CSV path.
- `aiida://analysis/<name>/<dataset>[?k=v&k2=v2]` in `data_dirs` reads a summary-shaped
  result (`period_summary`) as that label's summary rows.
- Both select by params as `analyse show` does: the result under exactly those params
  (none: the defaults); results stored only under other params are refused, named.
- The plotter never computes an analysis; see
  [Stored analyses as plot sources](#stored-analyses-as-plot-sources).

---

## Plot documents

A minimal document:

```json
{
  "data_dirs": ["/data/campaign_a"],
  "labels": ["campaign A"],
  "plots": [
    {
      "plot_type": "summary",
      "x_axis": "force",
      "y_axis": "lf",
      "filter_size": "100x100y",
      "filename": "lf_vs_load.png",
      "add_linear_fit": true
    }
  ]
}
```

`data_dirs` holds directories containing an `outputs/`, and must name at least one — a
document with none can draw nothing. `labels` has one entry per `data_dir`. Start one
with:

```bash
FrictionSim2D postprocess plot --init plots.json --data-dir /data/campaign_a
```

### `db://` data directories (removed)

`db://` URIs were removed with the central database; a `data_dir` that starts with
`db://` raises `PlotDataError`. Read the local AiiDA store with
`aiida://dataset/<name>` (a dataset made by `FrictionSim2D aiida dataset`, see
[aiida.md](aiida.md#6-postprocessing-datasets-and-queries)), or export a tree and
use that directory:

```bash
FrictionSim2D aiida export-results <label> <dir>
```

### Stored analyses as plot sources

- `aiida://analysis/<name>/<dataset>` in `data_dirs`: a summary-shaped analysis result
  (`period_summary`) becomes that label's summary rows, as `aiida://dataset/` does.
- A table field (`ranking_file`, `traces_file`, ...) takes a path or
  `{"analysis": <name>, "dataset": <dataset>, "version": <optional>, "params": {...}}`.
  URI query values are text, parsed by the analysis' params model (a dict or list as
  JSON text).
- `stickslip_period_histogram` and `contact_cof_scatter` read the load from `load_nN`, or
  from `force` as `stick_slip` and `contact` name it. Those analyses hold every layer:
  `filter_layer` keeps one.
- `sheet_profile` and `integrity_panel` take `dataset` + `material` + `layer` instead
  of `run_dir`; the integrity panel draws failed points too.
- The plotter never computes an analysis: a missing result raises naming
  `FrictionSim2D analyse run <name> <dataset>`.

### External series from a dataset

An `external_series` entry (`summary`, `cof_histogram`) reads either a JSON `file` (+
`path`) or a `dataset`: a `data_dirs` label read from the store. With `dataset`, the
series is the mean of `x_key` and `y_key` over that label's summary rows at each load,
by increasing load; none of the plot's filters apply. Example (the MLIP bars of the
publication histogram):

```json
{"dataset": "deepmd", "x_key": "nf", "y_key": "lf_sx_sy", "label": "DeePMD + D3", "solid": true}
```

---

## Plot schema

Unknown keys are **errors**, not ignored. `postprocess plot --validate` checks a document
without rendering it, and `postprocess plot --schema` regenerates the tables below
straight from the models, so this section cannot drift from what the CLI validates
against.

## Document

| field | type | meaning |
|---|---|---|
| `data_dirs` | `List[str]` | Directories holding outputs/, aiida://dataset/<name> or aiida://analysis/<name>/<dataset> URIs |
| `labels` | `List[str]` | One per data_dir |
| `dataset_display_labels` | `Dict[str, str]` |  |
| `series_color_map` | `Dict[str, Optional[str]]` |  |
| `comment` | `Optional[str]` | Documentation only |
| `plots` | `List[Annotated[Union[frictionsim2d.postprocessing.plot_config.SummaryPlot, frictionsim2d.postprocessing.plot_config.TimeseriesPlot, frictionsim2d.postprocessing.plot_config.StickSlipPlot, frictionsim2d.postprocessing.plot_config.ScatterComparisonPlot, frictionsim2d.postprocessing.plot_config.RankFrictionPlot, frictionsim2d.postprocessing.plot_config.CorrelationPlot, frictionsim2d.postprocessing.plot_config.CofHistogramPlot, frictionsim2d.postprocessing.plot_config.StickslipPeriodHistogramPlot, frictionsim2d.postprocessing.plot_config.ContactCofScatterPlot, frictionsim2d.postprocessing.plot_config.SheetProfilePlot, frictionsim2d.postprocessing.plot_config.IntegrityPanelPlot], FieldInfo(annotation=NoneType, required=True, discriminator='plot_type')]]` |  |

## `summary`

| field | type | meaning |
|---|---|---|
| `plot_type` | `Literal['summary']` |  |
| `source` | `Optional[str]` | Documentation only: the config or script this plot was first made with |
| `filename` | `Optional[str]` |  |
| `title` | `Optional[str]` |  |
| `datasets` | `Optional[List[str]]` |  |
| `use_dataset_colors` | `Optional[bool]` |  |
| `x_axis` | `Optional[str]` |  |
| `y_axis` | `Optional[str]` |  |
| `x_label` | `Optional[str]` | Overrides label_for() |
| `y_label` | `Optional[str]` | Overrides label_for() |
| `x_limits` | `Optional[List[Optional[float]]]` |  |
| `y_limits` | `Optional[List[Optional[float]]]` |  |
| `legend_location` | `Optional[str]` |  |
| `legend_outside` | `Union[bool, Literal['left', 'top']]` | True: right of the axes; 'left' or 'top' for those sides |
| `legend_top_align` | `Optional[Literal['right']]` | With legend_outside='top': right-align instead of centring |
| `legend_ncol` | `Optional[int]` |  |
| `legend_outside_left_anchor` | `Optional[float]` | x anchor of a 'left' legend in axes units (default -0.16) |
| `legend_order` | `Optional[List[str]]` | These labels first, in this order; the rest keep theirs |
| `figure_size` | `Optional[List[float]]` |  |
| `filter_size` | `Optional[str]` |  |
| `filter_layer` | `Optional[int]` |  |
| `filter_speed` | `Optional[float]` |  |
| `filter_tip_radius` | `Optional[float]` |  |
| `filter_materials` | `Optional[List[str]]` |  |
| `filter_forces` | `Optional[List[float]]` |  |
| `filter_pressures` | `Optional[List[float]]` |  |
| `pressures` | `Optional[List[float]]` |  |
| `angle` | `Optional[float]` |  |
| `force` | `Optional[float]` |  |
| `filter_force_range` | `Optional[List[float]]` |  |
| `filter_nf_range` | `Optional[List[float]]` |  |
| `lf_source` | `Optional[Literal['as_was', 'sx_sy', 'fx_fy']]` |  |
| `family` | `Optional[str]` | Keep only runs of this family (new in v2) |
| `tier` | `Literal['window', 'raw', 'peratom']` | Which sampling tier to plot (new in v2) |
| `use_scientific_notation` | `Optional[bool]` |  |
| `average_first_n_timesteps` | `Optional[int]` |  |
| `average_timestep_range` | `Optional[List[int]]` |  |
| `add_linear_fit` | `Optional[bool]` |  |
| `axes_margins_in` | `Optional[Dict[Literal['left', 'right', 'bottom'], float]]` | Space (in) between that side of the axes and the image edge, so panels line up; never clips |
| `append_slope_to_legend` | `Optional[bool]` |  |
| `clip_fit_to_data` | `Optional[bool]` |  |
| `extend_fit_to_origin` | `Optional[bool]` |  |
| `external_series` | `Optional[List[frictionsim2d.postprocessing.plot_config.ExternalSeries]]` |  |
| `external_outline_width` | `Optional[float]` |  |
| `fit_constraint` | `Optional[Literal['first_point', 'origin', 'none']]` |  |
| `fit_only` | `Optional[bool]` |  |
| `fit_x_range` | `Optional[List[float]]` |  |
| `label_prefix` | `Optional[str]` |  |
| `label_suffix` | `Optional[str]` |  |
| `legend_figure_size` | `Optional[List[float]]` |  |
| `legend_filename` | `Optional[str]` |  |
| `legend_separate_png` | `Optional[bool]` |  |
| `marker_by_potential_type` | `Optional[bool]` |  |
| `plot_by` | `Optional[str]` |  |
| `plot_style` | `Optional[str]` |  |
| `potential_markers` | `Optional[Dict[str, str]]` |  |
| `shift_external_series_to_origin` | `Optional[bool]` |  |
| `shift_series_to_origin` | `Optional[bool]` |  |
| `show_dataset_in_legend` | `Optional[bool]` |  |
| `show_error_bands` | `Optional[bool]` |  |
| `slope_decimals` | `Optional[int]` |  |
| `slope_fit_constraint` | `Optional[Literal['first_point', 'origin', 'none']]` |  |
| `slope_label` | `Optional[str]` |  |
| `use_visible_xrange_for_fit` | `Optional[bool]` |  |

## `timeseries`

| field | type | meaning |
|---|---|---|
| `plot_type` | `Literal['timeseries']` |  |
| `source` | `Optional[str]` | Documentation only: the config or script this plot was first made with |
| `filename` | `Optional[str]` |  |
| `title` | `Optional[str]` |  |
| `datasets` | `Optional[List[str]]` |  |
| `use_dataset_colors` | `Optional[bool]` |  |
| `x_axis` | `Optional[str]` |  |
| `y_axis` | `Optional[str]` |  |
| `x_label` | `Optional[str]` | Overrides label_for() |
| `y_label` | `Optional[str]` | Overrides label_for() |
| `x_limits` | `Optional[List[Optional[float]]]` |  |
| `y_limits` | `Optional[List[Optional[float]]]` |  |
| `legend_location` | `Optional[str]` |  |
| `legend_outside` | `Union[bool, Literal['left', 'top']]` | True: right of the axes; 'left' or 'top' for those sides |
| `legend_top_align` | `Optional[Literal['right']]` | With legend_outside='top': right-align instead of centring |
| `legend_ncol` | `Optional[int]` | Columns of a 'top' legend (default 5) |
| `legend_outside_left_anchor` | `Optional[float]` | x anchor of a 'left' legend in axes units (default -0.16) |
| `legend_order` | `Optional[List[str]]` | These labels first, in this order; the rest keep theirs |
| `figure_size` | `Optional[List[float]]` |  |
| `filter_size` | `Optional[str]` |  |
| `filter_layer` | `Optional[int]` |  |
| `filter_speed` | `Optional[float]` |  |
| `filter_tip_radius` | `Optional[float]` |  |
| `filter_materials` | `Optional[List[str]]` |  |
| `filter_forces` | `Optional[List[float]]` |  |
| `filter_pressures` | `Optional[List[float]]` |  |
| `pressures` | `Optional[List[float]]` |  |
| `angle` | `Optional[float]` |  |
| `force` | `Optional[float]` |  |
| `filter_force_range` | `Optional[List[float]]` |  |
| `filter_nf_range` | `Optional[List[float]]` |  |
| `lf_source` | `Optional[Literal['as_was', 'sx_sy', 'fx_fy']]` |  |
| `family` | `Optional[str]` | Keep only runs of this family (new in v2) |
| `tier` | `Literal['window', 'raw', 'peratom']` | Which sampling tier to plot (new in v2) |
| `use_scientific_notation` | `Optional[bool]` |  |
| `average_first_n_timesteps` | `Optional[int]` |  |
| `average_timestep_range` | `Optional[List[int]]` |  |
| `plot_all_forces` | `Optional[bool]` |  |
| `primary_line_alpha` | `Optional[float]` |  |
| `primary_line_style` | `Optional[str]` |  |
| `primary_line_width` | `Optional[float]` |  |
| `run_label_mode` | `Optional[str]` |  |
| `secondary_dash_pattern` | `Optional[List[float]]` |  |
| `secondary_line_alpha` | `Optional[float]` |  |
| `secondary_line_style` | `Optional[str]` |  |
| `secondary_line_width` | `Optional[float]` |  |
| `secondary_y_axis` | `Optional[str]` |  |
| `secondary_y_label` | `Optional[str]` |  |
| `secondary_y_scale` | `Optional[float]` |  |
| `shift_time_to_zero` | `Optional[bool]` |  |
| `time_scale` | `Optional[float]` |  |
| `time_unit` | `Optional[str]` |  |
| `use_run_labels` | `Optional[bool]` |  |

## `stick_slip_analysis`

| field | type | meaning |
|---|---|---|
| `plot_type` | `Literal['stick_slip_analysis']` |  |
| `source` | `Optional[str]` | Documentation only: the config or script this plot was first made with |
| `filename` | `Optional[str]` |  |
| `title` | `Optional[str]` |  |
| `datasets` | `Optional[List[str]]` |  |
| `use_dataset_colors` | `Optional[bool]` |  |
| `x_axis` | `Optional[str]` |  |
| `y_axis` | `Optional[str]` |  |
| `x_label` | `Optional[str]` | Overrides label_for() |
| `y_label` | `Optional[str]` | Overrides label_for() |
| `x_limits` | `Optional[List[Optional[float]]]` |  |
| `y_limits` | `Optional[List[Optional[float]]]` |  |
| `legend_location` | `Optional[str]` |  |
| `legend_outside` | `Union[bool, Literal['left', 'top']]` | True: right of the axes; 'left' or 'top' for those sides |
| `legend_top_align` | `Optional[Literal['right']]` | With legend_outside='top': right-align instead of centring |
| `legend_ncol` | `Optional[int]` | Columns of a 'top' legend (default 5) |
| `legend_outside_left_anchor` | `Optional[float]` | x anchor of a 'left' legend in axes units (default -0.16) |
| `legend_order` | `Optional[List[str]]` | These labels first, in this order; the rest keep theirs |
| `figure_size` | `Optional[List[float]]` |  |
| `filter_size` | `Optional[str]` |  |
| `filter_layer` | `Optional[int]` |  |
| `filter_speed` | `Optional[float]` |  |
| `filter_tip_radius` | `Optional[float]` |  |
| `filter_materials` | `Optional[List[str]]` |  |
| `filter_forces` | `Optional[List[float]]` |  |
| `filter_pressures` | `Optional[List[float]]` |  |
| `pressures` | `Optional[List[float]]` |  |
| `angle` | `Optional[float]` |  |
| `force` | `Optional[float]` |  |
| `filter_force_range` | `Optional[List[float]]` |  |
| `filter_nf_range` | `Optional[List[float]]` |  |
| `lf_source` | `Optional[Literal['as_was', 'sx_sy', 'fx_fy']]` |  |
| `family` | `Optional[str]` | Keep only runs of this family (new in v2) |
| `tier` | `Literal['window', 'raw', 'peratom']` | Which sampling tier to plot (new in v2) |
| `use_scientific_notation` | `Optional[bool]` |  |
| `average_first_n_timesteps` | `Optional[int]` |  |
| `average_timestep_range` | `Optional[List[int]]` |  |
| `shift_time_to_zero` | `Optional[bool]` | Start the time axis at 0 (default) or keep the LAMMPS clock |
| `apply_hann_window` | `Optional[bool]` |  |
| `bottom_ylabel` | `Optional[str]` |  |
| `displacement_axis` | `Optional[str]` |  |
| `displacement_scale` | `Optional[float]` |  |
| `displacement_unit` | `Optional[str]` |  |
| `display_periods` | `Optional[float]` |  |
| `display_time_start` | `Optional[float]` |  |
| `display_time_window` | `Optional[List[float]]` |  |
| `extrema_min_distance_points` | `Optional[int]` |  |
| `extrema_min_prominence` | `Optional[float]` |  |
| `extrema_smooth_window` | `Optional[int]` |  |
| `extrema_use_prominence` | `Optional[bool]` |  |
| `fft_peak_min_distance_points` | `Optional[int]` |  |
| `fft_peak_min_prominence` | `Optional[float]` |  |
| `fft_peak_rank` | `Optional[int]` |  |
| `fft_peaks_to_report` | `Optional[int]` |  |
| `freq_max` | `Optional[float]` |  |
| `freq_min` | `Optional[float]` |  |
| `freq_plot_max` | `Optional[float]` |  |
| `freq_plot_min` | `Optional[float]` |  |
| `freq_xlabel` | `Optional[str]` |  |
| `max_intervals_to_draw` | `Optional[int]` |  |
| `results_filename` | `Optional[str]` |  |
| `run_index` | `Optional[int]` |  |
| `secondary_y_axis` | `Optional[str]` |  |
| `secondary_y_scale` | `Optional[float]` |  |
| `show_peak_trough_intervals` | `Optional[bool]` |  |
| `single_panel_figure_size` | `Optional[List[float]]` |  |
| `smooth_for_extrema` | `Optional[bool]` |  |
| `spectrum_filename` | `Optional[str]` |  |
| `time_scale` | `Optional[float]` |  |
| `time_unit` | `Optional[str]` |  |
| `top_ylabel` | `Optional[str]` |  |

## `scatter_comparison`

| field | type | meaning |
|---|---|---|
| `plot_type` | `Literal['scatter_comparison']` |  |
| `source` | `Optional[str]` | Documentation only: the config or script this plot was first made with |
| `filename` | `Optional[str]` |  |
| `title` | `Optional[str]` |  |
| `datasets` | `Optional[List[str]]` |  |
| `use_dataset_colors` | `Optional[bool]` |  |
| `x_axis` | `Optional[str]` |  |
| `y_axis` | `Optional[str]` |  |
| `x_label` | `Optional[str]` | Overrides label_for() |
| `y_label` | `Optional[str]` | Overrides label_for() |
| `x_limits` | `Optional[List[Optional[float]]]` |  |
| `y_limits` | `Optional[List[Optional[float]]]` |  |
| `legend_location` | `Optional[str]` |  |
| `legend_outside` | `Union[bool, Literal['left', 'top']]` | True: right of the axes; 'left' or 'top' for those sides |
| `legend_top_align` | `Optional[Literal['right']]` | With legend_outside='top': right-align instead of centring |
| `legend_ncol` | `Optional[int]` | Columns of a 'top' legend (default 5) |
| `legend_outside_left_anchor` | `Optional[float]` | x anchor of a 'left' legend in axes units (default -0.16) |
| `legend_order` | `Optional[List[str]]` | These labels first, in this order; the rest keep theirs |
| `figure_size` | `Optional[List[float]]` |  |
| `filter_size` | `Optional[str]` |  |
| `filter_layer` | `Optional[int]` |  |
| `filter_speed` | `Optional[float]` |  |
| `filter_tip_radius` | `Optional[float]` |  |
| `filter_materials` | `Optional[List[str]]` |  |
| `filter_forces` | `Optional[List[float]]` |  |
| `filter_pressures` | `Optional[List[float]]` |  |
| `pressures` | `Optional[List[float]]` |  |
| `angle` | `Optional[float]` |  |
| `force` | `Optional[float]` |  |
| `filter_force_range` | `Optional[List[float]]` |  |
| `filter_nf_range` | `Optional[List[float]]` |  |
| `lf_source` | `Optional[Literal['as_was', 'sx_sy', 'fx_fy']]` |  |
| `family` | `Optional[str]` | Keep only runs of this family (new in v2) |
| `tier` | `Literal['window', 'raw', 'peratom']` | Which sampling tier to plot (new in v2) |
| `use_scientific_notation` | `Optional[bool]` |  |
| `average_first_n_timesteps` | `Optional[int]` |  |
| `average_timestep_range` | `Optional[List[int]]` |  |
| `add_linear_fit` | `Optional[bool]` |  |
| `color_by_material_class` | `Optional[bool]` |  |
| `iterative_outlier_removal` | `Optional[bool]` |  |
| `point_label_fontsize` | `Optional[float]` |  |
| `show_error_bars` | `Optional[bool]` |  |
| `show_identity_line` | `Optional[bool]` |  |
| `show_point_labels` | `Optional[bool]` |  |
| `show_r_squared` | `Optional[bool]` |  |
| `x_source` | `Optional[frictionsim2d.postprocessing.plot_config.SourceSpec]` |  |
| `y_source` | `Optional[frictionsim2d.postprocessing.plot_config.SourceSpec]` |  |

## `rank_friction`

| field | type | meaning |
|---|---|---|
| `plot_type` | `Literal['rank_friction']` |  |
| `source` | `Optional[str]` | Documentation only: the config or script this plot was first made with |
| `filename` | `Optional[str]` |  |
| `title` | `Optional[str]` |  |
| `datasets` | `Optional[List[str]]` |  |
| `use_dataset_colors` | `Optional[bool]` |  |
| `x_axis` | `Optional[str]` |  |
| `y_axis` | `Optional[str]` |  |
| `x_label` | `Optional[str]` | Overrides label_for() |
| `y_label` | `Optional[str]` | Overrides label_for() |
| `x_limits` | `Optional[List[Optional[float]]]` |  |
| `y_limits` | `Optional[List[Optional[float]]]` |  |
| `legend_location` | `Optional[str]` |  |
| `legend_outside` | `Union[bool, Literal['left', 'top']]` | True: right of the axes; 'left' or 'top' for those sides |
| `legend_top_align` | `Optional[Literal['right']]` | With legend_outside='top': right-align instead of centring |
| `legend_ncol` | `Optional[int]` | Columns of a 'top' legend (default 5) |
| `legend_outside_left_anchor` | `Optional[float]` | x anchor of a 'left' legend in axes units (default -0.16) |
| `legend_order` | `Optional[List[str]]` | These labels first, in this order; the rest keep theirs |
| `figure_size` | `Optional[List[float]]` |  |
| `filter_size` | `Optional[str]` |  |
| `filter_layer` | `Optional[int]` |  |
| `filter_speed` | `Optional[float]` |  |
| `filter_tip_radius` | `Optional[float]` |  |
| `filter_materials` | `Optional[List[str]]` |  |
| `filter_forces` | `Optional[List[float]]` |  |
| `filter_pressures` | `Optional[List[float]]` |  |
| `pressures` | `Optional[List[float]]` |  |
| `angle` | `Optional[float]` |  |
| `force` | `Optional[float]` |  |
| `filter_force_range` | `Optional[List[float]]` |  |
| `filter_nf_range` | `Optional[List[float]]` |  |
| `lf_source` | `Optional[Literal['as_was', 'sx_sy', 'fx_fy']]` |  |
| `family` | `Optional[str]` | Keep only runs of this family (new in v2) |
| `tier` | `Literal['window', 'raw', 'peratom']` | Which sampling tier to plot (new in v2) |
| `use_scientific_notation` | `Optional[bool]` |  |
| `average_first_n_timesteps` | `Optional[int]` |  |
| `average_timestep_range` | `Optional[List[int]]` |  |
| `aggregate_over_force_range` | `Optional[bool]` |  |
| `fit_x_range` | `Optional[List[float]]` |  |
| `force_range` | `Optional[List[float]]` |  |
| `rank_by` | `Optional[str]` |  |
| `rank_order` | `Optional[Literal['ascending', 'descending']]` |  |

## `correlation`

| field | type | meaning |
|---|---|---|
| `plot_type` | `Literal['correlation']` |  |
| `source` | `Optional[str]` | Documentation only: the config or script this plot was first made with |
| `filename` | `Optional[str]` |  |
| `title` | `Optional[str]` |  |
| `datasets` | `Optional[List[str]]` |  |
| `use_dataset_colors` | `Optional[bool]` |  |
| `x_axis` | `Optional[str]` |  |
| `y_axis` | `Optional[str]` |  |
| `x_label` | `Optional[str]` | Overrides label_for() |
| `y_label` | `Optional[str]` | Overrides label_for() |
| `x_limits` | `Optional[List[Optional[float]]]` |  |
| `y_limits` | `Optional[List[Optional[float]]]` |  |
| `legend_location` | `Optional[str]` |  |
| `legend_outside` | `Union[bool, Literal['left', 'top']]` | True: right of the axes; 'left' or 'top' for those sides |
| `legend_top_align` | `Optional[Literal['right']]` | With legend_outside='top': right-align instead of centring |
| `legend_ncol` | `Optional[int]` | Columns of a 'top' legend (default 5) |
| `legend_outside_left_anchor` | `Optional[float]` | x anchor of a 'left' legend in axes units (default -0.16) |
| `legend_order` | `Optional[List[str]]` | These labels first, in this order; the rest keep theirs |
| `figure_size` | `Optional[List[float]]` |  |
| `filter_size` | `Optional[str]` |  |
| `filter_layer` | `Optional[int]` |  |
| `filter_speed` | `Optional[float]` |  |
| `filter_tip_radius` | `Optional[float]` |  |
| `filter_materials` | `Optional[List[str]]` |  |
| `filter_forces` | `Optional[List[float]]` |  |
| `filter_pressures` | `Optional[List[float]]` |  |
| `pressures` | `Optional[List[float]]` |  |
| `angle` | `Optional[float]` |  |
| `force` | `Optional[float]` |  |
| `filter_force_range` | `Optional[List[float]]` |  |
| `filter_nf_range` | `Optional[List[float]]` |  |
| `lf_source` | `Optional[Literal['as_was', 'sx_sy', 'fx_fy']]` |  |
| `family` | `Optional[str]` | Keep only runs of this family (new in v2) |
| `tier` | `Literal['window', 'raw', 'peratom']` | Which sampling tier to plot (new in v2) |
| `use_scientific_notation` | `Optional[bool]` |  |
| `average_first_n_timesteps` | `Optional[int]` |  |
| `average_timestep_range` | `Optional[List[int]]` |  |
| `correlate_by` | `Optional[Literal['size', 'force', 'pairwise']]` |  |
| `correlation_force` | `Optional[float]` |  |
| `filename_prefix` | `Optional[str]` |  |
| `sizes_to_compare` | `Optional[List[str]]` |  |

## `cof_histogram`

| field | type | meaning |
|---|---|---|
| `plot_type` | `Literal['cof_histogram']` |  |
| `source` | `Optional[str]` | Documentation only: the config or script this plot was first made with |
| `filename` | `Optional[str]` |  |
| `title` | `Optional[str]` |  |
| `datasets` | `Optional[List[str]]` |  |
| `use_dataset_colors` | `Optional[bool]` |  |
| `x_axis` | `Optional[str]` |  |
| `y_axis` | `Optional[str]` |  |
| `x_label` | `Optional[str]` | Overrides label_for() |
| `y_label` | `Optional[str]` | Overrides label_for() |
| `x_limits` | `Optional[List[Optional[float]]]` |  |
| `y_limits` | `Optional[List[Optional[float]]]` |  |
| `legend_location` | `Optional[str]` |  |
| `legend_outside` | `Union[bool, Literal['left', 'top']]` | True: right of the axes; 'left' or 'top' for those sides |
| `legend_top_align` | `Optional[Literal['right']]` | With legend_outside='top': right-align instead of centring |
| `legend_ncol` | `Optional[int]` | Columns of a 'top' legend (default 5) |
| `legend_outside_left_anchor` | `Optional[float]` | x anchor of a 'left' legend in axes units (default -0.16) |
| `legend_order` | `Optional[List[str]]` | These labels first, in this order; the rest keep theirs |
| `figure_size` | `Optional[List[float]]` |  |
| `filter_size` | `Optional[str]` |  |
| `filter_layer` | `Optional[int]` |  |
| `filter_speed` | `Optional[float]` |  |
| `filter_tip_radius` | `Optional[float]` |  |
| `filter_materials` | `Optional[List[str]]` |  |
| `filter_forces` | `Optional[List[float]]` |  |
| `filter_pressures` | `Optional[List[float]]` |  |
| `pressures` | `Optional[List[float]]` |  |
| `angle` | `Optional[float]` |  |
| `force` | `Optional[float]` |  |
| `filter_force_range` | `Optional[List[float]]` |  |
| `filter_nf_range` | `Optional[List[float]]` |  |
| `lf_source` | `Optional[Literal['as_was', 'sx_sy', 'fx_fy']]` |  |
| `family` | `Optional[str]` | Keep only runs of this family (new in v2) |
| `tier` | `Literal['window', 'raw', 'peratom']` | Which sampling tier to plot (new in v2) |
| `use_scientific_notation` | `Optional[bool]` |  |
| `average_first_n_timesteps` | `Optional[int]` |  |
| `average_timestep_range` | `Optional[List[int]]` |  |
| `annotate_bars` | `Optional[bool]` |  |
| `bar_width` | `Optional[float]` |  |
| `external_series` | `Optional[List[frictionsim2d.postprocessing.plot_config.ExternalSeries]]` |  |
| `fit_x_range` | `Optional[List[float]]` |  |
| `slope_decimals` | `Optional[int]` |  |
| `slope_fit_constraint` | `Optional[Literal['first_point', 'origin', 'none']]` |  |
| `x_label_rotation` | `Optional[int]` |  |
| `external_outline_width` | `Optional[float]` | Ring width (pt) of a hollow external-series bar (default 10) |
| `show_adhesion_pairs` | `Optional[bool]` | Two bars per series: mean LF/NF (with adhesion) and the slope |
| `ratio_x_range` | `Optional[List[float]]` | Load window of the mean LF/NF (default: the fit window) |
| `with_adhesion_label` | `Optional[str]` |  |
| `without_adhesion_label` | `Optional[str]` |  |
| `without_adhesion_legend_first` | `Optional[bool]` |  |
| `no_adhesion_hatch` | `Optional[str]` | Hatch of the slope bar (default '///') |

## `stickslip_period_histogram`

| field | type | meaning |
|---|---|---|
| `plot_type` | `Literal['stickslip_period_histogram']` |  |
| `source` | `Optional[str]` | Documentation only: the config or script this plot was first made with |
| `filename` | `Optional[str]` |  |
| `title` | `Optional[str]` |  |
| `datasets` | `Optional[List[str]]` |  |
| `use_dataset_colors` | `Optional[bool]` |  |
| `x_axis` | `Optional[str]` |  |
| `y_axis` | `Optional[str]` |  |
| `x_label` | `Optional[str]` | Overrides label_for() |
| `y_label` | `Optional[str]` | Overrides label_for() |
| `x_limits` | `Optional[List[Optional[float]]]` |  |
| `y_limits` | `Optional[List[Optional[float]]]` |  |
| `legend_location` | `Optional[str]` |  |
| `legend_outside` | `Union[bool, Literal['left', 'top']]` | True: right of the axes; 'left' or 'top' for those sides |
| `legend_top_align` | `Optional[Literal['right']]` | With legend_outside='top': right-align instead of centring |
| `legend_ncol` | `Optional[int]` | Columns of a 'top' legend (default 5) |
| `legend_outside_left_anchor` | `Optional[float]` | x anchor of a 'left' legend in axes units (default -0.16) |
| `legend_order` | `Optional[List[str]]` | These labels first, in this order; the rest keep theirs |
| `figure_size` | `Optional[List[float]]` |  |
| `filter_size` | `Optional[str]` |  |
| `filter_layer` | `Optional[int]` |  |
| `filter_speed` | `Optional[float]` |  |
| `filter_tip_radius` | `Optional[float]` |  |
| `filter_materials` | `Optional[List[str]]` |  |
| `filter_forces` | `Optional[List[float]]` |  |
| `filter_pressures` | `Optional[List[float]]` |  |
| `pressures` | `Optional[List[float]]` |  |
| `angle` | `Optional[float]` |  |
| `force` | `Optional[float]` |  |
| `filter_force_range` | `Optional[List[float]]` |  |
| `filter_nf_range` | `Optional[List[float]]` |  |
| `lf_source` | `Optional[Literal['as_was', 'sx_sy', 'fx_fy']]` |  |
| `family` | `Optional[str]` | Keep only runs of this family (new in v2) |
| `tier` | `Literal['window', 'raw', 'peratom']` | Which sampling tier to plot (new in v2) |
| `use_scientific_notation` | `Optional[bool]` |  |
| `average_first_n_timesteps` | `Optional[int]` |  |
| `average_timestep_range` | `Optional[List[int]]` |  |
| `loads` | `Optional[List[float]]` | Loads (nN), one colour each |
| `load_colors` | `Optional[List[str]]` |  |
| `line_width` | `Optional[float]` |  |
| `stickslip_file` | `Optional[frictionsim2d.postprocessing.plot_config.TableSource]` | material, load_nN (or force), lf_period_A, a_x_A, lf_period_clear |
| `ranking_file` | `Optional[frictionsim2d.postprocessing.plot_config.TableSource]` | material, eligible |
| `bins_arange` | `Optional[List[float]]` | [start, stop, step] |
| `reference_line` | `Optional[float]` | x of the dashed line; null for none |

## `contact_cof_scatter`

| field | type | meaning |
|---|---|---|
| `plot_type` | `Literal['contact_cof_scatter']` |  |
| `source` | `Optional[str]` | Documentation only: the config or script this plot was first made with |
| `filename` | `Optional[str]` |  |
| `title` | `Optional[str]` |  |
| `datasets` | `Optional[List[str]]` |  |
| `use_dataset_colors` | `Optional[bool]` |  |
| `x_axis` | `Optional[str]` |  |
| `y_axis` | `Optional[str]` |  |
| `x_label` | `Optional[str]` | Overrides label_for() |
| `y_label` | `Optional[str]` | Overrides label_for() |
| `x_limits` | `Optional[List[Optional[float]]]` |  |
| `y_limits` | `Optional[List[Optional[float]]]` |  |
| `legend_location` | `Optional[str]` |  |
| `legend_outside` | `Union[bool, Literal['left', 'top']]` | True: right of the axes; 'left' or 'top' for those sides |
| `legend_top_align` | `Optional[Literal['right']]` | With legend_outside='top': right-align instead of centring |
| `legend_ncol` | `Optional[int]` | Columns of a 'top' legend (default 5) |
| `legend_outside_left_anchor` | `Optional[float]` | x anchor of a 'left' legend in axes units (default -0.16) |
| `legend_order` | `Optional[List[str]]` | These labels first, in this order; the rest keep theirs |
| `figure_size` | `Optional[List[float]]` |  |
| `filter_size` | `Optional[str]` |  |
| `filter_layer` | `Optional[int]` |  |
| `filter_speed` | `Optional[float]` |  |
| `filter_tip_radius` | `Optional[float]` |  |
| `filter_materials` | `Optional[List[str]]` |  |
| `filter_forces` | `Optional[List[float]]` |  |
| `filter_pressures` | `Optional[List[float]]` |  |
| `pressures` | `Optional[List[float]]` |  |
| `angle` | `Optional[float]` |  |
| `force` | `Optional[float]` |  |
| `filter_force_range` | `Optional[List[float]]` |  |
| `filter_nf_range` | `Optional[List[float]]` |  |
| `lf_source` | `Optional[Literal['as_was', 'sx_sy', 'fx_fy']]` |  |
| `family` | `Optional[str]` | Keep only runs of this family (new in v2) |
| `tier` | `Literal['window', 'raw', 'peratom']` | Which sampling tier to plot (new in v2) |
| `use_scientific_notation` | `Optional[bool]` |  |
| `average_first_n_timesteps` | `Optional[int]` |  |
| `average_timestep_range` | `Optional[List[int]]` |  |
| `loads` | `Optional[List[float]]` | Loads (nN), one colour each |
| `load_colors` | `Optional[List[str]]` |  |
| `line_width` | `Optional[float]` |  |
| `contact_file` | `Optional[frictionsim2d.postprocessing.plot_config.TableSource]` | material, load_nN (or force), state, <x_column> |
| `ranking_file` | `Optional[frictionsim2d.postprocessing.plot_config.TableSource]` | material, eligible, cof_<L>nN |
| `x_column` | `Optional[str]` | Contact column on x (required) |
| `state` | `Optional[str]` | 'load' (default) or 'final' |
| `marker_scale` | `Optional[float]` | times markers.size**2 (0.8) |
| `alpha` | `Optional[float]` |  |
| `y_scale` | `Optional[Literal['log', 'linear']]` |  |
| `y_ticks` | `Optional[List[float]]` |  |

## `sheet_profile`

| field | type | meaning |
|---|---|---|
| `plot_type` | `Literal['sheet_profile']` |  |
| `source` | `Optional[str]` | Documentation only: the config or script this plot was first made with |
| `filename` | `Optional[str]` |  |
| `title` | `Optional[str]` |  |
| `datasets` | `Optional[List[str]]` |  |
| `use_dataset_colors` | `Optional[bool]` |  |
| `x_axis` | `Optional[str]` |  |
| `y_axis` | `Optional[str]` |  |
| `x_label` | `Optional[str]` | Overrides label_for() |
| `y_label` | `Optional[str]` | Overrides label_for() |
| `x_limits` | `Optional[List[Optional[float]]]` |  |
| `y_limits` | `Optional[List[Optional[float]]]` |  |
| `legend_location` | `Optional[str]` |  |
| `legend_outside` | `Union[bool, Literal['left', 'top']]` | True: right of the axes; 'left' or 'top' for those sides |
| `legend_top_align` | `Optional[Literal['right']]` | With legend_outside='top': right-align instead of centring |
| `legend_ncol` | `Optional[int]` | Columns of a 'top' legend (default 5) |
| `legend_outside_left_anchor` | `Optional[float]` | x anchor of a 'left' legend in axes units (default -0.16) |
| `legend_order` | `Optional[List[str]]` | These labels first, in this order; the rest keep theirs |
| `figure_size` | `Optional[List[float]]` |  |
| `filter_size` | `Optional[str]` |  |
| `filter_layer` | `Optional[int]` |  |
| `filter_speed` | `Optional[float]` |  |
| `filter_tip_radius` | `Optional[float]` |  |
| `filter_materials` | `Optional[List[str]]` |  |
| `filter_forces` | `Optional[List[float]]` |  |
| `filter_pressures` | `Optional[List[float]]` |  |
| `pressures` | `Optional[List[float]]` |  |
| `angle` | `Optional[float]` |  |
| `force` | `Optional[float]` |  |
| `filter_force_range` | `Optional[List[float]]` |  |
| `filter_nf_range` | `Optional[List[float]]` |  |
| `lf_source` | `Optional[Literal['as_was', 'sx_sy', 'fx_fy']]` |  |
| `family` | `Optional[str]` | Keep only runs of this family (new in v2) |
| `tier` | `Literal['window', 'raw', 'peratom']` | Which sampling tier to plot (new in v2) |
| `use_scientific_notation` | `Optional[bool]` |  |
| `average_first_n_timesteps` | `Optional[int]` |  |
| `average_timestep_range` | `Optional[List[int]]` |  |
| `loads` | `Optional[List[float]]` | Loads (nN), one colour each |
| `load_colors` | `Optional[List[str]]` |  |
| `line_width` | `Optional[float]` |  |
| `run_dir` | `Optional[str]` | AFM run directory holding <layer_dir>/ (or give dataset) |
| `layer_dir` | `Optional[str]` |  |
| `dataset` | `Optional[str]` | Stored dataset holding the run (instead of run_dir) |
| `material` | `Optional[str]` | Material, with dataset |
| `layer` | `Optional[int]` | Layers, with dataset (default 2) |
| `tip_types` | `Optional[List[int]]` |  |
| `tip_apex_depth` | `Optional[float]` |  |
| `far_field_radius` | `Optional[float]` |  |
| `sublayer_dz` | `Optional[float]` |  |
| `radial_bins_arange` | `Optional[List[float]]` | [start, stop, step] |
| `top_label` | `Optional[str]` |  |
| `bottom_label` | `Optional[str]` |  |

## `integrity_panel`

| field | type | meaning |
|---|---|---|
| `plot_type` | `Literal['integrity_panel']` |  |
| `source` | `Optional[str]` | Documentation only: the config or script this plot was first made with |
| `filename` | `Optional[str]` |  |
| `title` | `Optional[str]` |  |
| `datasets` | `Optional[List[str]]` |  |
| `use_dataset_colors` | `Optional[bool]` |  |
| `x_axis` | `Optional[str]` |  |
| `y_axis` | `Optional[str]` |  |
| `x_label` | `Optional[str]` | Overrides label_for() |
| `y_label` | `Optional[str]` | Overrides label_for() |
| `x_limits` | `Optional[List[Optional[float]]]` |  |
| `y_limits` | `Optional[List[Optional[float]]]` |  |
| `legend_location` | `Optional[str]` |  |
| `legend_outside` | `Union[bool, Literal['left', 'top']]` | True: right of the axes; 'left' or 'top' for those sides |
| `legend_top_align` | `Optional[Literal['right']]` | With legend_outside='top': right-align instead of centring |
| `legend_ncol` | `Optional[int]` | Columns of a 'top' legend (default 5) |
| `legend_outside_left_anchor` | `Optional[float]` | x anchor of a 'left' legend in axes units (default -0.16) |
| `legend_order` | `Optional[List[str]]` | These labels first, in this order; the rest keep theirs |
| `figure_size` | `Optional[List[float]]` |  |
| `filter_size` | `Optional[str]` |  |
| `filter_layer` | `Optional[int]` |  |
| `filter_speed` | `Optional[float]` |  |
| `filter_tip_radius` | `Optional[float]` |  |
| `filter_materials` | `Optional[List[str]]` |  |
| `filter_forces` | `Optional[List[float]]` |  |
| `filter_pressures` | `Optional[List[float]]` |  |
| `pressures` | `Optional[List[float]]` |  |
| `angle` | `Optional[float]` |  |
| `force` | `Optional[float]` |  |
| `filter_force_range` | `Optional[List[float]]` |  |
| `filter_nf_range` | `Optional[List[float]]` |  |
| `lf_source` | `Optional[Literal['as_was', 'sx_sy', 'fx_fy']]` |  |
| `family` | `Optional[str]` | Keep only runs of this family (new in v2) |
| `tier` | `Literal['window', 'raw', 'peratom']` | Which sampling tier to plot (new in v2) |
| `use_scientific_notation` | `Optional[bool]` |  |
| `average_first_n_timesteps` | `Optional[int]` |  |
| `average_timestep_range` | `Optional[List[int]]` |  |
| `material` | `Optional[str]` | Material (required) |
| `run_dir` | `Optional[str]` | AFM run directory holding L<layer>/ (or give dataset) |
| `dataset` | `Optional[str]` | Stored dataset holding the run (instead of run_dir) |
| `layer` | `Optional[int]` |  |
| `loads` | `Optional[List[float]]` |  |
| `traces_file` | `Optional[frictionsim2d.postprocessing.plot_config.TableSource]` | run_id, step, temp_sheet_3d, comz_L1, comz_L2 |
| `status_file` | `Optional[frictionsim2d.postprocessing.plot_config.TableSource]` | run_id, force, status |
| `title_status` | `Literal['completion', 'verdict']` | Dataset panels: 'completion' titles a failed-integrity run as finished (as the published panels); 'verdict' shows its integrity verdict |
| `dpi` | `Optional[int]` |  |

## Settings and presets

Plot style is a `PlotSettings` document. Every section is optional; what you set is
merged over the defaults.

```jsonc
{
  "figure":      {"size": [10, 7], "dpi": 150},
  "fonts":       {"family": "sans-serif", "title": 26, "axis_label": 24,
                  "tick_label": 22, "legend": 16},
  "colors":      {"palette": "gem12"},
  "markers":     {"style": "o", "size": 12},
  "lines":       {"width": 1.3, "fit_style": "--", "fit_alpha": 0.8},
  "grid":        {"show": true, "which": "both", "major_style": "-",
                  "minor_style": ":", "major_alpha": 0.5, "minor_alpha": 0.3},
  "error_bands": {"alpha": 0.2},
  "legend":      {"location": "best"},
  "layout":      {"constrained_layout": false},
  "axes":        {"use_scientific_notation": true, "scilimits": [-3, 4]},
  "export":      {"formats": ["png"], "transparent": false}
}
```

### The `publication` preset

Sized for a single journal column (3.4 in, the APS/Elsevier width), vector-first, with
type that sits correctly against a 9–10 pt body:

```json
{
  "figure": {"size": [3.4, 2.6], "dpi": 600},
  "fonts": {"family": "sans-serif", "title": 9, "axis_label": 9, "tick_label": 8, "legend": 7},
  "colors": {"palette": "gem12"},
  "markers": {"style": "o", "size": 4},
  "lines": {"width": 1.0, "fit_style": "--", "fit_alpha": 0.9},
  "grid": {"show": false, "which": "major", "major_style": "-", "minor_style": ":", "major_alpha": 0.25, "minor_alpha": 0.15},
  "error_bands": {"alpha": 0.18},
  "legend": {"location": "best"},
  "layout": {"constrained_layout": true},
  "axes": {"use_scientific_notation": true, "scilimits": [-3, 4]},
  "export": {"formats": ["pdf", "svg", "png"], "transparent": false}
}
```

```bash
FrictionSim2D postprocess plot plots.json --preset publication
```

**Precedence**, lowest to highest: the defaults, then `--preset`, then `--settings`.

**A list in an override REPLACES the default list; it never extends it.**
`--settings '{"figure": {"size": [8, 6]}}'` gives `[8, 6]`, not `[10, 7, 8, 6]`. There is
no setting for which appending is the right answer.

### Axis labels

An axis with no `x_label`/`y_label` on the plot spec is labelled from the observable
registry: `label_for('nf', 'afm')` is `Normal Load (nN)`, `label_for('comz_L4', ...)` is
`Layer 4 COM z (A)`. An explicit label on the spec still wins — a paper figure sometimes
needs a shorter axis than the honest one.

---

## Commands

### `postprocess read`

```bash
FrictionSim2D postprocess read RESULTS_DIR [--export] [--skip-fraction FLOAT] [--complete-only]
                                [--integrity-settings FILE] [--no-integrity]
```

- `RESULTS_DIR` — a simulation root, the directory holding `afm/`, `sheetonsheet/`,
  `indent/`, `pes_tip/`, `pes_sheet/`.
- `--export` — write `outputs/output_full_<size>.json` (schema 3) **and**
  `outputs/summary_<size>.csv`, echoing each path.
- `--skip-fraction FLOAT` (default `0.2`) — the transient fraction dropped before the
  summary statistics, forwarded to `summarise_run`.
- `--complete-only` — export and summarise only `complete` load points (the
  pre-schema-3 behaviour); `metadata.status_counts` still counts every point.
- `--integrity-settings FILE` — YAML with an `integrity:` mapping that replaces the user
  settings' thresholds for this read; `--no-integrity` skips the check. See
  [integrity.md](integrity.md).

The completeness reports are written either way.

### `postprocess plot`

```bash
FrictionSim2D postprocess plot [PLOT_CONFIG] [OPTIONS]
```

- `-o, --output-dir DIR` (default: `plots`)
- `--settings PATH` — a partial `PlotSettings` JSON, merged over `--preset`
- `--preset NAME` — a shipped preset (currently: `publication`)
- `--validate` — validate the document and exit, without importing matplotlib
- `--schema` — print the plot schema as Markdown and exit; needs no `PLOT_CONFIG`
- `--init PATH` — write a valid two-plot starter document and exit
- `--data-dir DIR` — with `--init`, a dataset directory to reference (repeatable)

Every file written is echoed, and the count is reported at the end.

### `postprocess pes-scan`

```bash
FrictionSim2D postprocess pes-scan RUN_ROOT [OPTIONS]
```

- `-d, --data-dir PATH` (default: `data`) — output dir for `pes_scan_{sheet,tip}.csv`
- `--grid-dir PATH` — also copy tidied per-material grids
- `--kind [sheet|tip|both]` (default: `both`)

See [pes_scan.md](pes_scan.md).

### `postprocess indent`

```bash
FrictionSim2D postprocess indent RUN_ROOT [OPTIONS]
```

- `-o, --out-dir PATH` (default: `outputs/poke_sims`)
- `-d, --data-dir PATH` (default: same as `--out-dir`) — for `indent_features.csv`

The per-load `output_indent_f<load>_s<seed>.json` is the v2 export document restricted to
one load, plus `metadata.indent_load_nN` and `metadata.indent_base_seed`. See
[indent.md](indent.md).
