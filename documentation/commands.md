# Commands

Complete reference for every CLI command group, mirroring `src/cli.py`.

For the most current option list, `FrictionSim2D <GROUP> <COMMAND> --help` is always authoritative.

## Top level

```bash
FrictionSim2D --help
FrictionSim2D --version
```

## run

### run afm

```bash
FrictionSim2D run afm CONFIG_FILE [OPTIONS]
```

Options:

- `--settings-file PATH` — per-run settings.yaml override
- `-o, --output-dir TEXT` (default: `simulation_output`)
- `--hpc-scripts` — generate PBS/SLURM scripts in the same step

### run sheetonsheet

```bash
FrictionSim2D run sheetonsheet CONFIG_FILE [OPTIONS]
```

Same options as `run afm`.

### run pes-sheet

```bash
FrictionSim2D run pes-sheet CONFIG_FILE [OPTIONS]
```

Generate a static interlayer PES scan: a frozen bilayer whose top layer is
grid-scanned laterally over one surface unit cell, producing a per-cell normalized
energy map instead of a sliding run.

Requires `[2D] layers = [2]` and a `[pes]` section. Same options as `run afm`. See
[pes_scan.md](pes_scan.md).

### run pes-tip

```bash
FrictionSim2D run pes-tip CONFIG_FILE [OPTIONS]
```

Generate a static tip-on-sheet surface PES scan: assembles the tip, sheet, and
substrate stack, places the rigid tip at a fixed contact gap, then grid-scans it
laterally, recording energy and the lateral force on the tip.

Requires a `[pes]` section. Omitting `[sub]` selects the freestanding builder. Same
options as `run afm`. See [pes_scan.md](pes_scan.md).

### run indent

```bash
FrictionSim2D run indent CONFIG_FILE [OPTIONS]
```

Generate an indentation simulation: the standard AFM build and `system.in` ramp to
each load in `[general] force`, then per load a short damped finite-temperature hold
followed by a quasi-static retract of the rigid tip to full detachment. The sliding
phase is never invoked.

Records both the out-of-plane penetration channels and the force–distance retract
curve, with a fresh recorded velocity seed per `(material, layer)`. Requires an
`[indent]` section; loads come from `[general] force`. Same options as `run afm`.

Replaces the former separate `run poke` (hold-only) and `run adhesion`
(retract-only) commands. See [indent.md](indent.md).

## settings

### settings show

```bash
FrictionSim2D settings show [--settings-file PATH] [--origin]
```

Options:

- `--settings-file PATH` — inspect a specific file rather than the resolved default
- `--origin` — also print which settings file was loaded

### settings init

```bash
FrictionSim2D settings init [--global] [--force]
```

Options:

- `--global` — write to `~/.config/FrictionSim2D/settings.yaml`
- `--force` — overwrite without prompting

### settings reset

```bash
FrictionSim2D settings reset [--global]
```

Options:

- `--global` — remove the global settings file instead of the local one

## hpc

### hpc generate

```bash
FrictionSim2D hpc generate SIMULATION_DIR [OPTIONS]
```

Options:

- `--settings-file PATH`
- `-s, --scheduler {pbs|slurm}` (default: `pbs`)
- `-o, --output-dir TEXT` (default: `SIMULATION_DIR/hpc`)

### hpc enrich-provenance

```bash
FrictionSim2D hpc enrich-provenance -r ROOT          # sweep a whole tree
FrictionSim2D hpc enrich-provenance SIM_DIR [LOG_PATH]   # one run
```

Options:

- `-r, --recursive` — treat the argument as a tree root and enrich every run
  beneath it

Folds a finished run's LAMMPS log into its `provenance.json`: the AFM settle
loop's `# RAMP_STATUS converged=... r=... iterations=...` and
`# UNLOADED_REF layer=... z=...` lines become the record's
`ramp_converged`/`ramp_r`/`ramp_iterations` and `unloaded_reference_z` fields,
plus `lammps_version`. Without this step a tip that never settled leaves no
trace anywhere you would look.

**The recursive sweep is the expected workflow today.** Simulations are built
on the workstation, transferred to the cluster, and run there; `FrictionSim2D`
itself is *not* installed on the cluster, so the guarded call the generated job
scripts carry never fires. The merge therefore happens here, after the results
come back:

```bash
# 1. results back from the cluster (the submit helper's own download line --
#    it includes run*.log, which is where RAMP_STATUS lives)
rsync -avzhP --include="*/" --include="results/***" --include="visuals/***" \
      --include="run*.log" --exclude="*" HPC_HOST:HPC_HOME/simulation_20260908_101500 ./

# 2. one sweep over the whole tree
FrictionSim2D hpc enrich-provenance -r ./simulation_20260908_101500
```

That prints one line per run directory and a closing tally:

```
enriched afm/h_MoS2/L1: from run_system.log
enriched afm/h_MoS2/L2: from run_slide.log
skipped afm/t_CoTe2/L1: no run log in .../lammps
2 enriched, 0 unchanged, 1 skipped.
```

A run whose only log is a slide log still reports `enriched`: every LAMMPS run
opens with a `LAMMPS (<version>)` banner, so the record gains a
`lammps_version` even when no settle-loop line is present. It gains nothing
else — the merge only writes fields the log actually supplied, so a slide log
can never invent, or overwrite, a settle result. `unchanged` is what a repeat
pass over an already-swept tree reports.

Run directories are found with the same discovery helper `hpc generate` uses,
but the sweep is deliberately broader: `hpc generate` filters by the script
names configured in `settings.yaml`, while the sweep accepts any directory with
a `lammps/` holding an `.in` file. That is the right breadth after a run —
directories the generator would have filtered out simply report `skipped`, at
no cost. Re-running the sweep is safe: it re-reads the same logs and writes the
same values.

**Single-run mode.** `SIM_DIR` is the directory holding `provenance.json` (the
parent of `lammps/`). `LOG_PATH` is optional — omitted, it defaults to
`SIM_DIR/lammps/run_system.log`, falling back to the sole `run*.log` when that
name is absent. Several logs with none named `run_system.log` are reported and
skipped rather than guessed at. Pointing it at a slide log is harmless: it can
only re-record `lammps_version`, never touch the settle fields. (If the two
phases ran against different LAMMPS builds, the last log passed wins that one
field; both phases belong to the same run, so this is accepted rather than
worked around.)

The job scripts name each log after its input script (`run_system.log`,
`run_slide.log`, and one per split for per-load/per-angle slide scripts), so
the system phase's log is never overwritten by a later phase.

It always exits 0 on missing data — a missing log, a missing or unreadable
record, a log with no settle-loop lines, and a tree with no runs in it are all
reported and ignored. A completed simulation is never reported as failed over
a provenance nicety. (Passing `-r` together with a `LOG_PATH` is a usage error,
since one log cannot describe a whole tree.)

If `FrictionSim2D` is later installed in a conda env on the cluster, the job
scripts' own call starts working by itself and this step becomes optional; the
templates already carry it, guarded by `command -v`.

## aiida

> AiiDA commands require `aiida-core`. Install with `pip install -e .[aiida]`.

The store is import-only: `run` generates, the scheduler runs, and `aiida
import` registers the finished tree (one CalcJob per deck, an integrity
verdict per point, the set's rulings). Nothing submits through AiiDA.
See [aiida.md](aiida.md) for the node model.

Commands that read settings (`import`, `postprocess`, `dataset`) take
`--settings-file PATH` to read that file instead of
`~/.config/FrictionSim2D/settings.yaml`; its `aiida.profile` also names the
profile when `--profile` is not given.

### aiida status

```bash
FrictionSim2D aiida status
```

### aiida preflight

Verify profile, database and entry points before generating a campaign; tells
apart "no profile initialised here" (fine on a generation-only machine) from
"profile broken". Exits 0 only when both are healthy:

```bash
FrictionSim2D aiida preflight
```

### aiida setup

Creates (or loads) the AiiDA profile — only the profile: the machine
`Computer` and the LAMMPS `Code` of each run are created when it is imported.

```bash
FrictionSim2D aiida setup [OPTIONS]
```

Options:

- `-p, --create-profile TEXT` — name of the profile to **create** (default:
  `friction2d`). Every other `aiida` command spells `--profile`, which
  **loads** an existing one.

### aiida import

Ingest a `simulation_XXXXXXXX` directory as a new (or updated) simulation set.
Idempotent — re-importing adds only new or changed load points:

```bash
FrictionSim2D aiida import SIMULATION_FOLDER [OPTIONS]
```

Options:

- `-l, --label TEXT` — label for the set (default: the folder name)
- `--machine TEXT` — the machine (Computer) the runs were produced on
  (default: `settings.aiida.machine`, else each run's `settings.hpc.hpc_host`,
  else `localhost`)
- `--profile TEXT`
- `--settings-file PATH`
- `--dry-run` — print what would be ingested; create nothing

The report counts decks created, skipped, superseded and already imported
(under another label), points, verdicts and rulings, and lists every deck with
no LAMMPS version (no Code), every failure and every warning.

Exit code: non-zero if any run failed. A real (non-dry-run) import also exits
non-zero if any run could not be parsed.

### aiida rule

Apply a rulings file to a set: a `pass`/`fail` ruling per quarantined point,
or — with `override: true` and a note — per any point:

```bash
FrictionSim2D aiida rule LABEL [--rulings PATH] [--withdraw] [--profile TEXT]
```

- `--rulings PATH` — the file, or its directory (default:
  `integrity_rulings.yaml` in the root the set was imported from)
- `--withdraw` — also withdraw every ruling the file no longer makes. The
  set's own file always withdraws them; another file does only with this flag.

Idempotent. Prints `applied N, skipped N, withdrawn N`, every live override of
the set (`override <point>: <original> -> <new> (<note>)`), the withdrawn
points, and rulings that match no point.

### aiida postprocess

Summarise every point of a set whose latest verdict is `complete` (one tracked
summary per point, summary options and code version; a second pass makes none):

```bash
FrictionSim2D aiida postprocess LABEL [--profile TEXT] [--settings-file PATH]
```

The summary options come from `settings.aiida.skip_fraction`.

### aiida dataset

Make the dataset `friction2d/datasets/NAME` from one or more sets: one row per
point whose latest verdict is `complete`, with the summary the policy selects
and the units:

```bash
FrictionSim2D aiida dataset NAME LABEL [LABEL ...] [--profile TEXT] [--settings-file PATH]
```

Datasets are never rewritten: an existing NAME is refused.

### aiida reindex

Recompute every friction2d index extra (block, schema, file and folder hashes;
run, deck and token extras) from the stored nodes — after an archive import
that dropped or edited extras:

```bash
FrictionSim2D aiida reindex [--profile TEXT]
```

### aiida verify

Report hashes held by more than one node and hash extras that disagree with
their node's content; exits 1 when it finds either:

```bash
FrictionSim2D aiida verify [--profile TEXT]
```

### aiida stamp-uuid

Give every legacy `provenance.json` under ROOT a `run_uuid`. Runs built
before provenance schema 2.2.0 carry no run identity, and the CalcJob import
refuses them (`MissingRunUUID`). Needs no aiida-core:

```bash
FrictionSim2D aiida stamp-uuid ROOT [--dry-run]
```

- Each file without a `run_uuid` gets a fresh uuid4 (each AFM `L{n}` record
  its own) and `run_uuid_stamped_at` (ISO date), so stamped legacy runs stay
  distinguishable. Every other key keeps its value; the write is atomic.
- An existing `run_uuid` is never overwritten: a second pass stamps nothing.
- `--dry-run` reports what would be stamped and writes nothing.
- Stamping rewrites `provenance.json`. Stamp once, before the first import,
  and keep the stamped tree: a separately stamped copy of the same runs gets
  new `run_uuid` values and imports as different runs.
- Exit code 1 if any `provenance.json` was unreadable, or holds a `run_uuid`
  that is not a UUID (listed under `invalid`, never overwritten).

### aiida list-sets

```bash
FrictionSim2D aiida list-sets [OPTIONS]
```

Options:

- `--profile TEXT`
- `--format {table|json|csv}` (default: `table`)

`Runs` counts distinct runs of live decks and `Results` live points;
superseded nodes are kept in the store but not counted.

### aiida export-results

Write a stored set back out as an ordinary simulation tree — the finished
outputs, ready for `postprocess read`/the GUI download unchanged:

```bash
FrictionSim2D aiida export-results SET_LABEL OUTPUT_DIR [--profile TEXT]
```

### aiida rebuild

Regenerate a set's LAMMPS *inputs* from the store: exports each run's stored
`config.json` and the set's `settings.yaml`, then reruns the real builder
router — the same one `FrictionSim2D run` uses:

```bash
FrictionSim2D aiida rebuild SET_LABEL [OPTIONS]
```

Options:

- `-o, --output-dir PATH` (default: current directory)
- `--hpc-scripts` — also generate PBS/SLURM scripts
- `--profile TEXT`

### aiida query

```bash
FrictionSim2D aiida query [OPTIONS]
```

Options:

- `-m, --material TEXT` — repeatable; a matching run must contain **all** given materials
- `--family TEXT` (e.g. `afm`, `hetero`)
- `--model TEXT` (e.g. `afm`, `sheetonsheet`)
- `--set-label TEXT`
- `--potential TEXT`
- `-l, --layers INTEGER`
- `--force-min FLOAT` / `--force-max FLOAT`
- `--pressure-min FLOAT` / `--pressure-max FLOAT`
- `--angle-min FLOAT` / `--angle-max FLOAT`
- `--speed-min FLOAT` / `--speed-max FLOAT`
- `--temperature-min FLOAT` / `--temperature-max FLOAT`
- `--limit INTEGER`
- `--order-by TEXT` (prefix `-` for descending)
- `--format {table|csv|json}` (default: `table`). `json` without `-o` prints
  one object per result row; with `-o` it writes the query parameters and the
  matching runs.
- `-o, --output PATH`
- `--profile TEXT`

One row per live point. The `table` and `csv` formats lead with the point
(`set_label`, `rel_path`, `deck_name`, `load`, `load_unit`, `angle`, `speed`)
and the verdict the policy selects (`status`, `reasons`, `verdict_kind`,
`provisional`, `thresholds_version`), then the run tokens, the identity
columns and the selected summary.

Any other `RunTokens` filter is Python-API only (`src/aiida/query.py`), not
exposed as a CLI option.

### aiida delete

Delete a single simulation set and all linked nodes (irreversible):

```bash
FrictionSim2D aiida delete SET_LABEL [--profile TEXT] [--yes]
```

### aiida clear

Delete **all** FrictionSim2D nodes from the profile (irreversible):

```bash
FrictionSim2D aiida clear [--profile TEXT] [--yes]
```

### aiida export

Archive one simulation set (its Group, CalcJobs, verdicts, rulings and
summaries) to a portable `.aiida` file, with `<OUTPUT>.params/` block
sidecars and `<OUTPUT>.README.md`:

```bash
FrictionSim2D aiida export SET_LABEL [-o OUTPUT] [--profile TEXT] [--force]
                                     [--include-peratom] [--keep-hpc-paths]
```

- `-o, --output PATH` (default: `friction2d.aiida`; an existing file is refused)
- `--force` — write even when live points have no integrity verdict (the
  export refuses otherwise)
- `--include-peratom` — keep the per-atom nodes (left out by default)
- `--keep-hpc-paths` — do not anonymise hostnames, executables and paths

The AiiDA User email is kept for attribution. Datasets are not included; they
export separately (`src.aiida.export.export_dataset`, Python only). See
[aiida.md](aiida.md#9-archives).

### aiida import-archive

Import a previously exported `.aiida` archive into the current profile.
Creates no extra `friction2d/<label>` group of its own — the group, if any,
came across inside the archive:

```bash
FrictionSim2D aiida import-archive ARCHIVE_PATH [--profile TEXT]
```

## postprocess

### postprocess read

Walk a simulation root and export its runs:

```bash
FrictionSim2D postprocess read RESULTS_DIR [--export] [--skip-fraction FLOAT]
                                [--integrity-settings FILE] [--no-integrity]
```

Options:

- `--export` — write `outputs/output_full_<size>.json` (schema 2) **and**
  `outputs/summary_<size>.csv`
- `--skip-fraction FLOAT` (default `0.2`) — transient fraction dropped before the
  summary statistics
- `--integrity-settings FILE` — YAML with an `integrity:` mapping that replaces the
  thresholds of the user settings for this read
- `--no-integrity` — skip the integrity check ([integrity.md](integrity.md))

See [postprocessing.md](postprocessing.md) for both schemas.

### postprocess plot

Generate plots from a JSON plot configuration file:

```bash
FrictionSim2D postprocess plot PLOT_CONFIG [OPTIONS]
```

Options:

- `-o, --output-dir TEXT` (default: `plots`)
- `--settings PATH` — a partial `PlotSettings` JSON, merged over `--preset`
- `--preset NAME` — a shipped settings preset (currently: `publication`)
- `--validate` — validate the document against the plot schema and exit
- `--schema` — print the plot schema as Markdown and exit (no `PLOT_CONFIG` needed)
- `--init PATH` — write a valid starter plot document and exit
- `--data-dir DIR` — with `--init`, a dataset directory to reference (repeatable)

`PLOT_CONFIG` must define `data_dirs`, `labels`, and `plots`; unknown keys are errors.
See [postprocessing.md](postprocessing.md) for the full schema, [examples.md](examples.md),
and the reference config at [publication/plots_publication.json](publication/plots_publication.json).

### postprocess pes-scan

Reduce PES-scan energy/force grids to per-material descriptor CSVs:

```bash
FrictionSim2D postprocess pes-scan RUN_ROOT [OPTIONS]
```

Options:

- `-d, --data-dir PATH` (default: `data`) — output dir for `pes_scan_{sheet,tip}.csv`
- `--grid-dir PATH` — also copy tidied per-material grids to `<grid-dir>/{sheet,tip}/<material>.csv`
- `--kind [sheet|tip|both]` (default: `both`)

Scans `RUN_ROOT` for `pes_sheet/` and `pes_tip/` results and writes one row per
material, merge key `material`. See [pes_scan.md](pes_scan.md).

### postprocess indent

Reduce an indent run to penetration JSON + adhesion features + tidied curves:

```bash
FrictionSim2D postprocess indent RUN_ROOT [OPTIONS]
```

Options:

- `-o, --out-dir PATH` (default: `outputs/poke_sims`) — destination for the
  per-load penetration JSON, the tidied `indent/<material>_l<layer>.csv` retract
  curves and `indent_manifest.csv`
- `-d, --data-dir PATH` (default: same as `--out-dir`) — destination for
  `indent_features.csv`

Produces both descriptor sets from the one hold-and-retract run:

- `output_indent_f<load>_s<seed>.json` — penetration, one per load, nested campaign schema
- `indent_features.csv` — `pulloff_force`, `work_of_adhesion`, `contact_stiffness`, keyed `material, layer, load`
- `indent/<material>_l<layer>.csv` — the tidied per-condition retract curves
- `indent_manifest.csv` — `material, layer, load, seed, hold_steps, z_step, n_steps, wall_time`

See [indent.md](indent.md).

## compare

Compare the parameter blocks of two runs or campaigns, field by field, each in
its own schema's units and defaults:

```bash
FrictionSim2D compare A B [--profile TEXT] [--rtol FLOAT] [--atol FLOAT]
```

A and B are each a CalcJob pk or uuid, a point uuid, a set (group) label, or
an `aiida export` sidecar. Prefix one with `pk:`, `uuid:`, `point:`, `group:`
or `sidecar:` to say which (`sidecar:` takes the `<archive>.params/`
directory, a directory holding one, or one deck's JSON in it). Unprefixed, a
directory counts as a sidecar only when it is, or holds, a `.params` directory
with its `schema-*.json`; anything else is looked up in the store, so a local
folder that shares a set's name never shadows the set.

## db and api (removed)

The `db` and `api` command groups, the central PostgreSQL database and the REST
API were removed. Share results as AiiDA archives instead (`FrictionSim2D aiida
import`, see [aiida.md](aiida.md)).

## Practical notes

- Every `run` command exposes `--hpc-scripts` to generate scheduler scripts in one step.
- `postprocess plot` requires a JSON config file, not a results directory.
- `aiida export-results` and `postprocess read --export` both produce a tree `postprocess read` can consume, but `aiida export-results` reads from the AiiDA store while `postprocess read` walks the raw LAMMPS output files on disk.

## Related docs

- [configuration.md](configuration.md)
- [examples.md](examples.md)
- [aiida.md](aiida.md)
