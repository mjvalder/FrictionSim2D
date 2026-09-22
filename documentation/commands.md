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
- `--aiida` — register generated simulations in AiiDA
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

`run <model> ... --aiida --label LABEL` registers the simulation set and its
run nodes at generation time (before any results exist); a later
`aiida import` on the finished tree attaches results to those same nodes.
See [aiida.md](aiida.md) for the node model and run modes.

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

Creates the AiiDA profile, the `localhost` computer, and the LAMMPS code.
Localhost only — run this inside the HPC's own conda environment for
hpc-direct mode.

```bash
FrictionSim2D aiida setup [OPTIONS]
```

Options:

- `-p, --create-profile TEXT` — name of the profile to **create** (default:
  `friction2d`). Every other `aiida` command spells `--profile`, which
  **loads** an existing one.
- `--lammps-path PATH` — explicit path to the LAMMPS executable

### aiida import

Ingest a `simulation_XXXXXXXX` directory as a new (or updated) simulation set.
Idempotent — re-importing adds only new or changed load points:

```bash
FrictionSim2D aiida import SIMULATION_FOLDER [OPTIONS]
```

Options:

- `-l, --label TEXT` — label for the set (default: the folder name)
- `--profile TEXT`
- `--dry-run` — print what would be ingested; create nothing

Exit code: non-zero if any run failed. A real (non-dry-run) import also exits
non-zero if any run was left empty or could not be parsed; a `--dry-run`
preview of a fresh, pre-HPC tree is legitimately all-empty and exits 0.

### aiida list-sets

```bash
FrictionSim2D aiida list-sets [OPTIONS]
```

Options:

- `--profile TEXT`
- `--format {table|json|csv}` (default: `table`)

`Results` counts live results only; superseded nodes are kept in the store but
not counted.

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

Archive one simulation set (set + runs + results — the whole
`friction2d/<label>` group) to a portable `.aiida` file:

```bash
FrictionSim2D aiida export SET_LABEL [-o OUTPUT] [--profile TEXT]
```

### aiida import-archive

Import a previously exported `.aiida` archive into the current profile.
Creates no extra `friction2d/<label>` group of its own — the group, if any,
came across inside the archive:

```bash
FrictionSim2D aiida import-archive ARCHIVE_PATH [--profile TEXT]
```

### aiida package

Create a `tar.gz` of simulation inputs for cluster transfer (excludes `.lammpstrj`):

```bash
FrictionSim2D aiida package SIMULATION_DIR [-o OUTPUT]
```

### aiida submit

```bash
FrictionSim2D aiida submit SIMULATION_DIR [OPTIONS]
```

`SIMULATION_DIR` is a `simulation_*` root or any directory inside one (a model
directory, a single run). The root is found by walking up to the nearest
directory with `<model>/.../provenance.json` under it, and every run at or
below `SIMULATION_DIR` is submitted. Register the tree first (`run --aiida` or
`aiida import`): a run with no run node still runs and its results are copied
back into the tree, but nothing is stored until you `aiida import` it.

In-process mode (the default) reports `Ran N job(s): K finished ok, M failed`
with each failed job's exit status, and exits 1 if any failed. With `--daemon`
the jobs are only queued and the command reports `Submitted`.

Options:

- `--profile TEXT` — AiiDA profile to load (default: `settings.aiida.profile`,
  then AiiDA's default profile)

- `--settings-file PATH`
- `-c, --code TEXT` (auto-detects if not specified, falling back to `settings.aiida.lammps_code_label`)
- `--scripts TEXT` (comma-separated list)
- `--array` — submit a single array job for all simulations
- `--machines INTEGER`
- `--mpiprocs INTEGER`
- `--walltime TEXT` (`HH:MM:SS` or hours)
- `--queue TEXT`
- `--project TEXT`
- `--dry-run` — preview configuration without submitting
- `--daemon` — submit to the AiiDA daemon (needs RabbitMQ). Default: run
  in-process with `aiida.engine.run`, which needs no broker.

## postprocess

### postprocess read

Walk a simulation directory and export time-series data to JSON:

```bash
FrictionSim2D postprocess read RESULTS_DIR [--export]
```

Options:

- `--export` — write `output_full_*.json` files alongside results

### postprocess plot

Generate plots from a JSON plot configuration file:

```bash
FrictionSim2D postprocess plot PLOT_CONFIG [OPTIONS]
```

Options:

- `-o, --output-dir TEXT` (default: `plots`)
- `--settings PATH` — additional JSON plot-style settings

`PLOT_CONFIG` must define `data_dirs`, `labels`, and `plots`. See [examples.md](examples.md) and the reference config at [publication/plots_publication.json](publication/plots_publication.json).

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

## db

The `db` command group manages the **central PostgreSQL database** — one
shared summary-statistics store (schema 3). It is **not** the local AiiDA
store; for full provenance and time-series access, use the `aiida` command
group. There is now a single central database, so there is no `--profile`
option on this group at all — the connection is `settings.database` plus the
`--host`/`--port`/`--dbname`/`--user`/`--password` overrides below.

Where an `aiida`-holding set label is needed (`db upload`, `db stage`), the
option is spelled `--aiida-profile`, never `--profile`: in this group
`--profile` used to mean the *database* profile that no longer exists, so
reusing the spelling would silently change its meaning.

**Two library entry points, reconciled.** The `db` CLI group always talks
straight PostgreSQL (`_make_db`/`psycopg2`, `src/cli.py`) — that path is only
meant to run on the machine hosting the database. `src.data.get_client()`
(the library entry point everything else should use) defaults to
`mode='api'`, an HTTP client against the REST API; `mode='direct'` is the
same psycopg2 path. The `--api-key` on `db stage` (and the key `db
create-key` generates) does not gate read or write access by itself —
Postgres role permissions do that — it records which contributor a row came
from.

**Common DB options** (available on every `db` subcommand):

```text
--host TEXT           $FRICTION_DB_HOST
--port INTEGER        $FRICTION_DB_PORT
--dbname TEXT         $FRICTION_DB_NAME
-u, --user TEXT       $FRICTION_DB_USER
--password TEXT       $FRICTION_DB_PASSWORD
```

### db setup

First-time setup: initializes schema, creates an API key, and saves it to the
per-user settings file `load_settings` reads
(`$XDG_CONFIG_HOME/FrictionSim2D/settings.yaml`, else
`~/.config/FrictionSim2D/settings.yaml`):

```bash
FrictionSim2D db setup [--name NAME] [DB OPTIONS]
```

### db init

Create (or verify) the central schema. Safe to re-run without `--drop` —
existing tables/columns are left untouched. An existing schema-2 table
without `--drop` refuses with a clear message rather than silently stamping
`schema_version = 3` over data that has not actually been migrated: there is
no in-place migration to schema 3, so `--drop` (which recreates
`simulations` from scratch) is the only supported upgrade path.

```bash
FrictionSim2D db init [--drop [--yes]] [DB OPTIONS]
```

`--drop` deletes every row, so it asks for confirmation first; `--yes`/`-y`
skips the prompt (for scripts).

### db create-key

Generate a new API key for write access (printed once — store it; it can only be revoked, not retrieved again):

```bash
FrictionSim2D db create-key --name NAME [DB OPTIONS]
```

### db upload

Upload one stored AiiDA simulation set to the central database. `SET_LABEL`
is a **positional** argument (an existing AiiDA simulation-set label) and
reads the local AiiDA store only (spec U4) — there is no automated fetch, so
every central row instead carries a run UUID and the producing host, so a
reader can contact the owner for the full time-series data:

```bash
FrictionSim2D db upload SET_LABEL [-n/--uploader NAME] [--contact EMAIL] \
    [--aiida-profile PROFILE] [--dry-run] [DB OPTIONS]
```

Rows are validated automatically; a row already present (matched by its
`time_series_hash`) is counted **"skipped (already in central DB)"**, not an
error. `--dry-run` reports what would be uploaded and connects to nothing.

### db stage

Same idea as `db upload`, with an API key; rows are validated and published
or rejected automatically:

```bash
FrictionSim2D db stage SET_LABEL -n/--uploader NAME [--contact EMAIL] \
    [--api-key KEY] [--aiida-profile PROFILE] [DB OPTIONS]
```

### db query

```bash
FrictionSim2D db query [OPTIONS] [DB OPTIONS]
```

Options:

- `-m, --material TEXT`
- `--type TEXT` (`afm`, `indent`, `sheetonsheet`, `hetero`, `flake`, `pes_tip`, or `pes_sheet`)
- `-l, --layers INTEGER`
- `-n, --uploader TEXT`
- `--limit INTEGER` (default: 50)
- `--csv PATH`

### db stats

```bash
FrictionSim2D db stats [DB OPTIONS]
```

### db publish

Promote a staged/validated result to published (curator action):

```bash
FrictionSim2D db publish ROW_ID [DB OPTIONS]
```

### db reject

Reject a staged or validated result:

```bash
FrictionSim2D db reject ROW_ID [--reason TEXT] [DB OPTIONS]
```

### db delete

Delete all rows belonging to a given uploader:

```bash
FrictionSim2D db delete --uploader NAME [DB OPTIONS]
```

## api

### api serve

```bash
FrictionSim2D api serve [OPTIONS]
```

Options:

- `--host TEXT`
- `-p, --port INTEGER`
- `--profile TEXT`
- `--reload` — auto-reload on code changes (development only)
- `--settings-file PATH`

`GET /results` takes the range filters `force_min/force_max`,
`pressure_min/pressure_max`, `temp_min/temp_max` and `angle_min/angle_max`.
Each bound is optional: `force_min=5` alone means "at least 5 nN". The same
holds for the Python clients (`query(force_range=(5.0, None))`, over the API
or directly on `FrictionDB`). A `materials=[...]` filter on `FrictionDB`
matches rows that contain **all** the given materials, like `aiida query -m`.

## Practical notes

- Every `run` command exposes `--hpc-scripts` to generate scheduler scripts in one step.
- `postprocess plot` requires a JSON config file, not a results directory.
- `aiida export-results` and `postprocess read --export` both produce a tree `postprocess read` can consume, but `aiida export-results` reads from the AiiDA store while `postprocess read` walks the raw LAMMPS output files on disk.

## Related docs

- [configuration.md](configuration.md)
- [examples.md](examples.md)
- [aiida.md](aiida.md)
