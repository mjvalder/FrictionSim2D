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
FrictionSim2D hpc enrich-provenance SIM_DIR LOG_PATH
```

Folds a finished run's LAMMPS log into its `provenance.json`: the AFM settle
loop's `# RAMP_STATUS converged=... r=... iterations=...` and
`# UNLOADED_REF layer=... z=...` lines become the record's
`ramp_converged`/`ramp_r`/`ramp_iterations` and `unloaded_reference_z` fields,
plus `lammps_version`.

`SIM_DIR` is the directory holding `provenance.json` (the parent of `lammps/`);
`LOG_PATH` is the stdout log of the `system.in` phase — `lammps/run.log` for
array jobs, `lammps/run_system.log` for combined ones.

The generated job scripts call this themselves after every run, so it rarely
needs to be typed by hand; run it manually to backfill a tree that predates
that wiring. It always exits 0 — a missing log, a missing or unreadable
record, or a log with no settle-loop lines are reported and ignored.

## aiida

> AiiDA commands require `aiida-core`. Install with `pip install -e .[aiida]`.

### aiida status

```bash
FrictionSim2D aiida status
```

### aiida setup

```bash
FrictionSim2D aiida setup [OPTIONS]
```

Options:

- `-p, --profile TEXT`
- `--lammps-path PATH`
- `--use-remote` — configure a remote HPC computer using `settings.yaml`
- `--hpc-config PATH` *(deprecated — use `--use-remote` instead)*
- `--settings-file PATH`

### aiida import

Import a `simulation_XXXXXXXX` directory as a new simulation set:

```bash
FrictionSim2D aiida import SIMULATION_FOLDER [OPTIONS]
```

Options:

- `-l, --label TEXT` — required label (prompted if omitted)
- `-d, --description TEXT`
- `--profile TEXT`

### aiida list-sets

```bash
FrictionSim2D aiida list-sets [OPTIONS]
```

Options:

- `--profile TEXT`
- `--format {table|json|csv}` (default: `table`)

### aiida dump

Export time-series data from an AiiDA set to `output_full_*.json` files, ready for `postprocess plot`:

```bash
FrictionSim2D aiida dump SET_LABEL --output-dir OUTPUT_DIR [OPTIONS]
```

Options:

- `-o, --output-dir PATH` *(required)*
- `--profile TEXT`

Output layout: `OUTPUT_DIR/outputs/output_full_<size>.json`

### aiida rebuild

Reconstruct a full simulation directory tree from AiiDA provenance data alone:

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

- `-m, --material TEXT`
- `-l, --layers INTEGER`
- `-f, --force FLOAT`
- `-s, --set TEXT` — filter by simulation set label
- `--format {table|csv|json}` (default: `table`)
- `-o, --output PATH`
- `--profile TEXT`

### aiida delete

Delete a single simulation set and all linked nodes (irreversible):

```bash
FrictionSim2D aiida delete SET_LABEL [--profile TEXT]
```

### aiida clear

Delete **all** FrictionSim2D nodes from the database (irreversible):

```bash
FrictionSim2D aiida clear [--profile TEXT]
```

### aiida export

Export nodes to a portable AiiDA archive file:

```bash
FrictionSim2D aiida export [OPTIONS]
```

Options:

- `-o, --output PATH` (default: `friction2d.aiida`)
- `-m, --material TEXT`

### aiida import-archive

Import a previously exported AiiDA archive into the current profile:

```bash
FrictionSim2D aiida import-archive ARCHIVE_PATH
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

Options:

- `--settings-file PATH`
- `-c, --code TEXT`
- `--scripts TEXT` (comma-separated list)
- `--array`
- `--machines INTEGER`
- `--mpiprocs INTEGER`
- `--walltime TEXT` (`HH:MM:SS` or integer seconds)
- `--queue TEXT`
- `--project TEXT`
- `--dry-run`

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

The `db` command group manages the **central (remote) PostgreSQL database** — the shared summary-statistics store.
It is **not** the local AiiDA database; for full provenance and time-series access, use the `aiida` command group.

> **Two-database architecture**
> - `FrictionSim2D aiida ...` → local AiiDA database (full time-series, complete provenance)
> - `FrictionSim2D db ...` → central remote PostgreSQL (summary statistics: mean COF, forces, conditions)

All `db` commands accept a common set of connection options. Pass `--profile central` to target the shared remote DB; `--profile local` targets a local instance.

**Common DB options** (available on every `db` subcommand):

```text
--profile TEXT        Profile from settings.yaml ('local' or 'central')
--host TEXT           $FRICTION_DB_HOST
--port INTEGER        $FRICTION_DB_PORT
--dbname TEXT         $FRICTION_DB_NAME
-u, --user TEXT       $FRICTION_DB_USER
--password TEXT       $FRICTION_DB_PASSWORD
```

### db setup

First-time setup: initializes schema, creates an API key, and saves it to settings:

```bash
FrictionSim2D db setup [--name NAME] [DB OPTIONS]
```

### db init

Initialize or verify the database schema (safe to re-run):

```bash
FrictionSim2D db init [DB OPTIONS]
```

### db create-key

Generate a new API key for write access:

```bash
FrictionSim2D db create-key --name NAME [DB OPTIONS]
```

### db upload

Upload simulation results to the shared database:

```bash
FrictionSim2D db upload RESULTS_DIR [--uploader NAME] [DB OPTIONS]
```

Each row is validated automatically; valid rows are published.

### db stage

Upload with an API key; rows are validated and published or rejected automatically:

```bash
FrictionSim2D db stage RESULTS_DIR --uploader NAME [--api-key KEY] [DB OPTIONS]
```

### db query

```bash
FrictionSim2D db query [OPTIONS] [DB OPTIONS]
```

Options:

- `-m, --material TEXT`
- `--type TEXT` (`afm` or `sheetonsheet`)
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

## Practical notes

- Every `run` command exposes `--hpc-scripts` to generate scheduler scripts in one step.
- `postprocess plot` requires a JSON config file, not a results directory.
- `aiida dump` and `postprocess read --export` both produce `output_full_*.json` files, but `aiida dump` reads from the AiiDA database while `postprocess read` walks the raw LAMMPS output files on disk.

## Related docs

- [configuration.md](configuration.md)
- [examples.md](examples.md)
- [aiida.md](aiida.md)
