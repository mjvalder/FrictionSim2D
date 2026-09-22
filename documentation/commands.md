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

Upload one stored AiiDA simulation set to the shared central database. Reads the
LOCAL AiiDA store only (spec U4) — SET_LABEL must already exist there:

```bash
FrictionSim2D db upload SET_LABEL [--uploader NAME] [--contact EMAIL] \
    [--aiida-profile PROFILE] [--dry-run] [DB OPTIONS]
```

Each row carries a run UUID and the host that produced it, so a reader can
contact the owner for the full time-series data — there is no automated
fetch. Each row is validated automatically; valid rows are published.

### db stage

Upload with an API key; rows are validated and published or rejected automatically:

```bash
FrictionSim2D db stage SET_LABEL --uploader NAME [--contact EMAIL] \
    [--api-key KEY] [--aiida-profile PROFILE] [DB OPTIONS]
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
