# AiiDA workflows

FrictionSim2D uses [AiiDA](https://www.aiida.net/) as its **local** provenance
store. There is no separate local database: a stored run's full time series,
inputs and build metadata live only in the AiiDA profile. The central summary
database (`FrictionSim2D db ...`, see [commands.md](commands.md)) holds
aggregate statistics only, contributed from this store.

## 1. The node model

Every simulation set is three kinds of node:

| node | one per | holds |
|---|---|---|
| `FrictionSimulationSetData` | one `simulation_*` root (or `--label`) | `settings.yaml`, the `.ini` config, and attributes `label`, `model`, `source_root`, `created`, `git_commit`, `git_dirty`, `materials[]`, `n_runs`, `source_host` |
| `FrictionSimulationData` | one run (one `provenance.json` — an `L{n}` layer directory for AFM-like families, or a run directory for sheet-like families) | the run's LAMMPS inputs and provenance files, plus every `RunTokens` field as an attribute (`model`, `family`, `materials[]`, `size_x/y`, `substrate`, `tip_material`, `temperature`, `layers`, `pot_token`, `rel_path`, ...) and the full provenance record under `provenance` |
| `FrictionResultsData` | one load point (`(load, angle, speed)` for slide families; one grid for a PES scan; one load for indent) | the original result files verbatim (`friction_*`, `raw_*`, `peratom_*`, `pes_scan*.csv`, `indent_*.csv`, logs, final/hold data), plus summary attributes (`family`, `load_kind`, `load`, `angle`, `speed`, `layers`, `tiers[]`, mean/COF statistics, `run_uuid`, `set_uuid`) |

**Numbers are attributes, files are in the repository.** Time series are
never stored as attributes — `FrictionResultsData.frame(tier)` re-parses the
stored file with the same reader `postprocess read` uses, so the store and
the CLI never disagree about how to read a results file.

Attributes are immutable once an AiiDA node is stored. Anything that changes
after the fact — import timestamps, job status — lives in **extras**, not
attributes. Re-importing a run whose result content changed does not edit the
old `FrictionResultsData` node in place: the old node is marked
`extras['superseded'] = True` and a fresh node takes its place. Queries,
exports and re-import matching all skip superseded nodes, so a stale result
is never returned or double-counted, but it is never deleted either.

The same goes for a run's `provenance.json`. `hpc enrich-provenance` rewrites
it after the run node was created (`ramp_*`, `unloaded_reference_z`,
`lammps_version`); the next `aiida import` of the tree sees that the file
changed and records the new record, its sha256 and the file text in
`extras['provenance_updated']` (reported as `provenance updated: N`).
`export-results`, the provenance bundle and `db upload` all use that newer
record; re-importing an unchanged file records nothing.

Every node belonging to one simulation set is tagged with the AiiDA group
`friction2d/<label>`. `aiida export <label>` archives that whole group
(the group itself included); `aiida import-archive` does **not** recreate
the group — importing an archive is a pure node import, so re-importing
someone else's export never accumulates loose groups of its own.

**No process nodes on `aiida import`.** Nodes link to each other by
`set_uuid`/`run_uuid` attributes, not by AiiDA `CalcJobNode` provenance links.
This is deliberate: faking a calculation node for data that was never run
through AiiDA would be misleading provenance. A run submitted through
`aiida submit` (§3 below) *does* get real CalcJob provenance links, for free,
because it actually went through the engine.

### What is never stored

`visuals/` and any `*.lammpstrj` trajectory file are **never** ingested —
they exist for visual/manual verification only and stay on disk. LAMMPS run
logs (`lammps/run_*.log`) **are** kept, because the settle-loop and
build-identity diagnostics folded into provenance come from them.

What *is* stored is one rule, `src/aiida/retention.py`, shared by
`aiida import` and the CalcJob's `retrieve_list`, so both paths keep the same
files: `provenance.json`, `provenance/**`, `lammps/*.in`,
`lammps/system.in.settings`, `lammps/run_*.log`, `results/*`,
`data/{hold_*,final_*,load_*}`, `log*.lammps` and `*.out`. The run node takes
the inputs and provenance, each load point its own tier files, and every other
retained file (including a `results/*` file no load point claims) rides along
on each load point. Anything over `aiida.max_stored_file_mb` is recorded by
path, size and sha256 instead of stored.

## 2. Idempotent import

```bash
FrictionSim2D aiida import ./simulation_20260421_143404 --label 251113-afm
```

`SIMULATION_FOLDER` is the `simulation_XXXXXXXX` directory produced by one
`FrictionSim2D run`. `-l/--label` names the set (default: the folder name).

Re-running the same import is safe: only new or changed load points are
added, nothing already stored is duplicated. It exits non-zero if any run
failed. On a real (non-dry-run) import it also exits non-zero if any run was
left empty or could not be parsed — but `--dry-run` on a fresh, pre-HPC tree
legitimately previews an all-empty set and still exits 0 unless it hit an
actual failure:

```bash
FrictionSim2D aiida import ./simulation_20260421_143404 --dry-run
```

`--profile` loads an existing AiiDA profile (default: `settings.aiida.profile`,
or AiiDA's own default profile if that is unset too).

`FrictionSim2D run <model> ... --aiida --label 251113-afm` runs the same
ingest at generation time: it registers the set and run nodes immediately
(with no results yet), and a later `aiida import` on the finished tree
attaches results to those same nodes rather than creating new ones.

## 3. Run modes

| mode | workflow |
|---|---|
| **local** | `FrictionSim2D run <model> ... --aiida` then `FrictionSim2D aiida submit`, which runs a real CalcJob on the `localhost` computer — or generate + run LAMMPS yourself and `FrictionSim2D aiida import` the finished tree. Either way the results end up as the same `FrictionResultsData` nodes. |
| **remote-manual** | `FrictionSim2D run` + `FrictionSim2D hpc generate` → rsync the tree to the cluster → submit the PBS/Slurm scripts there → rsync the results back → `FrictionSim2D aiida import <root>` locally. Nothing AiiDA-specific ever runs on the cluster. |
| **hpc-direct** | Same as local, but with AiiDA itself installed inside the HPC's own conda environment: `FrictionSim2D aiida setup` there, run and ingest on the cluster, `FrictionSim2D aiida export <label>` to build a portable archive, then `FrictionSim2D aiida import-archive` on your workstation. |

There is no SSH-connected remote-submission mode: the HPC blocks direct
inbound AiiDA connections, so "remote" here always means one of the two
workflows above, never a live `core.ssh` transport.

**Submission is broker-less by default.** `FrictionSim2D aiida submit` runs
each simulation in-process with `aiida.engine.run` — no RabbitMQ required.
Pass `--daemon` to submit to the AiiDA daemon instead (`aiida.engine.submit`),
which does need a running broker. `aiida submit` takes `--profile` like every
other command, accepts the simulation root or any directory inside it, and
finds each run's node by its `rel_path` under that root -- so register the
tree first (`run --aiida`, or `aiida import`). The Python API
`run_with_aiida(...)` registers the fresh tree itself before submitting.

## 4. One-time setup

```bash
FrictionSim2D aiida status                       # check install + profile health
FrictionSim2D aiida setup                        # create the profile, computer, LAMMPS code
FrictionSim2D aiida setup --lammps-path /usr/local/bin/lmp
```

`aiida setup` uses `-p/--create-profile` (default `friction2d`) because it
*creates* a profile — every other `aiida` command spells `--profile` and
*loads* an existing one. `setup` is localhost-only: on the HPC, run it inside
the HPC's own conda environment for hpc-direct mode, then bring the result
home with `aiida export`.

`FrictionSim2D aiida preflight` checks profile, database and entry points
before you generate a campaign — it tells apart "no profile initialised here"
(fine on a generation-only machine) from "profile broken" and exits 0 only
when both a healthy profile and the `friction2d` entry points are present.

## 5. Query, export, rebuild

```bash
FrictionSim2D aiida query --material h-MoS2 --layers 2 --format csv
FrictionSim2D aiida query --set-label 251113-afm --format table
FrictionSim2D aiida list-sets
```

`aiida query` filters on `RunTokens` fields exposed as CLI options
(`--material`, repeatable — a matching run must contain **all** given
materials — `--family`, `--model`, `--potential`, `--layers`, plus the
force/pressure/angle/speed/temperature min/max ranges) and `--set-label`.
Anything else `RunTokens` carries is reachable only from the Python API
(`src/aiida/query.py`), not from the CLI.

```bash
FrictionSim2D aiida export-results 251113-afm ~/exported/251113-afm
FrictionSim2D aiida rebuild 251113-afm -o ~/rebuild_test --hpc-scripts
```

- `aiida export-results SET_LABEL OUTPUT_DIR` writes a stored set back out as
  an ordinary simulation tree (`results/`, `provenance.json`, `provenance/`)
  — it works unchanged with `postprocess read` and the GUI download.
- `aiida rebuild SET_LABEL [-o DIR] [--hpc-scripts]` instead regenerates a
  set's *inputs*: it exports each run's stored `config.json` and the set's
  `settings.yaml`, then reruns the real builder router (the same one
  `FrictionSim2D run` uses) — not a duplicated, drifting copy of it. Add
  `--hpc-scripts` to also generate PBS/Slurm scripts after rebuilding.

These are different operations: `export-results` gets you back the finished
outputs; `rebuild` gets you back a fresh, re-runnable input tree.

## 6. Archives

```bash
FrictionSim2D aiida export 251113-afm -o ~/transfer/251113-afm.aiida
FrictionSim2D aiida import-archive ~/transfer/251113-afm.aiida
```

`aiida export SET_LABEL` archives every node of that set's
`friction2d/<label>` group, the group itself included, to a portable `.aiida`
file. `aiida import-archive ARCHIVE_PATH` imports it into the current
profile — this is the way an hpc-direct store gets mirrored onto a local
workstation. As noted above, importing an archive creates no extra group of
its own; the group came across inside the archive.

## 7. Deleting sets

```bash
FrictionSim2D aiida delete 260312-force_rebomos    # one set and its linked nodes (irreversible)
FrictionSim2D aiida clear                          # every FrictionSim2D node in the profile (irreversible)
```

Both prompt for confirmation unless `--yes` is passed.

## 8. Packaging inputs for transfer

```bash
FrictionSim2D aiida package ./simulation_output/simulation_YYYYMMDD_HHMMSS -o bundle.tar.gz
```

Builds a `tar.gz` of simulation inputs for cluster transfer; excludes
`.lammpstrj` files by design.

## Known gaps

- **The flake family has no normal-load channel.** `channel_by_role('flake',
  'normal_load')` has nothing to resolve, so `summarise_run` records COF as
  `None` for every flake run. A flake result can still be ingested and
  queried locally, but it cannot be published to the central database (which
  requires a COF).
- **Array-mode submission (`aiida submit --array`, `submit_array`) is not
  ingested by the parser.** It also lacks the AFM shared-build staging
  (`build/`, `provenance/` staged once and reused by every layer) that a
  normal per-run submission gets — an array job stages each run
  independently.
- **AFM's `provenance/` directory is not retrieved by the CalcJob.** The run
  node created at generation time (`run --aiida`) already holds the same
  provenance data as an attribute, so retrieving the directory again on
  completion would be redundant; only `provenance.json` and the LAMMPS
  inputs/results travel back through `retrieve_list`.
- **A duplicate central-DB upload is not an error, at either client.** The
  REST API responds `409 Conflict` to a duplicate `time_series_hash`, and
  `FrictionHTTPClient.upload_result`/`upload_record` (the `mode='api'`
  client — see [commands.md](commands.md)) return `None` for it rather than
  raising, mirroring the direct-`psycopg2` client's own `Optional[int]`
  return for the same case. `db upload`/`db stage` report it as
  `skipped (already in central DB)`, not a failure.

## Related docs

- [commands.md](commands.md)
- [settings.md](settings.md)
- [hpc_jobs.md](hpc_jobs.md)
- [provenance.md](provenance.md)
