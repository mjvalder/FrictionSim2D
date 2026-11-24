# AiiDA provenance store

FrictionSim2D uses [AiiDA](https://www.aiida.net/) as its **local** provenance
store and its only database. AiiDA never builds or runs a simulation: the
builders write the LAMMPS decks and job scripts, the jobs run outside AiiDA
(locally or on an HPC), and `FrictionSim2D aiida import` records the finished
run directory afterwards. Results are shared as `.aiida` archives, with no
server in between.

What the store gives you:

- a graph showing which inputs produced which result, down to the number on a
  plot;
- shared input nodes, so runs that use the same material, potential file or
  thermostat setup are connected, and "the runs that differ only in X" is one
  query (`related()`);
- archives that open in plain aiida-core, with JSON sidecars that need no
  AiiDA at all (Materials Cloud Archive is the publication target).

## 1. One-time setup

```bash
FrictionSim2D aiida status                     # is aiida-core installed, which profile is active
FrictionSim2D aiida setup                      # create (or load) the profile 'friction2d'
FrictionSim2D aiida setup -p myprofile         # another profile name
FrictionSim2D aiida setup --postgres           # try PostgreSQL first (needs a server)
FrictionSim2D aiida preflight                  # profile + entry points healthy? exit 0 only if so
```

- `aiida setup` creates the **profile only**, on **SQLite** by default
  (`verdi presto`). With `--postgres` it runs `verdi presto --use-postgres`
  first and falls back to SQLite when PostgreSQL is not available.
- No RabbitMQ broker and no daemon are needed: every import runs in-process
  (`engine.run`). The test tier runs against a broker-less SQLite profile.
- The machine `Computer` and the LAMMPS `Code` of each run are created when the
  run is imported (§2.4), not at setup.
- `setup` spells `-p/--create-profile` because it *creates* a profile. Every
  other command spells `--profile`, which *loads* an existing one (default:
  `settings.aiida.profile`, else AiiDA's own default profile).

The workflows the store supports:

| workflow | steps |
|---|---|
| **local** | `FrictionSim2D run ...`, run LAMMPS, `FrictionSim2D aiida import ROOT` |
| **remote-manual** | `run` + `hpc generate`, rsync the tree to the cluster, submit the job scripts there, rsync the results back, `aiida import ROOT` locally. Nothing AiiDA-specific runs on the cluster. See [hpc_jobs.md](hpc_jobs.md). |
| **hpc-direct** | install FrictionSim2D + aiida-core in the HPC's conda env, `aiida setup` and `aiida import` there, `aiida export LABEL` to an archive, `aiida import-archive` on your workstation |

There is no AiiDA submission. `aiida submit`, `run --aiida` and `run_with_aiida`
were removed.

## 2. The graph

### 2.1 One CalcJob per LAMMPS deck

Each LAMMPS deck of a run is recorded as one `LammpsFrictionCalcJob`
(entry point `friction2d.lammps`):

- a sheet-on-sheet run has one deck;
- an AFM run has a `system` deck plus slide decks: one `slide_f<F>N` deck per
  load in array mode, or a single `slide` deck in combined mode;
- local runs follow the same deck pattern.

A deck's log is the first of `lammps/run_<deck>.log` (array mode),
`lammps/run_system.log` / `lammps/run_slide.log` (combined mode),
`results/log_<deck minus 'slide_'>.lammps` (the MLIP sheet runs:
`slide_p1gpa` -> `results/log_p1gpa.lammps`), and the single root
`log*.lammps` of a one-deck local run.

The CalcJob is launched in AiiDA's *import mode*: `engine.run` with a
`remote_folder` pointing at the run directory and no upload or submit. aiida-core
only retrieves and parses. Retrieval copies into a temporary sandbox, so the
run directory is never written. Its `process_state` and exit status describe
the parse, not the LAMMPS run:

| exit code | meaning |
|---|---|
| 320 `ERROR_NO_RESULTS` | the deck produced neither load points nor structures |
| 330 `ERROR_PARSING_FAILED` | the parser raised |
| 410 `ERROR_LAMMPS_NOT_FINISHED` | the log has no `Total wall time` line |

**Inputs** of a deck CalcJob:

| link label | node | content |
|---|---|---|
| `deck` | `SinglefileData` | `lammps/<deck>.in` |
| `includes__*` | `SinglefileData` | files the deck includes (`lammps/system.in.settings`) |
| `provenance` | `FolderData` | the run's `provenance/` directory plus `provenance.json`, credential-redacted |
| `blocks__<label>` | `Dict` | the parameter blocks (§3) |
| `derived` | `Dict` | builder-computed values: `drive_k`, relaxed `lat_c`, `damp_ev`, ... Not hashed. |
| `schema` | `Dict` | the block schema: kind, unit and default of every field |
| `potentials__*` | `SinglefileData` | the potential files the run used |
| `structures__*` | `SinglefileData` | `build/*.lmp` for the first deck. For a slide deck, the very `load_*.data` nodes the system deck output for its loads: the system → slide handoff is a link in the graph. |
| `parameters` | `Dict` | `deck_name`, `rel_path`, `run_uuid`, `point_filter`, `family`, `n_layers`, `log_name`, `writes_structures`, `block_names`, `build_level` |
| `code` | `InstalledCode` | the LAMMPS Code of (machine, version); absent when the version is unknown (§2.4) |
| `remote_folder` | `RemoteData` | the run directory, on the machine `Computer` the run was produced on |

**Outputs** of a deck CalcJob:

| link label | node | content |
|---|---|---|
| `points__<key>` | `FolderData` | one load point: its window and raw tiers (and the indent retract curve). `<key>` is `<load kind><load>_a<angle>_s<speed>`, e.g. `force10_a0_s2` (`m` = minus, `p` = decimal point). |
| `snapshots__<key>` | `SinglefileData` | the point's end snapshot, `final_*.data` |
| `peratom__<key>` | `FolderData` | the point's per-atom tier |
| `structures__*` | `SinglefileData` | `load_*.data` written by a system deck |
| `aux__*` | `SinglefileData` | other retained files of a system deck (`hold_*.txt`) |
| `run_info` | `Dict` | `lammps_version`, `wall_time_s`, `n_procs`, `log_missing`, `log_has_version`, `not_stored` |
| `log` | `SinglefileData` | the deck's LAMMPS log |

A point carries the attributes `family`, `load_kind`, `load`, `angle`,
`speed`, `layers`, `status`, `units`, `point_uuid`, `rel_path`, `deck_name`,
`has_snapshot` and `has_peratom`. A deck's `point_filter` comes from its stem
(`slide_f30N` covers load 30). The configured points are read from
`provenance/config.json`, so a configured point that wrote nothing is an empty
`FolderData` with `status='no_output'`, never a silent gap.

**Steps on the points** are calcfunctions, so the provenance runs from a point
to every number derived from it:

```
point + snapshot + bond reference + context + thresholds ─► check_integrity ─► verdict Dict
verdict + ruling Dict ────────────────────────────────────► apply_ruling    ─► ruled verdict Dict
ruled verdict + verdict ──────────────────────────────────► withdraw_ruling ─► withdrawn verdict Dict
point + options (+ snapshot, provenance) ─────────────────► summarise       ─► summary Dict (with units)
```

A dataset is not a calcfunction: it is a Group of the selected summaries plus
one table `Dict` (§6).

**Groups.** Each campaign is the Group `friction2d/<label>`, with extras
`source_root`, `source_host`, `git_commit`, `git_dirty`, `model`, `materials`
and `created`. Each dataset is the Group `friction2d/datasets/<name>`.

### 2.2 What is stored and what is not

One rule, `frictionsim2d/aiida/layout.py`, decides which files of a run are kept:
`provenance.json`, `provenance/**`, `lammps/*.in`, `lammps/system.in.settings`,
`lammps/run_*.log`, `results/*`, `data/hold_*.txt`, `data/final_*.data`,
`data/load_*.data` and `build/*.lmp`.

- Root-level `log*.lammps` and `*.out` (from a multi-deck local run) also match
  the retention rule. They are kept on disk but are **not** stored in the
  CalcJob yet.

- `visuals/` and every `*.lammpstrj` trajectory are **never** stored. They stay
  on disk for visual checks only.
- A file over `aiida.max_stored_file_mb` is recorded in `run_info.not_stored`
  by path, size and sha256 instead of being stored.
- Syncthing `*.sync-conflict-*` copies are skipped; the original file is kept.

### 2.3 Credentials never enter the store

Every file staged into a node goes through `frictionsim2d.aiida.data.redact.stage_file`.

- A settings or config document (`settings.yaml`, `config.json`, `*.ini`, any
  of those with an extra suffix, any file under `provenance/` outside `cif/`
  and `potentials/`) is parsed and re-serialised with every credential-named
  key blanked: `password`, `passwd`, `api_key`, `apikey`, `secret`, `token`,
  or a name ending in `_password`, `_secret`, `_token` or `_api_key`
  (case-insensitive), wherever the key sits.
- A candidate document that cannot be parsed is refused
  (`UnparsableDocumentError`), never stored unexamined.
- This is the second of two gates. The first blanks the same keys when the
  run tree is written (`redact_secrets`, see [settings.md](settings.md)).

Content-addressed file hashes are taken over the **stored** (redacted) bytes,
so a published hash can be checked against the published content.

### 2.4 Machine and LAMMPS version: Computer and Code

- **Computer.** One per machine label, created at first import and left
  unconfigured: it records where a run came from and never executes anything.
  The label is `aiida import --machine`, else `settings.aiida.machine`, else the
  run's `settings.hpc.hpc_host`, else `localhost`. Its hostname and scheduler
  come from the run's `settings.hpc`. The run's settings are
  `provenance/settings.yaml`, else the `settings` section of
  `provenance/config.json`, the same fallback the parameter blocks use.
- **Code.** One `InstalledCode` per (machine, LAMMPS version), labelled
  `lammps-<version>@<machine>`. The version comes from the deck's log, falling
  back to `provenance.json`. The executable is the `lmp*` token of the run's job
  script, else `lmp`.
- **The no-version rule.** A deck whose LAMMPS version is unknown gets **no**
  Code, so an "unknown" Code never merges unrelated binaries. The import report
  lists each such deck under `no LAMMPS version (no Code)`.
- `metadata.options.resources` come from the process count in the log;
  `max_wallclock_seconds` from `settings.hpc.walltime_hours`.
- Known limitation: Computers and Codes are found by label. Two profiles that
  each created `lammps-<version>@archer2` hold two different nodes, and an
  archive import adds the second one beside the first.

## 3. Parameter blocks

A run's `config.json`, `settings.yaml` and `provenance.json` are split into
named parameter blocks, each stored once as an `orm.Dict`. Every run with an
identical block links to the same node, which is what connects runs across
campaigns.

| block | one per | holds |
|---|---|---|
| `material__<role>_<i>` | material entry (sheet, tip, substrate, flake) | material name, CIF sha256 |
| `potential__<role>_<i>` | material entry | potential type and potential-file sha256 (what the build resolved) |
| `interaction` | run | interlayer style and file sha256, `interlayer_g`, `lj_override`, `potential.*` settings |
| `thermostat` | run | temperature, thermostat type and boundaries, tip `dspring`/`damping_ratio` |
| `integration` | run | timestep, minimiser, neighbour list, `slide_run_steps`, hold and ramp steps |
| `drive` | run | drive method, constraint mode, springs, tip position |
| `geometry` | run | sheet size, layers, stacking, tip radius, substrate, flake, lattice matching, twist, finite sheet, the amorphous quench recipe |
| `protocol` | run | PES and indent protocol, halt factor, interlayer comparison mode |
| `output` | run | dumps, frequencies, channels, `thermo` |
| `realization` | run | the RNG seeds (`seed_thermal_tip`, `seed_thermal_sub`, `seed_velocity`, `seed_sheet_center`, the indent `seed`) and `substrate_realization_id` |

The full field table, units and defaults are in `frictionsim2d/aiida/block_schema.py`
(`FIELD_MAP`, `UNITS`). How they are derived from a run is in
[provenance.md](provenance.md#from-provenance-to-blocks).

**Block rules:**

- **Requested values only.** A block holds what the config asked for, coerced
  to the model's type (`300` and `300.0` alike for a float field).
- **Swept axes are not block content.** Load, angle and speed are point
  attributes, so a load sweep does not put every deck on its own `drive`
  block. The deck shape (`outer_loop`) is not identity either.
- **Builder-computed values go to `derived`**, an unhashed input: the
  relaxed `lat_c`, `drive_k`, `damp_ev`, the built geometry, the interlayer
  details.
- **Defaults are elided.** A field equal to its family default is left out, so
  an absent field and its default hash alike, and a field added later changes
  no existing hash. Readers refill elided fields from the run's own `schema`
  node, never from today's defaults.
- **Files enter by sha256, never by path.** An unresolvable file is recorded
  as `unresolved:<basename>`, with a warning.
- **The hash covers kind and content, not the role.** Si used as a tip and Si
  used as a substrate share one material node.
- **Coverage is enforced.** A test fails when a config or settings model gains
  a field that is neither assigned to a block nor listed in `EXCLUDED`
  (`hpc.*`, `aiida.*`, `integrity.*`, `quench.run_local`/`n_procs`/
  `allow_generate`, `materials_list`, `outer_loop`, `scan_angle_force`, and the
  record's own identity fields).
- **Every numeric field has a unit**, ASCII, metal units: lengths in
  `angstrom`, springs in the drive block in `N/m`, `drive_k` in
  `eV/angstrom^2`, temperatures in `K`. A unit change is a `block_schema`
  version bump with a converter, never silent.

`kind` and `block_schema` sit inside each block's content, so a block's hash
can be recomputed from its stored attributes alone. Content-addressed nodes
(blocks, files, provenance folders, schemas) also get a deterministic UUID
from their hash, so the same block built by two people in two profiles is one
node after an archive import.

**Unit check at import.** The import refuses a run whose `derived.drive_k` is
not its requested `driving_spring` (N/m) divided by the exact factor
`NM_TO_EV_A2 = 16.02176634` (a spring recorded in N/m, or converted twice).
AFM builds before provenance schema 2.2.0, and stamped legacy runs, divided by
a rounded `16.02`; for those the check also accepts that factor.

## 4. Identity and import

```bash
FrictionSim2D aiida import ./simulation_20260421_143404 --label 260421-afm --machine archer2
FrictionSim2D aiida import ./simulation_20260421_143404 --dry-run
```

Options: `-l/--label` (default: the folder name), `--machine`, `--profile`,
`--settings-file` (whose `aiida.machine`, `aiida.profile` and `integrity`
thresholds apply), `--dry-run` (classify every deck against the store; create
nothing).

### 4.1 Identity never comes from a path or a label

- **A run** is its `provenance.json` `run_uuid`, a uuid4 drawn at build time
  (provenance schema 2.2.0). A run without one is refused with `MissingRunUUID`
  and reported as a failure; the other runs go on.
- **A deck** is `<run_uuid>|<deck_name>` (the CalcJob extra
  `friction2d_deck_key`).
- **A point** is `point_uuid = uuid5(run_uuid, <key>)`, e.g.
  `uuid5(run_uuid, 'force10_a0_s2')`. It is path-free: the same point of the
  same run has the same uuid wherever the run directory sits and under whatever
  label it is imported.
- **The label only names the Group.** The same runs rsynced to a new path
  import into their set as before (`source_root` is updated). A label that
  already names a non-empty set holding none of the tree's runs is refused
  (`SetRootMismatch`): pass another `--label`, or delete the old set.

### 4.2 Legacy runs: stamp, then import

Runs built before provenance schema 2.2.0 carry no `run_uuid`. Give them one,
once, before their first import.
Stamp them with the standalone `stamp_uuid.py` kept beside the data (it is not part of
FrictionSim2D).

- Each `provenance.json` without a `run_uuid` gets a fresh uuid4 (each AFM
  `L{n}` record its own) and `run_uuid_stamped_at` (the ISO date). Every other
  key keeps its value.
- Stamping rewrites `provenance.json`, and the provenance is part of each
  deck's fingerprint. Stamp a copy of a tree, and keep using that stamped copy:
  a second, separately stamped copy of the same runs gets new `run_uuid`
  values, so it imports as different runs.

### 4.3 Re-import and supersession

Each imported deck carries a fingerprint, the extra `friction2d_deck_sha256`: a
sha256 over its deck file, includes, log (or its absence), retained output
files, the run's provenance and every structure handed to it. It is path-free,
so a moved tree has the same fingerprints.

- **Unchanged deck** → skipped. If it was imported under another label, it is
  added to this Group too (`already imported`).
- **Changed deck** (a log that arrived later, a re-run, `hpc enrich-provenance`
  rewriting `provenance.json`) → a new CalcJob. The old CalcJob, its outputs and
  every calcfunction result derived from them get the extras `superseded=True`
  and `superseded_at`.
- Nothing is deleted or overwritten. Queries, comparisons, exports and the
  integrity pass skip superseded nodes unless asked for them.
- **Supersession is local.** It is an index in this profile's extras, not
  provenance. Supersession across archives from different profiles is not
  tracked.
- A superseded CalcJob that another set also holds is reported as
  `superseded <deck>, also held by set '<other>'`.

### 4.4 The config must match `config_hash`

`provenance.json` records `config_hash`, the sha256 of the build's compact
`config.json` (`model_dump_json()`). Every import, dry runs included, checks
it before anything is stored. An AFM run's `provenance/` is shared by its
`L{n}` layers, so a one-layer rerun can overwrite the `config.json` the other
layer ran with.

- `provenance/config.json` matches: imported as before.
- It does not: the importer takes the first of these that matches:
  - `provenance/config.sync-conflict-*.json` (Syncthing copies);
  - `<run parent>/_v3_crashed/provenance/config.json` (for an AFM run, the pot
    level).

  That file is stored as `provenance/config.json` in the run's provenance
  folder, and so drives the blocks, the parser and the integrity check. The
  report warns:
  `WARNING <run>: provenance/config.json does not match the config_hash of provenance.json; using <file>, which does`.
- None matches: the run goes to `failures` as
  `ConfigMismatch: provenance/config.json does not match the config_hash ...`,
  naming every file tried, and the other runs go on. Repair the run (restore
  its config) before importing it.
- A record without `config_hash`, or a run with no `config.json` at all, is
  not checked.

### 4.5 The import report

The import report counts decks created, skipped, superseded and already
imported, points created, decks with a missing log, decks with no LAMMPS
version, unparseable runs, runs with no decks, and the integrity verdicts and
rulings of §5. It exits 1 if any run failed; a real (non-dry-run) import also
exits 1 if any path could not be parsed.

- **A run with no decks** (no `lammps/*.in`, e.g. a results-only copy) has
  nothing to import: no CalcJob, so no point either. It is not a failure, but
  it is never silent: the report counts it (`runs with no decks:`) and names
  it (`no decks (nothing imported): <run>`).

## 5. Integrity verdicts and rulings

### 5.1 A verdict for every point, at import

Every import (not a `--dry-run`) ends with an integrity pass over the whole set:

- every live point with no live verdict gets one from `check_integrity`, the
  ported checker of [integrity.md](integrity.md), with the thresholds of the
  `integrity` block of `--settings-file` (else the per-user settings);
- `no_output` and `not_reached` points get a verdict too, so "no verdict"
  never means "not looked at";
- a point whose stage-2 context changed (a later import changed its status,
  layers, snapshot or reference) is re-checked; its old verdict and that
  verdict's rulings are superseded, and the report lists
  `re-checked <point>: <old> -> <new>`;
- a point whose stage-2 status could not be computed is checked
  *provisionally* (extra `friction2d_provisional`) and re-checked on the next
  import.

An import does **not** re-check points after the integrity settings change.
Edit the `integrity:` block (bump its `version` when thresholds change), then run:

```bash
FrictionSim2D aiida recheck 260421-afm --settings-file ./settings.yaml
```

- `aiida recheck` re-checks every live point whose live verdict has another
  `thresholds_version`, or whose stored settings differ from the current ones.
- The old verdict is superseded with its rulings, exactly as for a context
  change, and the set's own rulings file is applied again to the new verdicts.
- It lists `re-checked <point>: <old> -> <new>` for every status that changed.
  A second run checks nothing.
- A ruling applied from another file (`--rulings`) is re-applied from that
  file as well, or kept as pending when it cannot be (§5.3).

A verdict holds `status`, `failed`, `observed`, `reasons`, `quarantine`,
`not_applicable`, `measures`, `thresholds_version`, `code_version` and
`point_uuid`.

The store applies the rules of [integrity.md](integrity.md) unchanged. That
includes the 2-layer case: a bilayer sheet-on-sheet point (hetero, flake,
sheetonsheet) gets **no `gap_rise`** (it needs three or more layers) and **no
heating check** (`n_heat = 0`; only the centre layers are counted). Its verdict
rests on the gate, `collapse` and the snapshot rules. Both rules show up in
`not_applicable`.

### 5.2 Rulings: `integrity_rulings.yaml`

Rulings live in `integrity_rulings.yaml` in the simulation root, in the format
of [integrity.md](integrity.md#quarantine-and-rulings). The point id is
`<rel_path>|<load_kind>=<load>|angle=<a>|speed=<s>`.

```bash
FrictionSim2D aiida rule 260421-afm                                  # the set's own file
FrictionSim2D aiida rule 260421-afm --rulings ~/reviews/260421.yaml  # another file
FrictionSim2D aiida rule 260421-afm --rulings ~/reviews/             # a directory
```

- Every `aiida import` also applies the set's own file, after the integrity
  pass. `aiida rule` applies it (or another file) on demand.
- Each (verdict, ruling) pair becomes one `apply_ruling` calcfunction. The
  ruling `Dict` carries the extra `friction2d_author='user'`: a human decided
  it, not the checker. It also records the file it came from: `source` (the
  resolved path; its basename in an anonymised archive) and `own_file` (whether
  that is the set's own `integrity_rulings.yaml`).
- Idempotent: a ruling already applied to a verdict is skipped (the extra
  `friction2d_ruling_key = <verdict uuid>|<sha256 of the ruling>`). Editing a
  ruling's note or date applies it anew, superseding the previous ruled
  verdict.
- A quarantined point takes `pass` or `fail`.

### 5.3 Expert overrides of any verdict

You can overrule **any** verdict, not only a quarantined one. Such a ruling
needs `override: true` **and** a non-empty `note`:

```yaml
- point: "afm/h_MoS2/.../L1|force=30|angle=0|speed=2"
  ruling: pass
  override: true
  note: the ejected atoms are a known thermostat artefact
  date: 2026-11-16
```

- Without `override: true`, or with an empty note, the ruling changes nothing,
  and the report lists the point and warns. `aiida rule` prints
  `not quarantined: <point>`. The import report prints
  `ruling on a point not quarantined (needs override: true and a note): <point>`.
- What `pass` and `fail` do to each status, and that `pass` is refused for a
  point with no data, is the rule of [integrity.md](integrity.md#overriding-any-verdict).
- **Overrides are always listed.** Every import report and every `aiida rule`
  report prints each live override of the set:
  `override <point>: <original> -> <new> (<note>)`. The ruled verdict carries
  `overridden: true` and `original_status`.
- **Withdrawing.** Remove the entry (or its `override: true`, or its note)
  and re-apply the file (`aiida rule LABEL`, or the next import). The ruling is
  withdrawn: `withdraw_ruling` puts the checker's verdict back in force,
  stamped `withdrawn_ruling=<ruled uuid>`. `aiida rule` prints
  `withdrawn: <point>`; the import report prints `ruling withdrawn: <point>`.
  A ruling naming no point of the set is printed as `unmatched: <point>`
  (`ruling unmatched: <point>` in the import report).
- **Re-checks re-apply rulings.** When a later import or `aiida recheck`
  re-checks a point, the old ruled verdict is superseded with the old checker
  verdict, and the rulings pass that follows applies the point's ruling to the
  new verdict, whatever file it came from. A re-import that supersedes a
  changed deck does the same: rulings are kept by point id, not by node. If a
  re-check moves a ruled point out of quarantine, its ruling (without
  `override: true`) no longer applies, and the report warns that it now needs
  `override: true` and a note.

#### The ruling ledger: an override is never silently lost or flipped

Every set keeps a **ruling ledger** (Group extra `friction2d_rulings_ledger`):
every ruling of every rulings file, by point id and file, including the ones
not in force. Each entry records the ruling, its note and date, its file
(`source`), when it was **seen** (first added, or its decision last changed),
whether the file is `present`, `missing` or `archived`, and which entry is in
force. A set ruled before the ledger existed gets one at its next pass, built
from its live rulings.

**Which files are read.** Every import, `aiida rule` and `aiida recheck`
reads:

- the set's own `integrity_rulings.yaml`;
- the `--rulings` file of `aiida rule`, if given;
- every other file the ledger already holds that still exists.

**What reading a file does.**

- A file that parses replaces its entries. A point it no longer rules loses
  that file's ruling: removing an entry is how a file withdraws a ruling, for
  the own file and any other alike. The report prints
  `ruling retracted by its file: <point> (<file>)`.
- A malformed entry is skipped with a warning. The file's other entries still
  count, but nothing of that file is withdrawn on that pass
  (`WARNING no ruling of <file> withdrawn: it has entries that cannot be read`).
- A file that cannot be read at all changes nothing, with a warning. So does a
  file that is empty (0 bytes, blank, or only comments), as an editor may
  leave it: write `[]` to withdraw every ruling of a file.
- A file that is **gone** keeps its entries, marked `missing`. They **still
  count**: an unmounted drive never costs you an override. Every report lists
  them as pending:
  `ruling pending: <file> missing: <point>; it still counts. Restore it with 'FrictionSim2D aiida rule LABEL --rulings <file>', or drop it with '--withdraw-pending <file>'`.
  When the file comes back, the next pass reads it again by itself.
- **An unreachable root withdraws nothing.** If the set's `source_root`
  directory is missing, the own file counts as gone: its rulings stay in
  force, are listed as pending, and the report warns
  `WARNING no ruling withdrawn: <file> is not reachable (its directory <root> is missing)`.
  A missing file inside an existing root does mean "no rulings".
- **A moved root.** When a set is re-imported from another root, the rulings
  the own file made at the old root rank as another file's from then on, at
  their old path (`missing` if it is gone). The new own file overrides them
  only where it rules a point. The report warns once:
  `WARNING former own rulings file <path> now ranks as another file: ...`.
  A dry run judges this against the root being imported.
- **Archived rulings.** A set loaded from an anonymised archive has basenames
  in its ledger (`x.yaml`, `x.yaml (2)`). They are never read from disk: they
  stay in force, marked `archived`, and the first pass prints
  `ruling archived: <point> (<file>): it came with an archive and still counts; no file is read for it`.
  They are not pending and carry no restore hint.
- `aiida rule LABEL --withdraw-pending FILE` deletes every entry of FILE from
  the ledger (a directory means `DIR/integrity_rulings.yaml`; the file need
  not exist). The report prints `pending ruling withdrawn: <point> (<file>)`.
  The own file cannot be dropped this way: edit it instead.

**Which ruling is in force.** For each point:

1. the set's own file's ruling, if it rules the point;
2. else the **most recently seen** ruling of another file. Re-reading an
   unchanged file does not make it newer, and neither does editing an entry's
   note or date; changing its `ruling` or `override` does.

A ruling that does not apply to the point's verdict (no `override: true` on a
point that is not quarantined) is skipped and the next one counts, with
`WARNING integrity ruling ignored: <point> (<file>): <why>`. A lower-ranked
ruling that does not apply while another is in force is not reported.

**Every change of decision is printed**, whatever caused it:
`ruling changed: <point> <old> -> <new> (<file>, <why>)`, where a decision is
`pass`, `fail` or `none`. Every pass
makes the store match: it applies the ruling, replaces another one, or
withdraws one no file makes any more. A second identical pass makes nothing.
The reports count `applied`, `replaced`, `skipped` and `withdrawn`.

**Conflicts are always listed.** When two or more files rule a point with
different decisions, every report prints one line naming the ruling in force
and why:
`conflict <point>: <file> rules fail (in force: the set's own integrity_rulings.yaml wins); <other> rules pass (not applied)`
(`ruling conflict ...` in the import report; `most recently seen (<time>)`
when no own ruling decides). Files that agree are not a conflict, but both
stay in the ledger: if the own file drops its ruling, the other file's comes
back.

`aiida import --dry-run` on an existing set prints the rulings pass it would
make (counts, conflicts, pending) and writes nothing.

In an anonymised archive every file path in the ledger is its basename.

### 5.4 Which verdict counts: the policy

A point can hold several live verdicts (checker, ruled, withdrawn, and those of
a newer threshold set). `frictionsim2d/aiida/rulings.py` picks one by **declared version,
never by creation time**:

1. the highest `thresholds_version`, then the highest `code_version`: a newer
   generation wins over every verdict of an older one, a ruled one included;
2. within one generation: ruled > withdrawn > the checker's own;
3. then the uuid, so the choice is reproducible.

A new generation comes from `aiida recheck` (§5.1). A ruling does not carry
over to it by itself: the rulings pass that `recheck` runs re-applies it to the
new generation's verdict. A `Policy` can also pin an exact
`thresholds_version` (verdicts) or `code_version` (summaries) instead of
`'newest'`.

## 6. Postprocessing, datasets and queries

```bash
FrictionSim2D aiida postprocess 260421-afm
FrictionSim2D aiida dataset mos2-loads 260421-afm 260512-afm
FrictionSim2D aiida query --set-label 260421-afm --format csv -o points.csv
FrictionSim2D aiida list-sets
```

- **`aiida postprocess LABEL`** runs `summarise` on every live point whose
  selected verdict is `complete` (checked, ruled or overridden so). A summary
  is `RunSummary` (mean/std COF, lateral and normal force, ...) plus `units`,
  `options` (`skip_fraction`, from `settings.aiida.skip_fraction`),
  `options_hash`, `code_version` and `point_uuid`. Its identity is (point,
  options hash, code version): a second pass makes none; new options or a new
  code version make new summaries beside the old ones. It warns about points
  summarised on a provisional verdict and about verdicts from an older
  threshold set. It does not re-check them: run `aiida recheck LABEL` first
  (§5.1).
- **`aiida dataset NAME LABEL...`** makes the Group `friction2d/datasets/NAME`
  from one or more sets: for every live point whose selected verdict is
  `complete`, the summary the policy selects, plus one table `Dict` holding
  `name`, `labels`, the `policy` (with the summary options and their hash),
  `units` per family, `columns` and one row per point (set, family,
  `point_uuid`, `calc_uuid`, `summary_uuid`, ...). A figure built from it names
  the verdict and summary generation it used. Datasets are never rewritten: an
  existing NAME is refused. Plot one with the plot source
  `aiida://dataset/<name>` ([postprocessing.md](postprocessing.md)).
- **`aiida query`** prints one row per live point: the point, the verdict the
  policy selects (`status`, `reasons`, `verdict_kind`, `provisional`,
  `thresholds_version`), the run tokens, identity columns and the selected
  summary. Filters are in [commands.md](commands.md#aiida-query).

### 6.1 `related()`: the runs that differ only in X

From Python (`frictionsim2d.aiida.query`):

```python
from frictionsim2d.aiida import load_aiida_profile
from frictionsim2d.aiida.query import point_table, related

load_aiida_profile()
table = point_table()                                   # one row per live point
others = related(point_uuid, 'thermostat', table=table) # differ only in the thermostat block
loads = related(point_uuid, 'load', table=table)        # the same physics at other loads
dt = related(point_uuid, 'settings.simulation.timestep', table=table)
reps = related(point_uuid, 'realization', table=table)  # pure replicates: seeds only
```

`point_table()` has one row per point and one column per requested field
(`<block name>.<field path>`, e.g. `integration.settings.simulation.timestep`,
`material__sheet_0.name`), every elided field refilled from the run's own
schema, plus the swept axes `load`, `angle`, `speed`, their units and the
identity columns. It is built from bulk projections, not a query per point.

`related(point, vary)` returns the current points equal to `point` everywhere
except along `vary`:

- **Point level and field level.** `vary` is a swept axis (`'load'`,
  `'angle'`, `'speed'`), a block kind (`'thermostat'`), a block name
  (`'material__sheet_0'`), a field path (`'settings.simulation.timestep'`, bare
  or with its block), `'realization'`, or a list of these. Identity columns
  (`family`, `rel_path`, ...) cannot be varied.
- **Tolerance.** Floats are equal when they agree to 6 significant figures
  (`0.1 + 0.2` and `0.3` match; `1.0` and `1.001` do not). Integers and bools
  compare exactly, so seeds and step counts never round. Swept axes compare to
  9 decimals. This is the same tolerance `compare` uses for block fields, so
  the two never disagree on whether a value changed.
- **Units.** A field stored in another unit by the other run's schema is never
  equal, and the swept axes' units always have to match (a load in nN and one
  in GPa are not two values of one quantity).
- **Seeds and realization.** The `realization` block (seeds) is ignored: seeds
  differ between every pair of runs, so comparing them would relate nothing. A
  *replicate*, equal on every field but the seeds, is returned only when
  `'realization'` is in `vary`.
- **Deck shape and groups do not matter.** An array-mode point and a
  combined-mode point of the same physics relate, across campaigns.
- The point itself is never returned; results are ordered by `rel_path`,
  `load`, `angle`, `speed`.

### 6.2 `compare`: two runs or campaigns, field by field

```bash
FrictionSim2D compare 260421-afm 260512-afm               # two sets
FrictionSim2D compare point:<point_uuid> pk:1234          # a point against a CalcJob
FrictionSim2D compare sidecar:~/dl/260421-afm.aiida.params 260512-afm
```

A and B are each a CalcJob pk or uuid, a `point_uuid`, a set label, or an
archive's `.params` sidecar. A prefix says which: `pk:`, `uuid:`, `point:`,
`group:` or `sidecar:`.

- **`sidecar:`** takes the `<archive>.params/` directory, a directory holding
  one, or one deck's JSON in it. Without a prefix, a directory is read as a
  sidecar only when it is, or holds, a `.params` directory with its
  `schema-*.json`. Anything else is looked up in the store, so a local folder
  that shares a set's name never shadows the set. An unprefixed all-digit
  reference is a pk (else a group label).
- **Field by field, with units.** The report lists, per block, each differing
  field as `path: a -> b <unit>`, the blocks present on one side only, and the
  derived values that differ.
- **Units and defaults per side.** Each side is read in its own schema's units
  and defaults (the CalcJob's `schema` node, or the sidecar's
  `schema-<version>.json`), so a comparison reads the units a run was stored
  with, not today's. An elided field is refilled from its own side's default.
  A field the two sides store in different units is listed as a unit
  difference and never compared.
- **Tolerance.** Block fields: floats to 6 significant figures, integers
  exactly (as in `related()`). Derived values are equal when
  `|a - b| <= max(atol, rtol * max(|a|, |b|))`, or when both are floats with
  the same significant-figure key (6 figures at the default `rtol`, the key of
  `related()`). `--rtol` (default `1e-6`) and `--atol` (default `1e-9`) apply to
  derived values only.
- **Seeds.** Unlike `related()`, `compare` shows the `realization` block like
  any other, so two runs that differ only in their seeds are reported as such.
- **Campaigns.** A group gives the fields shared by all its current decks; a
  field that varies inside it shows as `<varies>`. The swept axes are point
  attributes, not blocks, so a load sweep does not show as a difference.

## 7. Maintenance: `reindex` and `verify`

Identity lookups go through node extras (`friction2d_block_hash`,
`friction2d_block_kind`, `friction2d_sha256`, `friction2d_folder_sha256`,
`friction2d_schema_hash`, and the CalcJob extras `friction2d_run_uuid`,
`friction2d_deck_key`, `friction2d_tokens`). Extras are an index, recomputable
from the nodes themselves.

```bash
FrictionSim2D aiida verify      # duplicate hashes, or hash extras that disagree with content; exit 1 if any
FrictionSim2D aiida reindex     # recompute every index extra from the stored nodes
```

Run `reindex` after an archive import that dropped or edited extras.
`point_table()` leaves out the points of a CalcJob whose blocks lack their hash
extra, with a warning that names `reindex`.

## 8. Getting data back out

```bash
FrictionSim2D aiida export-results 260421-afm ~/exported/260421-afm
FrictionSim2D aiida rebuild 260421-afm -o ~/rebuild_test --hpc-scripts
```

- **`export-results LABEL OUTPUT_DIR`** writes a set's live CalcJobs back as an
  ordinary simulation tree (`provenance/`, `lammps/`, `results/`, `data/`,
  logs). Halted points get their `results/halted.txt` line back and user
  rulings their `integrity_rulings.yaml` entry, so `postprocess read` gives the
  statuses it gave on the source tree.
- **`rebuild LABEL`** regenerates a set's *inputs*: it hands each run's stored
  `config.json` and `settings.yaml` to the builder router `FrictionSim2D run`
  uses. `--hpc-scripts` also writes the job scripts.

## 9. Archives

```bash
FrictionSim2D aiida export 260421-afm -o ~/transfer/260421-afm.aiida
FrictionSim2D aiida export 260421-afm -o 260421-afm.aiida --include-peratom
FrictionSim2D aiida export 260421-afm -o 260421-afm.aiida --keep-hpc-paths
FrictionSim2D aiida import-archive ~/transfer/260421-afm.aiida
```

`aiida export LABEL` writes three things:

- `<output>` — the `.aiida` archive: the set's Group, its CalcJobs and every
  calcfunction reached from their outputs (verdicts, rulings, summaries), the
  stored analysis results of every dataset built from the set (Groups
  `friction2d/analyses/<dataset>`, see
  [postprocessing.md](postprocessing.md#analyses)), plus what aiida's export
  closure adds (their inputs and outputs, so an analysis brings its
  `record_analysis` node and the dataset table `Dict` it read). Every node is a
  core AiiDA type, so the archive imports and its data loads **without
  FrictionSim2D installed**; only `calc.process_class` and `calc.res` need the
  plugin.
- `<output>.params/` — the JSON sidecars: `<rel_path>/<deck>.json` per live
  deck (`{block name: block content, 'derived': ...}`) and
  `schema-<version>.json`. `FrictionSim2D compare` reads them without AiiDA.
- `<output>.README.md` — the link-label conventions (`points__*`, `blocks__*`,
  ...), the schema nodes, the archive size and the personal data it carries.

**The integrity gate.** The export refuses while any live point of the set has
no live verdict (`UncheckedPointsError`, listing the points): import the set
again to check them, or pass `--force` to write anyway. It also refuses when the
closure would carry a CalcJob of another set (`ExportClosureError`).

**Per-atom data is left out** by default: the `peratom__*` nodes are dropped
from the archive. Pass `--include-peratom` to keep them.

**HPC details are anonymised** by default. The anonymisation rewrites the
written archive, never the store, which keeps everything:

- a Computer's hostname becomes its machine label, a Code's executable its
  basename;
- `RemoteData.remote_path` and each CalcJob's `remote_workdir` become
  `<machine label>:<rel_path>`; `hpc_run_path` leaves `parameters`;
- the set Group's `source_root` becomes its basename and `source_host` the
  machine label (or `unknown`);
- in the provenance documents, `hpc.hpc_home`, `scratch_dir`, `log_dir`,
  `conda_env` and `account` and `database.user`/`host` are dropped,
  `hpc.hpc_host` becomes the machine label, and every absolute path becomes its
  basename;
- node logs and process failure text are scrubbed of local paths;
- rewritten nodes keep their UUIDs and carry the extra
  `friction2d_anonymised`. Their `friction2d_folder_sha256` is recomputed from
  the rewritten content, so `aiida verify` agrees on the far side.

Pass `--keep-hpc-paths` to keep all of these as recorded.

**Attribution.** The AiiDA User (name and email) of whoever imported the runs
is kept on every node, for attribution, and the README says so.

**Datasets export separately.** A set archive never includes dataset Groups or
their tables. `frictionsim2d.aiida.export.export_dataset(name, path)` (Python only, no CLI
command yet) archives one dataset on its own, with the verdicts of every member
point, across every set it was built from; it is gated, anonymised, per-atom
free and documented the same way.

**`aiida import-archive PATH`** imports an archive into the current profile.
The set's Group comes across inside the archive, so no extra import group is
made. Content-addressed nodes that already exist (same UUID) are not
duplicated. Run `aiida verify` (and `aiida reindex` if needed) afterwards.

### 9.1 Publishing on Materials Cloud

Materials Cloud Archive (DOI, long-term hosting) is the publication target:

1. Import every run, make sure the import report shows no failures and every
   point has a verdict (the export refuses otherwise).
2. Review the overrides listed by `aiida rule LABEL`.
3. `aiida postprocess LABEL` and, for a figure, `aiida dataset NAME LABEL...`.
4. `aiida export LABEL -o LABEL.aiida` (anonymised; add `--include-peratom` only
   if the per-atom data is part of the record).
5. Upload `LABEL.aiida`, `LABEL.aiida.params/` and `LABEL.aiida.README.md`
   together. A reader can browse the graph after `verdi archive import`, or
   compare against their own runs with `FrictionSim2D compare` on the sidecars.

## 10. Deleting

```bash
FrictionSim2D aiida delete 260312-force_rebomos    # one set (irreversible)
FrictionSim2D aiida clear                          # every FrictionSim2D node (irreversible)
```

- `delete` removes the set's Group and every CalcJob of it that no other set
  holds, with that CalcJob's outputs and their calcfunction results. Shared
  inputs (blocks, files, provenance, schema, Codes, Computers) are never
  deleted. A set that a dataset draws on is refused whole.
- **Cascade.** AiiDA deletes everything downstream of a deleted node. A CalcJob
  whose deletion would cascade into a CalcJob another set holds is kept, with
  the warning
  `Keeping CalcJob <pk> of '<label>': deleting it would delete CalcJobs another set holds`.
- **Stranded copies.** If another set holds one of the deleted decks only as a
  superseded CalcJob, that set is left with no live copy of the deck. The
  delete warns:
  `Deleting set '<label>' removed the live CalcJobs of N deck(s) that set(s) '<other>' hold only as superseded copies. Recover by re-importing the tree under each surviving label (FrictionSim2D aiida import <root> --label <label>).`
- `clear` removes every friction2d CalcJob with its outputs, steps, summaries
  and datasets, the inputs they leave unused, and every `friction2d/*` Group.
  Computers are kept.
- Both prompt for confirmation unless `--yes` is passed.

## Related docs

- [commands.md](commands.md)
- [provenance.md](provenance.md)
- [integrity.md](integrity.md)
- [settings.md](settings.md)
- [hpc_jobs.md](hpc_jobs.md)
