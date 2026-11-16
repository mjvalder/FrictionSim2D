# Settings

Every `settings.yaml` section and its defaults.

Settings are global to a run and loaded through `load_settings()` in `src/core/config.py`. Per-run parameters live in `.ini` files instead — see [configuration.md](configuration.md).

## Where settings come from

`load_settings()` takes the **first readable** of these, and nothing else:

1. An explicit path, passed as `--settings-file` to `run`, `hpc generate` and
   `settings show`.
2. `$XDG_CONFIG_HOME/FrictionSim2D/settings.yaml` — in practice
   `~/.config/FrictionSim2D/settings.yaml`.

With neither present, the `GlobalSettings` defaults defined in code are used unchanged.

**A `settings.yaml` in your project root or working directory is not read.** This is the
one thing to know about this file, because `FrictionSim2D settings init` (without
`--global`) writes exactly such a file — deliberately, as a per-run template you then
pass with `--settings-file`. It has no effect on its own.

```bash
FrictionSim2D settings init            # -> ./settings.yaml, for use with --settings-file
FrictionSim2D settings init --global   # -> ~/.config/FrictionSim2D/settings.yaml, machine-wide
FrictionSim2D settings show --origin   # which file is actually in effect
```

Use `--global` for machine-level values (HPC host, queue names, database credentials) and
a per-run file for anything that belongs to one campaign.

For reproducibility, keep a copy of the exact settings file used in your simulation root.

## CLI helpers

```bash
FrictionSim2D settings show
FrictionSim2D settings init
FrictionSim2D settings reset
```

## Top-level sections

- `simulation`
- `thermostat`
- `geometry`
- `output`
- `potential`
- `quench`
- `hpc`
- `aiida`
- `database`
- `integrity`

## simulation

Defaults:

```yaml
simulation:
  timestep: 0.001
  thermo: 100000
  min_style: cg
  minimization_command: minimize 1e-4 1e-8 1000000 1000000
  neighbor_list: 0.3
  neigh_modify_command: neigh_modify every 1 delay 0 check yes
  slide_run_steps: 500000
  drive_method: virtual_atom
  constraint_mode: none
  halt_temperature_factor: 10.0
```

`halt_temperature_factor` is a physics guard, not a thermostat setting: every
template that integrates in time emits a `fix halt` that ends the run the first
time the global temperature crosses `factor x max([general] temp, 300 K)`.
LAMMPS already aborts on lost atoms; this catches the blow-up that keeps all its
atoms and would otherwise spend the whole step budget. The halt costs only the
point that blew up, since a blow-up is often specific to one load or pressure:

- **Slide** — that point writes no final state, and the script moves on to the
  next load, angle or speed.
- **AFM `system.in` ramp** — the ramp stops at the halted load, since every
  later load would ramp from the blown-up state. The loads already written
  still slide; `slide.in` skips any load with no `load_*.data`.
- Each halt appends a `# HALT stage=… load=… T=… step=…` line to the run's
  `results/halted.txt`. Postprocessing reads it and exports the point with
  status `halted`, with its partial frames dropped.

The HPC jobs run `slide.in` even when `system.in` failed (the two-phase
submission uses `afterany`), so a hard LAMMPS error mid-ramp also keeps the
loads written before it.

The 300 K floor is not cosmetic: a 1 K-target run legitimately reaches tens of
K once minimisation strain energy is released (13–119 K measured across the
LAMMPS test set), so a bare `factor x 1 K` threshold would abort healthy runs.
Set the factor to `0` to disable the guard.

`constraint_mode` options:

- `atom_bonds`
- `com_spring`
- `none`

For sheet-on-sheet with potentials that include internal interlayer interactions, use `constraint_mode: none`.

## thermostat

Defaults:

```yaml
thermostat:
  type: langevin
  time_int_type: nve
  langevin_boundaries:
    tip:
      fix: [3.0, 0.0]
      thermo: [6.0, 3.0]
    sub:
      fix: [0.0, 0.3]
      thermo: [0.3, 0.6]
```

## geometry

Defaults:

```yaml
geometry:
  tip_reduction_factor: 2.25
  rigid_tip: false
  lat_c_default: 6.0
```

## output

Defaults:

```yaml
output:
  dump:
    system_init: true
    slide: true
  dump_frequency:
    system_init: 10000
    slide: 10000
  results_frequency: 1000
  channels:
    window: true
    raw: true
    peratom: true
  optional_channels: []
```

### `output.channels`

Gates which observable sampling tiers get a `fix ave/time` in the generated
LAMMPS input. All three default to `true`, preserving previous behaviour.

| channel | what it samples | cost |
|---|---|---|
| `window` | time-averaged over `results_frequency` — the COF source, and the only tier postprocessing currently ingests | cheap; leave on |
| `raw` | instantaneous samples every 100 fs | runtime + disk |
| `peratom` | per-atom computes, including the neighbour-list-rebuilding `group/group` | the most expensive of the three |

Turning `peratom` off also drops its `group/group` computes, not just the
output file. Turn `raw` and `peratom` off when you are not analysing them —
they cost runtime and disk on every run. See `tests/builders/test_output_channels.py`.

### `output.optional_channels`

Names of opt-in observable channels to enable, beyond the default set.
Channels declared `opt_in=True` in the registry cost measurable runtime and
are emitted only when named here. A name that matches no registry channel is
a build error, not a silent no-op. Empty by default.

## potential

Defaults:

```yaml
potential:
  LJ_type: LJ_base
  LJ_cutoff: 11.0
  reaxff_safezone: 1.2
  reaxff_mincap: 50
```

## quench

Defaults:

```yaml
quench:
  run_local: true
  n_procs: 16
  quench_slab_dims: [200, 200, 50]
  quench_rate: 1e12
  quench_melt_temp: 2500.0
  quench_target_temp: 300.0
  timestep: 0.002
  melt_steps: 50000
  equilibrate_steps: 20000
```

Implementation note: model fields also support aliases such as `melt_temp` and `quench_temp`.

## hpc

Defaults:

```yaml
hpc:
  scheduler_type: pbs
  queue: null
  partition: null
  account: ""
  hpc_host: null
  hpc_home: null
  log_dir: null
  scratch_dir: $TMPDIR
  num_nodes: 1
  num_cpus: 32
  memory_gb: 62
  walltime_hours: 20
  max_array_size: 300
  modules: null
  conda_env: null        # e.g. $HOME/envs/lammps -- its bin/ and lib/ go on PATH
  mpi_command: mpirun
  use_tmpdir: true
  preflight: false       # also write hpc/preflight.sh + its submit wrapper
  lammps_scripts: [system.in, slide.in]
```

Job-script generation needs `modules`, `conda_env`, or both.

Job scripts -- PBS and SLURM alike -- load the environment, copy the simulation
to `scratch_dir` when `use_tmpdir` is on, run LAMMPS with
`-log lammps/run_<deck>.log`, copy `results/`, `visuals/`, `data/` and the run
logs back, and remove scratch. Use `queue` for PBS and `partition` for SLURM;
the other is ignored.

With `preflight` (or `hpc generate --preflight`) you also get a check-only
`preflight.sh` (run locally: `bash hpc/preflight.sh`) and a submit wrapper that
runs it on a compute node -- `preflight.pbs` (`qsub hpc/preflight.pbs`) or
`preflight.slurm` (`sbatch hpc/preflight.slurm`), from the simulation root.
Both verify modules/conda, `lmp -h`, array ranges vs manifests, every deck and
the files it reads, pair styles, and log/scratch directories.

For sheet-on-sheet runs, script ordering is adjusted automatically to slide scripts only when needed.

## aiida

Five fields, all reachable. The old SSH-connected HPC mode is gone: the HPC
blocks direct connections, so the supported remote workflows are
remote-manual (rsync out, `aiida import` on return) and hpc-direct (AiiDA
runs in the HPC's own conda env, archive brought home) — see
[aiida.md](aiida.md).

Defaults:

```yaml
aiida:
  profile: null                     # null = AiiDA's own default profile
  computer_label: localhost
  max_stored_file_mb: 20
  skip_fraction: 0.2
  machine: null                     # null = resolved at ingest
```

- `profile` — the AiiDA profile every profile-taking function falls back to
  when its own `profile` argument (Python) or `--profile` option (CLI) is
  omitted. `null` means "AiiDA's own default profile."
- `computer_label` — the AiiDA computer label `aiida setup` registers
  (`localhost` by default; every supported mode runs its CalcJobs on
  `localhost`, even hpc-direct, because that *is* the HPC in that mode).
- `max_stored_file_mb` — files larger than this are recorded by sha256 + path
  instead of being copied into the AiiDA repository.
- `skip_fraction` — leading fraction of each friction trace discarded as
  transient before the COF statistics.
- `machine` — the machine a run tree was produced on; `null` lets ingest
  resolve it.
- `enabled` and `lammps_code_label` are removed (AiiDA is import-only); an
  old settings file naming them loads with one warning.

Several fields that used to live here are gone: the settings that named a
second CalcJob for postprocessing on the HPC (never built — postprocessing
always runs locally), a provenance on/off switch (provenance is always
written now), a results-import on/off switch (an AiiDA WorkChain step that
ran unconditionally regardless of the switch), a mode enum consolidating
three flags that already agreed with each other, and the whole SSH-transport
block (host/workdir/user/port/key file) for the SSH-connected remote mode,
which is removed (see [aiida.md](aiida.md) for the two supported remote
workflows). See `docs/superpowers/specs/TO-DO/future-works.md`
§15.3/§15.4/§15.6 for the closed triage of each.

## database

The central summary database, its REST API and the `db`/`api` command groups
were removed. The local store is AiiDA, configured entirely under `## aiida` above
and via `FrictionSim2D aiida setup`. The `database` section below is a leftover:
nothing reads it any more, and it is deleted in Task 11.

Eight fields, unused:

```yaml
database:
  host: ""
  port: 5432
  dbname: frictionsim2ddb
  user: ""
  password: ""
  api_key: ""
  api_url: http://localhost:8000
  skip_fraction: 0.2
```

- `host`, `port`, `dbname`, `user`, `password`, `api_key`, `api_url` — the former
  central-database connection (env defaults `FRICTION_CENTRAL_DB_*`,
  `FRICTION_DB_API_KEY`). Unused.
- `skip_fraction` — the same warm-up fraction the postprocessing COF
  calculation uses, applied when the store computes summary statistics.

`password` and `api_key` -- and any other mapping key named like a credential,
anywhere in the document -- are blanked in the copy of the settings written
into every run tree (`provenance/settings.yaml` and the `settings` block of
`provenance/config.json`): those files are stored in the AiiDA repository and
travel in archives and export bundles. The other `database` fields are kept.
This happens twice: once when the tree is written (`redact_secrets`), and
again -- independently, and by the same rule -- the moment any settings/config
document is staged into the AiiDA store (see
[aiida.md](aiida.md#credentials-never-enter-the-store) for the precise
guarantee), so a tree built before the first gate existed, or a hand-edited
settings file with credentials still in it, cannot leak them either.

## integrity

Thresholds of the post-run integrity check (stage 3 of `postprocess read`). Every threshold is
optional; `null` turns that rule off. The rules, statuses and quarantine are described in
[integrity.md](integrity.md).

Defaults:

```yaml
integrity:
  hole_bonds: 1.5               # largest atom-free disk, in bond lengths
  isolated_radius_bonds: 1.5    # neighbour radius for "isolated"
  isolated_atoms: 2             # atoms with no neighbour
  ejected_atoms: 2              # atoms off the layer's median plane
  ejected_bonds: 8.0            # distance that counts as ejected
  heating_sigma: 8.0            # peak sheet temperature, in sigma_T
  gap_rise: 0.05                # interlayer gap change, fraction of its start
  collapse: true                # fail when layers pass through each other
  frozen_std: 1.0e-6            # a channel with a smaller standard deviation is frozen
  gap_start_samples: 25         # samples averaged for the starting gap
  sheet_heating_skip_samples: 5 # leading samples skipped (sheet-on-sheet heating peak)
  sheet_gap_end_samples: 50     # samples averaged for the end-of-slide gap (sheet-on-sheet)
  margins: {}                   # rule name -> fraction of its threshold; empty = none
  version: "1.0.0"              # version of the rule set, stamped into exports
```

`margins` is empty by default. A rule with margin `m` quarantines a passing point whose value
is in `[m * threshold, threshold)`. `postprocess read --integrity-settings <file>` replaces this
block for one read; `--no-integrity` skips the check.

## Related docs

- [configuration.md](configuration.md)
- [commands.md](commands.md)
- [aiida.md](aiida.md)
- [integrity.md](integrity.md)
