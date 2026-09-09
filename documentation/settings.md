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
```

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
  mpi_command: mpirun
  use_tmpdir: true
  lammps_scripts: [system.in, slide.in]
```

For sheet-on-sheet runs, script ordering is adjusted automatically to slide scripts only when needed.

## aiida

Defaults:

```yaml
aiida:
  enabled: false
  lammps_code_label: lammps@my_hpc
  postprocess_code_label: python@my_hpc
  postprocess_script_path: ""
  create_provenance: true
  auto_import_results: false
  hpc_mode: offline
  computer_label: localhost
  transport: local
  hostname: null
  workdir: null
  username: null
  ssh_port: 22
  key_filename: null
```

`hpc_mode` values:

- `offline`: generate and run outside AiiDA, import later
- `local`: submit where daemon/computer is local
- `remote`: submit to configured remote computer

## database

FrictionSim2D uses a **two-database architecture**:

| Database | Backend | Contents | CLI group |
|---|---|---|---|
| **Local (AiiDA)** | AiiDA-managed SQLite/PostgreSQL | Full time-series data, complete provenance | `FrictionSim2D aiida` |
| **Central (remote)** | Shared PostgreSQL on remote server | Summary statistics only (mean COF, forces, conditions) | `FrictionSim2D db` |

The `database` section in `settings.yaml` configures connections for the **central DB** and an optional local mirror.
The AiiDA database is configured separately via `verdi` / `FrictionSim2D aiida setup`.

Defaults:

```yaml
database:
  active_profile: local   # 'local' = local mirror for testing; 'central' = shared remote DB
  local:
    host: localhost
    port: 5432
    dbname: frictionsim2ddb
    user: ""
    password: ""
    api_key: ""
  central:
    host: ""              # hostname of your remote PostgreSQL server
    port: 5432
    dbname: frictionsim2ddb
    user: ""
    password: ""
    api_key: ""
  skip_fraction: 0.2
  api_url: http://localhost:8000
  api_host: 0.0.0.0
  api_port: 8000
```

These values back the `db` and `api` command groups when explicit CLI options are not provided.

> **Note**: `active_profile: local` points to a local PostgreSQL mirror of the central DB schema —
> useful for offline testing. It is **not** the AiiDA database. To target the shared remote DB,
> set `active_profile: central` or pass `--profile central` to `db` commands.

## Related docs

- [configuration.md](configuration.md)
- [commands.md](commands.md)
- [aiida.md](aiida.md)
