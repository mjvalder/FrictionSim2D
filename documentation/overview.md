# Overview

FrictionSim2D generates structured simulation folders and LAMMPS input files for atomistic friction studies on 2D materials.

## Simulation models

| Model | Command | System |
|---|---|---|
| AFM sliding | `run afm` | Tip, sheet, and substrate; the tip is loaded and dragged laterally. |
| Sheet-on-sheet | `run sheetonsheet` | An N-layer stack (minimum 3 layers) sheared under pressure. |
| Interlayer PES scan | `run pes-sheet` | A frozen bilayer, grid-scanned laterally over one surface unit cell. |
| Tip PES scan | `run pes-tip` | Tip, sheet, and substrate held at fixed contact and grid-scanned. |
| Indentation | `run indent` | Tip pressed to load, held, then retracted to detachment. |

The two PES models and the indentation model are static or quasi-static: they reuse
the sliding geometry but never invoke the sliding phase. See [pes_scan.md](pes_scan.md)
and [indent.md](indent.md).

## What FrictionSim2D produces

For each generated case:

| Directory | Contents |
|---|---|
| `lammps/` | LAMMPS input scripts (`system.in`, `slide*.in`, settings include files) |
| `data/` | Structure and combined data files |
| `results/` | Destination for simulation outputs and postprocessing |
| `visuals/` | Trajectory and visual outputs |
| `provenance/` | Copied input assets and `manifest.json` |

Optionally, an `hpc/` directory of PBS or SLURM submission scripts, and AiiDA
registration, submission, and import workflows.

## Standard lifecycle

1. Write a config file — see [configuration.md](configuration.md).
2. Generate the simulation tree with `FrictionSim2D run ...`.
3. Generate scheduler scripts, either with `--hpc-scripts` during step 2 or with
   `FrictionSim2D hpc generate` afterwards.
4. Run the LAMMPS jobs locally or on a cluster.
5. Import completed results into the local AiiDA database — full timeseries plus
   provenance: `FrictionSim2D aiida import`.
6. Optionally upload summary statistics to the shared central database, reading
   the local AiiDA store directly by set label:
   `FrictionSim2D db upload SET_LABEL --uploader NAME --contact EMAIL`.

## Quick start

```bash
# Generate AFM simulations
FrictionSim2D run afm examples/afm_config.ini --output-dir ./simulation_output

# Generate sheet-on-sheet simulations
FrictionSim2D run sheetonsheet examples/sheet_config.ini --output-dir ./simulation_output

# Generate scheduler scripts for an existing simulation root
FrictionSim2D hpc generate ./simulation_output/simulation_YYYYMMDD_HHMMSS --scheduler pbs
```

## Output structure

A typical AFM root:

```text
simulation_YYYYMMDD_HHMMSS/
  afm/
    <material>/
      <size>/
        <sub_tip>/
          K<temp>/
            L1/
              lammps/
              data/
              results/
              visuals/
            L2/
            ...
            provenance/
  hpc/                # optional
  logs/
```

A typical sheet-on-sheet root:

```text
simulation_YYYYMMDD_HHMMSS/
  sheetonsheet/
    <material>/
      <size>/
        K<temp>/
          lammps/
          data/
          results/
          visuals/
          provenance/
  hpc/                # optional
  logs/
```

AFM produces per-layer `L<N>` subfolders because `[2D] layers` may hold several
values. Sheet-on-sheet takes exactly one layer count, so it produces one directory
per expanded parameter set.

## Command families

| Group | Purpose |
|---|---|
| `run` | Generate simulations of any of the five models |
| `settings` | Inspect, initialise, or reset global settings |
| `hpc` | Build scheduler scripts from generated simulations |
| `aiida` | Setup, submit, import, query, export archives |
| `postprocess` | Read results and produce plots or descriptor tables |
| `db` | Manage keys and records in the shared database |
| `api` | Serve a REST API on top of the database |

Full reference in [commands.md](commands.md).

## Two-database architecture

FrictionSim2D uses two separate databases with different roles.

| | Local (AiiDA) | Central (remote) |
|---|---|---|
| **Backend** | AiiDA-managed | PostgreSQL on a remote server |
| **Contents** | Full time-series data, complete provenance | Summary statistics only (mean COF, forces, conditions) |
| **CLI** | `FrictionSim2D aiida ...` | `FrictionSim2D db ...` |
| **Setup** | `FrictionSim2D aiida setup` | `FrictionSim2D db setup --profile central` |
| **Use for** | Analysis, plotting, rebuilding simulations | Cross-study comparison, sharing results |

## Reproducibility

Every simulation directory carries a `provenance/` area holding copies of the
source files and a manifest. To archive a study, keep at minimum:

- The original config file
- The generated simulation root
- `provenance/manifest.json`
- The settings file used at generation time
- An environment description, for example a package list

Details in [provenance.md](provenance.md).

## Related docs

- [installation.md](installation.md)
- [configuration.md](configuration.md)
- [commands.md](commands.md)
- [examples.md](examples.md)
