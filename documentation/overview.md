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

A sixth generator, `run quench`, makes no friction run: it writes melt-quench jobs
that produce the amorphous blocks an `amorph = a` tip or substrate is cut from
(see [examples.md](examples.md#11-make-an-amorphous-block-melt-quench)).

## What FrictionSim2D produces

For each generated case:

| Directory | Contents |
|---|---|
| `lammps/` | LAMMPS input scripts (`system.in`, `slide*.in`, settings include files) |
| `build/` | The built structures (`*.lmp`: sheet, tip, substrate) |
| `data/` | Data files LAMMPS writes at run time (`load_*.data`) |
| `results/` | Destination for simulation outputs and postprocessing |
| `visuals/` | Trajectory and visual outputs |
| `provenance/` | Copied inputs (`cif/`, `potentials/`, `config.json`, `settings.yaml`) and `manifest.json` |
| `provenance.json` | The versioned run record (`run_uuid`, materials, parameters) the AiiDA import reads |

Optionally, an `hpc/` directory of PBS or SLURM submission scripts. Recording
finished runs in AiiDA is a separate, later step (`aiida import`).

## Standard lifecycle

1. Write a config file — see [configuration.md](configuration.md).
2. Generate the simulation tree with `FrictionSim2D run ...`.
3. Generate scheduler scripts, either with `--hpc-scripts` during step 2 or with
   `FrictionSim2D hpc generate` afterwards.
4. Run the LAMMPS jobs locally or on a cluster.
5. Import completed results into the local AiiDA database — full timeseries plus
   provenance, one integrity verdict per point: `FrictionSim2D aiida import`
   (`--integrity KEY=VALUE` overrides a threshold for that import; `aiida recheck`
   re-judges a set later).
6. Check, summarise and compare them in the store (`aiida postprocess`,
   `aiida dataset`, `FrictionSim2D compare`), and share them as AiiDA archives
   (`aiida export`, `aiida import-archive`).

## Quick start

```bash
# Generate AFM simulations
FrictionSim2D run afm examples/afm/basic.ini --output-dir ./simulation_output

# Generate sheet-on-sheet simulations
FrictionSim2D run sheetonsheet examples/sheetonsheet/basic.ini --output-dir ./simulation_output

# Generate scheduler scripts for an existing simulation root
FrictionSim2D hpc generate ./simulation_output/simulation_YYYYMMDD_HHMMSS --scheduler pbs
```

## Output structure

A typical AFM root:

```text
simulation_YYYYMMDD_HHMMSS/
  afm/
    <material>/
      <size>/                  # e.g. 100x_100y
        <sub_tip>/             # e.g. sub_amorph-Si_tip_Si_r25
          K<temp>/
            pot_<type>_<name>/ # e.g. pot_sw_MoS2_wen
              L1/
                lammps/
                data/
                results/
                visuals/
                provenance.json
              L2/
              ...
              build/
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
          pot_<type>_<name>/
            lammps/
            data/
            build/
            results/
            visuals/
            provenance.json
            provenance/
  hpc/                # optional
  logs/
```

AFM produces per-layer `L<N>` subfolders because `[2D] layers` may hold several
values; `build/` and `provenance/` are shared by them. Sheet-on-sheet takes exactly
one layer count, so it produces one directory per expanded parameter set. The size
token is `<x>x_<y>y`. A quench root is `quench_YYYYMMDD_HHMMSS/`.

## Command families

| Group | Purpose |
|---|---|
| `run` | Generate simulations of any of the five models, or the quench jobs for amorphous blocks (`run quench`) |
| `settings` | Inspect, initialise, or reset global settings |
| `hpc` | Build scheduler scripts from generated simulations; `enrich-provenance` folds finished logs into the records |
| `aiida` | Setup, import, integrity rulings, summaries, datasets, query, archives |
| `postprocess` | Read results and produce plots or descriptor tables |
| `compare` | Compare the parameters of two runs or campaigns, field by field |
| `quench` | The amorphous block library: `quench import`, `quench list` |
| `fetch` | Download add-ons: `fetch materials` (the pre-quenched materials pack), `fetch model` (the assistant's model) |
| `analyse` | Run and read the on-demand analyses of a dataset in the store |
| `gui` | Launch the desktop simulation builder |

Full reference in [commands.md](commands.md).

## Storage

Results live in the local AiiDA store (`FrictionSim2D aiida ...`): full
time-series data and complete provenance. The former central PostgreSQL database
and REST API were removed; exchange results as AiiDA archives
(`FrictionSim2D aiida export` / `aiida import-archive`).

## Materials and potentials

Config files name structures and potentials as `materials/<file>.cif` and
`potentials/<type>/<file>`. A name is looked up as given, then next to the config,
then in the package data (`frictionsim2d/data/`), then in the materials pack
(`FrictionSim2D fetch materials`, installed under `$XDG_DATA_HOME/FrictionSim2D/`),
then in the amorphous block library, and last in a legacy `examples/` folder, which
works but logs a deprecation warning.

## Reproducibility

Every simulation directory carries a `provenance/` area holding copies of the
source files and a manifest. To archive a study, keep at minimum:

- The original config file
- The generated simulation root
- `provenance.json` and `provenance/manifest.json`
- The settings file used at generation time
- An environment description, for example a package list

Details in [provenance.md](provenance.md).

## Related docs

- [installation.md](installation.md)
- [configuration.md](configuration.md)
- [commands.md](commands.md)
- [examples.md](examples.md)
