# FrictionSim2D

FrictionSim2D generates and manages LAMMPS friction simulation workflows for 2D materials.
It supports AFM sliding, sheet-on-sheet shear, static PES scans, and indentation setups,
with HPC script generation and an optional AiiDA provenance store for finished runs.
Sheets can be driven by classical potentials (Stillinger-Weber, Tersoff, ReaxFF, ...)
or by a **machine-learned interatomic potential** (DeePMD or MACE) selected the
same way, with one `pot_type` line. Amorphous tips and substrates are cut from
melt-quenched blocks made with `run quench`.

Full documentation lives in [documentation/](documentation/) — start with
[documentation/README.md](documentation/README.md).

## Installation

FrictionSim2D is not yet published on conda-forge. Create an environment, then install
from source. Requires Python 3.9+ and LAMMPS on your path; Atomsk is optional.

```bash
conda create -n frictionsim2d python=3.11
conda activate frictionsim2d
pip install -e .          # core
pip install -e ".[all]"   # with aiida, plotting, and gui extras
```

Verify:

```bash
FrictionSim2D --help
FrictionSim2D --version
lmp -h
```

Individual extras, external-binary checks, AiiDA bootstrap, and troubleshooting are
covered in [documentation/installation.md](documentation/installation.md). The AiiDA
store needs no RabbitMQ broker and no daemon: `FrictionSim2D aiida setup` creates the
profile only, and an SQLite profile is enough.

The conda-forge installation scheme below is planned for a future release and is
intentionally hidden for now.

<!--
Future conda-forge installation draft

FrictionSim2D is distributed as a Conda package for maximum reproducibility and minimal dependency conflicts.

### Option A – Base package (recommended for most users)

Install LAMMPS, Atomsk, and core dependencies:

```bash
conda create -n frictionsim2d -c conda-forge frictionsim2d
conda activate frictionsim2d
```

Verify:
```bash
FrictionSim2D --help
lmp -h
atomsk --version
```

### Option B – With AiiDA (provenance store)

```bash
conda create -n frictionsim2d-aiida -c conda-forge frictionsim2d-aiida
conda activate frictionsim2d-aiida
FrictionSim2D aiida setup
```

`aiida setup` creates the profile only (SQLite is enough). There is no broker and
no daemon to start.

### Option D – Postprocessing / plotting

```bash
conda create -n frictionsim2d-plotting -c conda-forge frictionsim2d-plotting
conda activate frictionsim2d-plotting
```

### Option F – All-in-one

```bash
conda create -n frictionsim2d-all -c conda-forge frictionsim2d-all
conda activate frictionsim2d-all
FrictionSim2D aiida setup
```
-->

## Quick start

Generate AFM simulations:

```bash
FrictionSim2D run afm examples/afm/basic.ini --output-dir ./simulation_output
```

Generate sheet-on-sheet simulations:

```bash
FrictionSim2D run sheetonsheet examples/sheetonsheet/basic.ini --output-dir ./simulation_output
```

Drive the sheet with a machine-learned potential instead — same command, one
extra key in the `.ini`:

```bash
./scripts/fetch_mlip_models.sh deepmd       # one-time: fetch + freeze the model
FrictionSim2D run sheetonsheet examples/sheetonsheet/mlip_full.ini --output-dir ./simulation_output
```

See [documentation/mlip.md](documentation/mlip.md) and
[frictionsim2d/data/potentials/mlip/README.md](frictionsim2d/data/potentials/mlip/README.md).

Generate static PES (corrugation-barrier) scans — cheap energy-vs-lateral-position maps:

```bash
FrictionSim2D run pes-sheet examples/pes-sheet/basic.ini --output-dir ./simulation_output
FrictionSim2D run pes-tip   examples/pes-tip/basic.ini   --output-dir ./simulation_output
```

See [documentation/pes_scan.md](documentation/pes_scan.md).

Generate an indentation run — penetration on hold, adhesion on retract:

```bash
FrictionSim2D run indent examples/indent/basic.ini --output-dir ./simulation_output
```

See [documentation/indent.md](documentation/indent.md).

Import a finished run tree into the AiiDA provenance store:

```bash
FrictionSim2D aiida import ./simulation_output/simulation_YYYYMMDD_HHMMSS -l my-campaign
```

Generate HPC scripts:

```bash
FrictionSim2D hpc generate ./simulation_output/simulation_YYYYMMDD_HHMMSS --scheduler pbs
```

## CLI overview

- `FrictionSim2D run afm|sheetonsheet|pes-sheet|pes-tip|indent|quench ...`
- `FrictionSim2D quench import|list`: the amorphous block library
- `FrictionSim2D hpc generate|enrich-provenance ...`
- `FrictionSim2D settings show|init|reset`
- `FrictionSim2D aiida status|setup|preflight|import|rule|recheck|postprocess|dataset|query|list-sets|export-results|rebuild|reindex|verify|delete|clear|export|import-archive`
- `FrictionSim2D analyse list|run|show|export`: on-demand analyses of a dataset
- `FrictionSim2D postprocess read|plot|pes-scan|indent`
- `FrictionSim2D compare LABEL_A LABEL_B`
- `FrictionSim2D fetch materials|model`: the materials pack, the assistant model
- `FrictionSim2D gui`

Full reference: [documentation/commands.md](documentation/commands.md).

## Machine-learned potentials (DeePMD / MACE)

An MLIP is a third potential category alongside the classical ones, not a separate
workflow. Select it as you select Stillinger-Weber:

```ini
[2D]
pot_type = deepmd                      # or: mace
pot_path = /home/you/opt/models/dpa3_alex2d.pth
dispersion = d3                        # Grimme-D3 overlay; the only vdW lever
lat_c = 6.112                          # interlayer distance, MANDATORY for MLIP
```

- Models are third-party weights and are not in this repository:
  `./scripts/fetch_mlip_models.sh deepmd` or `mace --elements Mo S`.
- Every model can host an MLIP; every component of a cell must name the same one.
  Unsupported combinations are refused at generation time.
- MACE runs on the GPU (`device`) backend by default; `hpc.frameworks.mace.backend: host`
  selects the CPU one. `precision = float32` is opt-in and adds `-f32` to the run directory.
- Tip-bearing MLIP cells generate, but the two frameworks disagree by ~44% RMS on
  forces for the reference cell. Read
  [documentation/mlip.md §13](documentation/mlip.md) before trusting an
  AFM, pes-tip or indent number.
- LAMMPS builds for the two frameworks are in [`scripts/hpc/`](scripts/hpc/); read
  its README first.

Details: [documentation/mlip.md](documentation/mlip.md).

## AiiDA notes

- The former central database and REST API were removed; share results as AiiDA archives (`FrictionSim2D aiida export`).
- The store is import-only: runs are generated and run outside AiiDA, then registered with `FrictionSim2D aiida import`.
- `FrictionSim2D aiida setup` creates the profile only (SQLite is enough; no broker or daemon). Computers and Codes are created at import.
- For secure HPC clusters, running AiiDA natively on the HPC environment is often more robust than local SSH transport.
- `aiida import` and `aiida recheck` accept `--integrity KEY=VALUE` to override an integrity threshold for that call.
- Use archive transfer when needed: `aiida export` on source environment and `aiida import-archive` on target.

## Output structure

Generated simulation roots typically contain:

- `afm/`, `sheetonsheet/`, `pes_sheet/`, `pes_tip/`, or `indent/` runs
- per-run `lammps/`, `build/`, `data/`, `results/`, `visuals/`, `provenance/`, `provenance.json`
- optional `hpc/` submission scripts

Full layout in [documentation/overview.md](documentation/overview.md).

## Documentation

The index at [documentation/README.md](documentation/README.md) lists everything in
reading order. Most-used entries:

- [documentation/installation.md](documentation/installation.md) — setup and troubleshooting
- [documentation/overview.md](documentation/overview.md) — models, output layout, lifecycle
- [documentation/configuration.md](documentation/configuration.md) — writing `.ini` configs
- [documentation/commands.md](documentation/commands.md) — full CLI reference
- [documentation/examples.md](documentation/examples.md) — end-to-end workflows
- [documentation/mlip.md](documentation/mlip.md) — machine-learned potentials (DeePMD, MACE): capability matrix, models, builds

## Development

Run tests:

```bash
pytest                          # fast tier; real-LAMMPS tests are deselected
pytest -m lammps                # the real-LAMMPS set (slow)
```

See [documentation/testing.md](documentation/testing.md).

Run lint (example):

```bash
pylint frictionsim2d/aiida
```
