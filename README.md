# FrictionSim2D

FrictionSim2D generates and manages LAMMPS friction simulation workflows for 2D materials.
It supports AFM sliding, sheet-on-sheet shear, static PES scans, and indentation setups,
with HPC script generation and optional AiiDA provenance and submission workflows.

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
covered in [documentation/installation.md](documentation/installation.md). RabbitMQ and
PostgreSQL are required for full AiiDA workflows.

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

### Option B – With AiiDA (provenance tracking, HPC submission)

Install base package + AiiDA with bundled PostgreSQL and RabbitMQ:

```bash
conda create -n frictionsim2d-aiida -c conda-forge frictionsim2d-aiida
conda activate frictionsim2d-aiida
```

Then bootstrap AiiDA and start services (one-time setup):

```bash
frictionsim2d-start-aiida
```

This:
- Starts RabbitMQ with automatic `consumer_timeout` patching (36000000 ms default)
- Starts PostgreSQL
- Creates your first AiiDA profile (`friction2d`)

Options: `--profile NAME` to use a custom profile name, `--no-daemon` to skip daemon startup.

Override RabbitMQ timeout if needed:
```bash
export RABBITMQ_CONSUMER_TIMEOUT_MS=72000000
frictionsim2d-start-aiida
```

Optional: install conda activate/deactivate hooks for automatic service lifecycle:
```bash
frictionsim2d-install-hooks
```

After this, PostgreSQL and RabbitMQ start/stop automatically with `conda activate/deactivate`.

Verify AiiDA setup:
```bash
verdi --version
verdi profile show
verdi daemon status
```

### Option D – Postprocessing / plotting

```bash
conda create -n frictionsim2d-plotting -c conda-forge frictionsim2d-plotting
conda activate frictionsim2d-plotting
```

### Option F – All-in-one

```bash
conda create -n frictionsim2d-all -c conda-forge frictionsim2d-all
conda activate frictionsim2d-all
frictionsim2d-start-aiida
```
-->

## Quick start

Generate AFM simulations:

```bash
FrictionSim2D run afm examples/afm_config.ini --output-dir ./simulation_output
```

Generate sheet-on-sheet simulations:

```bash
FrictionSim2D run sheetonsheet examples/sheet_config.ini --output-dir ./simulation_output
```

Generate static PES (corrugation-barrier) scans — cheap energy-vs-lateral-position maps:

```bash
FrictionSim2D run pes-sheet examples/pes_sheet_config.ini --output-dir ./simulation_output
FrictionSim2D run pes-tip   examples/pes_tip_config.ini   --output-dir ./simulation_output
```

See [documentation/pes_scan.md](documentation/pes_scan.md).

Generate an indentation run — penetration on hold, adhesion on retract:

```bash
FrictionSim2D run indent examples/indent_config.ini --output-dir ./simulation_output
```

See [documentation/indent.md](documentation/indent.md).

Generate with AiiDA registration enabled:

```bash
FrictionSim2D run afm examples/afm_config.ini --aiida --output-dir ./simulation_output
```

Generate HPC scripts:

```bash
FrictionSim2D hpc generate ./simulation_output/simulation_YYYYMMDD_HHMMSS --scheduler pbs
```

## CLI overview

- `FrictionSim2D run afm|sheetonsheet|pes-sheet|pes-tip|indent ...`
- `FrictionSim2D hpc generate ...`
- `FrictionSim2D settings show|init|reset`
- `FrictionSim2D aiida status|setup|submit|import|list-sets|dump|rebuild|query|delete|clear|export|import-archive|package`
- `FrictionSim2D postprocess read|plot|pes-scan|indent`

Full reference: [documentation/commands.md](documentation/commands.md).

## AiiDA notes

- The former central database and REST API were removed; share results as AiiDA archives (`FrictionSim2D aiida import`).
- Use `FrictionSim2D aiida setup` for first-time profile/computer/code setup.
- For secure HPC clusters, running AiiDA natively on the HPC environment is often more robust than local SSH transport.
- Use archive transfer when needed: `aiida export` on source environment and `aiida import-archive` on target.

## Output structure

Generated simulation roots typically contain:

- `afm/`, `sheetonsheet/`, `pes_sheet/`, `pes_tip/`, or `indent/` runs
- per-run `lammps/`, `data/`, `results/`, `visuals/`, `provenance/`
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

## Development

Run tests:

```bash
pytest -v
```

Run lint (example):

```bash
pylint src/aiida
```
