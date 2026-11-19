# FrictionSim2D

FrictionSim2D generates and manages LAMMPS friction simulation workflows for 2D materials.
It supports AFM sliding, sheet-on-sheet shear, static PES scans, and indentation setups,
with HPC script generation and optional AiiDA provenance and submission workflows.
Sheets can be driven by classical potentials (Stillinger-Weber, Tersoff, ReaxFF, ...)
or by a **machine-learned interatomic potential** — DeePMD or MACE — selected the
same way, with one `pot_type` line.

Full documentation lives in [documentation/](documentation/) — start with
[documentation/README.md](documentation/README.md).

## Installation

FrictionSim2D is not yet published on conda-forge. Create an environment, then install
from source. Requires Python 3.9+ and LAMMPS on your path; Atomsk is optional.

```bash
conda create -n frictionsim2d python=3.11
conda activate frictionsim2d
pip install -e .          # core
pip install -e ".[all]"   # with aiida, db, plotting, and api extras
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

### Option C – Community database only

```bash
conda create -n frictionsim2d-db -c conda-forge frictionsim2d-db
conda activate frictionsim2d-db
```

### Option D – Postprocessing / plotting

```bash
conda create -n frictionsim2d-plotting -c conda-forge frictionsim2d-plotting
conda activate frictionsim2d-plotting
```

### Option E – REST API server

```bash
conda create -n frictionsim2d-api -c conda-forge frictionsim2d-api
conda activate frictionsim2d-api
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

Drive the sheet with a machine-learned potential instead — same command, one
extra key in the `.ini`:

```bash
./scripts/fetch_mlip_models.sh deepmd       # one-time: fetch + freeze the model
FrictionSim2D run sheetonsheet examples/mlip/sheet_mlip_full.ini --output-dir ./simulation_output
```

See [documentation/mlip.md](documentation/mlip.md) and
[examples/potentials/mlip/README.md](examples/potentials/mlip/README.md).

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
- `FrictionSim2D db setup|init|create-key|upload|stage|query|stats|publish|reject|delete`
- `FrictionSim2D api serve`

Full reference: [documentation/commands.md](documentation/commands.md).

## Machine-learned potentials (DeePMD / MACE)

An MLIP is a **third potential category** alongside the classical ones, not a
separate workflow. You select it exactly as you select Stillinger-Weber:

```ini
[2D]
pot_type = deepmd                      # or: mace
pot_path = /home/you/opt/models/dpa3_alex2d.pth
dispersion = d3                        # Grimme-D3 overlay; the only vdW lever
lat_c = 6.112                          # interlayer distance, MANDATORY for MLIP
```

Everything else — staging the model, the `pair_style`/`pair_coeff` lines, the D3
overlay, hiding the virtual drive atom from the model's neighbour graph, the
per-framework HPC launcher flags — is generated for you.

**Getting the models.** They are not in this repository (18-120 MB of
third-party weights). One command each:

```bash
./scripts/fetch_mlip_models.sh deepmd                 # 118 elements: one file, every material
./scripts/fetch_mlip_models.sh mace --elements Mo S   # element-specific: re-extract per material
```

See [examples/potentials/mlip/README.md](examples/potentials/mlip/README.md) for
the difference between the two and for why the defaults are what they are.

**What is supported today.** MLIP coverage must span the *whole cell* with one
instance — both frameworks reject the `NULL`-padded type maps a partitioned cell
would need. In practice that means **single-component sheet cells**:

| Model | MLIP? |
|---|---|
| `run sheetonsheet` | **Yes** |
| `run pes-sheet` | **Yes** |
| `run afm`, `run pes-tip`, `run indent` | No — tip + sheet + substrate is multi-component |
| hetero / flake | No — likewise |

Unsupported combinations are refused **at generation time** with a message that
names the rule and quotes the LAMMPS error behind it, not at 3 a.m. on a
cluster. Full capability matrix, the measured evidence, model provisioning,
builds and GPU profiles: **[documentation/mlip.md](documentation/mlip.md)**.

LAMMPS binaries for the two frameworks are mutually incompatible and are built
by [`scripts/hpc/`](scripts/hpc/) — read that directory's README before
building, especially the mandatory symmetrix patch for MACE.

## Shared database

FrictionSim2D supports an API-first shared database model for publications.

- Users query and upload through a public REST API URL.
- Database credentials stay only on the server.
- Each contributor gets a personal API key.

### Publication links to put on GitHub

- Public FrictionSim2DDB API: https://YOUR-DOMAIN-OR-IP:8000
- Local mirror API (for local development only): http://localhost:8000

Important: localhost is only reachable on the same machine. It is not a public link.

### Configure connection

For server admins (machine running PostgreSQL + API server), set database credentials:

```bash
export FRICTION_DB_HOST=db.example.com
export FRICTION_DB_PORT=5432
export FRICTION_DB_NAME=frictionsim2d
export FRICTION_DB_USER=myuser
export FRICTION_DB_PASSWORD=secret
```

For package users/collaborators, set API URL and personal API key (no DB password needed):

```bash
export FRICTION_API_URL=https://YOUR-DOMAIN-OR-IP:8000
export FRICTION_API_KEY=sk_your_personal_key
```

### Start the API server (server admin)

```bash
FrictionSim2D db init --profile local
FrictionSim2D api serve --host 0.0.0.0 --port 8000
```

### Create contributor keys (server admin)

```bash
FrictionSim2D db create-key --name alice --profile local
FrictionSim2D db create-key --name bob --profile local
```

### Upload results

```bash
# Upload through authenticated REST endpoint (example)
curl -X POST "$FRICTION_API_URL/results" \
	-H "X-API-Key: $FRICTION_API_KEY" \
	-H "Content-Type: application/json" \
	-d '{"material":"h-MoS2","simulation_type":"afm","layers":1,"mean_cof":0.03}'
```

### Query results

```bash
# Print table to terminal
FrictionSim2D db query --material h-MoS2 --layers 1

# Save to CSV
FrictionSim2D db query --material h-WS2 --csv output.csv
```

### Database statistics

```bash
FrictionSim2D db stats
```

### Delete your own rows

```bash
FrictionSim2D db delete --uploader alice
```

## AiiDA notes

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
- [documentation/mlip.md](documentation/mlip.md) — machine-learned potentials (DeePMD, MACE): capability matrix, models, builds

## Development

Run tests:

```bash
pytest -v
```

Run lint (example):

```bash
pylint src/aiida
```
