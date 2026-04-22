# Installation (Conda Package)

\anchor installation

\section install_scope Scope

FrictionSim2D is distributed as a Conda package to ensure reproducible, conflict-free installation
across different computing environments. All dependencies (LAMMPS, Atomsk, AiiDA services) are managed
through Conda.

\section install_prereq Prerequisites

- Linux system
- Conda or Miniconda installed
- Access to `conda-forge` channel (optional; conda-build can build locally)

\section install_options Installation Options

Six conda packages are available (install the ones you need):

| Package | Contents |
|---|---|
| `frictionsim2d` | Core: LAMMPS, Atomsk, CLI |
| `frictionsim2d-aiida` | + AiiDA, PostgreSQL, RabbitMQ, psycopg2 |
| `frictionsim2d-db` | + psycopg2 (community database commands only) |
| `frictionsim2d-plotting` | + matplotlib, seaborn, scipy |
| `frictionsim2d-api` | + FastAPI, uvicorn, httpx (REST server/client) |
| `frictionsim2d-all` | All of the above |

All packages install the `FrictionSim2D` console command.

For pip users, equivalent extras are `[aiida]`, `[db]`, `[plotting]`, `[api]`, `[all]`.
See the [Development from source](#install_source) section.

\section install_base Base Install (No AiiDA)

Install core package with LAMMPS and Atomsk:

```bash
conda install -c conda-forge frictionsim2d
```

Create and activate a named environment (recommended):

```bash
conda create -n frictionsim2d -c conda-forge frictionsim2d
conda activate frictionsim2d
```

Verify installation:

```bash
FrictionSim2D --help
lmp -h
atomsk --version
```

\section install_aiida AiiDA Install (With Services)

Install the AiiDA variant with PostgreSQL and RabbitMQ:

```bash
conda install -c conda-forge frictionsim2d-aiida
```

Or create a named environment:

```bash
conda create -n frictionsim2d-aiida -c conda-forge frictionsim2d-aiida
conda activate frictionsim2d-aiida
```

Run the one-time bootstrap script to initialize services and create your AiiDA profile:

```bash
frictionsim2d-start-aiida
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--profile NAME` | `friction2d` | AiiDA profile name to create or reuse |
| `--no-daemon` | — | Skip starting the AiiDA daemon |

This command:
- Starts RabbitMQ with automatic `consumer_timeout` patching (default: 36000000 ms)
- Starts PostgreSQL (initializes data directory if needed)
- Creates the named AiiDA profile
- Starts the AiiDA daemon (unless `--no-daemon`)

Override RabbitMQ timeout if needed:

```bash
export RABBITMQ_CONSUMER_TIMEOUT_MS=72000000
frictionsim2d-start-aiida --profile friction2d
```

Optional: Install conda hooks for automatic service lifecycle management:

```bash
frictionsim2d-install-hooks
```

After hook installation, PostgreSQL and RabbitMQ automatically start when you run `conda activate frictionsim2d-aiida`
and stop when you run `conda deactivate`.

Verify AiiDA setup:

```bash
verdi --version
verdi profile show
verdi daemon status
```

\section install_db Community Database Install

To use `FrictionSim2D db` commands without AiiDA:

```bash
conda create -n frictionsim2d-db -c conda-forge frictionsim2d-db
conda activate frictionsim2d-db
```

This provides the `db` CLI group: `init`, `create-key`, `upload`, `stage`, `query`, `stats`, `delete`, `publish`, `reject`.

Set database connection via environment variables or `settings.yaml`:

```bash
export FRICTION_DB_HOST=db.example.com
export FRICTION_DB_NAME=frictionsim2d
export FRICTION_DB_USER=myuser
export FRICTION_DB_PASSWORD=secret
```

\section install_plotting Plotting Install

To use `FrictionSim2D postprocess plot`:

```bash
conda create -n frictionsim2d-plotting -c conda-forge frictionsim2d-plotting
conda activate frictionsim2d-plotting
```

Verify:

```bash
python -c "import matplotlib, seaborn, scipy; print('ok')"
```

\section install_api REST API Install

To run the REST API server or use the API client:

```bash
conda create -n frictionsim2d-api -c conda-forge frictionsim2d-api
conda activate frictionsim2d-api
FrictionSim2D api serve --host 0.0.0.0 --port 8000
```

\section install_all All-in-one Install

Install everything:

```bash
conda create -n frictionsim2d-all -c conda-forge frictionsim2d-all
conda activate frictionsim2d-all
frictionsim2d-start-aiida
```

\section install_verify Verify Installation

Base environment:

```bash
FrictionSim2D --help
lmp -h
atomsk --version
```

AiiDA environment:

```bash
FrictionSim2D --help
verdi --version
rabbitmqctl status
verdi daemon status
```

\section install_aiida_services Check AiiDA Services

List profiles:

```bash
verdi profile list
```

Check daemon status:

```bash
verdi daemon status
```

Check RabbitMQ:

```bash
rabbitmqctl status
```

Stop services (if manual):

```bash
verdi daemon stop
rabbitmqctl stop
```

\section install_notes Notes for HPC Clusters

For HPC clusters **without local AiiDA installation**:

1. Install `frictionsim2d` (base) locally
2. Install `frictionsim2d-aiida` on the HPC resource
3. Use AiiDA's SSH transport to connect to HPC (if cluster policy permits)
4. For restricted environments, use archive export/import for workflow transfer

\section install_troubleshooting Troubleshooting

**`FrictionSim2D: command not found`**
- Confirm the environment is active: `conda activate frictionsim2d` or `frictionsim2d-aiida`
- Confirm package is installed: `conda list | grep frictionsim2d`

**`lmp: command not found`**
- Confirm environment is active
- Confirm `lammps` is installed: `conda list | grep lammps`

**AiiDA profile errors**
- Rerun bootstrap: `frictionsim2d-start-aiida`
- Check profile: `verdi profile show`

**PostgreSQL or RabbitMQ not starting**
- Check system resources (disk space, memory)
- Check logs in `$CONDA_PREFIX/var/postgres/` or `$CONDA_PREFIX/var/rabbit/`
- Manually start: Run `frictionsim2d-start-aiida` with verbose output

**Services not stopping** after `conda deactivate`
- If hooks are installed: `frictionsim2d-install-hooks` may need re-running
- Manual stop: `verdi daemon stop && rabbitmqctl stop`

\section install_source Development (from source)

To build and install locally from source:

```bash
cd /path/to/FrictionSim2D
conda-build conda/
```

Or install in development mode with pip extras:

```bash
# Core only
pip install -e .

# With specific extras
pip install -e ".[db]"        # community database (psycopg2)
pip install -e ".[plotting]"  # matplotlib, seaborn, scipy
pip install -e ".[api]"       # FastAPI, uvicorn, httpx
pip install -e ".[aiida]"     # aiida-core (services via conda)
pip install -e ".[all]"       # all of the above

# Development tools
pip install -e ".[dev]"       # pytest, pytest-cov, pylint
```

> **Note:** `aiida-core.services` (RabbitMQ + PostgreSQL) are conda-only.
> After `pip install -e ".[aiida]"`, install services separately:
> ```bash
> conda install -c conda-forge aiida-core.services
> frictionsim2d-start-aiida
> ```
