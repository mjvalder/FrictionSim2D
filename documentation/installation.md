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

Two packages are available:

- `frictionsim2d`: Core package with LAMMPS and Atomsk (no AiiDA)
- `frictionsim2d-aiida`: Full package including AiiDA with PostgreSQL and RabbitMQ

Both install the `FrictionSim2D` console command automatically.

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

This command:
- Starts PostgreSQL (initializes data directory if needed)
- Starts RabbitMQ with automatic `consumer_timeout` patching (default: 36000000 ms)
- Creates your first AiiDA profile

Override RabbitMQ timeout if needed:

```bash
export RABBITMQ_CONSUMER_TIMEOUT_MS=72000000
frictionsim2d-start-aiida
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

Or install in development mode:

```bash
pip install -e .  # after activating a conda environment with build tools
```
