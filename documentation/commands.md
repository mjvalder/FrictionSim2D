# Commands Reference

\anchor commands

Complete CLI command reference for FrictionSim2D.

## Main Entrypoint

```bash
FrictionSim2D --help        # Show all commands
FrictionSim2D --version     # Show version
```

---

## run - Generate Simulations

### run afm

Generate AFM (tip-on-substrate) simulation files.

```bash
FrictionSim2D run afm CONFIG_FILE [OPTIONS]
```

**Arguments**:
- `CONFIG_FILE`: Path to `.ini` configuration file

**Options**:
- `-o, --output-dir DIR`: Output directory (default: `simulation_output`)
- `--aiida`: Enable AiiDA provenance tracking
- `--hpc-scripts`: Also generate HPC submission scripts
- `--hpc NAME`: HPC configuration name (overrides settings)
- `--local`: Mark as local run (no HPC submission)

**Examples**:

```bash
# Basic generation
FrictionSim2D run afm afm_config.ini

# Custom output directory
FrictionSim2D run afm config.ini -o ./my_simulations

# With AiiDA provenance
FrictionSim2D run afm config.ini --aiida

# Generate HPC scripts inline
FrictionSim2D run afm config.ini --hpc-scripts

# Combined
FrictionSim2D run afm config.ini -o ./output --aiida --hpc-scripts
```

### run sheetonsheet

Generate sheet-on-sheet sliding simulation files.

```bash
FrictionSim2D run sheetonsheet CONFIG_FILE [OPTIONS]
```

**Arguments**:
- `CONFIG_FILE`: Path to `.ini` configuration file

**Options**: Same as `run afm`

**Examples**:

```bash
FrictionSim2D run sheetonsheet sheet_config.ini
FrictionSim2D run sheetonsheet config.ini -o ./sheet_output --hpc-scripts
```

---

## settings - Settings Management

### settings show

Display the current merged settings (package defaults overridden by local `settings.yaml`).

```bash
FrictionSim2D settings show
```

### settings init

Copy the package default `settings.yaml` into the current directory for local customisation.

```bash
FrictionSim2D settings init
```

Creates `./settings.yaml`. Edit this file to override simulation, HPC, or database defaults.

### settings reset

Remove the local `settings.yaml` so the package defaults take effect again.

```bash
FrictionSim2D settings reset
```

---

## hpc - HPC Script Management

### hpc generate

Generate PBS or SLURM array-job scripts for an existing simulation tree.

```bash
FrictionSim2D hpc generate SIMULATION_DIR [OPTIONS]
```

**Arguments**:
- `SIMULATION_DIR`: Root directory produced by `run afm` / `run sheetonsheet`

**Options**:
- `-s, --scheduler {pbs,slurm}`: Scheduler type (default: `pbs`)
- `-o, --output-dir DIR`: Output directory for scripts (default: `SIMULATION_DIR/hpc`)

**Examples**:

```bash
# PBS scripts (default)
FrictionSim2D hpc generate ./simulation_output

# SLURM scripts
FrictionSim2D hpc generate ./simulation_output --scheduler slurm

# Custom script output location
FrictionSim2D hpc generate ./simulation_output --output-dir ./hpc_scripts
```

**Generated files** (inside the output directory):
- `manifest.txt` — ordered list of simulation paths
- `run_array.pbs` / `run_array.sh` — array job script
- `submit_all.txt` — next-steps instructions

---

## aiida - AiiDA Workflow Commands

All `aiida` subcommands require `aiida-core` to be installed. The group will abort with an install hint if AiiDA is unavailable.

### aiida setup

First-time AiiDA configuration: create a profile, register a computer, and configure the LAMMPS code.

```bash
FrictionSim2D aiida setup [OPTIONS]
```

**Options**:
- `-p, --profile NAME`: AiiDA profile name (default: `friction2d`)
- `--lammps-path PATH`: Explicit path to LAMMPS executable
- `--use-remote`: Configure a remote HPC computer from the `aiida` section of `settings.yaml`
- `--hpc-config PATH`: *(Deprecated)* Use `--use-remote` instead

**Examples**:

```bash
# Localhost setup (most common)
FrictionSim2D aiida setup

# Custom LAMMPS binary
FrictionSim2D aiida setup --lammps-path /opt/lammps/bin/lmp_mpi

# Remote HPC computer (configure settings.yaml first)
FrictionSim2D aiida setup --use-remote
```

### aiida status

Check AiiDA installation and active profile.

```bash
FrictionSim2D aiida status
```

**Output**:
```
✅ AiiDA is installed
✅ Active profile: friction2d
   Storage: psql_dos
```

### aiida submit

Submit simulations to AiiDA with smart defaults.

```bash
FrictionSim2D aiida submit SIMULATION_DIR [OPTIONS]
```

**Arguments**:
- `SIMULATION_DIR`: Simulation root directory

**Options**:
- `-c, --code LABEL`: AiiDA code label (auto-detects if omitted)
- `--scripts CSV`: Comma-separated list of LAMMPS scripts to run in order
- `--array`: Submit a single array job for all simulations
- `--machines N`: Override number of nodes per job
- `--mpiprocs N`: Override MPI processes per node
- `--walltime TIME`: Override walltime (`HH:MM:SS` or integer seconds)
- `--queue NAME`: Override scheduler queue/partition
- `--project NAME`: Override scheduler account/project
- `--dry-run`: Preview configuration without submitting

**Examples**:

```bash
# Minimal — auto-detect code, use settings.yaml defaults
FrictionSim2D aiida submit ./simulation_output

# Resource overrides
FrictionSim2D aiida submit ./simulation_output --machines 4 --walltime 24:00:00

# Preview first
FrictionSim2D aiida submit ./simulation_output --dry-run

# Array job with explicit code
FrictionSim2D aiida submit ./simulation_output --array --code lammps@hpc
```

### aiida import

Import completed simulation results into the AiiDA database.

```bash
FrictionSim2D aiida import RESULTS_DIR [OPTIONS]
```

**Arguments**:
- `RESULTS_DIR`: Directory containing returned simulation outputs

**Options**:
- `--process` / `--no-process`: Run post-processing before import (default: `--process`)

**Examples**:

```bash
FrictionSim2D aiida import ./returned_results
FrictionSim2D aiida import ./results --no-process
```

### aiida query

Query the AiiDA simulation database.

```bash
FrictionSim2D aiida query [OPTIONS]
```

**Filter options**:
- `-m, --material MAT`: Filter by material name
- `-l, --layers N`: Filter by layer count
- `-f, --force F`: Filter by applied force (nN)

**Output options**:
- `--format {table,csv,json}`: Output format (default: `table`)
- `-o, --output FILE`: Save to file

**Examples**:

```bash
# All results (table)
FrictionSim2D aiida query

# Filtered
FrictionSim2D aiida query --material h-MoS2 --layers 2

# CSV export
FrictionSim2D aiida query --material h-MoS2 --format csv -o results.csv
```

### aiida export

Export the AiiDA database to a portable archive.

```bash
FrictionSim2D aiida export [OPTIONS]
```

**Options**:
- `-o, --output FILE`: Archive path (default: `friction2d.aiida`)
- `-m, --material MAT`: Export only simulations for this material

**Examples**:

```bash
FrictionSim2D aiida export
FrictionSim2D aiida export -o all_data.aiida
FrictionSim2D aiida export -m h-MoS2 -o mos2.aiida
```

### aiida import-archive

Import an AiiDA archive into the current profile.

```bash
FrictionSim2D aiida import-archive ARCHIVE_PATH
```

**Arguments**:
- `ARCHIVE_PATH`: Path to `.aiida` archive file

**Examples**:

```bash
FrictionSim2D aiida import-archive friction2d.aiida
FrictionSim2D aiida import-archive /path/to/shared_results.aiida
```

### aiida package

Create a `.tar.gz` archive of simulation input files (`.lammpstrj` trajectory files are excluded).

```bash
FrictionSim2D aiida package SIMULATION_DIR [OPTIONS]
```

**Arguments**:
- `SIMULATION_DIR`: Simulation root directory

**Options**:
- `-o, --output FILE`: Output archive path (default: `SIMULATION_DIR.tar.gz`)

**Examples**:

```bash
FrictionSim2D aiida package ./simulation_output
FrictionSim2D aiida package ./simulation_output -o transfer.tar.gz
```

---

## postprocess - Result Analysis

### postprocess read

Read simulation result files, generate issue reports, and optionally export full time-series data.

```bash
FrictionSim2D postprocess read RESULTS_DIR [OPTIONS]
```

**Arguments**:
- `RESULTS_DIR`: Directory containing simulation results

**Options**:
- `--export`: Export full time-series data to JSON (default: off)

**Examples**:

```bash
FrictionSim2D postprocess read ./simulation_output
FrictionSim2D postprocess read ./simulation_output --export
```

### postprocess plot

Generate plots from processed simulation data using a JSON config file.

```bash
FrictionSim2D postprocess plot PLOT_CONFIG [OPTIONS]
```

**Arguments**:
- `PLOT_CONFIG`: JSON file describing what to plot (must contain `data_dirs`, `labels`, and `plots` keys)

**Options**:
- `-o, --output-dir DIR`: Output directory for plots (default: `plots`)
- `--settings FILE`: Plot settings JSON file (colours, font sizes, etc.)

**Examples**:

```bash
FrictionSim2D postprocess plot plot_config.json
FrictionSim2D postprocess plot plot_config.json -o ./figures --settings plot_style.json
```

---

## db - Community Database

Interact with the shared PostgreSQL database. Connection settings are resolved in order:

1. Explicit CLI flags (`--host`, `--port`, …)
2. Environment variables (`FRICTION_DB_HOST`, `FRICTION_DB_PORT`, `FRICTION_DB_NAME`, `FRICTION_DB_USER`, `FRICTION_DB_PASSWORD`)
3. The active profile in `settings.yaml`

All `db` commands accept these common connection options:

| Option | Env var | Default |
|--------|---------|---------|
| `-p, --profile {local,central}` | — | settings.yaml active profile |
| `--host` | `FRICTION_DB_HOST` | `localhost` |
| `--port` | `FRICTION_DB_PORT` | `5432` |
| `--dbname` | `FRICTION_DB_NAME` | `frictionsim2ddb` |
| `-u, --user` | `FRICTION_DB_USER` | — |
| `--password` | `FRICTION_DB_PASSWORD` | — |

### db init

Create or verify the database schema. Safe to run repeatedly.

```bash
FrictionSim2D db init [CONNECTION OPTIONS]
```

```bash
FrictionSim2D db init --profile local
```

### db create-key

Generate an API key for authenticated write access. The raw key is printed once — store it securely.

```bash
FrictionSim2D db create-key --name NAME [CONNECTION OPTIONS]
```

```bash
FrictionSim2D db create-key --name alice --profile local
```

### db upload

Upload simulation results directly to the database (requires database credentials).

```bash
FrictionSim2D db upload RESULTS_DIR [OPTIONS]
```

**Arguments**:
- `RESULTS_DIR`: Directory with completed simulation outputs

**Options**:
- `-n, --uploader NAME`: Your name or identifier (stored with each row)
- Connection options (see above)

```bash
FrictionSim2D db upload ./simulation_output --uploader alice
```

### db stage

Upload results with status `staged` for curator review before they become publicly visible. Requires an API key.

```bash
FrictionSim2D db stage RESULTS_DIR --uploader NAME [OPTIONS]
```

**Options**:
- `-n, --uploader NAME` *(required)*: Your name or identifier
- `--api-key KEY` / env `FRICTION_DB_API_KEY`: API key for write access
- Connection options (see above)

```bash
export FRICTION_DB_API_KEY=<your-key>
FrictionSim2D db stage ./simulation_output --uploader alice
```

### db query

Query the database and print matching rows.

```bash
FrictionSim2D db query [OPTIONS]
```

**Filter options**:
- `-m, --material NAME`: Filter by material
- `--type {afm,sheetonsheet}`: Filter by simulation type
- `-l, --layers N`: Filter by layer count
- `-n, --uploader NAME`: Filter by uploader
- `--limit N`: Maximum rows to return (default: 50)

**Output options**:
- `--csv FILE`: Save results to a CSV file

```bash
FrictionSim2D db query --material h-MoS2 --layers 1
FrictionSim2D db query --type afm --limit 100 --csv results.csv
```

### db stats

Show aggregate statistics for the database.

```bash
FrictionSim2D db stats [CONNECTION OPTIONS]
```

```bash
FrictionSim2D db stats --profile central
```

### db delete

Delete all rows you uploaded (matched by uploader name). Prompts for confirmation.

```bash
FrictionSim2D db delete --uploader NAME [CONNECTION OPTIONS]
```

```bash
FrictionSim2D db delete --uploader alice
```

### db publish

*(Curator action)* Promote a result from `validated` to `published`.

```bash
FrictionSim2D db publish ROW_ID [CONNECTION OPTIONS]
```

```bash
FrictionSim2D db publish 42
```

### db reject

*(Curator action)* Reject a staged or validated result.

```bash
FrictionSim2D db reject ROW_ID [OPTIONS]
```

**Options**:
- `-r, --reason TEXT`: Reason for rejection

```bash
FrictionSim2D db reject 42 --reason "Duplicate of row 37"
```

---

## api - REST API Server

### api serve

Start the FastAPI REST server so collaborators can upload and query results without direct database credentials.

```bash
FrictionSim2D api serve [OPTIONS]
```

**Options**:
- `--host HOST`: Bind address (default: `settings.database.api_host` or `0.0.0.0`)
- `-p, --port PORT`: Port number (default: `settings.database.api_port` or `8000`)
- `--profile {local,central}`: Database profile to back the server
- `--reload`: Auto-reload on code changes (development only; requires `uvicorn`)

**Setup** (run once):
```bash
FrictionSim2D db init --profile local
FrictionSim2D db create-key --name alice --profile local
FrictionSim2D api serve --host 0.0.0.0 --port 8000
```

Collaborators can then browse the interactive API docs at `http://your-server:8000/docs`.

---

## Common Workflows

### Generate and submit via AiiDA

```bash
FrictionSim2D run afm config.ini --aiida
FrictionSim2D aiida submit ./simulation_output --dry-run
FrictionSim2D aiida submit ./simulation_output
FrictionSim2D aiida status
```

### Offline HPC workflow

```bash
# Generate inputs + HPC scripts
FrictionSim2D run afm config.ini --hpc-scripts

# Transfer to cluster, run, transfer results back, then:
FrictionSim2D postprocess read ./returned_results
FrictionSim2D db upload ./returned_results --uploader alice
```

### Share results with the community

```bash
# Stage for review (needs API key)
FrictionSim2D db stage ./returned_results --uploader alice --api-key $KEY

# Query public database
FrictionSim2D db query --material h-MoS2 --format csv -o community.csv
```

---

## See Also

- [Configuration Guide](configuration_guide.md) - Config file format
- [AiiDA Workflows](aiida_workflows.md) - Detailed AiiDA usage
- [Python API Guide](python_api_guide.md) - Programmatic usage
- [Settings Reference](settings_reference.md) - Settings configuration
- [Examples](examples.md) - Complete working examples
