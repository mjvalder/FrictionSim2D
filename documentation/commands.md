# Commands Reference

This document reflects the current CLI in `src/cli.py`.

## Top Level

```bash
FrictionSim2D --help
FrictionSim2D --version
```

## run

### run afm

```bash
FrictionSim2D run afm CONFIG_FILE [OPTIONS]
```

Options:

- `-o, --output-dir TEXT` (default: `simulation_output`)
- `--aiida`
- `--hpc-scripts`

### run sheetonsheet

```bash
FrictionSim2D run sheetonsheet CONFIG_FILE [OPTIONS]
```

Options are the same as `run afm`.

## settings

### settings show

```bash
FrictionSim2D settings show
```

### settings init

```bash
FrictionSim2D settings init
```

### settings reset

```bash
FrictionSim2D settings reset
```

## hpc

### hpc generate

```bash
FrictionSim2D hpc generate SIMULATION_DIR [OPTIONS]
```

Options:

- `-s, --scheduler {pbs|slurm}` (default: `pbs`)
- `-o, --output-dir TEXT` (default: `SIMULATION_DIR/hpc`)

## aiida

Note: AiiDA commands require AiiDA dependencies.

### aiida status

```bash
FrictionSim2D aiida status
```

### aiida setup

```bash
FrictionSim2D aiida setup [OPTIONS]
```

Options:

- `-p, --profile TEXT`
- `--lammps-path PATH`
- `--hpc-config PATH` (deprecated)
- `--use-remote`

### aiida submit

```bash
FrictionSim2D aiida submit SIMULATION_DIR [OPTIONS]
```

Options:

- `-c, --code TEXT`
- `--scripts TEXT` (comma-separated script list)
- `--array`
- `--machines INTEGER`
- `--mpiprocs INTEGER`
- `--walltime TEXT`
- `--queue TEXT`
- `--project TEXT`
- `--dry-run`

### aiida import

```bash
FrictionSim2D aiida import RESULTS_DIR [--process/--no-process]
```

### aiida query

```bash
FrictionSim2D aiida query [OPTIONS]
```

Options:

- `-m, --material TEXT`
- `-l, --layers INTEGER`
- `-f, --force FLOAT`
- `--format {table|csv|json}`
- `-o, --output PATH`

### aiida export

```bash
FrictionSim2D aiida export [OPTIONS]
```

Options:

- `-o, --output PATH` (default: `friction2d.aiida`)
- `-m, --material TEXT`

### aiida import-archive

```bash
FrictionSim2D aiida import-archive ARCHIVE_PATH
```

### aiida package

```bash
FrictionSim2D aiida package SIMULATION_DIR [OPTIONS]
```

Options:

- `-o, --output PATH`

## postprocess

### postprocess read

```bash
FrictionSim2D postprocess read RESULTS_DIR [--export]
```

### postprocess plot

```bash
FrictionSim2D postprocess plot PLOT_CONFIG [OPTIONS]
```

Options:

- `-o, --output-dir TEXT` (default: `plots`)
- `--settings PATH` (JSON plot settings)

## db

Database commands accept connection/profile options. See `FrictionSim2D db --help` for full argument list.

### db init

```bash
FrictionSim2D db init [DB OPTIONS]
```

### db create-key

```bash
FrictionSim2D db create-key --name NAME [DB OPTIONS]
```

### db setup

```bash
FrictionSim2D db setup [--name NAME] [--profile PROFILE] [DB OPTIONS]
```

Initializes schema, creates an API key, and stores it in `~/.config/FrictionSim2D/settings.yaml`.

### db upload

```bash
FrictionSim2D db upload RESULTS_DIR [--uploader NAME] [DB OPTIONS]
```

Current behavior: each uploaded row is validated automatically. Valid rows are published; invalid rows are rejected.

### db stage

```bash
FrictionSim2D db stage RESULTS_DIR --uploader NAME [--api-key KEY] [DB OPTIONS]
```

Current behavior: staged rows are validated automatically and then published when valid (or rejected when invalid).

### db query

```bash
FrictionSim2D db query [OPTIONS] [DB OPTIONS]
```

Important options:

- `--material TEXT`
- `--type TEXT`
- `--layers INTEGER`
- `--uploader TEXT`
- `--limit INTEGER`
- `--csv PATH`

### db stats

```bash
FrictionSim2D db stats [DB OPTIONS]
```

### db delete

```bash
FrictionSim2D db delete --uploader NAME [DB OPTIONS]
```

### db publish

```bash
FrictionSim2D db publish ROW_ID [DB OPTIONS]
```

### db reject

```bash
FrictionSim2D db reject ROW_ID [--reason TEXT] [DB OPTIONS]
```

## api

### api serve

```bash
FrictionSim2D api serve [OPTIONS]
```

Options:

- `--host TEXT`
- `-p, --port INTEGER`
- `--profile TEXT`
- `--reload`

## Practical Notes

- For the latest generated command usage, run `FrictionSim2D <group> <command> --help`.
- `run` commands currently expose `--hpc-scripts` and do not expose `--hpc` or `--local` flags.
- `postprocess plot` expects a JSON plot configuration file, not a results directory path.
- API `POST /results` submissions run automatic validation and will end in `published` or `rejected`.

## Related Docs

- [configuration_guide.md](configuration_guide.md)
- [examples.md](examples.md)
- [aiida_workflows.md](aiida_workflows.md)
