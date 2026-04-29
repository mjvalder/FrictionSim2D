# Examples

These examples are written against the current CLI behavior.

## 1. Generate AFM Simulations

```bash
FrictionSim2D run afm examples/afm_config.ini --output-dir ./simulation_output
```

Result: a timestamped simulation root under `./simulation_output/`.

## 2. Generate Sheet-on-Sheet Simulations

```bash
FrictionSim2D run sheetonsheet examples/sheet_config.ini --output-dir ./simulation_output
```

## 3. Generate HPC Scripts After Build

```bash
FrictionSim2D hpc generate ./simulation_output/simulation_YYYYMMDD_HHMMSS --scheduler pbs
```

SLURM version:

```bash
FrictionSim2D hpc generate ./simulation_output/simulation_YYYYMMDD_HHMMSS --scheduler slurm
```

## 4. Generate and Include HPC Scripts In One Step

```bash
FrictionSim2D run afm examples/afm_config.ini --hpc-scripts --output-dir ./simulation_output
```

## 5. AiiDA Setup and Submission

```bash
FrictionSim2D aiida setup
FrictionSim2D aiida submit ./simulation_output/simulation_YYYYMMDD_HHMMSS
```

Dry-run first:

```bash
FrictionSim2D aiida submit ./simulation_output/simulation_YYYYMMDD_HHMMSS --dry-run
```

## 6. Import and Query AiiDA Results

Import a completed simulation directory into the local AiiDA database:

```bash
FrictionSim2D aiida import ./simulation_20260421_143404 --label "251113-afm"
```

Dump stored time-series to JSON files for plotting:

```bash
FrictionSim2D aiida dump 251113-afm --output-dir ~/results/aiida_dump/251113-afm
```

Query the database:

```bash
FrictionSim2D aiida query --material h-MoS2 --format table
FrictionSim2D aiida query --set 251113-afm --format csv --output afm_results.csv
```

Export the full database to a portable archive:

```bash
verdi -p friction2d archive create --all friction2d_results.aiida
```

## 7. Postprocess Results

Read and export full time-series JSON:

```bash
FrictionSim2D postprocess read ./simulation_output/simulation_YYYYMMDD_HHMMSS --export
```

Plot from JSON plot config:

```bash
FrictionSim2D postprocess plot plot_config.json --output-dir plots
```

## 8. Database Upload and Query

```bash
FrictionSim2D db init
FrictionSim2D db upload ./simulation_output/simulation_YYYYMMDD_HHMMSS --uploader my_name
FrictionSim2D db query --material h-MoS2 --limit 20
```

## 9. Serve REST API

```bash
FrictionSim2D api serve --host 0.0.0.0 --port 8000
```

Then inspect API docs at `/docs` on that host.

## Related Docs

- [commands.md](commands.md)
- [configuration_guide.md](configuration_guide.md)
- [aiida_workflows.md](aiida_workflows.md)
