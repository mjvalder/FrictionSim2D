# AiiDA Workflows

FrictionSim2D integrates with AiiDA for provenance-aware submission, import, query, and archive exchange.

## 1. One-Time Setup

Check install:

```bash
FrictionSim2D aiida status
```

Initialize profile/computer/code:

```bash
FrictionSim2D aiida setup
```

Remote setup using settings:

```bash
FrictionSim2D aiida setup --use-remote
```

`--hpc-config` is still accepted but marked deprecated by the CLI.

## 2. Generate Simulations

```bash
FrictionSim2D run afm afm_config.ini --output-dir ./simulation_output --aiida
```

## 3. Submit Generated Runs

Minimal submit (auto-detect code/defaults):

```bash
FrictionSim2D aiida submit ./simulation_output/simulation_YYYYMMDD_HHMMSS
```

Common overrides:

```bash
FrictionSim2D aiida submit ./simulation_output/simulation_YYYYMMDD_HHMMSS \
  --machines 2 --mpiprocs 32 --walltime 24:00:00 --queue normal
```

Preview without submitting:

```bash
FrictionSim2D aiida submit ./simulation_output/simulation_YYYYMMDD_HHMMSS --dry-run
```

## 4. Import Completed Results

```bash
FrictionSim2D aiida import ./returned_results
```

Skip processing pass:

```bash
FrictionSim2D aiida import ./returned_results --no-process
```

## 5. Query Database

```bash
FrictionSim2D aiida query --material h-MoS2 --layers 3 --format table
```

Export query to CSV:

```bash
FrictionSim2D aiida query --format csv --output results.csv
```

## 6. Export/Import Archives

Export:

```bash
FrictionSim2D aiida export --output friction2d.aiida
```

Import on another machine/profile:

```bash
FrictionSim2D aiida import-archive friction2d.aiida
```

## 7. Package Simulation Inputs for Transfer

```bash
FrictionSim2D aiida package ./simulation_output/simulation_YYYYMMDD_HHMMSS --output bundle.tar.gz
```

This excludes `.lammpstrj` files by design.

## Recommended Modes

- Offline: generate and run manually, import later.
- Local: AiiDA daemon and compute endpoint on same environment.
- Remote: local controller with SSH-backed remote computer.

Choose based on cluster policy and authentication constraints.

## Related Docs

- [commands.md](commands.md)
- [settings_reference.md](settings_reference.md)
- [HPC_TWO_PHASE_JOBS.md](HPC_TWO_PHASE_JOBS.md)
