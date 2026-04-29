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

## 4. Import a Simulation Set

Import a completed `simulation_XXXXXXXX` directory into the local AiiDA database. Stores full time-series data with provenance.

```bash
FrictionSim2D aiida import ./simulation_20260421_143404 --label "251113-afm"
```

The label is required and will be prompted if omitted.

## 5. Export (Dump) Time-Series Data

Export AiiDA-stored time-series to `output_full_*.json` files ready for `postprocess plot`:

```bash
FrictionSim2D aiida dump 251113-afm --output-dir ~/results/aiida_dump/251113-afm
```

Output: `OUTPUT_DIR/outputs/output_full_<size>.json`

This validates the full import round-trip and is the recommended way to regenerate plots from the AiiDA database.

## 6. Rebuild Simulation Inputs from Provenance

Reconstruct a complete simulation directory tree from AiiDA provenance data alone — without needing the original INI config or simulation directory:

```bash
FrictionSim2D aiida rebuild 251113-afm --output-dir ~/rebuild_test/ --hpc-scripts
```

The reconstructed tree mirrors the output of `FrictionSim2D run` and can be submitted directly to an HPC cluster.

## 7. Query Database

```bash
FrictionSim2D aiida query --material h-MoS2 --layers 3 --format table
```

Filter by simulation set:

```bash
FrictionSim2D aiida query --set 251113-afm --format table
```

Export query to CSV:

```bash
FrictionSim2D aiida query --format csv --output results.csv
```

## 8. Export/Import Archives

Create a portable AiiDA archive of the entire database:

```bash
verdi -p friction2d archive create --all friction2d_archive.aiida
```

Or export a material-specific subset:

```bash
FrictionSim2D aiida export -m h-MoS2 --output mos2_results.aiida
```

Import on another machine or profile:

```bash
FrictionSim2D aiida import-archive friction2d_archive.aiida
```

## 9. Delete Simulation Sets

Delete a single set and all its linked nodes (irreversible):

```bash
FrictionSim2D aiida delete 260312-force_rebomos
```

Delete all FrictionSim2D nodes (irreversible):

```bash
FrictionSim2D aiida clear
```

## 10. Package Simulation Inputs for Transfer

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
