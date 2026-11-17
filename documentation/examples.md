# Examples

End-to-end workflows, from generating a simulation tree through to plots and the AiiDA store.

## 1. Generate AFM simulations

```bash
FrictionSim2D run afm examples/afm_config.ini --output-dir ./simulation_output
```

Result: a timestamped simulation root under `./simulation_output/`.

## 2. Generate sheet-on-sheet simulations

```bash
FrictionSim2D run sheetonsheet examples/sheet_config.ini --output-dir ./simulation_output
```

### Sweeping over heterostructure pairs

`[2D-1]`/`[2D-2]` stack two materials, and a material list of `mat1 + mat2`
lines sweeps the pair:

```bash
FrictionSim2D run sheetonsheet examples/hetero_pair_sweep.ini --output-dir ./hetero_output
```

The three lines in `examples/hetero_pairs.txt` become three runs, one per pair,
under `sheetonsheet/h-MoS2+h-WS2/`, `sheetonsheet/h-MoSe2+h-WSe2/` and
`sheetonsheet/h-MoTe2+h-WTe2/`. See
[configuration.md](configuration.md#heterostructure-pair-sweeps) for the list
format and which models accept these sections.

## 3. Generate PES scans

Interlayer scan on a frozen bilayer, then a tip-on-sheet scan:

```bash
FrictionSim2D run pes-sheet examples/pes_sheet_config.ini --output-dir ./pes_output
FrictionSim2D run pes-tip   examples/pes_tip_config.ini   --output-dir ./pes_output
```

Reduce the finished grids to per-material descriptors:

```bash
FrictionSim2D postprocess pes-scan ./pes_output/simulation_YYYYMMDD_HHMMSS \
    --data-dir data --grid-dir outputs/pes_scan
```

See [pes_scan.md](pes_scan.md).

## 4. Generate an indentation run

```bash
FrictionSim2D run indent examples/indent_config.ini --output-dir ./indent_output
```

Reduce the finished runs to penetration and adhesion descriptors:

```bash
FrictionSim2D postprocess indent ./indent_output/simulation_YYYYMMDD_HHMMSS \
    --out-dir outputs/poke_sims --data-dir data
```

See [indent.md](indent.md).

## 5. Generate HPC scripts after the build

```bash
FrictionSim2D hpc generate ./simulation_output/simulation_YYYYMMDD_HHMMSS --scheduler pbs
```

SLURM version:

```bash
FrictionSim2D hpc generate ./simulation_output/simulation_YYYYMMDD_HHMMSS --scheduler slurm
```

## 6. Generate simulations and HPC scripts in one step

```bash
FrictionSim2D run afm examples/afm_config.ini --hpc-scripts --output-dir ./simulation_output
```

## 7. AiiDA setup

AiiDA only records finished runs; nothing is submitted through it. Create the
profile once:

```bash
FrictionSim2D aiida setup
```

## 8. Import and query AiiDA results

Import a finished simulation directory into the local AiiDA store (one CalcJob
per LAMMPS deck, an integrity verdict per point). Preview it first:

```bash
FrictionSim2D aiida import ./simulation_20260421_143404 --label 251113-afm --dry-run
FrictionSim2D aiida import ./simulation_20260421_143404 --label 251113-afm
```

A tree built before provenance schema 2.2.0 needs a `run_uuid` first:

```bash
FrictionSim2D aiida stamp-uuid ./simulation_20260421_143404
```

Summarise the complete points, collect them into a dataset, and compare two
sets:

```bash
FrictionSim2D aiida postprocess 251113-afm
FrictionSim2D aiida dataset mos2-afm 251113-afm
FrictionSim2D compare 251113-afm 251120-afm
```

Write a stored set back out as an ordinary simulation tree, ready for `postprocess read`/plotting:

```bash
FrictionSim2D aiida export-results 251113-afm ~/results/exported/251113-afm
```

Query the database:

```bash
FrictionSim2D aiida query --material h-MoS2 --format table
FrictionSim2D aiida query --set-label 251113-afm --format csv --output afm_results.csv
```

Export one simulation set to a portable archive (with `.params/` sidecars and
a README; HPC details anonymised):

```bash
FrictionSim2D aiida export 251113-afm -o friction2d_results.aiida
```

## 9. Postprocess results

Read and export full time-series JSON:

```bash
FrictionSim2D postprocess read ./simulation_output/simulation_YYYYMMDD_HHMMSS --export
```

Plot from JSON plot config:

```bash
FrictionSim2D postprocess plot plot_config.json --output-dir plots
```

## 10. Sharing results

The former central database and REST API (`FrictionSim2D db ...`,
`FrictionSim2D api ...`) were removed. Share results as AiiDA archives:
`FrictionSim2D aiida export` on one side, `FrictionSim2D aiida import-archive`
on the other (see [aiida.md](aiida.md#9-archives)).

## Related docs

- [commands.md](commands.md)
- [configuration.md](configuration.md)
- [aiida.md](aiida.md)
