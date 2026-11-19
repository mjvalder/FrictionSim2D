# HPC jobs

Scheduler script generation and the two-phase submission model.

## Why two phases exist

AFM workflows usually need initialization before sliding:

1. `system.in` phase
2. `slide*.in` phase

Generated scripts separate these phases so slide jobs start only after successful initialization.

## Generated files

In `hpc/` you can expect files such as:

- `manifest.json`
- `manifest_system.txt` (AFM)
- `manifest_slide.txt`
- `run_system.pbs` or `run_system.sh` (AFM)
- `run_slide.pbs` or `run_slide.sh`
- `submit_jobs.sh`

Exact names can vary with scheduler and script selection.

### MLIP (GPU) job groups

A campaign that mixes classical and MLIP runs (`pot_type = mace | deepmd`) is
split into one job group per `(needs_gpu, framework)`, read back from each
run's `pot_<framework>[-d3]_<stem>` path segment. Each group gets its own script
and manifest with a group suffix (`run_cpu.pbs`, `run_gpu_mace.pbs`,
`manifest_combined_gpu_deepmd.txt`, two-phase `run_system_cpu.sh`, ...); a
single-group campaign keeps the plain names. MLIP groups request one MPI rank
plus `hpc.gpu`, use the framework's binary and launcher flags from
`hpc.frameworks`, and MACE groups export `OMP_NUM_THREADS` = the requested
cores. See [mlip.md](mlip.md) §8.

### The manifest is a script list

`manifest.json` (`src/hpc/manifest.py`) records, per generated LAMMPS script,
which simulation it belongs to and which script within it (`system`/`slide`,
material, layers, force, angle, speed). It is a **script list**, not a live
job tracker: there is no AiiDA-uuid field linking an entry back to a stored
node, and none of the status-mutation methods that used to update a job's
lifecycle (`submitted` → `running` → `completed`) as jobs progressed remain —
FrictionSim2D is not installed on the cluster in the normal workflow, so
nothing local was ever in a position to update those fields anyway. The
script-list half — which script belongs to which simulation, in what order —
is what `hpc_dispatch` actually needs, and that stays.

## Submission model

AFM:

- Submit system phase first.
- Submit slide phase with scheduler dependency on system completion.

Sheet-on-sheet:

- Usually slide-only phase.

## Generate scripts

From a generated simulation root:

```bash
FrictionSim2D hpc generate ./simulation_output/simulation_YYYYMMDD_HHMMSS --scheduler pbs
```

Or produce scripts during generation:

```bash
FrictionSim2D run afm afm_config.ini --hpc-scripts
```

## Checks before submission

1. Confirm scheduler type (`pbs` or `slurm`).
2. Confirm module list and MPI launcher in settings (for MLIP groups also
   `hpc.gpu` and the `hpc.frameworks.<framework>.binary`).
3. Confirm walltime/resources match queue policies.
4. Confirm script discovery includes expected `system.in`/`slide*.in` files.

## Getting results back into AiiDA: remote-manual

The supported "remote" workflow when the cluster blocks direct AiiDA
connections (which is the normal case — see [aiida.md](aiida.md)) is
**remote-manual**: rsync out, run on the cluster with the generated scripts,
rsync the results back, then ingest locally with a plain import — no AiiDA
component ever runs on the cluster itself:

```bash
# 1. generate + hpc scripts, then transfer out
FrictionSim2D run afm afm_config.ini --hpc-scripts -o ./simulation_output
rsync -avzhP ./simulation_output/simulation_YYYYMMDD_HHMMSS HPC_HOST:HPC_HOME/

# 2. submit on the cluster (qsub/sbatch the generated scripts), then bring results home
rsync -avzhP --include="*/" --include="results/***" --include="data/***" \
      --include="visuals/***" --include="run*.log" --exclude="*" \
      HPC_HOST:HPC_HOME/simulation_YYYYMMDD_HHMMSS ./

# 3. ingest the finished tree
FrictionSim2D aiida import ./simulation_YYYYMMDD_HHMMSS --label 251113-afm
```

If AiiDA itself is installed in a conda environment on the HPC instead, use
hpc-direct mode: run and `aiida import` there, `aiida export` to build a
portable archive, and `aiida import-archive` it on your workstation. See
[aiida.md](aiida.md) for both modes.

## Troubleshooting

No simulation directories found:

- Ensure generated folders contain `lammps/` and at least one `.in` script.

Unexpected script ordering:

- Review `hpc.lammps_scripts` in settings and generated manifests.

Dependency errors at submit time:

- Check scheduler syntax and whether wrapper script captured the first job ID correctly.

## Related docs

- [commands.md](commands.md)
- [settings.md](settings.md)
- [aiida.md](aiida.md)
- [overview.md](overview.md)
