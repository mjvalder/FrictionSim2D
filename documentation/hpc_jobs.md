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
2. Confirm module list and MPI launcher in settings.
3. Confirm walltime/resources match queue policies.
4. Confirm script discovery includes expected `system.in`/`slide*.in` files.

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
- [overview.md](overview.md)
