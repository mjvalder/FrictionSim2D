# Essentials

## Standard Lifecycle

1. Prepare a config file (`afm_config.ini` or `sheet_config.ini`).
2. Generate simulation folders with `FrictionSim2D run ...`.
3. Optionally generate HPC scripts (`--hpc-scripts` or `hpc generate`).
4. Run LAMMPS scripts on local or cluster resources.
5. Optionally postprocess and/or import results with AiiDA.

## Model Differences

AFM model:

- Uses `[2D]`, `[tip]`, `[sub]`, and `[general]` sections.
- Supports layer sweeps through `2D.layers`.
- Produces per-layer subfolders like `L1`, `L2`, `L3`.

Sheet-on-sheet model:

- Uses `[2D]` and `[general]` sections.
- Requires exactly one `layers` value and that value must be at least 3.
- Uses one simulation directory per expanded parameter set.

## Output Structure

Typical AFM root:

```text
simulation_YYYYMMDD_HHMMSS/
  afm/
    <material>/
      <size>/
        <sub_tip>/
          K<temp>/
            L1/
              lammps/
              data/
              results/
              visuals/
            L2/
            ...
            provenance/
  hpc/                # optional
  logs/
```

Typical sheet-on-sheet root:

```text
simulation_YYYYMMDD_HHMMSS/
  sheetonsheet/
    <material>/
      <size>/
        K<temp>/
          lammps/
          data/
          results/
          visuals/
          provenance/
  hpc/                # optional
  logs/
```

## Reproducibility Basics

Each simulation directory includes a provenance area with copied source files and a manifest.

Recommended minimum to archive per study:

- Original config file
- Generated simulation root
- `provenance/manifest.json`
- Settings used at generation time
- Environment description (for example, package list)
