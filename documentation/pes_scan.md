# PES scans

Static energy-versus-lateral-position maps that measure the corrugation energy barrier of an interface.

PES scans reuse the sliding-simulation geometry but replace the dynamic sliding run
with an N×N lateral grid scan over one surface unit cell. Each grid point is a single
energy minimization, so a whole scan takes seconds to minutes per material — orders of
magnitude cheaper than a sliding run. The corrugation barrier *U* is the descriptor the
Prandtl–Tomlinson picture relies on, where friction scales with barrier height.

There are two flavours.

| Scan | Command | System | Records |
|---|---|---|---|
| Interlayer | `run pes-sheet` | Frozen bilayer, no tip or substrate | Energy per surface cell |
| Tip surface | `run pes-tip` | AFM tip, sheet, and substrate | Energy plus lateral force on the tip |

## How it works

### Sheet scan

A two-layer stack is built at its equilibrium interlayer spacing. The bottom layer is
frozen (`setforce 0 0 0`); the top layer is held rigid in-plane and grid-scanned
laterally over `[0, a) × [0, b)`, the orthogonalized surface unit cell. At each point
the out-of-plane coordinate either relaxes (`z_relax = true`, the relaxed PES) or the
layers stay rigid (`z_relax = false`, the bare PES). The total energy, normalized per
surface unit cell, is written to `results/pes_scan.csv` as `dx, dy, energy_eV`.

### Tip scan

Self-contained — there is no `system.in` indentation phase. The scan assembles the
tip, sheet, and substrate, places the rigid tip at a fixed firm contact gap above the
sheet, then holds it at each grid point and records the total energy and the lateral
force on the tip to `results/pes_scan.csv` as `x, y, energy_eV, fx, fy`.

The fixed gap is used rather than a dynamic force ramp because the ramp loses atoms.
The descriptor is therefore read at a fixed firm contact, not at a normal-load
setpoint. The rendered script always uses a 3.5 Å gap; `[pes] tip_load` (default 5 nN)
is the target an external gap-calibration step uses, and nothing in the generated
script enforces it. The recorded `fz` column reports the load actually felt.

The tip scan runs on whatever sheet `[2D] layers` builds, so `layers = [2]` presses the
tip on a bilayer with no code change. The AFM stack already splits per-layer types and
adds the interlayer LJ for a multi-layer SW sheet. Descriptor extraction is identical;
the `L2/` run directory is handled.

### Freestanding tip scan

Omitting the `[sub]` section from a `pes-tip` config selects the **freestanding**
builder. A self-supported N-layer slab of the material itself replaces the amorphous
substrate — set `[2D] layers = [4]`, with `layer_1` held fixed and the upper layers
relaxed. This removes the disordered sheet-substrate registry.

Each grid point is evaluated either by a static `minimize` (`[pes] eval_mode = minimize`,
the cheapest) or by a short finite-temperature MD run with the `layer_2` Langevin band
(`eval_mode = md`, a thermalized PES). Descriptor extraction is unchanged. Example:
[../examples/afm_freestanding_pes_config.ini](../examples/afm_freestanding_pes_config.ini).

### Multi-cell tip scan

`[pes] scan_cells = M` (default 1) makes the tip scan span M×M surface unit cells at
`grid_n` points per cell. Over a disordered substrate one cell is a random patch of a
non-periodic surface; M > 1 measures the cell-to-cell spread. The sheet footprint must
clear the tip diameter plus the scan travel plus about 12 Å, or the builder warns about
overlap with the tip's periodic image.

### Convergence

Both flavours write the scan as the production-run script `lammps/slide.in`, so the
existing HPC array and combined-job generation applies unchanged.

Each grid point uses a **force-converged** minimization (`minimize 0.0 1e-6 …`). The
sliding default's relative energy tolerance is far too coarse to resolve a sub-meV
corrugation. The tip scan defaults to a fast rigid grid, `z_relax = false`.

## Configuration

Add a `[pes]` section to an otherwise ordinary sheet-on-sheet or AFM config.

```ini
[pes]
grid_n = 12          # N for the N×N lateral grid over one surface unit cell
grid_n_refine = 20   # optional finer grid, emitted as slide_refine.in, for a convergence check
z_relax = true       # sheet scan: relax the top layer's z at each grid point
tip_load = 5.0       # tip scan: target load in nN for external gap calibration (not enforced)
scan_cells = 1       # tip scan: scan M x M surface cells
eval_mode = minimize # freestanding tip scan: minimize or md
md_steps = 2000      # freestanding tip scan: steps per grid point when eval_mode = md
```

- **Sheet scan** requires `[2D] layers = [2]`. Keep `x` and `y` larger than twice the
  LJ cutoff, about 24 Å, to avoid periodic self-interaction. A few unit cells suffice.
- **Tip scan** should set `[general] force` to a single low load, for example 2 nN;
  the corrugation is clearest before ploughing sets in.

Runnable examples: [../examples/pes_sheet_config.ini](../examples/pes_sheet_config.ini),
[../examples/pes_tip_config.ini](../examples/pes_tip_config.ini), and
[../examples/afm_freestanding_pes_config.ini](../examples/afm_freestanding_pes_config.ini).

```bash
FrictionSim2D run pes-sheet examples/pes_sheet_config.ini -o ./pes_output --hpc-scripts
FrictionSim2D run pes-tip   examples/pes_tip_config.ini   -o ./pes_output --hpc-scripts
```

## Descriptor extraction

After the scans run, reduce the grids to per-material descriptors.

```bash
FrictionSim2D postprocess pes-scan ./pes_output/simulation_YYYYMMDD_HHMMSS \
    --data-dir data --grid-dir outputs/pes_scan
```

`--kind sheet|tip|both` (default `both`) selects which scans to reduce. This writes
`data/pes_scan_sheet.csv` and `data/pes_scan_tip.csv` — one row per
material, merge key `material` — and, with `--grid-dir`, the tidied grids under
`outputs/pes_scan/{sheet,tip}/<material>.csv`.

| Descriptor | Definition |
|---|---|
| `pes_barrier_U` | `max(E) − min(E)`, the Prandtl–Tomlinson corrugation barrier |
| `pes_barrier_mep` | Easiest straight-channel barrier, the easy-sliding barrier |
| `pes_corr_rms` | RMS of `E − mean(E)`, overall corrugation strength |
| `pes_barrier_aniso` | Barrier along **a** versus **b**, frictional anisotropy |
| `pes_curv_min` | Curvature of `E` at the minimum, lateral stiffness *K* |
| `pes_eta` | `2π²·U / (K·a²)`, the dimensionless PT friction parameter |
| `pes_fmax` (tip only) | `max‖(fx, fy)‖`, peak lateral static-friction force |

The descriptor maths lives in [`frictionsim2d/postprocessing/pes_scan.py`](../frictionsim2d/postprocessing/pes_scan.py).

## Related docs

- [configuration.md](configuration.md)
- [commands.md](commands.md)
- [indent.md](indent.md)
