# Indentation simulations

A single run per condition that measures both vertical-contact quantities the lateral PES scans cannot see.

An **indent** run is the standard AFM simulation — tip, sheet, and substrate, with the
indentation ramp in `system.in` — run only up to the load, then held and retracted. The
sliding phase is never invoked. Per load it performs two phases back to back.

1. **Hold (penetration).** After the aveforce ramp presses the tip to the target load,
   a short damped finite-temperature hold records the out-of-plane channels. The
   thermally averaged **penetration depth**, `tip_z − sheet_com_z`, is the
   vertical-compliance descriptor: a deeper, softer contact means lower friction.
2. **Retract (adhesion).** The MD fixes are torn down, the rigid tip is frozen and
   retracted quasi-statically at T = 0 in small z-steps to full detachment, tracing the
   force–distance curve for **pull-off force**, **work of adhesion**, and **contact
   stiffness** — the one contact quantity neither sliding nor hold data holds.

This one simulation type subsumes the former separate `poke` (hold-only) and `adhesion`
(retract-only) commands: because the retract starts from the held contact, both
descriptor sets come out of one job.

## How it works

The indent reuses the full AFM build and the `system.in` indentation unchanged.
`system.in` builds the stack, minimises, and runs the aveforce ramp to each target
load, writing one `load_<f>N.data` per load. The run phase then swaps the sliding script
for `afm/indent.lmp`, emitted as `slide.in`, which per load does the following.

**Phase 1 — hold**

1. Reads `load_<f>N.data`, the loaded contact state.
2. Re-applies the target normal load via `aveforce` and holds the rigid, laterally
   fixed tip, free only in *z* — a pure vertical hold, no slide.
3. Records `nf`, `comz` (sheet COM *z*), and `tipz` (tip COM *z*) via `fix ave/time` at
   the 1 ps campaign cadence for a `hold_steps` damped hold, 10 000 steps or 10 ps by
   default. The output uses the same ten-column
   `results/friction_f<load>_a0_s<speed>_layer<N>.txt` layout as the sliding campaign,
   so the campaign reader ([`read_data.py`](../src/postprocessing/read_data.py)) parses
   it unchanged.

**Phase 2 — retract**

4. Unfixes every dynamics fix from the hold — langevin, aveforce, viscous drag, nve,
   the rigid tip — so the minimization sees only conservative forces.
5. Freezes the rigid tip (`tip_all`) with `setforce` and force-converged `minimize`s the
   held contact to a clean reference.
6. Repeats `n_steps` times: `displace_atoms` moves the tip up by `z_step`, then a
   force-converged `minimize` runs and `z_tip, fz_tip, fx/fy_tip, pe` are recorded —
   tracing the pull-off force–distance curve to `results/indent_f<load>N_l<layer>.csv`.

Everything downstream — two-phase and combined HPC job generation, `system.in` to
`slide.in` sequencing — applies with no change.

> **Two ensembles, by design.** The hold is finite-temperature damped MD, so the
> penetration is a thermal average; the retract is T = 0 minimization, a quasi-static
> force–distance curve. Phase 2 tears down the thermostat, aveforce, and viscous fixes
> before minimising so langevin random forces cannot corrupt the sub-nN pull-off signal.

### Velocity seeds

Each run needs a fresh velocity seed. A single campaign base seed, `[indent] seed`, fans
out to a distinct, reproducible seed per `(material, layer)` — SHA-256 of
`base:material:layers`. Each is injected into that run's `system.in` `velocity create`
and recorded in the manifest. Omit `[indent] seed` to draw a fresh base seed at build
time.

## Configuration

Add an `[indent]` section to an otherwise ordinary AFM config. Loads come from
`[general] force`, layers from `[2D] layers`, and the angle from `[general] scan_angle`.

```ini
[general]
force = [5, 10, 20, 30, 50]   # loads in nN
scan_angle = 0
outer_loop = force            # one HPC job per load; all share the system.in ramp

[tip]
dspring =                     # blank -> auto viscous damping, so the tip settles in the hold

[indent]
hold_steps = 10000            # 10 ps damped hold at the target load
z_step = 0.2                  # retract increment per step, Å; 0.1-0.2 Å recommended
n_steps = 100                 # retract steps; n_steps x z_step = 20 Å total
seed = 20260710               # campaign base seed; per-run seeds are derived from it
```

- Leave `[tip] dspring` blank so the tip's viscous damping is auto-set to a fraction of
  critical and it settles quasi-statically within the hold.
- `n_steps × z_step`, 20 Å by default, is the total retract distance. It must
  comfortably exceed the adhesive tail so the curve reaches full detachment. The
  postprocessor warns if a curve's `fz_tip` minimum sits at the last steps.

Runnable example: [../examples/indent_config.ini](../examples/indent_config.ini).

```bash
FrictionSim2D run indent examples/indent_config.ini -o ./indent_output --hpc-scripts
```

## Postprocessing

After the runs finish, reduce them to both descriptor sets in one pass.

```bash
FrictionSim2D postprocess indent ./indent_output/simulation_YYYYMMDD_HHMMSS \
    --out-dir outputs/poke_sims --data-dir data
```

| File | Contents |
|---|---|
| `outputs/poke_sims/output_indent_f<load>_s<seed>.json` | **Penetration** — one per load and seed batch, nested campaign schema `results → material → size → substrate → tip_mat → r25 → l{layer} → s{speed} → f{load} → a0`, each leaf `{columns, data}` with columns including `nf, comz, tipz` |
| `data/indent_features.csv` | **Adhesion** — one row per `(material, layer, load)`, merge key `material`, holding `pulloff_force`, `work_of_adhesion`, `contact_stiffness`, plus `f_adh_min`, `snapoff_z`, `work_of_adhesion_nnA`, `detached` |
| `outputs/poke_sims/indent/<material>_l<layer>.csv` | The tidied per-condition retract curve: `step, z_tip, fz_tip, fx_tip, fy_tip, pe` |
| `outputs/poke_sims/indent_manifest.csv` | One row per condition with `seed`, `hold_steps`, `z_step`, `n_steps`, and best-effort `wall_time` |

Adhesion descriptors, where `fz_tip` is positive when repulsive and negative when
adhesive, and `z_tip` increases as the tip retracts:

| Descriptor | Definition |
|---|---|
| `pulloff_force` | `−min(fz_tip)`, the peak tensile adhesive force, nN |
| `work_of_adhesion` | `∫` over the attractive branch of `(−fz_tip) dz`, the energy to separate, eV |
| `contact_stiffness` | Slope of the leading repulsive wall of `fz_tip` versus `z_tip`, nN/Å |
| `snapoff_z` | Tip COM *z* at the pull-off minimum, Å |

Snap-in distance and approach/retract hysteresis need the approach branch, which this
retract does not record — the approach is the `system.in` aveforce ramp, which stores no
per-step curve.

The delivery maths lives in [`src/postprocessing/indent.py`](../src/postprocessing/indent.py),
the builder in [`src/builders/indent.py`](../src/builders/indent.py), and the run-phase
template in [`src/templates/afm/indent.lmp`](../src/templates/afm/indent.lmp).

## Checks before trusting a run

- Penetration must deepen monotonically with load per `(material, layer)`. A
  non-monotone curve signals an unconverged hold or contact loss.
- Every retract curve must reach detachment, `detached = 1`. The postprocessor warns
  when `fz_tip`'s minimum is at the last steps; raise `[indent] n_steps` or `z_step`.

## Related docs

- [configuration.md](configuration.md)
- [commands.md](commands.md)
- [pes_scan.md](pes_scan.md)
