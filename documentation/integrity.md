# Integrity check

A load point can finish its run and still not describe the physical system. With the non-reactive
Stillinger-Weber potential a sheet that should tear instead blows up, ejects atoms, splits into
islands or has a hole punched through it, usually without tripping the temperature guard. The
integrity check finds those points so they can be left out before any friction is fitted.

It judges only whether the system is intact. Ploughing, tip deformation, pinned sheets and
departures from steady sliding are not judged; they stay part of your analysis.

The rules are a port of the checker used for the published run campaigns. `tools/integrity/compare_reference.py`
compares the port with that reference on the two campaigns: on every point both call `complete`,
the rule set and the snapshot measures are identical.

## Three stages

1. **In LAMMPS.** The temperature guard (`halt_temperature_factor`, default 10 times the target)
   ends a runaway load point and records it in `halted.txt`. The script then continues.
2. **Status from the files.** `DataReader` gives every configured point one of `complete`,
   `truncated`, `unreadable`, `no_output`, `halted` or `not_reached`.
3. **Integrity rules.** Run on every `complete` point. The point becomes `complete`,
   `failed_integrity` or `quarantined`. Points with any other status pass through unchanged.

A stage 3 failure never turns later loads into `not_reached`: the later loads ran, and their data
is kept.

## Rules

A point fails when any rule fires. Thresholds are dimensionless, scaled by a quantity of the run
itself. The bond length of a layer is its median nearest-neighbour distance in the reference
structure. Setting a threshold to `null` turns that rule off.

| rule | fires when | setting (default) |
|---|---|---|
| `non_finite` | any NaN or infinity in the core channels, or in `etotal` | always on |
| `frozen` | a core channel has a standard deviation below the limit | `frozen_std` (1e-6) |
| `hole` | the largest atom-free disk in a layer's xy projection, end of the slide, has radius at least this many bonds | `hole_bonds` (1.5) |
| `isolated_atoms` | at least this many sheet atoms, summed over layers, have no same-layer neighbour within `isolated_radius_bonds` | `isolated_atoms` (2), `isolated_radius_bonds` (1.5) |
| `ejected_atoms` | at least this many sheet atoms lie more than `ejected_bonds` from their layer's median plane | `ejected_atoms` (2), `ejected_bonds` (8.0) |
| `sheet_heating` | peak sheet temperature is at least this many sigma_T above target, sigma_T = T sqrt(2 / 3N) | `heating_sigma` (8.0) |
| `gap_rise` | the interlayer gap moves by at least this fraction of its starting value (needs two layers, three for sheet-on-sheet) | `gap_rise` (0.05) |
| `collapse` | any adjacent-layer gap is negative: the layers passed through each other (sheet-on-sheet only) | `collapse` (true) |

The snapshot rules (`hole`, `isolated_atoms`, `ejected_atoms`) read the end-of-slide `data/final_*.data`.
The others read the point's time series. A single stray atom is not a failure, which is why the
atom counts need at least 2. Bonds stretched under the tip without a hole are not a failure either.

### AFM and sheet-on-sheet differ

Both families get every rule. The inputs differ:

| | AFM family (afm, indent, pes_tip) | sheet-on-sheet family (sheetonsheet, hetero, flake, pes_sheet) |
|---|---|---|
| temperature channel | `temp_sheet_3d`, N = all sheet atoms | `temp_center_3d`, N = atoms of the centre layers (all but the first and last) |
| heating peak | over the whole window | after the first `sheet_heating_skip_samples` samples |
| gap | layer 2 minus layer 1 centre of mass (`comz_L2 - comz_L1`), for any stack of two or more layers | shear interface, one layer above minus the one below (`comz_L{k+1} - comz_L{k}`, `k = n_layers // 2`) |
| `gap_rise` | the highest rise over the start | a lasting shift of either sign: the median of the last `sheet_gap_end_samples` against the start |
| `collapse` | not applied | applied to every adjacent pair |
| core channels | `nf lfx lfy comx comy comz tipx tipy tipz` | `lfx lfy nf` and `comz_L<k>` of each layer; layer 1 is fixed, so is exempt from `frozen` |
| bond reference | `data/load_<F>N.data`, the loaded state of that load; else the first structure in `build/` | the first structure in the run's `build/` (the built sheet; no loaded state is written) |

Monolayers have no gap, so `gap_rise` and `collapse` do not apply to them.

A **2-layer sheet-on-sheet stack** (a bilayer hetero, flake or sheetonsheet point) gets fewer
rules than the table suggests. This is faithful to ML_taxonomy s51 and is not a bug:

| rule | 2-layer sheet stack | why |
|---|---|---|
| `gap_rise` | not applied | the shear-interface gap needs `n_layers >= 3` |
| `sheet_heating` | not applied | `n_heat = 0`: the temperature uses the centre layers only (all but the first and last), and a bilayer has none |
| `collapse`, `frozen`, snapshot rules | applied | |

So a bilayer sheet point is judged only on the gate, `collapse` and the snapshot rules. A
passing bilayer verdict does not mean its gap or heating was checked.

The sheet temperature uses each point's own atom count N.

## Statuses

Every point carries two flags besides its status:

| status | failed | observed |
|---|---|---|
| `complete` | no | yes |
| `quarantined` | no | yes |
| `failed_integrity` | yes | yes |
| `truncated`, `halted`, `unreadable` | yes | yes |
| `no_output`, `not_reached` | yes | no |

`failed` counts a point that never ran as a failure. `observed` says whether data exists for it.
Partial data of a failed point is kept, `halted` included.

`complete_only` exports (the `--complete-only` flag) keep `complete` points only, so they drop
`quarantined` points too.

## Quarantine and rulings

A **quarantined** point is held out of results until you rule on it. It is neither failed nor
passed. A point is quarantined when it did not fail a rule but:

- it is within a margin of a threshold (see below), or
- its snapshot rules could not run: the final snapshot or bond reference is missing or unreadable,
  or the layer types could not be resolved.

A run whose decks write no end snapshot, or a run with more than one speed (one `final_*` file
cannot tell the speeds apart), is not quarantined. Its snapshot rules are marked not applicable
and the detail says so.

**Margins.** `integrity.margins` maps a rule name to a fraction `m`. A point that passes is
quarantined when its value lies in `[m * threshold, threshold)`. Margins never apply to
`non_finite`, `frozen` or `collapse`. The margins are available but empty by default, so with
the default settings (rule set version 1.0.0) nothing is quarantined on a margin, only on missing
or unreadable inputs. Typical healthy values: an intact sheet already measures 0.75 to 1.0 bonds
for `hole`, so a margin of 0.6 or below puts most of a campaign in quarantine.

**Rulings.** `postprocess read` prints a stub for each quarantined point:

```
Integrity: 12 failed, 3 quarantined (thresholds v1.0.0)
- point: "afm/h_MoS2/.../L1|force=30|angle=0|speed=2"
  ruling:   # pass or fail -- hole 1.4 bonds within margin 0.9 of 1.5
```

Copy the stubs into `<results_dir>/integrity_rulings.yaml` and fill them in:

```yaml
- point: "afm/h_MoS2/.../L1|force=30|angle=0|speed=2"
  ruling: pass            # or fail
  note: looked at the snapshot, the hole is a viewer artefact
  date: 2026-11-12
```

On the next read, `pass` makes the point `complete` and `fail` makes it `failed_integrity`. Its
verdict is marked `ruled_by_user` and the status detail records the note.

### Overriding any verdict

You are the expert, so a ruling can also overrule a point the checker did not quarantine. Add
`override: true` and a non-empty `note`:

```yaml
- point: "afm/h_MoS2/.../L1|force=30|angle=0|speed=2"
  ruling: pass
  override: true          # required unless the point is quarantined
  note: the ejected atoms are a known thermostat artefact
  date: 2026-11-16
```

- `failed_integrity` or `complete`: `pass` gives `complete`, `fail` gives `failed_integrity`.
- `truncated`, `halted`, `unreadable`: `pass` gives `complete` (you accept the point); `fail` keeps the status.
- `no_output`, `not_reached`: `pass` is refused, there is no data to accept; `fail` keeps the status.
- Without `override: true`, or with an empty `note`, the ruling changes nothing and `postprocess read`
  prints one warning naming what is missing.

An overridden verdict has `overridden: true` and `original_status` (also in the export's `integrity`
block); its status detail reads `overridden by user <date>: <note>`. `postprocess read` prints
`Overrides: N` and one line per point (id, original -> new status, note); the assistant lists them
under `overridden by you`.

A ruling on a point that does not exist gives a warning. Unreadable or duplicate entries also give
a warning; for a duplicate, the last entry wins.

The AiiDA store applies the same file and the same rules: every `aiida import` checks each new
point and then applies the set's `integrity_rulings.yaml`, and `aiida rule LABEL` re-applies it on
demand. There, removing a ruling and re-applying the file withdraws it. See
[aiida.md](aiida.md#5-integrity-verdicts-and-rulings).

## Changing thresholds

The thresholds come from the `integrity:` block of your settings (see [settings.md](settings.md#integrity)).
For one read, point `--integrity-settings` at a YAML file holding an `integrity:` mapping. That
mapping replaces the whole `integrity` block of your settings: a key it omits falls back to its
built-in default (the table above), not to the value in your settings file.

```yaml
# strict.yaml
integrity:
  hole_bonds: 1.2
  heating_sigma: 6.0
  margins:
    sheet_heating: 0.7
```

```bash
FrictionSim2D postprocess read ./simulation_output --integrity-settings strict.yaml --export
FrictionSim2D postprocess read ./simulation_output --no-integrity     # stage 3 off
```

Thresholds are applied at read time, never frozen into the run. The ones used are written into
the export metadata as `integrity_thresholds`, with the rule set `version`.

## Reading `status_detail`

A point that fails, is quarantined or skips a rule gets an entry in `status_detail`, in the
`output_full_*.json` and in the CLI listing:

```
integrity: hole 3.6 bonds; sheet_heating 48 sigma_T
```

- Each fired rule is `<rule> <measured value><unit>`; `non_finite` and `frozen` carry no value.
- A quarantine reason follows, for example `hole 1.4 bonds within margin 0.9 of 1.5` or
  `missing final snapshot (data/final_*.data)`.
- A skip note follows, for example `snapshot rules skipped: multi-speed run`.
- A ruling adds `ruled pass by user 2026-11-12: <note>`.
- A point that failed at stage 2 keeps its own detail; the rules it fires are listed after it.

The full measurements of each point are in the `integrity` block of its export entry: `status`,
`failed`, `observed`, `reasons`, `quarantine`, `not_applicable`, `measures` (`hole`, `isolated`,
`ejected`, `T_sigma`, `gap_rise`, `gap_min_any`, `n_heat`), `thresholds_version` and
`ruled_by_user`. A `null` measure means the rule was not applicable, never a pass.

## Cost

About half a second per point for the snapshot rules, run one point after another.
Use `--no-integrity` (or `integrity=False` in `DataReader`) to skip stage 3.
