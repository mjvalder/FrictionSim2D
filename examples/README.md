# Examples

One folder per `FrictionSim2D run` subcommand, one file per feature. Every file
starts with what it demonstrates and the exact command that runs it. Run them
from the repository root:

```bash
FrictionSim2D run afm examples/afm/basic.ini -o ./afm_output
```

Copy the closest example and change one thing at a time. Lengths are in Å,
forces in nN, speeds in m/s, pressures in GPa.

These files are also the test suite's fixtures: `tests/helpers/examples.py`
discovers every `examples/<mode>/*.ini`, and
`tests/integration/test_example_configs.py` builds each one, so a new file here
is tested as soon as it exists. Tests that pin an example's exact values read
the frozen copy in `tests/fixtures/examples/` instead.

**Needs** column:
- *lammps* -- a heterostructure: building it runs `lmp_serial` to relax the
  coincidence cell, so LAMMPS must be on `PATH` at build time.
- *settings* -- pass the settings file named in its header with
  `--settings-file` (the option is a `settings.yaml` key, not an `.ini` one).
- *MLIP model* -- names a machine-learned potential file that is not shipped;
  point `pot_path` at your own model.

## afm/ -- tip sliding on a supported sheet (`run afm`)

| file | shows | needs |
|---|---|---|
| `afm/basic.ini` | tip + sheet + amorphous substrate; load and layer sweeps | |
| `afm/flake.ini` | a shaped flake between the tip and the sheet | |
| `afm/finite_sheet.ini` | a finite sheet (island, free edges) on a larger substrate | |
| `afm/finite_sheet_spring_edges.ini` | the same island with spring-tethered edge atoms | settings |
| `afm/crystalline_substrate.ini` | a crystalline Si substrate under a finite sheet | |
| `afm/hetero.ini` | MoS2/WS2 heterostructure on a lattice-matched crystalline substrate | lammps |
| `afm/hetero_finite.ini` | a finite MoS2/WS2 island on an amorphous substrate | lammps |
| `afm/rebomos.ini` | the sheet described by REBO-MoS instead of SW | |
| `afm/reaxff.ini` | the sheet described by ReaxFF instead of SW | |
| `afm/mlip_full.ini` | one DeePMD instance over tip, sheet and substrate | MLIP model |

`afm/basic.ini` is also the load- and scan-angle-sweep example: `force` and
`scan_angle` lists there sweep inside one run directory.

## sheetonsheet/ -- a driven sheet over a sheet (`run sheetonsheet`)

| file | shows | needs |
|---|---|---|
| `sheetonsheet/basic.ini` | driven top layer on a 3-layer stack | |
| `sheetonsheet/interlayer_styles.ini` | the same run under lj, lj_g and ilp interlayer styles | |
| `sheetonsheet/five_layers.ini` | a thicker 5-layer stack | |
| `sheetonsheet/hetero_pair_sweep.ini` | one run per `A + B` pair in `lists/hetero_pairs.txt` | lammps |
| `sheetonsheet/hetero_twist.ini` | twisted MoS2/WS2 heterostructure (`twist_angle`) | lammps |
| `sheetonsheet/hetero_alternating.ini` | layers interleaved A B A B (`hetero_stacking`) | lammps |
| `sheetonsheet/hetero_ilp.ini` | heterostructure with the ILP interlayer potential | lammps |
| `sheetonsheet/flake_slide.ini` | a finite MoS2 flake sliding on a bilayer | |
| `sheetonsheet/flake_slide_hetero.ini` | a two-material flake (`[Flake-1]`, `[Flake-2]`) sliding on a bilayer | lammps |
| `sheetonsheet/reaxff_speed_sweep.ini` | ReaxFF sheet, swept over two sliding speeds | |
| `sheetonsheet/mlip_full.ini` | full-cell DeePMD stack, interlayer physics from the model | MLIP model |

There is no sheet-on-sheet finite-edge example:
`geometry.finite_sheet_sheetonsheet_edge_mode: rigid` splits edge atom types
but no fix holds them yet, so it would not do what its name says.

## pes-tip/ -- static tip PES scans (`run pes-tip`)

| file | shows | needs |
|---|---|---|
| `pes-tip/basic.ini` | N x N energy / lateral-force map under a rigid tip | |
| `pes-tip/freestanding.ini` | the same scan on a freestanding slab, no substrate | |
| `pes-tip/freestanding_campaign.ini` | freestanding scans over every material in `lists/material_list.txt` | |

## pes-sheet/ -- static interlayer PES scans (`run pes-sheet`)

| file | shows | needs |
|---|---|---|
| `pes-sheet/basic.ini` | N x N interlayer-energy map of a frozen bilayer | |
| `pes-sheet/mlip.ini` | the same scan with a full-cell DeePMD model | MLIP model |

## indent/ -- press, hold, retract (`run indent`)

| file | shows | needs |
|---|---|---|
| `indent/basic.ini` | indentation to each load, hold, retract to pull-off | |
| `indent/campaign.ini` | indentation over every material in `lists/material_list.txt` | |

## quench/ -- amorphous blocks (`run quench`)

| file | shows | needs |
|---|---|---|
| `quench/basic.ini` | melt-quench decks for the amorphous block library | |

## lists/ and settings/

- `lists/material_list.txt` -- one material per line, for `materials_list =`.
- `lists/hetero_pairs.txt` -- one `A + B` pair per line, for pair sweeps.
- `settings/` -- partial `settings.yaml` files the examples above name; any key
  left out keeps its default.
