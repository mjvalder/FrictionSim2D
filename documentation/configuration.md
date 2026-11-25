# Configuration

Writing the per-run `.ini` files that drive simulation generation.

Global parameters live in `settings.yaml` and are documented separately in
[settings.md](settings.md).

## Section layout

| Model | Required sections | Additional |
|---|---|---|
| AFM sliding | `[2D]`, `[tip]`, `[sub]`, `[general]` | — |
| Sheet-on-sheet | `[2D]`, `[general]` | — |
| Interlayer PES scan | `[2D]`, `[general]` | `[pes]` |
| Tip PES scan | `[2D]`, `[tip]`, `[general]` | `[pes]`; omit `[sub]` for the freestanding builder |
| Indentation | `[2D]`, `[tip]`, `[sub]`, `[general]` | `[indent]` |

Optional in any model: `[lj_override]`. `run quench` takes a `[quench]` section
instead (see [`[quench]`](#quench)).

## File paths

`cif_path`, `pot_path` and `interlayer_file` are looked up in this order; the
first hit wins:

1. as given (absolute, or relative to the working directory);
2. next to the config file, then up to four directories above it;
3. the packaged library, `frictionsim2d/data/{materials,potentials}` — name a
   file `materials/h-MoS2.cif` or `potentials/sw/MoS2_wen.sw`;
4. the materials pack (`FrictionSim2D fetch materials`; materials only);
5. the user library, `$XDG_DATA_HOME/FrictionSim2D/amorphous` (materials only);
6. the old repository `examples/materials` / `examples/potentials` folders, with
   a deprecation warning. The spelling `examples/materials/...` still resolves
   through the same roots, with the same warning.

A file of your own never loses to a library one of the same name. A stored
config records library files by their portable `materials/...` /
`potentials/...` name, so it resolves on another install.

Every model's `[2D]` may instead be written as numbered `[2D-1]`/`[2D-2]`
sections, which stack the two materials into a heterostructure — see
[heterostructure pair sweeps](#heterostructure-pair-sweeps).

## Core fields

### [2D]

Common fields:

- `mat`
- `cif_path`
- `x`, `y`
- `layers`
- `pot_path`
- `pot_type`
- `stack_type` (default `AA` if omitted)
- `lat_c` (optional; **mandatory** for an MLIP sheet — the interlayer distance)
- `materials_list` (optional)
- `dispersion` (`none` or `d3`; MLIP sheets only)
- `precision` (`float64` default, or `float32`; MACE only; float32 adds `-f32` to the `pot_` directory)

`pot_type` takes the classical types (`sw`, `tersoff`, `reaxff`, ...) or a
machine-learned potential, `mace` or `deepmd`, with `pot_path` pointing at the
model (`.json` symmetrix MACE; `.pth`/`.pb` DeePMD). Under an MLIP every
component in the cell (tip, substrate, sheets, flake) must use the same MLIP,
`amorph = a` is refused (use `amorph = c, match = true` for an AFM substrate),
and `dispersion = d3` is the only vdW lever (`interlayer_style` must stay
`lj`). See [mlip.md](mlip.md).

### [tip]

- `mat`
- `r`
- `amorph` (`c` or `a`, default effectively `c`)
- `dspring`
- `pot_path`
- `pot_type`
- `cif_path`

`amorph = a` cuts the tip (or substrate, below) from a pre-made amorphous block
of the material. A block is reused only when material, potential and quench
conditions match; if none is found the run exits 1 before writing anything and
prints the `run quench --for` command to make it
([examples.md](examples.md) §11).

### [sub]

- `mat`
- `thickness`
- `amorph` (`c` or `a`, default effectively `c`)
- `pot_path`
- `pot_type`
- `cif_path`

### [general]

Common knobs include:

- `temp`
- `force` (AFM workflows)
- `pressure` (sheet-on-sheet workflows; **required** for every `run sheetonsheet` slide)
- `scan_angle`
- `scan_angle_force`
- `scan_speed`
- `bond_spring`
- `driving_spring`
- `outer_loop`

## Working examples

The shipped configs under [../examples/](../examples/) are the authoritative runnable
versions; the two below are trimmed to the minimum.

### Minimal AFM

```ini
[2D]
mat = h-MoS2
cif_path = materials/h-MoS2.cif
x = 100
y = 100
layers = [1,2,3]
pot_path = potentials/sw/MoS2_wen.sw
pot_type = sw
stack_type = AB

[tip]
mat = Si
r = 25
amorph = c
dspring = 1.6e-6
pot_path = potentials/sw/Si.sw
pot_type = sw
cif_path = materials/Si.cif

[sub]
mat = Si
thickness = 12
amorph = a
pot_path = potentials/sw/Si.sw
pot_type = sw
cif_path = materials/Si.cif

[general]
temp = 300
force = [5, 10, 20]
scan_angle = [0, 45, 90]
scan_speed = 2
driving_spring = 50
```

### Minimal sheet-on-sheet

```ini
[2D]
mat = h-MoS2
cif_path = materials/h-MoS2.cif
x = 100
y = 100
layers = [3]
pot_path = potentials/sw/MoS2_wen.sw
pot_type = sw
stack_type = AB

[general]
temp = 300
pressure = [0.1, 1.0, 10.0]
scan_angle = [0, 90]
scan_speed = 1
bond_spring = 5
driving_spring = 50
```

## Sweep behaviour

Important implementation detail:

- Runtime expansion only creates independent simulation directories for list values in `[general]` except `force`, `pressure`, `scan_angle`, and `scan_speed`.
- Those four are handled inside generated LAMMPS scripts.
- AFM layer lists are handled in the AFM builder and produce `L<N>` subdirectories.
- A sweep whose members would build into the same output directory is rejected before anything is built, and the error names the parameter: `[general] temp` values that differ only below 1 K (the directory token is `K<int(temp)>`), a `materials_list` over a section with no `{mat}` marker, and `pot_path` entries with the same type and file stem (or listed twice). A `materials_list` of one entry over a section that keeps a literal `mat` and carries no `{mat}` marker is rejected for the same reason.

### Splitting a sweep into per-value scripts (`outer_loop`)

By default the four in-script sweeps above run as a single looped script (one
`slide.in` that iterates over every value). Set `outer_loop` in `[general]` to
instead emit **one slide script per value**, so each becomes its own HPC job and
they run in parallel rather than back-to-back:

| `outer_loop` | Model | Emits | Per value |
|---|---|---|---|
| `force` | AFM | `slide_f<load>N.in` | one job per load |
| `pressure` | sheet-on-sheet | `slide_p<value>gpa.in` | one job per pressure |
| `scan_speed` | sheet-on-sheet | `slide_<value>ms.in` | one job per speed |
| `scan_angle` | AFM / sheet-on-sheet (dynamic modes) | `slide_a<angle>.in` | one job per angle |

Every row assumes the config routes to the homogeneous AFM or sheet-on-sheet
builder — see *Which builders honour `outer_loop`* below.

For AFM `outer_loop = force`, a single `system.in` (indentation phase) still
writes every `load_<load>N.data`, and each `slide_f<load>N.in` reads its own
load file. The HPC generator wires this as a two-phase submission: the per-load
slide array depends on the shared system job completing first. A single-value
`force` list stays as `slide.in` (nothing to split).

```ini
[general]
force = [2, 5, 10, 20, 30, 50, 70, 85, 100]
outer_loop = force
```

`outer_loop = scan_angle` works the same way, in AFM's `afm` (dynamic + indent)
model and sheet-on-sheet's `sheetonsheet` model: each configured angle becomes
its own `slide_a<angle>.in`, with `-` formatted as `m` and `.` as `p` in the
token (`slide_a30.in` for 30, `slide_am15.in` for -15, `slide_a2p5.in` for
2.5). As with `force`, a single-value `scan_angle` list stays as `slide.in` —
the combined system+slide HPC path depends on that literal name, so a lone
angle is never split. It cannot be combined with `scan_angle_force`: a
per-angle script has no per-force selector left to apply that filter to, so
the combination is refused before anything is built (and reported in the GUI
validation panel) rather than silently dropping the filter. It is refused
only for the builders that really split by angle — where the run phase stays
a single script the filter is honoured in-script, so the pair is accepted.

Only the angle is collapsed: each per-angle script still loops in-script over
every `force`/`pressure` and `scan_speed` value configured, exactly as the
unsplit `slide.in` would.

#### Which builders honour `outer_loop`

`outer_loop` is acted on by the builder the config routes to, not by the
`run` command alone. Where the selected builder ignores it, the run phase
stays one `slide.in` and one job — the predicted job count says so too.

| Config | Builder | Honours |
|---|---|---|
| `run afm` (dynamic + indent) | AFM (incl. hetero AFM) | `force`, `scan_angle` |
| `run indent` | indent | `force` only — the hold/retract never reads the angle |
| `run sheetonsheet`, single `[2D]`, no twist | sheet-on-sheet | `pressure`, `scan_speed`, `scan_angle` |
| `run sheetonsheet` with `[general] twist_angle`, `[2D-1]`/`[2D-2]`, or `[Flake]` | hetero / flake slide | nothing — every value stays an in-script loop |
| `run pes-tip`, `run pes-sheet` | PES scans | nothing — the run phase is replaced entirely |

The GUI reflects the last two rows: it shows no split control at all for the
PES modes, and its job count for a twisted or flake sheet-on-sheet config
stays at one job per build. A numbered `[2D-1]`/`[2D-2]` config is not
editable in the GUI at all — open it with the CLI.

```ini
[general]
scan_angle = [0, 30, 60, 90]
outer_loop = scan_angle
```

### Material list expansion

Use placeholder replacement from a text list:

```ini
[2D]
materials_list = examples/material_list.txt
mat = {mat}
cif_path = materials/{mat}.cif
pot_path = potentials/sw/sw_lammps/{mat}.sw
```

`materials_list` format: one material token per line, no header. Blank lines and
`#` comments (whole-line or trailing) are ignored. Potential and CIF paths
resolve like any other path (see [File paths](#file-paths)), so every `{mat}`
must have a matching file; a name whose files cannot be found is skipped with a warning, and
a list where none resolve is an error rather than a run that generates nothing.

A comma-separated line is several separate single materials: `h-MoS2, h-WS2`
means two runs, one per material.

### Heterostructure pair sweeps

A config with two numbered sheet sections, `[2D-1]` and `[2D-2]`, builds a
heterostructure. Its material list holds one **ordered pair** per line, members
joined with `+`:

```
h-MoS2 + h-WS2
h-MoS2 + h-WSe2
h-MoSe2 + h-WS2
```

```ini
[2D-1]
materials_list = examples/hetero_pairs.txt
mat = {mat1}
cif_path = materials/{mat1}.cif
pot_path = potentials/sw/sw_lammps/{mat1}.sw

[2D-2]
mat = {mat2}
cif_path = materials/{mat2}.cif
pot_path = potentials/sw/sw_lammps/{mat2}.sw
```

The runnable version is [../examples/hetero_pair_sweep.ini](../examples/hetero_pair_sweep.ini)
with [../examples/hetero_pairs.txt](../examples/hetero_pairs.txt); the three
lines above expand into three runs.

The desktop application writes exactly this form — a second material, the pair
table and the `.txt` it produces — see
[gui.md](gui.md#two-materials-heterostructures).

Points worth knowing:

- **`+`, not a comma.** A comma line still means two separate single materials —
  long-standing behaviour that pair support did not change. A line cannot mix
  the two separators, and a `+` line must name exactly two materials — there is
  no `[2D-3]` for a third.
- **Order matters.** Member 1 fills `[2D-1]`, member 2 fills `[2D-2]`. Pairs are
  never sorted or normalised, so `h-MoS2 + h-WS2` and `h-WS2 + h-MoS2` are
  different stacks and different runs.
- **Only `[2D-1]` carries `materials_list`.** `[2D-2]` is filled from the same
  line's second member.
- **Placeholders, one per sheet.** `{mat1}` and `{mat2}` are the pair's two
  members. `[2D-1]` may spell its own member either `{mat1}` or `{mat}` — the
  two substitute identically, so a single-material config that gains a second
  sheet keeps working — but `[2D-2]` accepts only `{mat2}`.
- **The placeholder is checked per key, not per section.** A `[2D-2]` whose
  `mat` reads `{mat2}` while its `cif_path` or `pot_path` still reads `{mat1}`
  is refused, even though the section does hold `{mat2}` somewhere. Left to
  run, it would load the first material's files for both sheets while the
  output directory kept the pair's name: every path resolves, nothing fails,
  and the results are quietly a homostructure. A sheet holding no placeholder
  at all is refused for the same reason — it would be pinned to one material
  for every member of the sweep. Both messages name the section and the line.
- **Output directories join with `+`**, e.g. `h-MoS2+h-WS2` — real material
  names contain `-`, so a `-` join could not be split back apart.
- **Arity must agree with the sections.** A pair line in a config with no
  `[2D-2]` is an error, and so is a single-name line in a config that has one.
  Both name the offending line.

#### Which models accept a heterostructure config

`[2D-1]`/`[2D-2]` are accepted by `sheetonsheet`, `afm`, `indent`, `pes_tip` and
`pes_sheet`; each routes to that model's heterostructure builder. Two
restrictions apply to the AFM family: a `[sub]` section is required (there is no
freestanding heterostructure), and `[general] finite_sheet` is supported only by
`afm`, not by `indent`.

Each material's `layers` holds exactly one value and the counts add up:
`[2] + [2]` is the four-layer sheet-on-sheet stack (fixed bottom, thermostatted
middle, driven top), while `pes_sheet` wants `[1] + [1]` — a frozen bottom
monolayer and a scanned top one. `[general] hetero_stacking` chooses whether
each material's layers stay together (`grouped`) or interleave (`alternating`),
and `[general] twist_angle` builds a moiré — including from a single `[2D]`
section, which is the one heterostructure config that needs no second material.

## Scan angle formats

Supported by current runtime logic:

- Scalar angle: `scan_angle = 0`
- Explicit angle list: `scan_angle = [0, 30, 60, 90]`

Legacy interval-style lists may still appear in older configs; explicit lists are clearer and recommended.

`scan_angle_force` can restrict angle lists to selected force/pressure values and must follow the same ordering as the target list.

## Layer constraints

- AFM: `layers` can contain multiple values.
- Sheet-on-sheet: `layers` must contain exactly one value and that value must be at least 3.
- Interlayer PES scan: `layers` must be exactly `[2]`.

## Optional sections

### [lj_override]

Override specific LJ pairs:

```ini
[lj_override]
Mo-S = [0.4124, 3.75114]
S-S = [0.1984, 3.62368]
```

### [pes]

Required by `run pes-sheet` and `run pes-tip`. Full reference in
[pes_scan.md](pes_scan.md).

```ini
[pes]
grid_n = 12          # N for the N x N lateral grid over one surface unit cell
grid_n_refine = 20   # optional finer grid for a convergence check
z_relax = true       # sheet scan: relax the top layer's z at each grid point
```

### [indent]

Required by `run indent`. Full reference in [indent.md](indent.md).

```ini
[indent]
hold_steps = 10000   # damped hold at the target load
z_step = 0.2         # retract increment per step, Å
n_steps = 100        # retract steps
seed = 20260710      # campaign base seed
```

### [Flake] (sheet-on-sheet flake slide)

A `[Flake]` section on a `run sheetonsheet` config builds a finite flake on a
>=2-layer substrate sheet and drives the flake across it (no tip). A single
`[Flake]` reuses the `[2D]` material; `[Flake-1]`/`[Flake-2]` make a hetero
bilayer flake with `[Flake]` as the shape spec.

```ini
[Flake]
shape = hexagon        # triangle | square | hexagon
edge_length = 20       # Å: triangle/square side, hexagon circumradius
rotation_deg = 0
corner_spring_k = 10.0
corner_hold = false    # default; see "Corner hold" below
```

**The cell has to clear the flake.** The flake is finite, the sheet under it is
periodic: what separates the flake from its own image is `cell - flake span`,
and while that gap is inside `settings.potential.lj_cutoff` the flake slides
against a copy of itself. The build refuses such a config and names the
offending axis, so enlarge `[2D] x`/`y` (sheet-on-sheet) or shrink
`edge_length`. On a hetero AFM the cell is the coincidence cell replicated from
the tip radius alone — a small `[tip] r` with a large flake is the combination
that trips it.

**Shapes.** Every shape is centred on the flake centre and rotated CCW about
it by `rotation_deg`:

- `triangle` — equilateral, side `a`, centred on its centroid; at
  `rotation_deg = 0` the apex points +y (at `2h/3`, base at `−h/3`,
  `h = sqrt(3)/2 · a`).
- `square` — side `a`, edges along x/y at `rotation_deg = 0`.
- `hexagon` — the regular hexagon of circumradius `R` (side also `R`); at
  `rotation_deg = 0` two vertices lie on the x axis at `±R` and two edges are
  horizontal at `y = ±sqrt(3)/2 · R`.

(The same outlines are used by an AFM `[flake]`.)

**Corner hold.** `corner_hold` (default `false`) tethers the flake's corner
atoms -- those within `corner_radius` Å of a vertex -- to their starting xy
position with a `corner_spring_k` eV/Å² spring while the flake is driven. It
does **not** keep the flake "rigid while sliding": it stops the flake
translating at all, since the tether typically outweighs the COM drive
spring. Leave it off for a normal slide (the flake is deformable and moves
under the drive); turn it on only for static/indent-like studies where the
flake must stay pinned in place. This is a sheet-on-sheet flake-slide setting
only -- AFM indentation always tethers its flake's corners unconditionally,
via a separate mechanism, and does not read this field.

**Normal load.** `[general] pressure` (GPa) is applied to the flake over its
projected in-plane area, not the substrate box area. The area is that of the
outline the builder carves:

| shape | area (Å²) |
|---|---|
| `triangle` (side `a`) | `sqrt(3)/4 · a²` |
| `square` (side `a`) | `a²` |
| `hexagon` (circumradius `R`) | `3·sqrt(3)/2 · R² ≈ 2.598 R²` |

Rotation and `twist_angle` do not change the area. The load `P·A` is spread
equally over the flake's atoms (`fix addforce`). The results file records the
felt normal force `v_fz` (nN, mean ≈ `P[GPa] · A[Å²] / 100`), so a flake run's
COF is lateral force / normal load, as for sheet-on-sheet. Each pressure writes
its own `friction_p<P>_…` file.

**A pressure is required.** Every `run sheetonsheet` slide -- plain, hetero
(`[2D-1]`/`[2D-2]` or `twist_angle`) or `[Flake]` -- is refused before
anything is built when `[general] pressure` is unset: `[general] pressure is required for
a loaded slide`. Without it the run would slide at 0 GPa and report a
meaningless COF. A `run pes-sheet` scan applies no load and needs none.

### [quench]

Required by `run quench` (see `examples/quench_config.ini`); reference:
[commands.md](commands.md#run-quench). Unset conditions come from `settings.quench`;
a list value makes one member per value.

```ini
[quench]
material = Si                       # a list is allowed, e.g. [Si, Ge]
potential = potentials/sw/{mat}.sw  # {mat} is replaced per material
pot_type = sw
cif = materials/{mat}.cif           # default <mat>.cif
slab = [200, 200, 50]               # crystal slab to melt, Angstrom
quench_rate = 1e12                  # K/s
```

Optional: `melt_temp`, `quench_temp`, `timestep`, `melt_steps`,
`equilibrate_steps`.

## Interlayer interactions

`interlayer_style` (in `[general]`, sheet-on-sheet and AFM) chooses how the 2D layers of a sheet interact with each other. It is sweepable: `interlayer_style = [lj, lj_g, ilp]`.

| Style | Interaction | Default parameters |
|---|---|---|
| `lj` (default) | `lj/cut`, UFF or `[lj_override]` | none needed |
| `lj_g` | the run's LJ plus the Jiang-Park Gaussian, via `kolmogorov/crespi/z` | g table below |
| `ilp` | `ilp/tmd` for Mo/W/S/Se (`TMD.ILP`), `ilp/graphene/hbn` for B/N/C/H (`BNCH.ILP`) | bundled |
| `kc` | `kolmogorov/crespi/full` | carbon only (`CC.KC-full`, `CH.KC`); other materials need `interlayer_file` |

For `kc`, `tap_flag` is 1 only for `*_taper.KC` files; `ilp` always uses 1.

### Gaussian strength (`lj_g`)

g (meV): MoS2 175.68, graphene 94.87, black phosphorus 123.0. Override per sheet with `interlayer_g` in `[2D]`. In a heterostructure g = sqrt(g1 g2). The LJ part is the run's own LJ, overrides included.

### Parameter files

`interlayer_file` is looked up like `pot_path` ([File paths](#file-paths)), then in the bundled `frictionsim2d/data/potentials/ilp/` or `.../kc/`.

### Scope

- Acts only between 2D layers. Tip and substrate stay `lj/cut`; in AFM only between the sheet's own layers. A single layer makes the style a no-op.
- Not allowed with a potential that has its own interlayer term (`rebomos`, `reaxff`, `airebo`, `comb`, ...), nor with a `constraint_mode` (set in `settings.simulation.constraint_mode`) other than `none`.
- `ilp` and `kc` need an in-plane periodic cell wider than the 16 A cutoff plus the neighbor skin; otherwise the build fails with a message naming the `x`/`y` that pass.
- `ilp/tmd` splits each layer's chalcogen into top and bottom types automatically when the SW file has a single chalcogen type.

### Friction measurement and comparison mode

- `lj` measures the interlayer force with `compute group/group` at the shearing interface.
- `lj_g`, `ilp`, `kc` use the bottom-layer reaction force (`group/group` cannot run under them).
- `interlayer_comparison = true` forces every style, `lj` included, onto the reaction-force measurement, so a style sweep compares like with like. See `examples/sheet_config_interlayer.ini`.

### Output and provenance

- Output directories get an extra `il-<style>` level (`il-<style>-cmp` in comparison mode) after the `pot_` level; plain `lj` adds none.
- Observables: `E_int_interlayer` (`lj`, shearing interface) vs `E_int_interlayer_all` (`ilp`/`kc`, all layer pairs). The PES CSV gains `e_interlayer_eV` under `ilp`/`kc`.
- Provenance records an `interlayer` block when the style is not `lj` or comparison is on.

### GUI

The potential type is a drop-down with `Custom...`; a custom `pot_type` is checked against the installed `lmp` pair-style list (accepted with a warning if no binary is found). Interlayer controls appear only for potentials without their own interlayer term.

### Known caveats

- Sevik SW + `TMD.ILP` bilayers relax to Mo-Mo 6.21 A.
- `mos2_jiang3.sw` (sw/mod parameters in a `.sw` file) collapses under ILP.
- In-process stacking minimisation under ILP may stop early; prefer giving `lat_c`.

## Validation notes

Common failures:

- Missing referenced CIF or potential files
- Non-ascending sweep values for force/pressure/angle selectors
- Invalid sheet-on-sheet layer count
- A `run sheetonsheet` slide (plain, hetero or `[Flake]`) with no `[general] pressure`
- Incompatible `constraint_mode` with potentials that already include internal interlayer interactions

## Tips

1. Start with one material and one load value.
2. Confirm folder output before launching large sweeps.
3. Keep absolute or clearly relative paths to avoid file resolution surprises.
4. Version your config files with each simulation campaign.

## Related docs

- [settings.md](settings.md)
- [commands.md](commands.md)
- [examples.md](examples.md)
