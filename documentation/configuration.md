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

Optional in any model: `[lj_override]`.

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
- `lat_c` (optional)
- `materials_list` (optional)

### [tip]

- `mat`
- `r`
- `amorph` (`c` or `a`, default effectively `c`)
- `dspring`
- `pot_path`
- `pot_type`
- `cif_path`

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
- `pressure` (sheet-on-sheet workflows)
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
cif_path = examples/materials/h-MoS2.cif
x = 100
y = 100
layers = [1,2,3]
pot_path = examples/potentials/sw/MoS2_wen.sw
pot_type = sw
stack_type = AB

[tip]
mat = Si
r = 25
amorph = c
dspring = 1.6e-6
pot_path = examples/potentials/sw/Si.sw
pot_type = sw
cif_path = examples/materials/Si.cif

[sub]
mat = Si
thickness = 12
amorph = a
pot_path = examples/potentials/sw/Si.sw
pot_type = sw
cif_path = examples/materials/Si.cif

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
cif_path = examples/materials/h-MoS2.cif
x = 100
y = 100
layers = [3]
pot_path = examples/potentials/sw/MoS2_wen.sw
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

### Material list expansion

Use placeholder replacement from a text list:

```ini
[2D]
materials_list = examples/material_list.txt
mat = {mat}
cif_path = examples/materials/{mat}.cif
pot_path = examples/potentials/sw/sw_lammps/{mat}.sw
```

`materials_list` format: one material token per line, no header. Potential and CIF
paths resolve literally — there is no search path — so every `{mat}` must have a
matching file.

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

### [settings_override]

Documented in legacy workflows but not currently merged in the active runtime path.
Prefer editing global settings directly when reproducibility is required.

## Validation notes

Common failures:

- Missing referenced CIF or potential files
- Non-ascending sweep values for force/pressure/angle selectors
- Invalid sheet-on-sheet layer count
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
