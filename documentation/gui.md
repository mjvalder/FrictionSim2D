# Desktop GUI

A window for configuring, previewing and generating simulation packages. It is
a client over the same code the CLI uses, so anything it produces can be run,
edited or re-read with `FrictionSim2D run …` unchanged, and vice versa.

## Install and launch

```bash
conda activate aiida
pip install -e '.[gui]'

FrictionSim2D gui --check     # verify the environment first
FrictionSim2D gui             # launch
```

`--check` reports the Qt platform, the ovito version, whether the viewport gets
a usable OpenGL context, and whether `lmp` and `atomsk` are on PATH. It exits
non-zero when the environment cannot run the GUI, so it is worth running once on
a new machine before wondering why the viewport is blank.

The GUI is an optional extra. `src/gui` deliberately imports neither Qt nor
ovito at module load, so a compute node without them is unaffected.

## Workspace

On launch you are asked for a working directory. It is an output root, not a
container: generated packages, saved configs and the disposable preview area
live under it, but you can open a config from anywhere.

```
<workspace>/
  configs/       configs saved from the GUI
  .preview/      disposable preview builds, pruned automatically
  settings.yaml  optional workspace-local settings
  <simulation_*> generated packages
```

## Choosing what to build

Two families, each with a mode:

| Family | Mode | Equivalent CLI |
|---|---|---|
| AFM tip | Dynamic + indent | `run afm` |
| | Indent only | `run indent` |
| | PES scan | `run pes-tip` |
| Sheet on sheet | Dynamic (slide) | `run sheetonsheet` |
| | PES scan | `run pes-sheet` |

The mode changes which sections are offered — `[indent]` only for an indent run,
`[pes]` only for a scan — and which run scripts the build produces.

## Reading the run count

The header shows three numbers, e.g. **1 config → 2 builds → 10 jobs**:

- **configs** — separate simulations, each with its own output directory;
- **builds** — the `L{n}` layer trees actually written;
- **jobs** — run scripts, i.e. what an HPC array would submit.

They differ, and the difference is the point. `[general] force = [5,10,20,30,50]`
does not multiply the config count, because the loads are looped inside one
LAMMPS script; but with `outer_loop = force` each load becomes its own run
script, so the job count multiplies.

### Which fields may hold a list

A parameter can only be swept if it changes the output directory name.
Otherwise every value would write into the same directory and overwrite the
last.

| Behaviour | Fields |
|---|---|
| **Sweep** — separate simulations | `[2D] materials_list`, `[general] temp` |
| **Layers** — separate `L{n}` trees | `[2D] layers` |
| **In-file loop** — looped inside LAMMPS | `force`, `pressure`, `scan_angle`, `scan_angle_force`, `scan_speed` |
| **Single value only** | everything else |

Putting a list in the last category is reported before anything is generated.

## Render

**Render** builds the configured components, assembles them as the generated
LAMMPS script specifies, and displays the result. It never minimises and never
runs a melt-quench: if an amorphous structure is not already cached, the build
refuses rather than spending minutes in `mpiexec` and writing into the packaged
materials directory.

What you see is parsed out of the build's own outputs — the assembly script for
the cell and component placement, `system.in.settings` for component membership
and the element legend — so the preview cannot drift from what LAMMPS will read.

The viewport controls:

- **Colour** by atom type (with a legend) or by component;
- **Box** toggles the simulation cell;
- **Show** hides individual components. Sheets are what most often generate
  incorrectly, and you cannot inspect one through a substrate;
- **Top / Front / Side / Perspective** camera presets. Drag to rotate, scroll
  to zoom.

A component that reaches a periodic boundary is called out: it will interact
with its own image, which is what the ≥2× tip-diameter rule exists to prevent.

Builds run in a separate process, so a LAMMPS failure reports an error instead
of closing the window. Repeating a render with unchanged inputs reuses the
cached build; the cache key includes a fingerprint of the builder and template
sources, so editing a template invalidates it.

## Generate

**Generate** writes the full package — every config in the sweep, every layer —
into the workspace, optionally with HPC job scripts. It asks for confirmation
with the three counts first.

Unlike Render, this may generate a missing amorphous structure, because here it
is what you asked for rather than a side effect of looking.

## Configs

**Open config…** reads any `.ini` the CLI accepts and fills in the form.
**Save config…** writes one back. Round-tripping preserves every value, and is
tested against every config in `examples/`.

Two limits worth knowing:

- Inline comments are not preserved. The reader discards them, so a round trip
  cannot keep yours; canonical unit and help comments are re-emitted instead.
- Heterostructure configs — those with numbered `[2D-1]`, `[2D-2]` sections —
  are refused rather than silently edited. Use the CLI for those.

Relative paths in a config are resolved against the config's own directory and
its parents, so a config written for the project root still works when opened
from elsewhere.

## Settings

The Settings tab shows the effective `GlobalSettings` and which file they came
from. It is read-only for now, deliberately: `load_settings` replaces rather
than merges, so writing a partial file would silently reset every value it
omitted to a built-in default. Use `FrictionSim2D settings` to change them.

## Not included

Job monitoring and the results database (Parts 2 and 3 of the requirements),
heterostructure configs, and standalone single-component builds.
