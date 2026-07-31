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

### Sweeping over materials

`[2D] materials_list` names a `.txt` of material names, one per line, and each
becomes a separate simulation. Setting all of that by hand means keeping four
things in sync — the `.txt`, and `{mat}` placeholders in `mat`, `cif_path` and
`pot_path` — which is why the field was effectively undiscoverable before.

Ticking **Sweep over many materials** in `[2D]` reveals two directory
pickers, **Structures** and **Potentials**, alongside the existing `mat`,
`cif_path` and `pot_path` fields — it does not replace or hide them.
Committing a sweep rewrites those three to `{mat}` templates (see below).
Pointing at the two directories mirrors a real run directory — a `cif/`
beside a `sw_lammps/` — so it is usually just pointing at the pair that
already exists.

The material list is the **intersection** of the two directories: a material
needs a structure file and a potential file to be usable. The potential file
extension is inferred from the commonest suffix in the potentials directory.
Names found in only one directory are still shown, greyed out, with the
reason (`no potential file` / `no structure file`) rather than silently
dropped — a material missing from a sweep is exactly the kind of thing worth
noticing.

The derived names appear as a checkable list, so the sweep can be narrowed
below "everything usable". A summary line reads e.g. `12 of 14 usable → 9
configs`. **Select all** / **None** act on the currently usable names.

Committing a sweep writes the checked names to `<config>_materials.txt`
beside the config, and fills `mat = {mat}`, `cif_path =
<structures>/{mat}.cif`, `pot_path = <potentials>/{mat}<ext>` — paths are
stored relative to the config's own directory where that is possible. If the
selection matches what is already on disk, nothing is rewritten.

Opening a config with an existing `materials_list` loads those names as the
current selection; the two directories (recovered from the `{mat}` templates
in `cif_path` / `pot_path`) only decide what is available to tick, not what
is selected. Paths that still contain `{mat}` are reported as pending
substitutions rather than as validation errors, so a templated config renders
and generates normally.

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
from. Editing a value and pressing Render or Generate applies it: the tab edits
a document, a build reads a file, so the build writes `settings.yaml` into the
workspace first and passes it explicitly.

**Save to workspace** writes the file without building. **Revert to defaults**
discards the file's values for the built-in ones. The file written is always
complete — `load_settings` replaces rather than merges, so a partial file would
silently reset every value it omitted.

Two groups are worth knowing about:

- **HPC** — with **Modules** empty, no job scripts are written at all, whatever
  the rest of the group says. The Generate dialog says so before it builds.
- **Thermostat** — the Langevin bands split each component into fixed,
  thermostatted and free, shown as a table rather than raw JSON. The two
  halves use different units: tip bands are absolute Å measured down from the
  tip's top face, substrate bands are a fraction of the substrate thickness
  measured up from the bottom. A table with a component missing entirely is
  flagged, since the builders read both unconditionally and fail without them.

A few other fields get purpose-built editors instead of a JSON blob: `output.dump`
and `output.dump_frequency` are key/value grids (rows can be added and removed,
since the keys are dump-phase names the user picks), and
`quench.quench_slab_dims` is three labelled x/y/z cells. Elsewhere,
`[2D] stack_type` (stacking) is a dropdown of `AA`/`AB` that preserves an
unrecognised hand-written value rather than silently snapping it to a choice —
the model types the field as a plain string, so a config with something else
in it round-trips unchanged.

## Not included

Job monitoring and the results database (Parts 2 and 3 of the requirements),
heterostructure configs, and standalone single-component builds.
