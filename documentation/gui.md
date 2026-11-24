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

The GUI is an optional extra. `frictionsim2d/gui` deliberately imports neither Qt nor
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
`[pes]` only for a scan — and which run scripts the build produces. The form
keeps its scroll position across the switch, so re-picking a mode does not
throw you back to the top of a long section list.

## Units

A couple of fields store a unit LAMMPS does not drive with directly — both
say so in their ⓘ bubble (see [Fields](#fields)), but it is worth stating up
front:

- **Scan speed** is metres per second. The generated script divides it by
  100 to get the Å/ps LAMMPS actually drives with: 100 m/s = 1 Å/ps.
- **Driving spring** and **bond spring** (sheet-on-sheet) are newtons per
  metre. Every builder divides by the exact factor `NM_TO_EV_A2 =
  16.02176634` to get eV/Å² before writing it into the LAMMPS input (AFM
  used a rounded 16.02 before provenance schema 2.2.0).

Separately, `[tip] damping_ratio` now defaults to 0.0 (off) rather than
0.0072. `dspring` — a raw viscous override — already takes precedence
whenever it is set, so this only changes anything for a config that leaves
`dspring` blank too.

## Fields

Every field's explanation lives behind an ⓘ button at the end of its row
rather than a caption underneath it: hover or click it for the same
information a caption used to carry — the override's help text, further
detail, the list-sweep behaviour from
[Which fields may hold a list](#which-fields-may-hold-a-list), and the
default, when there is one.

**Required fields** are marked with a star. Most are unconditionally
required; the driving spring (`[general] driving_spring`, sheet-on-sheet) is
starred only when Settings → Simulation → drive method is `virtual_atom`,
since it does nothing otherwise. An empty starred field blocks Generate for
both **force** (AFM) and **pressure** (sheet-on-sheet): an absent value
renders an unparseable LAMMPS line (`variable ... equal None`), not a
zero-load run. Pressure has a builder-level `[0.0]` fallback, but only on
the `outer_loop` expansion paths — the default single-script path passes
`general.pressure` straight into the template unguarded, so an absent
pressure is exactly as broken there as an absent force. Both block.

**Defaults** are shown two ways: a starred field with a default is
pre-filled with it, so a required field is never confusingly blank; every
other field with a default shows it as grey placeholder text (e.g.
`default: 300.0`) instead of making you go find it. An enum with a default
shows `(default: X)` as its unset entry, rather than a bare `(unset)`.

**Disabled rows** state why on screen, not only behind the ⓘ. A row gated by
another field (e.g. **Scan angles apply at these loads**, which needs a scan
angle set first) shows the ⓘ's `enable_hint` as the control's own grey
placeholder text while it is disabled, restoring whatever placeholder it had
(a default, say) once it is enabled again. A scan angle of `0.0` — a
straight-line scan, a real and meaningful value — counts as set for this: the
gate checks whether the other field holds a value at all, not whether it is
truthy, so pairing loads with a zero scan angle no longer leaves that row
permanently greyed out.

**Tip position** (`[tip] tip_x`, `tip_y`, and their two offsets) stays out
of the form until **Set tip position** is ticked. Unticking it removes
those keys from the config entirely, not just blanks them. One consequence
worth knowing: a hand-written config with an explicit `tip_x_offset = 0.0`
loads as untouched — 0.0 is the offset's own default — so it ticks no gate
and loses that key on the next save. This is a no-op semantically: the
builder already treats an absent offset as 0.0, so nothing about how the
config builds changes, only how it looks on disk.

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

### Splitting the run phase by scan angle

`scan_angle` is one of the fields **Split run phase by** (`outer_loop`) can
act on, and it behaves like `force`/`pressure`/`scan_speed`: by default every
angle in `[general] scan_angle` loops inside one run script; choosing
**Split run phase by → angle** instead emits one script — and one HPC
job — per angle, named `slide_a<token>.in` (`slide_a30.in` for 30°,
`slide_am15.in` for -15°, `slide_a2p5.in` for 2.5°). A single configured
angle is never split, whichever is chosen: the combined system+slide HPC path
depends on the literal name `slide.in`, so a lone angle always keeps it.

The dropdown reads **angle**; the value written to the `.ini` is still
`outer_loop = scan_angle`, after the `[general]` key it names.

The split cannot be combined with **Scan angles apply at these loads**
(`scan_angle_force`): a per-angle script has no per-force selector left to
apply that filter to, so the validation panel reports the pair — before
anything is built — rather than silently dropping the filter.

**Split run phase by** offers `scan_angle` only in the two modes whose slide
phase actually reads it — AFM's **Dynamic + indent** and sheet-on-sheet's
**Dynamic (slide)**. **Indent only** shares the AFM family's field but its
template never reads the angle variable, so it keeps `force` and drops
`scan_angle`; the two **PES scan** modes replace the run phase entirely and
never look at `outer_loop` at all, so they offer no split of any kind to
choose from (with nothing to offer, the control is not shown).

A sheet-on-sheet config that carries a **twist angle**, a `[Flake]` section or
a second material builds through a different builder, which ignores
`outer_loop` entirely — no split of any kind, and the job count shown reflects
that.

### Sweeping over materials

`[2D] materials_list` names a `.txt` of material names, one per line, and each
becomes a separate simulation. Setting all of that by hand means keeping four
things in sync — the `.txt`, and `{mat}` placeholders in `mat`, `cif_path` and
`pot_path` — which is why the field was effectively undiscoverable before.

**Sweep over materials** is the first field in `[2D]` — a bare checkbox now;
its row label carries the only copy of that text (it used to be repeated on
the checkbox itself). Ticking it does two things: it reveals a `.txt` picker
and a checkable material list right after the **Material** row below it, and
further down it switches the `cif_path` and `pot_path` cells from
single-file pickers to directory pickers — their browse button and tooltip
change to say so — and scans whatever the two cells hold. Pointing at the
two directories mirrors a real run directory — a `cif/` beside a
`sw_lammps/` — so it is usually just pointing at the pair that already
exists. Top-to-bottom the block reads: the checkbox, **Material**, the `.txt`
picker and tick list, then the structure/potential cells the checkbox
governs.

Committing a sweep rewrites `mat`, `cif_path` and `pot_path` to `{mat}`
templates (see below); from then on the same two cells show that templated
path rather than the bare directory, which is expected — it is what the
config actually holds now.

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

Above the list, a `.txt` path field with its own browse button (`…`) points
at an existing curated materials list instead of building one purely from
the two directories — leave it blank to do the latter. Pointing it at a file
before both directories have been scanned cannot check anything yet, so
rather than a bare "0 of 0 usable" (which reads as nothing having happened),
the loaded names appear as unusable, greyed placeholders with a summary
naming the count (e.g. "2 material(s) loaded from the list — choose both
directories to check which are usable"); scanning both directories replaces
them with the real, tickable entries.

Committing a sweep writes the checked names back to whatever file
`materials_list` currently resolves to — the same `.txt` the picker was
pointed at, or the one an already-opened config's `materials_list` names —
so pointing the picker at your own curated list and then changing a tick
updates that file in place rather than silently abandoning it for a fresh
one. Only when nothing is configured yet (a brand-new sweep) does it fall
back to creating `<config>_materials.txt` beside the config. Either way it
also fills `mat = {mat}`, `cif_path = <structures>/{mat}.cif`, `pot_path =
<potentials>/{mat}<ext>` — paths, including the materials-list path itself,
are stored relative to the config's own directory where that is possible. If
the selection matches what is already on disk, nothing is rewritten.

Opening a config with an existing `materials_list` loads those names as the
current selection; the two directories (recovered from the `{mat}` templates
in `cif_path` / `pot_path`) only decide what is available to tick, not what
is selected. Paths that still contain `{mat}` are reported as pending
substitutions rather than as validation errors, so a templated config renders
and generates normally.

The list may also hold `mat1 + mat2` pair lines, which sweep a heterostructure
config's `[2D-1]`/`[2D-2]` over ordered material pairs. The window builds and
edits those too — see [Two materials](#two-materials-heterostructures) below.
The file format is the CLI's, unchanged, and is specified in
[configuration.md](configuration.md#heterostructure-pair-sweeps).

## Two materials (heterostructures)

A config can hold two materials, stacked into one cell: `[2D-1]` and `[2D-2]`
in the `.ini`, **2D sheet** and **Material 2 (heterostructure)** in the form.
All five modes accept one — each routes to that mode's heterostructure
builder — so the second material survives a change of family or mode rather
than being dropped by the dropdown.

### Adding and removing the second material

At the bottom of the **2D sheet** section is one button: **Add a material**,
which becomes **Remove material 2** once there is one. Adding inserts a
**Material 2 (heterostructure)** section directly after `[2D]` — the two are
read together, so they are shown together. It carries the same fields as the
first sheet minus **Sweep over materials**: only the first sheet holds the
list, and each of its lines names *both* materials.

The button is available while a material sweep is configured, because the pair
sweep below is exactly the case that needs a second sheet. Adding one to a
config that is already templated for a sweep re-spells the first sheet from
`{mat}` to `{mat1}` and seeds the new one with `{mat2}`, pointing at the same
two directories; removing it puts `{mat}` back. Which button you press first
therefore does not decide whether the sweep is correct.

`[general] hetero_stacking` — **Layer stacking**, `grouped` (1,1,2,2) or
`alternating` (1,2,1,2) — is shown on the **Material 2** section rather than
under Run parameters, since it orders the layers of a two-material stack and
means nothing without one. It is hidden from Run parameters in every mode so
it is never offered twice. A hand-written single-material config that carries
a non-default `hetero_stacking` gets a note in the validation panel saying it
has no effect; it does not block.

### Sweeping over material pairs

Tick **Sweep over materials** on a config that has two materials and the tick
list is replaced by a two-column table, **Material 1** / **Material 2**, with
**Add pair**, **Remove pair** and **Generate pairs…**. Each row is one run:
the pair is written to the `.txt` as `a + b`, and `[2D-1]`/`[2D-2]` are
templated on `{mat1}`/`{mat2}` on commit. The summary line reads `N pair(s) →
N configs`.

Order is part of the pair. `a + b` and `b + a` stack the two materials the
other way up and are different runs, so nothing here sorts or dedupes them.
**Generate pairs…** builds every combination of the materials currently
available — the ticked ones, or everything usable when nothing is ticked — and
offers the two decisions that would otherwise be silent:

- **Include the reversed order (b + a)** — off by default, because doubling
  the sweep is a decision;
- **Include same-material pairs (a + a)** — off by default: a homostructure
  bilayer is a legitimate reference run, but not what "pair these up" asks
  for.

Each cell is a dropdown of the usable materials plus whatever that cell
already holds, so a curated list naming a material this machine cannot see
(a moved share, a potential not generated yet) is shown as it is rather than
silently rewritten to something else.

Both sheets share **one** structure folder and **one** potential folder — the
commit builds `[2D-2]`'s paths out of the folders `[2D]` holds — so while the
sweep is on, Material 2's own Structure and Potential pickers are greyed and
say the choice is made on the first material. Untick the sweep and they come
back, since without one each sheet names its own file.

**Removing material 2 while a pair list is configured** cannot leave the list
alone: its `a + b` lines are unusable by a one-sheet config, and the sweep
would silently stop sweeping. The list is collapsed to each pair's first
member, after a confirmation naming how many pairs there are and which second
materials are being dropped. The collapsed list is written to *this config's
own* list file — never over a curated `.txt` the picker was merely pointed at,
which the dialog says by name. An unsaved config with nowhere to write it
refuses the removal and asks you to save first; a failed write leaves the
second material in place rather than half-removing it.

### What the GUI checks, and what it does not

Both sheets must template on their own placeholder, and this is checked **per
key**, not per section: a `[2D-2]` whose `mat` reads `{mat2}` while its
`cif_path` still reads `{mat1}` builds the first material twice, inside
directories correctly named after the pair. Every path resolves and nothing
downstream notices, so the expansion refuses it by name — and Generate reports
that refusal (see [Generate](#generate)) instead of previewing an empty list of
output directories.

Any refusal that stops the sweep being counted — that one, a list whose lines
name the wrong number of materials for the sheet sections present, materials
whose files do not resolve — is reported in the issues panel on every
validation pass, Render included, in the expansion's own words. While one
stands, the panel says the `{mat}` placeholders *cannot* be substituted rather
than promising that they will be, and the config counts in the header read as
`—`.

A pair list left behind on a single-material config — hand-edited, or written
before the second material was dropped some other way — is not adopted as a
selection and not overwritten: the tick list opens empty, so committing the
section changes nothing on disk. It does not sweep either, and generation says
so, naming the line that asks for a pair the config has no `[2D-2]` to hold.

The same holds for the `.txt` picker, in both directions: pointing it at a
list of the wrong kind — ordered pairs while this config has one material,
single names while it has two — selects nothing and leaves the file alone.
The caption under the picker says which shape this config needs and that the
file was not touched. The selection stays empty through a later directory
scan too, so choosing the two folders afterwards cannot quietly turn the
curated file into the other kind of list. Fix it by editing the `.txt`, or by
adding/removing the second material so the config wants what the file holds.

Configs the window still declines to open are the ones it cannot represent:
three or more sheet sections, sheets numbered other than `[2D-1]`/`[2D-2]`,
and a plain `[2D]` mixed with numbered ones. Each is refused with the reason;
use the CLI for those.

A pair sweep is previewed one member at a time like any other sweep, and the
status line names the pair on screen: *"Previewing pair h_MoS2 + h_WS2 (1 of N
in the sweep)"*, or *"(the only pair in the sweep)"* for a list of exactly one
line. Which member that is comes from the same expansion Generate uses, so it
is genuinely one of the runs that would be produced — `{mat1}`/`{mat2}`
substituted, `materials_list` gone.

## Render

**Render** (**Ctrl+R**) builds the configured components, assembles them as
the generated LAMMPS script specifies, and displays the result. It never
minimises and never runs a melt-quench: if an amorphous structure is not
already cached, the build refuses rather than spending minutes in `mpiexec`
and writing into the packaged materials directory.

What you see is parsed out of the build's own outputs — the assembly script for
the cell and component placement, `system.in.settings` for component membership
and the element legend — so the preview cannot drift from what LAMMPS will read.

A render updates the existing viewport in place rather than tearing it down
and rebuilding it: the camera position, zoom, per-component visibility ticks
and the selected colour mode all survive a re-render. Only the first render,
or a cell whose largest dimension has drifted more than 10% from
the one last framed triggers a re-frame — anything smaller (recolouring,
toggling a component, toggling the box) leaves the camera exactly where you
put it. Validating the form no longer repaints every field on every pass,
either — a field only gets restyled when its error state actually changes,
so typing in one box does not flicker the rest of an already-validated
section.

The viewport controls:

- **Colour** by **Element**, **LAMMPS type** or **Component**, each with a
  swatch per type in the legend. LAMMPS type assigns every type its own
  colour off a golden-angle hue walk, which is what keeps a component with
  many LAMMPS types — an 18-type system, say — readable; colouring by
  element runs out of visually distinct assignments much sooner;
- **Box** toggles the simulation cell;
- **Show** hides individual components. Sheets are what most often generate
  incorrectly, and you cannot inspect one through a substrate;
- **Top / Front / Side / Perspective** camera presets. Drag to rotate, scroll
  to zoom.

A component that reaches a periodic boundary is called out: it will interact
with its own image, which is what the ≥2× tip-diameter rule exists to prevent.
The warning names the components that are *meant* to be finite — `tip` and
`flake` — rather than exempting the ones known to span. A sheet-family
heterostructure arrives as a single component (`all_atoms`) spanning the
coincidence cell: the cell *is* the interface's only in-plane periodicity, so
touching the wall is what it is supposed to do and there is nothing a user
could enlarge. Warning about it said only that a heterostructure had been
built.

Two things follow from naming components positively, and both are deliberate:

- `[general] finite_sheet` sizes the sheet as an island inside the substrate
  box while still reading it as `group 2D`. There — and only there — a `2D`
  touching the wall *is* the ≥2× failure, so the config adds `2D` to the list
  for that render. Nothing in an assembled system records which geometry it
  came from, so this is known from the config, not from the atoms.
- A flake in the **sheet** family has no component of its own: it is inside
  `all_atoms` with the substrate that legitimately spans. An oversized flake
  is therefore not called out there. Separating it would mean reading the
  builder's own per-layer placements rather than the run script.

On the heterostructure AFM path the warning is quiet by construction: the
builder replicates the coincidence cell until the footprint is at least four
tip radii, so a centred tip cannot reach the wall. The homogeneous AFM path
uses your `[2D] x`/`y` directly, where it can.

Builds run in a separate process, so a LAMMPS failure reports an error instead
of closing the window. Repeating a render with unchanged inputs reuses the
cached build; the cache key includes a fingerprint of the builder and template
sources, so editing a template invalidates it.

Editing the config after a render — without re-rendering — marks the preview
stale: the viewport shows a "config changed since last render" badge, so an
outdated preview does not pass for current.

## Generate

**Generate** (**Ctrl+G**) writes the full package — every config in the
sweep, every layer — into the workspace, optionally with HPC job scripts. It
asks for confirmation with the three counts first, followed by the output
directories it is about to create (up to 8, then "… and N more"). Any two
that coincide are called out with a warning — that means two runs would
overwrite each other, the exact hazard the
[loopability rule](#which-fields-may-hold-a-list) exists to prevent.

A config that cannot be expanded at all — a material list none of whose
materials resolve, a pair sweep whose sheets are templated on the wrong
placeholder — never reaches that confirmation. The reason the expansion gives
is reported in a dialog and written into the issues panel, and nothing is
generated: it names the section at fault and what to put in it, which an empty
list of output directories does not.

Unlike Render, this may generate a missing amorphous structure, because here it
is what you asked for rather than a side effect of looking.

Validation (not just the Generate confirmation) also warns — without
blocking — when an AFM-like sheet's footprint is under twice the tip
diameter: the same PBC self-interaction rule Render calls out on a built
system, checked here against the configured `x`/`y`/`r` values before you
have even rendered.

## Configs

**Open config…** reads any `.ini` the CLI accepts and fills in the form.
**Save config…** writes one back. Round-tripping preserves every value, and is
tested against every config in `examples/`.

Two limits worth knowing:

- Inline comments are not preserved. The reader discards them, so a round trip
  cannot keep yours; canonical unit and help comments are re-emitted instead.
- A heterostructure is exactly two materials. A config with three or more sheet
  sections, one numbering them anything other than `[2D-1]`/`[2D-2]`, or one
  mixing a plain `[2D]` with numbered sections, is refused with that reason
  rather than silently renumbered. Ordinary `[2D-1]`/`[2D-2]` pairs open
  normally — see [Two materials](#two-materials-heterostructures).

Relative paths in a config are resolved against the config's own directory and
its parents, so a config written for the project root still works when opened
from elsewhere.

A section can occasionally **refuse its commit** — the form holds something the
document will not accept, such as a pair selection that no longer matches the
config's shape. When that happens the form and the config disagree, and what
would be written is not what is on screen. **Save**, **Generate** and applying
settings therefore stop rather than proceed: you get a dialog naming the
disagreement, and the same text stays in the issues panel, which the status bar
could not do — it is overwritten by the next message, and a saved `.ini` that
silently disagreed with the form would be indistinguishable from a correct one.
**Render** is the exception: it persists nothing and launches nothing, and
looking at the geometry is often how the disagreement gets diagnosed, so there
the refusal is reported without blocking. There is no override — a refusal has
to be resolved in the form.

Closing the window with an unsaved config or settings edit prompts
**Save / Discard / Cancel** rather than discarding it silently. **Cancel**
leaves the window open; so does a **Save** that is itself cancelled (e.g. the
file-save dialog) or fails — the window only closes once nothing is left
unsaved.

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

## LJ overrides

The optional `[lj_override]` table (see [configuration.md](configuration.md))
gets a dedicated editor. Each row's two elements are dropdowns pre-filled with whatever the
*configured* simulation actually places — read from the CIF a component
points at, or its `mat` text when the CIF cannot be read — so typing a pair
no longer means guessing symbols right by hand. Typing something the scan
never saw still works; the scan is advisory, not a validator.

Two things get a non-blocking note in the row rather than a rejection:

- a **duplicate pair** — `Mo-S` and `S-Mo` normalise to the same key, the
  way `PotentialManager._pair_key` treats them, so the second occurrence is
  flagged as a duplicate of the first;
- an **unknown element** — one the scan never saw in this configuration, in
  case it is a typo.

**Shared overrides file** — `[lj_override] file = lj.txt` — covers a whole
material sweep from one plain-text table instead of retyping the same pairs
into every config:

```
# comment
Mo-S 0.4124 3.75114
S-S  0.1984 3.62368
```

One `El1-El2 epsilon sigma` line per pair (whitespace- or comma-separated),
`#` starts a comment, blank lines are ignored. Inline rows already in the
`.ini` win over the file's — a single run can still override one pair
without touching the shared table. A missing or unreadable file is skipped
silently, the same as a half-typed `materials_list` path. **Import…** pulls
the file's rows into the table as ordinary editable rows for a quick look
or a one-off tweak, but does not replace the file reference — it is still
resolved independently when the config is expanded, and the merged table
(inline rows plus whatever the file adds) is what gets inlined into every
generated config.

## Database

The window has two top-level tabs. **Builder** is everything above; **Database**
browses what has already been run.

**Source.** The Local store is this machine's AiiDA store — every run
generated or imported here, files included. The former *Central* source (the
shared summary database over its REST API) was removed; a workspace that
remembered it falls back to the Local store. A store whose backend is not
installed cannot be selected: its radio button is disabled and the message
panel says what to install.

**Filters.** Family, model, material, potential, layers, a load range (force
in nN or pressure in GPa — a run carries one or the other, never both),
temperature range and set. They run against the store, so pressing **Search**
re-queries. The **Find** box is different: it narrows the rows already
fetched, instantly, without going back to the store. There is no
Completeness filter — neither store's query takes one — but an incomplete
row is not hidden; it is called out in the table instead (below).

**The table.** Sortable by any column, multi-row selection. The columns shown
are a curated subset; **All columns** widens it to every stored field. The
Status column shows the row's own status word (`imported`, `staged`,
`published`, `failed`, …); a row the store marks incomplete gets
`(incomplete)` appended to that word and is coloured as a warning, and a
failed or rejected row is coloured as an error regardless of completeness.
Selecting a row fills the panel below it with every attribute of the run and
its results, plus the stored files and their sizes — which answers "is this
worth downloading" without downloading it. A query the row limit cut short
reports it honestly, e.g. "5000 results (limit 5000)", rather than implying
that is the whole store.

**Downloads.** Four buttons, each asking once for a destination directory —
there is no separate confirmation dialog:

| Button | What lands in the directory |
|---|---|
| **Results files** | the stored tier files and logs of the selected runs, verbatim |
| **Provenance bundle** | `provenance.json`, `config.json`, `settings.yaml`, the cif and the potential files |
| **Summary CSV** | one row per visible table row, in the columns on screen |
| **Recreate run tree** | a directory tree `FrictionSim2D postprocess read` can read; tick *rebuild lammps/* to regenerate the inputs too |

**Recreate run tree** exports whole sets, not only the selected rows — one
selected row pulls in every run beside it in the same set. Rather than a
second dialog, a warning naming the affected sets and their run counts is
written to the message panel before the destination chooser opens; cancelling
that chooser is how you back out.

**Import run directory…** ingests a simulation directory into the local store
and prints the ingest report in the message panel, including any results
still incomplete, any run directory with no results at all, and any path it
could not even parse. It is the only write action here — deleting and
submitting stay in the CLI.

**Connection settings are not edited in this pane.** The local store uses
`settings.aiida.profile` from the Settings tab. **Check connection** reports
what it found — profile loaded, how many sets are stored — in
the message panel, which is where every message from this pane appears;
nothing here opens a dialog except the destination chooser. A local profile
that fails to load points you at `FrictionSim2D aiida setup`.

A **list** query's result is cached, keyed on the filters plus the connection
it ran against — the AiiDA profile —
since the same filters against a different store are a different
answer. A result served from the cache says so in the message panel — its
timing is the original query's, not a fresh one's. **Refresh** bypasses the
cache, and a successful **Import run directory…** clears every cached list,
since it just changed what the store holds. The ten most recent cached
queries are kept; older ones are deleted as new queries arrive. Every other
request kind (a row's detail, a download, a connection check) is never cached
and runs from its own scratch directory. The source, the filters and the last
download directory are remembered per workspace.

Switching source empties the table: a row belongs to the store it came from,
and the two stores identify a row differently.

## Plots

The third top-level tab builds a **plot document** -- the same JSON
`FrictionSim2D postprocess plot` reads -- and renders it. A document written
here runs unchanged on a cluster, and one written by hand or by
`postprocess plot --init` opens here.

### Datasets

*Available datasets* lists every directory the workspace holds that looks
plottable: one with an `outputs/` containing at least one
`output_full_*.json`. Two places are scanned -- the workspace root, where
Generate writes its simulation packages, and `.preview/_datasets/`, which is
where a set pulled out of the local AiiDA store belongs:

```bash
FrictionSim2D aiida export-results <label> \
    --output-dir <workspace>/.preview/_datasets/<label>
```

Select one and press **Add dataset** to put it in the document; the pane does
not run `export-results` itself.

If a directory you expect is missing, it has no export yet:

```bash
FrictionSim2D postprocess read <that directory> --export
```

### Plots and the form

*Plots* lists the document's plots. **Add plot** appends one of the type in
the combo beside it, and selecting a row opens it in the form below.

The form is generated from `frictionsim2d/postprocessing/plot_config.py`, the same
Pydantic schema the CLI validates against, and is split in two groups:
**Common** (what every plot type takes -- filters, axes, labels, limits, the
sampling tier) and one named for the plot type (what only it takes). Changing
the type keeps the common values and drops the previous type's own.

A field added to the schema appears here with no GUI change at all. An invalid
value is reported in the status line at the bottom right, never in a dialog.

### Rendering

**Render** draws the **selected plot only**, as a PNG, and shows it in the
panel on the right. Previewing one figure should not cost the time of ten.

**Export** draws **every** plot with the chosen preset's own output formats --
`publication` writes PDF, SVG and PNG at 600 dpi, sized for a single journal
column. The status line names the formats and the directory they landed in.

The **Preset** combo offers the shipped presets; `(none)` uses the screen
defaults. For finer control, save the document and pass `--settings` on the
command line.

### Saving

**Open...**, **Save** and **Save as...** read and write the plot document
itself. `Save` writes back to the file the document came from and asks for a
path only the first time.

### Why there is no live canvas

The preview is a PNG in a scrolled label, not an embedded matplotlib canvas,
and that is deliberate. Rendering happens in a subprocess
(`python -m frictionsim2d.gui.plot_worker`), so:

- matplotlib never installs a second Qt backend beside the one OVITO's
  viewport already owns -- the failure mode there is a hung window with
  nothing to click;
- a slow render cannot freeze the interface, and can be abandoned by killing
  the process;
- the figures on disk are the same files the CLI produces, not a screen-only
  approximation of them.

## Not included

Job monitoring (Part 2 of the requirements) and standalone single-component
builds. Stacks of more than two materials are
not editable here either — see [Configs](#configs).
