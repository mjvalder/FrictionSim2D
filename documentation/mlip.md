# Machine-learned interatomic potentials (DeePMD and MACE)

FrictionSim2D can generate LAMMPS decks that drive a 2D sheet with a
machine-learned interatomic potential (**MLIP**) instead of a classical
Stillinger–Weber potential. Two frameworks are registered, as foundation models:
**DeePMD** (DPA family) and **MACE** (MACE-MP family). Training and fine-tuning
are out of scope.

Design authority for everything on this page:

- Specification (**v4, binding for the capability matrix**):
  [`docs/superpowers/specs/2026-08-11-mlip-post-m0-design.md`](../docs/superpowers/specs/2026-08-11-mlip-post-m0-design.md)
- Campaign design (**v5, 2026-08-24, revised 2026-08-31**):
  [`docs/superpowers/specs/2026-08-24-mlip-campaign-v5-design.md`](../docs/superpowers/specs/2026-08-24-mlip-campaign-v5-design.md)
  — supersedes v4 on **framework status** (§4.2: the symmetrix MACE stack, its
  measured energies and the `mace_types` patch). v4 still governs the coverage
  topology; where the two disagree about MACE, v5 wins.
- Implementation plan:
  [`docs/superpowers/plans/2026-08-11-mlip-post-m0.md`](../docs/superpowers/plans/2026-08-11-mlip-post-m0.md)
- Executed feasibility results:
  [`documentation/milestone0.md`](milestone0.md)
- **Executed AFM physics gate (v6 §6):**
  [`documentation/mlip_afm_gate.md`](mlip_afm_gate.md) — run 2026-09-09.
  Gates 1–3 (deck shape, Si sanity, energy agreement vs ASE) **pass**; Gate 4
  (cross-framework) is **open**: MACE-MPA-0 and the DPA-3.2 MPtrj head disagree
  by 44 % RMS on forces on the reference cell. **A tip-bearing MLIP result is
  not framework-independent on current evidence** — read that page before
  reporting an AFM, `pes-tip` or `indent` MLIP number.

> **v4 supersedes the 2026-08-06 (v3) design.** v3's matrix rested on the
> assumption that DeePMD accepts `NULL` entries in a `pair_coeff` type map, so
> that an MLIP could share a cell with SW/LJ under `pair_style hybrid`. That
> assumption was tested on real hardware and **is false**. Three of v3's four
> headline setups depended on it and are struck. If you are reading an older
> note, a `mlip_interlayer` key, or an `examples/mlip/*_lj.ini` filename, it is
> describing a topology no LAMMPS build accepts.

---

## 1. Capability matrix (binding)

Capability is decided by **potential-coverage topology**, not by builder family.
A run is either

- **full-cell MLIP** — every atom type in the cell is covered by *one* MLIP
  instance with a full element list, or
- **partial-cell MLIP** — the MLIP shares the cell with SW/LJ through
  `pair_style hybrid` and a `NULL`-padded type map.

**Partial-cell is unavailable for both frameworks.** That is the whole matrix:

| Coverage | DeePMD base | DeePMD + D3 | MACE base | MACE + D3 |
|---|---|---|---|---|
| **Full-cell** — one instance, full type map, every atom | **Yes** ¹ | **Yes** ² | **Yes** ³ | **Yes** ³ |
| Partial-cell (MLIP shares the cell with SW/LJ) | **No** ⁴ | No | **No** ⁵ | No |

¹ Executed (spec §1, E1): `pair_style hybrid deepmd <model> lj/cut` with
`pair_coeff * * deepmd Mo S Mo S Mo S` on a 36-atom MoS2 trilayer gives
`E_pair = -256.90309 eV` (−7.14 eV/atom), model resident on GPU 0.

² Executed (spec §3 footnote 1, item **E8**, 2026-08-11) on the backported
binary of §7: `pair_style hybrid/overlay deepmd <model> dispersion/d3 bj pbe
16.0 8.0` on the 24-atom bilayer gives `E_pair = -180.15327 eV`, decomposing via
`compute pair` into `c_edp = -171.42832` and `c_ed3 = -8.7249526` — a D3
contribution of 4.84 % of the total. 20 NVE steps stable, TotEng drift 9e-5 eV,
**no segfault**: the community reports of DeePMD crashing under
`hybrid/overlay` did not reproduce. The d3 term is **bit-identical** to the same
quantity measured standalone on an upstream LAMMPS 2025 build, which is what
certifies the backport as numerically faithful rather than merely compiling.

³ **Built and measured 2026-08-24 / 2026-08-31** (this footnote replaces the
"untested" text that stood while the stack was unbuilt). MACE is driven through
[wcwitt/symmetrix](https://github.com/wcwitt/symmetrix) — `pair_style
symmetrix/mace`, **not** the ACEsuit fork's `pair_style mace` — because
symmetrix is the implementation ML-MIX supports for hybrid ML/MM, whereas
`mliappy mace` is listed incompatible with it. On MACE-MPA-0-medium: an h-MoS2
bilayer binds unaided at **6.717 Å / 0.068 J/m²** (weak against the 0.25–0.30
literature value, but genuinely bound — unlike DPA-3, which has no minimum at
all), and with D3 it *over*-binds at 0.474 J/m². So for MACE `dispersion = none`
is a legitimate control rather than a broken run. **Two traps gate every MACE
number — see §3 and `scripts/hpc/README.md` before quoting one.**

⁴ Executed (spec §1, E2/E3). `pair_coeff * * deepmd Mo S NULL NULL NULL NULL`
dies with `ERROR: Pair hybrid sub-style deepmd is not used` — and it fails for
**one** instance exactly as it does for three, so `NULL` is the cause, not
multi-instancing. Restricting by type range instead
(`pair_coeff 1*2 1*2 deepmd Mo S`) dies with `deepmd requires that the scale
should be set to all atom types, i.e. pair_coeff * *` (`pair_deepmd.cpp:816`).
There is no third spelling.

⁵ `pair_style mace` rejects `NULL` type maps by *segfaulting* rather than
erroring, and the maintainer states the architecture cannot selectively ignore
atom types (ACEsuit #1314).

### What this struck

v3's setups **1** (sheet-on-sheet MLIP with an explicit `lj/cut` interlayer),
**3** and **4** (MLIP sheet with SW tip/substrate) are all partitions and all
require `NULL`. They are not "conditional" or "pending a build" — they are
**unavailable**, and the four example configs that demonstrated them have been
deleted rather than annotated, because a shipped example is a recommendation.

### Overlay-LJ: rejected on *physics*, not unavailable

Keep this distinction — "we can't" and "we shouldn't" have different future
work.

`pair_style hybrid/overlay deepmd <model> lj/cut` with a full type map **runs**
(spec §1, E4: stable 20-step MD, `c_edp = -256.898`, `c_elj = -1.684`). So it is
mechanically tempting to rescue setup 1 by swapping `hybrid` for
`hybrid/overlay`. That is wrong, and the reason is physical:

- the MLIP's cutoff (6 Å for DPA-3) **exceeds** the gap between facing S planes
  (~3.1 Å at a 6.15 Å interlayer spacing), so the model *already* describes the
  interlayer interaction;
- a UFF-mixed `lj/cut` parameterised to **be** the interlayer potential
  therefore double-counts it.

To use LJ in overlay legitimately, ε/σ would have to be re-fitted as a
*residual* correction to whatever the MLIP gets wrong — i.e. a bespoke,
unpublished dispersion correction, fitted per material and per model. **D3 does
that job better**: published C₆ coefficients, damping fitted per functional, one
canonical parameterisation. Overlay-LJ is strictly dominated and is not a
supported mode.

**Consequence: LJ has no role as an interlayer potential in an MLIP cell.**
`dispersion = d3` is the only vdW lever.

### "Full-cell" means one instance, NOT one component (updated, v6)

The builders used to register **one MLIP instance per component** (sheet, tip,
sub, flake, each hetero sheet) and partition the atom types between them with
`NULL`-padded maps. Post-E2 that topology is invalid for DeePMD as well as for
MACE, so v4 §6 restricted full-cell to a **single-component** cell.

**That restriction is gone.** The builders now emit **one instance whose type
map spans every atom type in the cell** (`TypeRegistry.build_full_map`), which
is exactly the shape `pair_coeff * *` wants. An AFM cell comes out as:

```
pair_style hybrid/overlay deepmd <model> dispersion/d3 bj pbe 16.0 8.0
pair_coeff * * deepmd Si Si Si Si Si Si Mo Mo Mo S S S Mo  # sub, tip, sheet
pair_coeff * * dispersion/d3 Si Si Si Si Si Si Mo Mo Mo S S S Mo
```

(13 types: sub Si x3 and tip Si x3 — Langevin splits each into
mobile/fix/thermo — then the 3-layer sheet as Mo(L1..L3), S(L1..L3), then the
virtual drive atom taking the placeholder element.) So **AFM, pes-tip, indent,
hetero and flake can all host an MLIP**, and `examples/mlip/afm_mlip_full.ini`
is a worked AFM example.

What replaces the old rule is **uniformity**, not component count:

> **Every component in a cell must use the SAME MLIP.** One `pair_coeff * *`
> claims every atom type, so a classical component beside an MLIP would have
> its types silently absorbed, and a second MLIP framework's line would simply
> override the first. Both are refused at generation time —
> `mlip.validate_mlip_cell` (rules 2 and 3a) and again in
> `PotentialManager` at emission.

Two further constraints on an MLIP cell:

- **`amorph = 'a'` is refused** (v6 §5). Amorphisation is melt-quench MD run at
  *generation* time, so it can evaluate neither the `pair_style zero` build
  surrogate (no forces) nor the MLIP itself. For AFM use a lattice-matched
  crystalline substrate — `amorph = c` with `match = true` — which
  `AFMSimulationConfig` accepts in place of an amorphous one.
- **The model must cover every element in the cell.** For MACE this is checked
  at generation time by reading the symmetrix `.json`; for DeePMD the type map
  is inside the serialized `.pth` and cannot be read without torch, so it is
  not enforced. Naming an element the model lacks does **not** error at run
  time — LAMMPS maps those types to the model's first N elements and returns a
  plausible, wrong energy (E7).

See `docs/superpowers/specs/2026-09-03-mlip-multicomponent-fullcell-design.md`
(v6), which closes v4 §6 and implements v5 §8.

### Drive method and constraint modes

**Virtual-atom drive** (`drive_method = virtual_atom`, the project's preferred
drive) survives an MLIP. Executed for both halves (milestone0 M0.4):

| | isolated-atom energy of the placeholder | force on the virtual atom |
|---|---|---|
| `dispersion/d3` | **zero** — PotEng identical to the no-virtual run | 0, 0, 0 |
| `deepmd` (DPA-3.1-3M, `Alex2D` head) | **-4.64264 eV**, a nonzero *constant* | 0, 0, 0 |

Constant means constant: it was unchanged to all printed digits after two
separate `displace_atoms` of the virtual atom.

> **Consequence.** Absolute `E_pair` from a `virtual_atom` run is **not**
> comparable to an `smd` / `fix_move` run of the same system without subtracting
> that constant. Forces, force differences, and therefore every friction
> observable are unaffected.

`constraint_mode = atom_bonds` and `com_spring` are **unsupported** with an MLIP
sheet and are rejected by `validate_mlip_cell`: they convert the sheet to a
molecular atom style with harmonic bonds and `special_bonds` exclusions, whose
interaction with MLIP neighbour lists is unvalidated. Use
`constraint_mode = none`.

---

## 2. Config surface

Three keys in `[2D]`, plus a mandatory `lat_c`:

```ini
[2D]
pot_type = deepmd          ; or mace -- a THIRD potential category alongside sw/tersoff/reaxff
pot_path = ../models/dpa3_alex2d.pth
dispersion = none          ; 'none' | 'd3' -- the ONLY vdW lever
lat_c = 6.15               ; MANDATORY for MLIP pot_types -- INTERLAYER distance
```

### `mlip_interlayer` is gone and is a hard error

The key only ever selected between a partition (`lj`) and full coverage
(`native`), and the partition is unavailable for both frameworks. Full-cell is
the only topology, so the key carries no information.

Pydantic ignores unknown keys by default, which here would silently change a
run's physics, so `SheetConfig.reject_removed_mlip_keys` rejects it **by name**
(`REMOVED_SHEET_KEYS` in `frictionsim2d/core/config.py`) with a `ValidationError` naming
spec §5. Delete the line from any old config; do not translate it.

### Hard validation rules

| Rule | Where it is enforced |
|---|---|
| `mlip_interlayer` present at all → `ValidationError` naming spec §5 | `SheetConfig.reject_removed_mlip_keys` |
| `dispersion` requires an MLIP `pot_type` | `SheetConfig.validate_mlip_capability_rules` |
| MLIP `pot_type` requires an explicit `lat_c` | `SheetConfig.validate_mlip_capability_rules` |
| Model extension must match the framework (`.pb`/`.pth` DeePMD; `.json` MACE, extracted by `symmetrix_extract_mace`) | `ComponentConfig.validate_mlip_model_extension` |
| **No MLIP in a cell that also holds a classical component** (either framework) | `mlip.validate_mlip_cell` |
| **Every component in the cell must use the SAME MLIP** (no second framework) | `mlip.validate_mlip_cell` (rule 3a) + `PotentialManager` at emission |
| **`amorph = 'a'` refused** under an MLIP (generation-time melt-quench) | `mlip.validate_mlip_cell` (rule 3c) + `get_single_component_commands` |
| Model must cover every element in the cell (**MACE only**; DeePMD unknowable without torch) | `PotentialManager._check_model_covers` |
| Every instance of one framework in a cell must name the **same** model file | `mlip.validate_mlip_cell` |
| `constraint_mode` in {`atom_bonds`, `com_spring`} rejected with an MLIP sheet | `mlip.validate_mlip_cell` |

Every message names the rule it enforces and points at the spec section, so a
rejected config is self-diagnosing.

### `lat_c` is the INTERLAYER distance

> **This is the single easiest thing to get wrong.** In FrictionSim2D `lat_c` is
> the **layer-to-layer distance** (`settings.geometry.lat_c_default` is `6.0` Å),
> *not* an in-plane lattice constant. MoS2's in-plane *a* ≈ 3.16 Å is never
> written into `lat_c`; the bulk 2H interlayer spacing ≈ 6.15 Å is. Putting the
> in-plane constant there starts the stack with the layers interpenetrating.

`lat_c` is mandatory for MLIP sheets because it cannot be found by minimizing:
MLIP stacking runs under a `pair_style zero` surrogate (§6), so there is no
force to minimize against. Get the number from
[`scripts/mlip_bilayer_benchmark.py`](../scripts/mlip_bilayer_benchmark.py),
which reports it as `equilibrium_separation_A` (echoed as
`lat_c_recommendation_A`). The **separate** key `lat_c_in_plane_A` in the same
JSON is the in-plane lattice constant and does **not** go into a config's
`lat_c`.

### `settings.yaml` additions

```yaml
potential:
  d3_functional: pbe    # DFT functional whose D3 parameters are used
  d3_damping: bj        # canonical name, enum-validated: original | bj | zerom | bjm
  d3_cutoff: 16.0       # Angstrom  (rthr)   -- D3 pair cutoff
  d3_cn_cutoff: 8.0     # Angstrom  (cn_thr) -- coordination-number cutoff
```

All four are ordinary settings you can change per machine or per campaign; they
flow into the emitted line through `frictionsim2d/core/mlip.py::d3_style_token` and
nowhere else.

**Where 16.0 and 8.0 come from — and what is *not* established.** They map onto
the third and fourth positional arguments of `pair_style dispersion/d3`
(`<damping> <functional> <rthr> <cn_thr>`; both are squared internally). The
values were fixed at the start of Milestone 0 and the *emitted string* was then
executed and verified (`E_pair = -8.7249526 eV` on the 24-atom bilayer, item
M0.2a). **No convergence study selects them**, and they are well below the
`dftd3` reference defaults (~50 Å pair / ~21 Å CN) — they are a cost/accuracy
truncation inherited as a default, not a measured optimum. Two consequences:

- If a published D3 comparison matters to you, converge `d3_cutoff` yourself
  first; the interlayer binding energies quoted here inherit this choice.
- **`d3_cutoff` constrains your geometry.** The communication cutoff becomes
  `d3_cutoff + neighbour skin`, and LAMMPS cannot send ghosts further than one
  box length, so 16.0 forces a periodic cell of at least ~16.3 Å in x and y. An
  `x = y = 15` sheet builds a 15.9 x 16.5 Å box and aborts.
  `mlip.check_d3_fits_box` catches this at generation time and tells you to
  either shrink the cutoff or grow the footprint — but on a small screening
  cell, shrinking the cutoff is a physics decision, not a formality.

`d3_damping` is a **canonical** name that `frictionsim2d/core/mlip.py::d3_style_token`
turns into the token a given build accepts. See §5 for the verified argument
order.

### Output directories

An MLIP run is told apart by main's ordinary **potential segment**,
`pot_<type-slug>_<stem>`, which `frictionsim2d/run/naming.py::output_dir_for` appends below
`K<temp>` whenever the sheet names a `pot_path` (always true for an MLIP). The
`K<temp>` leaf itself carries no suffix. The MLIP's dispersion choice rides in
the slug, so with/without-D3 runs of one model never share a directory:

```
<...>/K<temp>/pot_<framework>_<stem>        dispersion = none
<...>/K<temp>/pot_<framework>-d3_<stem>     dispersion = d3
```

`<stem>` is the model filename without its extension (`.pth`/`.pb`/`.json`).
Real examples produced by the configs in §4:

```
sheetonsheet/h-MoS2/30x_30y/K0/pot_deepmd-d3_dpa3_alex2d
pes_sheet/h-MoS2/30x_30y/K0/pot_deepmd-d3_dpa3_alex2d
afm/h-MoS2/30x_30y/sub_Si_tip_Si_r15/K300/pot_deepmd-d3_dpa32_omat24
```

The slug is produced by `frictionsim2d/potentials/utils.py::pot_dir_token` and decoded by
`frictionsim2d/run/naming.py::mlip_flavor_from_pot_token` (exposed on `parse_run_path`'s
`RunTokens` as `mlip_framework` and `dispersion`, both `None` for a classical
run) and by `frictionsim2d/hpc/manifest.py::parse_mlip_flavor`, which returns
`(framework, dispersion)`. Classical paths are unchanged.

---

## 3. Model acquisition

Models are downloaded and converted **once**; the converted artefact is what
LAMMPS reads. Neither step needs a GPU. Put the file somewhere shared (the
examples assume `../models/` relative to the simulation root) and point
`pot_path` at it.

MLIP models are staged **once per simulation root** under a shared
`potentials/` directory, with a `potentials/` prefix in the emitted
`pair_style` / `pair_coeff` lines — not copied into every run directory, which
for a 100-run sweep would be gigabytes. Classical potentials keep their per-run
staging.

`provenance/potentials/` also holds the model, per run — provenance has to carry
a real, openable file or it does not establish anything. But those entries are
**hard links**, not copies (`frictionsim2d/potentials/manager.py::stage_model_file`), so a
20-run campaign pays for the model once rather than twenty times. Each run
directory still contains a genuine file, hashed into `manifest.json` exactly as
before; the hash is what pins identity, and a byte copy adds nothing to it.
Linking falls back to a copy across filesystems, so this can only save space,
never fail a build. Classical potentials are KB-scale text and keep their plain
per-run copy.

> **One transfer caveat:** hard links survive a move but not every copy.
> `rsync` collapses them only with `-H`, and `tar` must be told as well;
> without those flags the links re-expand into N independent copies at the far
> end. The bytes still arrive — this is a transfer-size question, not a
> provenance one.

### Are the models material-specific?

The two frameworks differ, and it matters for how you provision them:

| | DeePMD `.pth` | MACE `.json` |
|---|---|---|
| Element coverage | **All 118 elements** — one file serves every 2D material | **Only the elements you extract for** (`--chemical-symbols Mo S`) |
| Per-material work | none | re-extract per element set |
| Size (as used here) | ~18 MB (`dpa3_alex2d.pth`) | ~120 MB (`mace-mpa-0-medium-Mo-S.json`) |

Neither is shipped in `examples/potentials/`. They are foundation-model weights
with their own licences and release channels, they are far too large for the
repository, and the DeePMD one is a *derived* artefact (a freeze of a
checkpoint at a chosen head) rather than a file with a canonical download URL.
Both recipes below produce them from published sources in one command.

### DeePMD — `dpa3_alex2d.pth` (the model actually used here)

DeePMD needs a **frozen** model; a training checkpoint will not load. The route
that works, end to end:

```bash
dp pretrained download DPA-3.1-3M
dp --pt freeze -c DPA-3.1-3M.pt -o dpa3_alex2d.pth --head Alex2D
```

- **DPA models are multitask.** A bare `dp freeze` fails with `Head must be
  specified in multitask mode`. `--head Alex2D` selects the Alexandria
  2D-materials head, which is the right one for layered 2D systems.
- `DP_ENABLE_PYTORCH=1` must have been set when deepmd-kit was installed, or the
  PyTorch custom OPs are missing and the freeze silently produces a model LAMMPS
  cannot run.
- **Never** write `pair_style deepmd model1.pth model2.pth`. Two model paths
  switch DeePMD into model-deviation (uncertainty) mode, in which only the first
  model drives the forces — not an error, just wrong physics.

> A published MoS2-specific DeePMD model is acceptable **only** as a code-path
> test fallback. It is not a scientific substitute for a DPA foundation model,
> and results obtained with it must not be quoted as physics.

### MACE — `mace-mpa-0-medium-Mo-S.json` (symmetrix)

symmetrix does **not** read a MACE `.model` or the ACEsuit fork's
`.model-lammps.pt`. It reads a plain `.json`, extracted for a specific element
set:

```bash
source "$PREFIX/mace-venv/bin/activate"   # STAGE=model ./build_lammps_mace.sh

# 1. Fetch a MACE-MP / MACE-MPA foundation checkpoint (lands in ~/.cache/mace),
#    e.g. via mace.calculators.foundations_models.mace_mp(model="medium",
#    default_dtype="float64", device="cpu").
# 2. Extract the symmetrix model FOR YOUR ELEMENTS.
symmetrix_extract_mace -m mace-mpa-0-medium.model --chemical-symbols Mo S
```

- **The element list is baked in.** `--chemical-symbols Mo S` produces a model
  that knows only Mo and S; a different material needs a different extraction.
  Name the file accordingly (`mace-mpa-0-medium-Mo-S.json`) — the shared
  `potentials/` store is keyed by filename, so two element sets coexist happily
  but two files both called `mace.json` would collide.
- The extraction is the **only** step that needs `mace-torch` and Python. The
  cluster that runs LAMMPS needs neither: ship it the `.json`.
- `pair_coeff` carries the model path for MACE (`pair_coeff * * <file> Mo S`),
  unlike DeePMD, which carries it in `pair_style`. The registry handles this —
  see `MLIPSpec.model_in_style_line`.
- Keep **float64**; float32 shifts energies at the meV level.

#### Traps that decide whether your MACE numbers are real

**1. The CUDA backend is silently wrong on a stock checkout.** symmetrix
allocates `mace_types` with one slot per *model element* but fills one per
*LAMMPS atom type*; they agree only when `ntypes == n_elements`, which is false
for any multi-layer cell. Unpatched CUDA returns **-126.5489 eV** where the
correct answer is **-173.40779 eV** (24-atom h-MoS2 bilayer) - a 27 % error with
no diagnostic. `scripts/hpc/symmetrix-mace_types-overflow.patch` fixes it and
`build_lammps_mace.sh` refuses to build without it. The fix is **not upstream**.
The host (OpenMP) backend the registry defaults to is correct either way, which
is exactly why it is the default.

**2. Runs exit non-zero after succeeding.** A teardown heap abort (`rc=134`)
fires *after* the output is written, even on a bare `lmp -h`. Harmless to the
physics; fatal to PBS job dependencies. Check that the run completed rather than
blanket-ignoring the exit code.

**3. `atom_modify map yes` is required.** Without it `pair_symmetrix_mace` aborts
with a hard run-time error (`pair_symmetrix_mace.cpp:172`). The registry's
`deck_directives` emit it, together with `newton on`.

**4. A bundled `liblammps.so` can shadow your build.** The conda `mlip` env ships
one with deepmd-kit; put the build directory first on `LD_LIBRARY_PATH` or the
run aborts with `undefined symbol: lammps_plugin_finalize`.

The `mace_types` overflow (trap 1) also corrupts the heap on the host backend, so
`hybrid/overlay` + `dispersion/d3` can die within seconds in
`malloc_consolidate(): invalid chunk size`. symmetrix needs LAMMPS >= 10 Sep 2025,
which also makes `dispersion/d3` native (no backport, unlike DeePMD).

---

## 4. The example configs

All live in [`examples/mlip/`](../examples/mlip) and are validated on every test
run by `tests/integration/test_examples_mlip.py`
(parse → sweep-expand → pydantic → `validate_mlip_cell`, plus the builder
selection, the layer floor and the `pot_` path token).

| File | Command | Demonstrates | D3 toggleable? |
|---|---|---|---|
| [`sheet_mlip_full.ini`](../examples/mlip/sheet_mlip_full.ini) | `run sheetonsheet` | Full-cell DeePMD over a 3-layer stack; interlayer physics from the model, no LJ at all | **Yes** (needs the §7 binary) |
| [`pes_sheet_mlip.ini`](../examples/mlip/pes_sheet_mlip.ini) | `run pes-sheet` | Full-cell DeePMD static corrugation scan; exactly 2 layers, no sliding, no drive atom — the cheapest MLIP run in the package | **Yes** (needs the §7 binary) |
| [`afm_mlip_full.ini`](../examples/mlip/afm_mlip_full.ini) | `run afm` | v6 multi-component full cell: Si tip + crystalline matched Si substrate + MoS2 sheet on ONE DeePMD instance. Read [mlip_afm_gate.md](mlip_afm_gate.md) (Gate 4 open) before quoting a number | **Yes** |

**The v3 partition examples are deleted, not deprecated:** `sheet_mlip_lj.ini`
and `afm_mlip_lj.ini` described partitioned cells (matrix row 2) and would be
rejected by `validate_mlip_cell` before a builder touched them. The current
`afm_mlip_full.ini` is a new v6 file (one instance over the whole cell), not the
v3 one of the same name.

Layer-count floors are enforced inside `build()`: sheet-on-sheet needs **≥ 3**
layers (`SheetOnSheetSimulation.MIN_LAYERS`) and the static PES sheet scan needs
**exactly 2** (`PESSheetSimulation.MIN_LAYERS`).

Run one:

```bash
FrictionSim2D run sheetonsheet examples/mlip/sheet_mlip_full.ini -o ./out
# or, without installing the package (see CLAUDE.md):
python -m frictionsim2d.cli run sheetonsheet examples/mlip/sheet_mlip_full.ini -o ./out
```

The two sheet examples name `../models/dpa3_alex2d.pth`, the frozen DPA-3.1-3M/`Alex2D`
model of §3. That file is multi-GB and deliberately not in the repo.

---

## 5. What the generator emits

These are the exact lines, taken from the golden tests in
`tests/potentials/test_potential_manager.py::TestMLIPEmission`.

**Atom types are ELEMENT-first, not layer-first.** `_assign_types` assigns all
layers of the first element, then all layers of the next — matching the LAMMPS
renumbering loop in `stack_multilayer_sheet`. A 3-layer MoS2 sheet is therefore

```
type 1 = Mo layer 1   type 4 = S layer 1
type 2 = Mo layer 2   type 5 = S layer 2
type 3 = Mo layer 3   type 6 = S layer 3
```

(The hand-written Milestone-0 decks in `scripts/milestone0/` use a *different*,
per-layer-pair layout — 1 = Mo L1, 2 = S L1, 3 = Mo L2, … — because ASE writes
them directly; do not copy a type map from one context into the other.)

**Full-cell DeePMD, no drive atom.** A lone full-cell MLIP needs no `hybrid`
wrapper, so there is no sub-style selector in the `pair_coeff` line:

```
pair_style deepmd <model>
pair_coeff * * Mo Mo Mo S S S # sheet
```

**Full-cell DeePMD with a virtual drive atom** (type 7). The `hybrid` wrapper
appears because the drive atom needs its own `lj/cut`, but the deepmd map is
still `* *` with one element per type — E3 forbids any type-range form — and the
virtual type takes the framework's placeholder element:

```
pair_style hybrid deepmd <model> lj/cut 11.0
pair_coeff * * deepmd Mo Mo Mo S S S Mo # sheet
```

**Full-cell + D3 + a virtual drive atom** (shown for MACE, whose model path
lives in `pair_coeff` rather than `pair_style`):

```
pair_style hybrid/overlay symmetrix/mace dispersion/d3 bj pbe 16.0 8.0 lj/cut 11.0
pair_coeff * * mace <model> Mo Mo Mo S S S Mo # sheet
pair_coeff * * dispersion/d3 Mo Mo Mo S S S Mo
pair_coeff * 7 lj/cut 1e-100 1e-100
neigh_modify exclude type 7 1
neigh_modify exclude type 7 2
...
neigh_modify exclude type 7 7
```

**No `NULL` appears in any emitted line, ever.** The per-layer and NULL-padded
goldens that used to pin that text have been deleted, because they pinned output
no LAMMPS build accepts.

### BENCHMARK RESULT: D3 is mandatory, not optional

`scripts/mlip_bilayer_benchmark.py` was run against **DPA-3.1-3M / `Alex2D`** on
the backported binary (2026-08-11, 24-atom bilayer, rigid single points
5.5–9.0 Å). This is the go/no-go physics gate, and it produced an unambiguous
operational rule.

| | equilibrium separation | interlayer binding | in-plane *a* |
|---|---|---|---|
| Experiment / literature (2H-MoS₂) | 6.15 Å | 0.25–0.30 J/m² | 3.16 Å |
| SW + LJ (classical reference) | 5.947 Å | 0.362 J/m² | — |
| **DPA-3.1 base (no D3)** | **none — unbound** | **~0** (5.7e-7 J/m²) | 3.179 Å |
| **DPA-3.1 + D3** | **6.112 Å** (−0.6 %) | **0.265 J/m²** | 3.136 Å |

**The bare foundation model has no interlayer binding whatsoever.** Its binding
curve decreases monotonically to the edge of the scan — the reported 8.95 Å
"equilibrium" is the scan boundary, not a minimum. That is exactly what a
PBE-trained model without a dispersion term should do, and it means:

> **Running a full-cell MLIP sheet stack with `dispersion = none` is
> scientifically invalid for any interlayer property.** The layers are unbound;
> they will drift apart, and any interlayer friction, adhesion or corrugation
> number taken from such a run is meaningless. Use `dispersion = d3` for every
> multilayer run. `dispersion = none` is for single-layer cells and for
> deliberately isolating the intralayer contribution.

With D3 the model lands **within 0.6 % of the experimental interlayer spacing
and inside the literature binding range** — beating the classical SW+LJ
reference on both. Note the in-plane constant goes the other way (base 3.179 Å
is closer to experiment than D3's 3.136 Å): intralayer bonding is covalent and
needs no dispersion, so D3 slightly over-contracts it. Feed the **D3** number
(6.112 Å) into `lat_c` when running with D3.

Raw JSON: `/home/mv923/opt/bench/deepmd_{base,d3}.json`.

### The verified `dispersion/d3` argument order

`PairDispersionD3::settings()` takes exactly four arguments, in this order:

```
dispersion/d3 <damping> <functional> <rthr> <cn_thr>
```

`units metal` is required, both cutoffs are squared internally, and the accepted
damping tokens are `original | zero | zerom | bj | bjm` (`original` and `zero`
are synonyms). **This matches what `frictionsim2d/core/mlip.py::d3_style_token` emits** —
verified by reading `PairDispersionD3::settings` and by executing the exact
emitted string (`pair_style dispersion/d3 bj pbe 16.0 8.0`, 24-atom bilayer,
`E_pair = -8.7249526 eV`). No change is needed and no hedging is warranted for
the mainline/backported build.

The one remaining caveat is per-build, not per-argument. The MACE stack is
LAMMPS `patch_10Sep2025`, where `dispersion/d3` is native mainline; the
DeePMD+D3 stack is a *backport* into a 29 Aug 2024 tree, whose d3 term was
verified bit-identical to upstream 2025. Both accept the order above today.
`d3_damping` remains a canonical name the registry translates, and the emitted
line is golden-tested per build, so a future divergence surfaces as a failure. `d3_style_token` is the
**single** place the order is written; if a build ever needs a different
spelling, change it there and only there.

Stock `dispersion/d3` requires a **real element for every atom type** — it does
not accept `NULL`. That is a second, independent reason D3 is full-cell-only.

---

## 6. Which builder families are reachable

MLIP `pot_type` is accepted by every builder family, gated only by the
coverage topology: one MLIP instance over the whole cell, the same MLIP in every
component (§1).

| Family | Command | MLIP reachable? | Why |
|---|---|---|---|
| Sheet-on-sheet | `run sheetonsheet` | **Yes** | One component (the sheet stack). ≥ 3 layers. Pressure-controlled (GPa). |
| Interlayer PES scan | `run pes-sheet` | **Yes** | One component, exactly 2 layers, static single points, no sliding, no drive atom — the cheapest and least constrained MLIP case in the package. |
| Hetero sheet-on-sheet | `run sheetonsheet` with `[2D-1]`/`[2D-2]` | **Yes** (v6) | Two sheet components, ONE instance spanning both materials. The headline hetero case: no reliable cross-material SW parametrization, one model covers both. |
| AFM sliding | `run afm` | **Yes** (v6) | tip + sheet + substrate, all on ONE model. An SW tip beside an MLIP sheet is still refused (E2/E3). Needs `[sub] amorph = c, match = true`. Force-controlled (nN). See `examples/mlip/afm_mlip_full.ini`. |
| Tip PES scan | `run pes-tip` | **Yes** (v6) | Same AFM stack, inherited for free. |
| Indentation | `run indent` | **Yes** (v6) | Same component structure as AFM. |
| Freestanding / flake | `run afm` variants | **Yes** (v6) | The flake is simply one more component in the same instance's type map. |

The rejections are raised by `validate_mlip_cell` (`frictionsim2d/core/mlip.py`), called
from `frictionsim2d/builders/sheetonsheet/base.py`, `sheetonsheet/flake_slide.py`,
`afm/base.py`, `afm/hetero.py` and `construction/hetero_stack.py` — i.e. at
generation time, not at 3 a.m. on the cluster. Separately, config validation
refuses `[general] interlayer_style` other than `lj` and `interlayer_comparison`
whenever any component is an MLIP (`dispersion = d3` is the vdW lever).

Two mechanisms you will see in generated trees:

- **`pair_style zero` stacking surrogate.** Multilayer stacking and
  hetero/flake assembly run LAMMPS *at generation time*, and those runs
  `include` the cell's potential settings file. For an MLIP that would demand an
  MLIP-enabled python-LAMMPS on the generation machine and in CI. Instead the
  stacking step writes `pair_style zero 10.0` + `pair_coeff * *` + masses. It is
  stock LAMMPS, defined for every element, and exerts exactly zero force — which
  is why `lat_c` must be supplied explicitly (nothing can be minimized under it)
  and why the surrogate can never influence geometry.
- **Virtual-atom placeholder + neighbour exclusion.** The massless `Virtual`
  drive type has no chemical element, which `pair_mace` and `dispersion/d3` both
  reject and which a DeePMD full element list cannot leave blank. Its slot in
  the type map is filled with the framework's placeholder element
  (`MLIPSpec.virtual_atom_element`, `Mo` for both), and its neighbour list is
  then emptied so it contributes only a constant isolated-atom energy and zero
  force — measured, see §1. **The exclusion is one line per type**:
  `neigh_modify exclude type M N` takes two explicit numeric type IDs and has
  **no wildcard form**; `neigh_modify exclude type 7 *` is a parse error that
  would kill the run.

---

## 7. Builds: getting `deepmd` and `dispersion/d3` into one binary

This section is the part that cost the most time to establish. Every statement
here was verified on this workstation.

### The blocker

- **`dispersion/d3` entered LAMMPS in `stable_22Jul2025`.** The v3 pin
  `stable_29Aug2024_update3` contains zero d3 sources.
- **deepmd-kit is ABI-locked to LAMMPS 2024.08.29.** Confirmed twice
  independently: the pip plugin loaded into a 2025 build fails with `undefined
  symbol: _ZN9LAMMPS_NS5utils11getsyserrorEv`, and conda-forge `deepmd-kit
  3.1.3` carries the hard constraint `lammps 2024.08.29`, so the solver refuses
  `lammps 2025.07.22` outright.

No single upstream release has both styles.

### The fix: backport d3, don't fight the ABI

`pair_dispersion_d3` is a self-contained pair style depending only on
long-stable internals. So
[`scripts/hpc/build_lammps_deepmd_d3.sh`](../scripts/hpc/build_lammps_deepmd_d3.sh)
builds the ABI-matching **2024** release with `pair_dispersion_d3.{cpp,h}` and
`d3_parameters.h` copied in from `stable_22Jul2025_update1`:

```bash
cd scripts/hpc
PREFIX=$HOME/opt TARGET=workstation ./build_lammps_deepmd_d3.sh
# binary: $PREFIX/lammps-deepmd-d3/bin/lmp
```

One binary, `pair_style deepmd` (via the plugin) **and** `dispersion/d3`. It
works, and the E8 bit-identity check of §1 footnote 2 is what proves the
backport is numerically the 2025 code, not merely a successful compile.

Two things the spec did not anticipate, both now handled by the script:

1. **The 2025 d3 source does not compile clean against 2024 headers**, but the
   breakage is cosmetic: exactly **11 sites**, all inside `error->all(...)`
   abort paths — 9× `Error::NOLASTLINE`, 1× `Error::NOPOINTER`, 1×
   `Info::get_pair_coeff_status`. These are post-2024 error-printer decorations.
   No executed code path, energy or force depends on them; the D3 maths and the
   C6/parameter tables are byte-for-byte the 2025 originals. The script applies
   a guarded `sed` shim and **hard-fails if the shim misses a site**, so a
   silent half-patch cannot reach the compiler.
2. **The build must be `BUILD_MPI=ON`, not OFF** — the reverse of the advice in
   `build_lammps_deepmd.sh`. The deepmd plugin leaves ~20 `MPI_*` symbols
   undefined and expects the **C** ABI; LAMMPS's own `BUILD_MPI=OFF` stubs
   (`src/STUBS/mpi.cpp` in the *LAMMPS* tree) are **C++**, so the symbols come out mangled
   (`_Z13MPI_AllgatherPviiS_iii`) and do not satisfy it. Build against the same
   MPI the plugin was built against (here, the `mlip` conda env's MPICH) with
   `MPICH_CXX=/usr/bin/g++` so the wrapper still uses the system toolchain —
   which also sidesteps the broken system MPICH without needing
   `CMAKE_DISABLE_FIND_PACKAGE_MPI`.

### The plugin path differs by install method

There is **no `deepmd.lmp` module** in deepmd-kit v3 — the widely-copied
`import deepmd.lmp` incantation raises `ModuleNotFoundError`, and if you
interpolate it into an `export` the variable quietly becomes a traceback string.
The real layouts:

| install | plugin directory | file |
|---|---|---|
| conda | `$CONDA_PREFIX/lib/deepmd_lmp` | `dpplugin.so` |
| pip | `<site-packages>/deepmd/lib` | `libdeepmd_lmpplugin.so` |

`LauncherSpec.env_lines` for DeePMD (`frictionsim2d/core/mlip.py`) emits a line that
prefers the conda layout and falls back to the pip one, so a job script works
under either.

### All plugin failure modes look identical

A wrong plugin path, a missing `LD_LIBRARY_PATH`, and a `BUILD_MPI=OFF` host all
present as the *same* misleading trio:

```
Open of .../dpplugin.so failed: undefined symbol: MPI_Allgather
Loaded 0 plugins from .../deepmd_lmp
ERROR: Unrecognized pair style 'deepmd'
```

`Loaded 1 plugins` in the log is the **only** proof the plugin is live — `lmp -h`
cannot show a runtime plugin at all. The build script ends with a `plugin list`
probe deck that asserts on it.

### Running the backported binary

```bash
export LMP_DEEPMD_D3=$HOME/opt/lammps-deepmd-d3/bin/lmp
export LAMMPS_PLUGIN_PATH=/home/mv923/miniconda3/envs/mlip/lib/deepmd_lmp
export LD_LIBRARY_PATH=/home/mv923/miniconda3/envs/mlip/lib
env -u PYTHONPATH "$LMP_DEEPMD_D3" -in deck.in
```

> **`env -u PYTHONPATH` is not optional** for anything launched from this repo
> against the `mlip` conda env: the user's shell profile points `PYTHONPATH` at
> a *different* FrictionSim2D checkout, which shadows the editable install. The
> symptom is imports resolving to the wrong tree with no error.

---

## 8. HPC settings and GPU sizing

### Settings

MACE (symmetrix + Kokkos) and DeePMD (LAMMPS 2024 + deepmd-kit plugin) are two
*mutually incompatible* LAMMPS builds, so `settings.yaml` carries a
per-framework overlay directly under `hpc:`. There are no named profiles: to
target another cluster, keep one `settings.yaml` per cluster and select it with
`--settings-file` (on `hpc generate`, and on `run ... --hpc-scripts`).

```yaml
hpc:
  scheduler_type: pbs
  queue: gpu72
  num_cpus: 8
  memory_gb: 64
  modules: [tools/prod]
  env_lines: []                  # extra shell lines, rendered after the env setup
  lmp_binary: null               # LAMMPS binary for classical jobs (default: lmp)
  gpu: {ngpus: 1, gpu_type: L40S}   # GPU request for MLIP job groups
  frameworks:
    deepmd:
      binary: "$HOME/opt/lammps-deepmd-d3/bin/lmp"
      # conda_env: /path/to/env  # optional; replaces hpc.conda_env for this group
    mace:
      binary: "$HOME/opt/lammps-mace/build/lmp"
```

```bash
FrictionSim2D hpc generate ./out --settings-file ~/clusters/cx3.yaml
```

Point the `deepmd` overlay at the **backported** binary if any run in the
campaign sets `dispersion = d3`; no other build has both styles.

`frictionsim2d/core/config.py::apply_framework_overlay(hpc, framework)` builds each job
group's settings: the overlay's `binary` replaces `hpc.lmp_binary`, its
`conda_env` replaces `hpc.conda_env`, and its `modules` / `env_lines` are
**appended** to the base lists. A classical (CPU) group gets the base `hpc:`
block unchanged, so a campaign with no MLIP run renders byte-identically to the
pre-MLIP output.

### What gets rendered

For an MLIP job group (`frictionsim2d/hpc/scripts.py::HPCConfig`):

- **One MPI rank.** PBS:
  `select=<nodes>:ncpus=<n>:mem=<m>gb:mpiprocs=1:ngpus=<g>[:gpu_type=<t>]`;
  SLURM: `--ntasks-per-node=1`, `--cpus-per-task=<n>` and
  `#SBATCH --gres=gpu[:<type>]:<g>`. The rank count is the framework's
  `LauncherSpec.mpi_ranks` (**1** for both — MACE because symmetrix is
  OpenMP-threaded, DeePMD because one process owns the GPU and extra ranks are
  monotonically slower at ~13 GB VRAM each), *not* the CPU path's
  `mpiprocs=ncpus`. The GPU request comes from `hpc.gpu`.
- **Threads.** The environment block exports
  `OMP_NUM_THREADS=<n>` (the cores the single rank owns) for MACE, whose
  launcher exports `OMP_NUM_THREADS` / `OMP_PROC_BIND=spread` /
  `OMP_PLACES=threads`; DeePMD and classical jobs keep `OMP_NUM_THREADS=1`.
  `hpc.env_lines`, overlay `env_lines` and the framework's `LauncherSpec`
  lines (e.g. DeePMD's `LAMMPS_PLUGIN_PATH` export) render *after* that export,
  so they can override it. With a conda env and a framework set,
  `CONDA_PREFIX` is exported too (deepmd-kit's plugin lookup keys off it).
- **Launch line.** `{{ launch_command }} -in <deck> -log <log>`, where
  `launch_command` is `<mpi_command> <binary> <launcher flags>` — for MACE
  `-k on t ${OMP_NUM_THREADS:-8} -sf kk -pk kokkos neigh half` (Kokkos host
  backend; see §10 for why not CUDA by default). Preflight scripts use the same
  per-group binary and env lines.
- **Grouping.** Jobs are grouped by `(needs_gpu, framework)`, one script per
  group for array, combined *and* two-phase submission: e.g. `run_cpu.pbs`,
  `run_gpu_deepmd.pbs`, `run_gpu_mace.pbs` (two-phase: `run_system_cpu.sh`, ...),
  each with its own `manifest_combined_<suffix>.txt`. A campaign with a single
  group keeps the plain `run.pbs` / `manifest_combined.txt` names and bytes.

> **Open question — OpenMPI thread binding.** `mpirun` under OpenMPI binds a
> single rank to one core by default, which would pin all of MACE's OpenMP
> threads onto that core. `--bind-to none` (in `hpc.mpi_command`, or via a
> settings `env_lines` workaround) may be needed for MACE jobs. This is
> **unverified on the cluster**; check thread utilisation on the first MACE job.

**How a job knows it needs a GPU.** `hpc generate` builds its manifest by
*discovering* a built tree (`JobManifest.from_simulation_directory`), never by
re-reading the configs — so the only surviving record of which framework a run
used is the `pot_<framework>[-d3]_<stem>` segment from §2.
`parse_mlip_flavor` decodes it into `(framework, dispersion)` and populates
`needs_gpu` and `framework`. Every path segment is checked, not just the leaf. A
classical path yields no flavour and keeps the `False` / `None`
defaults, so a wholly classical campaign still renders as a single CPU script
byte-identically.

### GPU sizing (Imperial RCS)

**CX3 is the default GPU target.**

| Card | VRAM | Where |
|---|---|---|
| L40S | 48 GB | CX3 `gpu72` default — start here |
| RTX6000 | 24 GB | CX3, smaller cells only |
| A100 | 40 GB | CX3, limited availability |
| A100 | 80 GB | HX1 — **escape hatch only** |

- **12 concurrent GPUs per user** on `gpu72`. An array campaign larger than that
  serialises; size arrays accordingly.
- RCS policy says single-node/small jobs belong on CX3. Use an HX1
  `settings.yaml` (e.g. `queue: a100`, `gpu: {ngpus: 1, gpu_type: A100}`,
  selected with `--settings-file`) **only** when a cell genuinely exceeds 48 GB
  of L40S VRAM.
- Design point is **1 GPU per job** at 10³–10⁴ atoms; multi-GPU and multi-node
  runs are out of scope.
- MLIP inference memory is dominated by graph activations scaling with
  atoms × neighbours, not by weights. Do not guess: the benchmark script (§9)
  records measured VRAM against atom count per framework.

The Milestone-0 measurements on this page were taken on a much smaller card —
an RTX A4000 16 GB (CUDA 12.5, driver 580) — on 24- and 36-atom cells, so they
say nothing about production VRAM.

---

## 9. Local GPU workstation quickstart

Four steps, in order. Nothing later is trustworthy until the earlier steps pass.

**1. Build the LAMMPS stack you need.** See
[`scripts/hpc/README.md`](../scripts/hpc/README.md) for the full argument
reference (`TARGET`, `GPU_ARCH` per card, what each build enables).

```bash
cd scripts/hpc
PREFIX=$HOME/opt TARGET=workstation ./build_lammps_deepmd_d3.sh   # deepmd AND d3
PREFIX=$HOME/opt TARGET=workstation ./build_lammps_deepmd.sh      # standalone d3 only
PREFIX=$HOME/opt TARGET=workstation GPU_ARCH=AMPERE86 ./build_lammps_mace.sh

export PREFIX=$HOME/opt
export LMP_DEEPMD_D3="$PREFIX/lammps-deepmd-d3/bin/lmp"
export LMP_DEEPMD="$PREFIX/lammps-deepmd/bin/lmp"
export LMP_MACE="$PREFIX/lammps-mace/build/lmp"
```

`build_lammps_deepmd_d3.sh` is the one to use for anything combining the two
styles; `build_lammps_deepmd.sh` pins a 2025 tag at which the deepmd plugin will
**not** load, so it is useful for standalone `dispersion/d3` only. All scripts
are idempotent; a re-run after a failure resumes.

**2. Run Milestone 0.** [`documentation/milestone0.md`](milestone0.md), top to
bottom — geometries first, then the items. The DeePMD and D3 items are already
executed and recorded there; the MACE items are not.

```bash
cd scripts/milestone0
python make_geometries.py --outdir .
```

**3. Run the physics gate.** The bilayer binding benchmark is the explicit
go/no-go before any production run: a foundation model that gets MoS2 interlayer
binding wrong will get friction wrong.

```bash
python scripts/mlip_bilayer_benchmark.py --framework sw \
    --pot examples/potentials/sw/MoS2_wen.sw --lmp lmp --out sw_reference.json

python scripts/mlip_bilayer_benchmark.py --framework deepmd \
    --model $HOME/models/dpa3_alex2d.pth --lmp "$LMP_DEEPMD_D3" \
    --out deepmd_base.json

python scripts/mlip_bilayer_benchmark.py --framework deepmd \
    --model $HOME/models/dpa3_alex2d.pth --lmp "$LMP_DEEPMD_D3" \
    --d3 --out deepmd_d3.json
```

It scans the interlayer separation 5.5–9.0 Å in rigid single points and reports
`equilibrium_separation_A` (→ your `lat_c`), `lat_c_in_plane_A` (the in-plane
constant — *not* `lat_c`), the binding energy per area, and measured VRAM vs
atom count. `--dry-run` prints the decks without invoking LAMMPS.

**4. First full-cell run.** Generate, then execute locally.

```bash
python -m frictionsim2d.cli run sheetonsheet examples/mlip/sheet_mlip_full.ini -o ./out
./scripts/run_local.sh out/<sim_root>/sheetonsheet/h-MoS2/30x_30y/K0/pot_deepmd-d3_dpa3_alex2d deepmd
```

[`scripts/run_local.sh`](../scripts/run_local.sh) resolves the binary from
`settings.yaml`'s `hpc.frameworks.<framework>.binary` (then `hpc.lmp_binary`,
`$FRICTIONSIM_SETTINGS` selects the file; falling back to `$LMP_DEEPMD` /
`$LMP_MACE`, then `lmp` on `PATH`), applies the registry's launcher flags and
env exports, `cd`s to the directory the generated relative paths are written
against, and runs `system.in` then `slide.in`. `RUN_LOCAL_DRY_RUN=1` prints the
commands without running them. `frictionsim2d/run/simulate.py` itself only *builds* inputs;
promoting local execution into it is future work.

---

## 10. Troubleshooting

**`pair_style deepmd` is an unknown style / `Loaded 0 plugins`.**
DeePMD is a *runtime plugin*, not compiled in, and three different problems
produce this one symptom (§7): wrong `LAMMPS_PLUGIN_PATH`, missing
`LD_LIBRARY_PATH` for the plugin's own libabsl/TensorFlow runtime, or a host
LAMMPS built with `BUILD_MPI=OFF`. Check for `Loaded 1 plugins` in the log —
nothing else proves the plugin is live.

```bash
export LAMMPS_PLUGIN_PATH=$CONDA_PREFIX/lib/deepmd_lmp        # conda layout
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
```

`run_local.sh` checks that `LAMMPS_PLUGIN_PATH` is non-empty and refuses to
start otherwise, because `export X=$(cmd)` returns the status of `export`, not
of the substitution — a failed lookup would otherwise be silent.

**`pair_style dispersion/d3` is an unknown style.**
Your binary predates `stable_22Jul2025`, or it is the plain deepmd build. Use
the backported binary from `build_lammps_deepmd_d3.sh` (§7). There is no
upstream release with both styles.

**Silently wrong DeePMD energies — the missing element list.**
`pair_coeff * * deepmd` with **no** element list does **not** error. LAMMPS types
are mapped to the model's first N elements (H, He, Li, …), which on the
reference 36-atom trilayer gives `-61.89 eV` instead of `-256.90 eV`. Always
pass the full explicit element list, one entry per LAMMPS type. The generator
always does.

**Silently wrong DeePMD forces — the model-deviation trap.**
`pair_style deepmd model1.pth model2.pth` is **not** two instances. Two model
paths switch DeePMD into model-deviation (uncertainty) mode, in which only the
**first** model drives the forces. Not an error, just wrong physics.

**`ERROR: Pair hybrid sub-style deepmd is not used`.**
You have a `NULL` in a deepmd type map. There is no fix at the deck level:
partial-cell DeePMD is unavailable (§1). Rewriting it as
`pair_coeff 1*2 1*2 deepmd Mo S` does not help either — that fails at
`pair_deepmd.cpp:816`. Make the whole cell one MLIP, or the whole cell
classical.

**`dp freeze` fails with "Head must be specified in multitask mode".**
DPA models are multitask. Pass `--head Alex2D` (§3).

**MACE segfaults instead of erroring.**
`pair_style mace` rejects `NULL` entries in its type map by crashing (ACEsuit
#1314). Config validation refuses every topology that would produce one, so a
segfault here means a hand-edited deck. Mapping tip/substrate Si to real Si in
MACE-MP is *not* a workaround: it double-counts against SW/LJ.

**`neigh_modify exclude type 7 *` is rejected.**
There is no wildcard form. `neigh_modify exclude type M N` takes two explicit
numeric type IDs; the generator emits one line per type (§6, and
`scripts/milestone0/m0_4_virtual_exclude.in`).

**Imports resolve to the wrong FrictionSim2D checkout.**
This machine's shell profile sets `PYTHONPATH` to a *different* checkout, which
shadows the editable install in the `mlip` conda env. Prefix anything run from
that env with `env -u PYTHONPATH` (§7).

**"`mlip_interlayer` is not accepted."**
Correct — it was removed (§2). Delete the line. If it said `native`, nothing
else changes; if it said `lj`, the run you were asking for is a partitioned cell
and is unavailable (§1).

**"Cannot mix '<framework>' with a classical potential in one cell."**
A component in your cell uses SW (or another classical potential) while another
uses an MLIP. One `pair_coeff * *` claims every atom type, so the classical
component's types would be silently absorbed. Make the whole cell one MLIP, or
the whole cell classical. Multi-component MLIP cells themselves are fine since
v6 — it is the *mixing* that is refused.

**"amorph = 'a' is not available under an MLIP."**
Amorphisation is melt-quench MD run at generation time and cannot evaluate the
model. For an AFM substrate use `amorph = c` with `match = true`, which
`AFMSimulationConfig` accepts in place of an amorphous one.

**"The <framework> model does not cover every element in this cell."**
Re-extract the model with every element the cell needs, e.g.
`symmetrix_extract_mace -m <model> --chemical-symbols Si Mo S`. This is checked
for MACE only; a DeePMD type map cannot be read without torch, so verify a
DeePMD cell by running the reference case.

**The sheet collapses, or the layers interpenetrate at step 0.**
This is the `lat_c` / initial-gap mismatch. Foundation models are attractive at
short range, and a starting interlayer gap smaller than the model's equilibrium
lets the layers fall together into an unphysical geometry that no amount of
sliding statistics will rescue. Workflow:

1. Run `scripts/mlip_bilayer_benchmark.py` for **your** model and flavour (base
   and D3 differ — D3 pulls the layers in).
2. Take `equilibrium_separation_A` from the JSON and put it in `[2D] lat_c`.
3. Do **not** put `lat_c_in_plane_A` there — that is the in-plane lattice
   constant (§2).
4. Re-check after any change of model, D3 setting or material.

**Absolute energies differ between a `virtual_atom` run and an `smd` run.**
Expected. Under DeePMD the virtual atom contributes a constant isolated-atom
energy (−4.64264 eV for the `Mo` placeholder in the `Alex2D` head); under D3 it
contributes exactly zero. Subtract the constant before comparing absolute
`E_pair`. Forces and every friction observable are unaffected (§1).

**A MACE number.**
Measured 2026-08-24 / 08-31 through `pair_style symmetrix/mace` on
MACE-MPA-0-medium. Reference energies, which every MACE run must reproduce
before its results are trusted:

| cell | correct | unpatched CUDA |
|---|---|---|
| h-MoS2 bilayer, 24 atoms | **−173.40779 eV** | −126.5489 eV |
| h-MoS2 trilayer, 810 atoms | **−5850.0183 eV** | −4785.9659 eV |

Interlayer binding: **6.717 Å / 0.068 J/m²** unaided (bound, but under-binding
the 0.25–0.30 literature value); with D3, 0.474 J/m² — an *over*-bind of
1.6–1.9×. Both arms are therefore meaningful for MACE, unlike DeePMD where the
no-D3 arm is unbound and meaningless.

Quote a MACE number **only** from a binary carrying
`symmetrix-mace_types-overflow.patch`, and only after the reference energy
above checks out. The unpatched CUDA figures are wrong by 27 % and are produced
with no error message.

---

## 11. Known future work

Documented, not implemented:

- **Patching `pair_deepmd` to honour `NULL` type maps.** This is the
  highest-value item — it would restore the struck partial-cell setups (an MLIP
  sheet beside a classical tip/substrate) — but it is real surgery: the model receives all atoms,
  so masking types means rebuilding the coord/type arrays and mapping forces
  back. Documented, not attempted (spec §7).
- **Building the MACE stack on this machine**, and with it MACE full-cell and
  MACE + D3 (spec §7).
- MACE partial-cell via the ML-IAP unified interface (`pair_style mliap
  unified`, undocumented hybrid/NULL behaviour) or a maintained fork of
  `pair_mace`. `build_lammps_mace.sh` already enables ML-IAP + ML-SNAP + PYTHON
  with `MLIAP_ENABLE_PYTHON=ON`, so the cheap experiment needs no rebuild.
- **Not** a goal: re-fitting LJ as a residual correction on top of an MLIP
  (§1 — D3 strictly dominates it).
- Promoting SW/ReaxFF into the same registry pattern; promoting
  `scripts/run_local.sh` into `frictionsim2d/run/`; GUI exposure of the MLIP config surface;
  multi-GPU scaling; a foundation model fine-tuned on MoS2 sliding
  configurations.
