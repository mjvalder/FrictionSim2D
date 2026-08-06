# MLIP Support for FrictionSim2D — Design Specification (v3)

**Date:** 2026-08-06
**Branch:** `mlip` (worktree off `gui-p2`)
**Status:** v3 — v2 (post-adversarial-review) revised per user feedback;
pending final user approval
**Review provenance:** v1 → adversarial review (25 findings, 5 blockers) +
moderation (all upheld, 4 moderator findings) → v2 → user feedback → v3.
User-directed changes in v3: DeePMD is the sole hybrid workhorse; MACE is
scoped to full-cell simulations; MLIP support across **all** builder families
is a core deliverable (screening tool); the virtual-atom drive is retained via
a compatibility mechanism rather than banned; the stacking surrogate is
`pair_style zero` (universal) instead of SW.

## Goal

Add machine-learned interatomic potential (MLIP) support to FrictionSim2D so
that MoS2 (and, for screening, arbitrary-material) simulations can be
generated, run on Imperial RCS clusters with correct GPU/CPU resource
requests, and reproduced on a local GPU workstation.

**Core friction setups (DeePMD unless noted):**

1. **Sheet-on-sheet, MLIP intralayer + LJ interlayer** — one DeePMD hybrid
   sub-style instance per layer (≥3 layers; builder enforces `MIN_LAYERS=3`),
   interlayer pairs on `lj/cut`.
2. **Sheet-on-sheet, full MLIP** — one MLIP describes the whole cell,
   intralayer and interlayer. **Both DeePMD and MACE** (this is the
   MACE-vs-DeePMD comparison point).
3. **AFM tip/sheet/substrate, MLIP sheet intralayer** — per-layer DeePMD
   instances for the sheet; SW tip and substrate; LJ between sheet layers and
   between sheet and tip/substrate.
4. **AFM tip/sheet/substrate, MLIP sheet intra+interlayer** — one DeePMD
   instance covering all sheet types (NULL-partitioned in hybrid); SW tip and
   substrate; LJ only between sheet and tip/substrate.

**Core screening deliverable:** MLIP `pot_type` works across **all builder
families** — sheetonsheet, afm (incl. freestanding/flake), pes-sheet,
pes-tip, indent, hetero — subject to the same capability rules (§ matrix).
Static PES scans and full-cell runs are where foundation models shine as a
screening tool, and hetero cells (two materials, one foundation model) are a
natural full-cell-MLIP fit.

Frameworks: **MACE** and **DeePMD**, foundation models (MACE-MP family;
DeePMD DPA family). Training/fine-tuning is out of scope. Dispersion: runs
without and with D3 where the matrix allows (base values are the comparison
baseline).

GUI integration is out of scope (backend only); §1 is the interface the GUI
will later expose.

## Capability matrix (binding)

Capability is determined by **potential-coverage topology**, not builder
family: any run is either *full-cell MLIP* (every real atom type covered by
one MLIP instance) or *partial-cell MLIP* (MLIP shares the cell with SW/LJ
via hybrid). The same rules then apply to every builder family:

| Coverage | DeePMD base | DeePMD + D3 | MACE base | MACE + D3 |
|---|---|---|---|---|
| Full-cell MLIP (setup 2; full-cell PES/hetero/indent screening) | Yes | Conditional¹ | Yes | Conditional¹˒² |
| Partial-cell, single MLIP instance (setup 4 and analogues) | Conditional¹ | No³ | No⁴ | No |
| Partial-cell, per-layer MLIP instances (setups 1/3 and analogues) | Conditional¹ | No³ | No⁴ | No |

¹ Mechanism documented/community-confirmed but unproven in our stack — gated
  on **Milestone 0** (§7). Nothing is "verified" until it passes.
² Full-cell MACE+D3 via `hybrid/overlay` is a community-confirmed pattern
  (matsci #62930), still Milestone-0-gated.
³ Stock `pair_style dispersion/d3` requires a real element for every atom
  type (no NULL): it cannot be restricted to MLIP pairs, so partial-cell D3
  would double-count dispersion over SW/LJ pairs. A NULL-aware fork is
  future work (§10).
⁴ `pair_style mace` rejects NULL type maps (segfault; maintainer states the
  architecture cannot selectively ignore atom types — ACEsuit #1314).
  **MACE is scoped to full-cell runs.** Mapping tip/sub Si to real Si in
  MACE-MP is not a workaround (double counting with SW/LJ). Possible future
  routes to MACE-in-hybrid, in order of plausibility (§7 stretch, §10):
  the **ML-IAP unified interface** (`pair_style mliap unified`, Kokkos-GPU,
  cuEquivariance) whose hybrid/NULL behavior is undocumented → cheap
  Milestone-0 experiment; or carrying a **fork of `pair_mace`** that masks
  NULL-mapped types out of the graph — physically well-defined for
  sheet-only coverage, but upstream has declined it, so it would be ours to
  maintain. Neither blocks the DeePMD path.

**Virtual-atom drive compatibility** (all cells of the matrix): see §1.4 —
the drive atom is made invisible to MLIPs/D3 via placeholder-element mapping
plus `neigh_modify exclude`, Milestone-0-verified, with `smd`/`fix_move` as
fallback if verification fails.

## Non-goals

- Training or fine-tuning MLIPs; GUI changes; multi-GPU/multi-node runs
  (design point: 1 GPU per job at 10³–10⁴ atoms).
- Patching LAMMPS pair styles in this iteration (NULL-aware d3 fork and
  pair_mace fork are documented future work).
- Scientific validation of foundation models beyond the bilayer benchmark
  gate (§5).

## 1. Configuration surface

### 1.1 Per-run `.ini` keys

```ini
[2D]
pot_type = mace            ; or deepmd (alongside existing sw, ...)
pot_path = ../models/mace-mp-0b3-medium.model-lammps.pt
mlip_interlayer = lj       ; 'lj' (partial-cell) | 'native' (full sheet stack)
dispersion = none          ; 'none' | 'd3'
lat_c = 3.16               ; MANDATORY for MLIP pot_types (see §3.3)
```

Validation rules (config layer, hard errors — derived from the matrix, not
from builder family):
- `mlip_interlayer` / `dispersion` require an MLIP `pot_type`.
- MLIP `pot_type` on tip or substrate is rejected **in hybrid setups** (SW by
  design there). Full-cell-MLIP variants of AFM-family runs (all components
  one MLIP) are permitted for MACE/DeePMD screening (§3.5).
- `dispersion = d3` requires full-cell MLIP coverage.
- `pot_type = mace` requires full-cell MLIP coverage
  (`mlip_interlayer = native` and no SW components in the cell).
- MLIP `pot_type` requires explicit `lat_c` and a model file whose extension
  matches the framework (`.model-lammps.pt`/`.pt` MACE; `.pb`/`.pth` DeePMD).
- Supported `constraint_mode` combinations per §3.4; unsupported ones
  rejected.

### 1.2 `settings.yaml` additions

```yaml
potential:
  d3_functional: pbe
  d3_damping: bj        # canonical name; enum-validated
  d3_cutoff: 16.0       # Å
  d3_cn_cutoff: 8.0     # Å
```

`d3_damping` is a **canonical** name the registry translates into the token
each LAMMPS build accepts: the ACEsuit fork ships an older `dispersion/d3`
(pre-rename tokens) than pinned mainline (`original|bj|zerom|bjm`). Emitted
syntax is pinned and golden-tested **per framework build**.

### 1.3 Output directory naming (sweep-loopability rule)

MLIP runs append a flavor token to the **innermost run-parameter segment**
of the output path (the segment already encoding temperature/spring
variants, so upstream segments used by postprocessing filters — material,
size — are untouched):

```
<...>/K<temp>_<pot_type>-<interlayer>-<disp>
disp token: 'base' when dispersion=none, 'd3' when dispersion=d3
e.g. .../K0_deepmd-lj-base, .../K0_mace-native-d3
```

Non-MLIP paths are byte-identical to today (regression-tested); a
postprocessing filter test proves existing segment parsing still works.

### 1.4 Virtual-atom drive: compatibility mechanism (not a ban)

`drive_method = virtual_atom` (the project's preferred drive) registers a
massless `Virtual` type with no chemical element — as-is this breaks
`pair_style mace` (cannot NULL-map it) and `dispersion/d3` (needs an element
per type). v2 banned virtual_atom for those runs; v3 instead makes it
compatible:

- **Placeholder-element mapping**: in full-cell MLIP and D3 coeff lines, the
  virtual type is mapped to a real element present in the model (e.g. the
  sheet metal). An atom with an **empty neighbor list** contributes only its
  constant isolated-atom energy — zero forces, no dynamics impact.
- **Neighbor-list exclusion**: `neigh_modify exclude type <virtual> *`
  removes every pair involving the virtual atom from all neighbor lists, so
  MLIP graphs and D3 sums never see it (its existing `lj/cut 1e-100`
  interaction was already physically null; the tether is a fix, not a pair
  interaction, and is unaffected).
- **Milestone 0 item**: verify pair_mace/deepmd/d3 tolerate an
  isolated (excluded) atom — graph models generally handle isolated nodes,
  but this is exactly the kind of claim v1 got burned on. If verification
  fails, validation falls back to requiring `smd`/`fix_move` for the
  affected combinations.
- The non-virtual drive paths (`smd`, `fix_move`) get golden + smoke tests
  regardless (they are under-exercised; `hetero.py` registers the Virtual
  type unconditionally — fixed as part of §3.5).

## 2. `src/core/mlip.py` — MLIP registry

```python
@dataclass(frozen=True)
class MLIPSpec:
    name: str                   # 'mace' | 'deepmd'
    model_in_style_line: bool   # deepmd: True; mace: False (model in pair_coeff)
    style_args: Callable        # (model_path, opts) -> pair_style argument string
    coeff_line: Callable        # (index_str, model_path, null_map) -> pair_coeff line
    supports_null: bool         # deepmd: True; mace: False → gates partial-cell
    supports_overlay: bool      # per-style flag; overlay layout derived from these
    launcher: LauncherSpec
    file_extensions: tuple      # validation (.pt/.model-lammps.pt vs .pb/.pth)
    d3_tokens: Callable         # canonical damping/functional -> build-specific tokens
    virtual_atom_element: str   # placeholder element for §1.4 mapping
    notes: str

@dataclass(frozen=True)
class LauncherSpec:
    binary: str                 # per-framework lmp binary — MACE (ACEsuit fork)
                                # and DeePMD (plugin build) are two incompatible
                                # LAMMPS builds; jobs must resolve the right one
    modules: list[str]          # per-framework module set (merged onto profile)
    env_lines: list[str]
    mpi_ranks: int              # 1 for MACE (no_domain_decomposition)
    lmp_flags: str              # only doc-verified flags pinned (MACE: '-k on g 1
                                # -sf kk'); more only after Milestone 0 verifies
```

- `MLIP_REGISTRY: dict[str, MLIPSpec]`, `is_mlip(pot_type)` helper.
- `d3_overlay(settings, spec) -> list[str]`: full-cell only; emits the
  build-correct `dispersion/d3` style token + a pair_coeff line listing a
  real element for **every** atom type (virtual type via
  `virtual_atom_element`, §1.4).

### `PotentialManager` / call-site integration (enumerated, not "surgical")

The interlayer decision must reach **five** string-only call sites; the
`is_sheet_lj` signature changes to accept component context (pot_type +
interlayer mode) and every site is updated:

1. `potential_manager.py` `register_component` (`potential_usage` accounting
   — per-layer MLIP counts as n_layers instances, native as 1),
2. `potential_manager.py` `add_self_interaction` routing,
3. `components.py` `stack_multilayer_sheet` (stacking pot file contents),
4. `sheetonsheet.py` `has_internal_lj` template flag (2 sites),
5. `afm.py` `has_internal_lj` template flag.

Other integration points:
- `write_file`: per-style argument assembly generalizes the ReaxFF
  special-case via the registry; `hybrid/overlay` chosen from
  `supports_overlay` flags when a D3 overlay is present; emits the §1.4
  `neigh_modify exclude` line when a virtual type coexists with MLIP/D3.
- **`count_atomtypes` fix (was a guaranteed crash):** binary model files
  crash the UTF-8 parse in `utils.py`. MLIP pot_types and extensions
  (`.pt`, `.pth`, `.pb`, `.model-lammps.pt`) join the single-type
  short-circuit — at **both** call sites, including the one that passes no
  `pot_type` (extension check is load-bearing there). Unit-tested.
- **Model staging:** MLIP models are staged **once at the simulation root**
  (shared `potentials/` with a relative prefix) instead of copied into every
  run directory (a 100-run sweep × a foundation model = gigabytes over
  rsync). Classical potentials keep today's per-run behavior.
- Per-layer instances use the numeric-suffix convention (`deepmd 1`,
  `deepmd 2`, …) — ≥3 instances for sheet-on-sheet. **Registry-documented
  trap:** the multi-model form `pair_style deepmd m1 m2` is model-deviation
  mode (only the first model drives forces); the generator must never emit
  it.

## 3. Builders

### 3.1 Routing

`sheetonsheet.py` / `afm.py`: parse the new keys; `native` → single
self-interaction across all sheet layers, no interlayer LJ; `lj` → existing
per-layer path (≥3 instances for sheet-on-sheet). Cross-component LJ
(sheet↔tip, sheet↔sub) unchanged in hybrid setups.

### 3.2 VRAM sizing (measured, not asserted)

MLIP inference memory is dominated by graph activations scaling with
atoms × neighbors (MACE-converted models default to float64), not weights.
The bilayer benchmark (§5) records measured VRAM vs atom count per framework
and instance count; job sizing guidance uses those measurements.

### 3.3 Stacking without an MLIP-enabled generation machine

`stack_multilayer_sheet` executes LAMMPS **in-process** (python `lammps`
module) for every multilayer build — even when `lat_c` is supplied — and its
rendered script `include`s a potential file derived from the sheet's
`pot_type`. Without intervention, merely generating MLIP inputs would
require an MLIP-enabled python-LAMMPS on the generation machine and in CI.

Design: for MLIP sheets the stacking step writes a **`pair_style zero`
surrogate** settings file (`pair_style zero <cutoff>` + `pair_coeff * *` +
masses). `pair_style zero` is stock LAMMPS, needs no potential file, and is
defined for every element — unlike the v2 SW surrogate, which assumed an SW
parametrization exists for the material (not true in general, and this is a
screening tool). Since `lat_c` is mandatory for MLIP configs (§1.1), no
minimization runs during stacking; the surrogate exists solely to make the
`include` loadable and **never influences geometry** (zero forces by
construction).

**Real caveat (benchmark-gated):** the user-supplied `lat_c` and LJ-derived
initial interlayer gap may mismatch the MLIP's equilibrium geometry; a
too-small initial spacing under an attractive foundation model is the known
"dome plow" failure mode. The bilayer benchmark reports the MLIP equilibrium
lattice constant and interlayer spacing so configs can be set consistently;
`documentation/mlip.md` states this workflow.

### 3.4 Supported-mode table

| Mode | MLIP support |
|---|---|
| `drive_method = virtual_atom` | All runs, via §1.4 mechanism (Milestone-0-gated; fallback: DeePMD-partial-cell only) |
| `drive_method = smd`, `fix_move` | All MLIP runs (tested regardless, as the §1.4 fallback) |
| `constraint_mode = atom_bonds`, `com_spring` | **Unsupported** with MLIP in this iteration (molecular conversion + harmonic bonds + special_bonds exclusions × MLIP neighbor lists is unvalidated; Milestone 0 stretch may lift this) |

### 3.5 Builder-family generalization (core deliverable)

MLIP `pot_type` is accepted in **every** builder family — sheetonsheet, afm,
afm_freestanding/flake, pes-sheet, pes-tip, indent, hetero, hetero_slide —
gated only by the capability matrix (coverage topology), not by family.
Rationale: static PES scans and indentation with a foundation model are
exactly the cheap screening runs MLIPs enable, and hetero cells (two
materials under one foundation model, full-cell) sidestep the
pair-potential-availability problem entirely.

Family-specific notes:
- **PES scans** (pes-sheet/pes-tip subclass the dynamic builders): static
  single-point scans; no drive atom in the scanned phase — the least
  constrained case. Full-cell MACE/DeePMD and partial-cell DeePMD both
  apply.
- **Indent**: same component structure as afm; the setup-3/4 analogues apply
  unchanged.
- **Hetero**: full-cell MLIP is the primary mode (one foundation model, two
  materials). The unconditional Virtual-type registration in `hetero.py`
  (regardless of `drive_method`) is fixed as part of this work — it is both
  a §1.4 dependency and a latent bug.
- **Flake/freestanding**: inherit from their parent builders; covered by the
  same routing + golden tests.

Each family gets at least one golden-block test config; the four friction
setups (§ Goal) remain the deep-test surface.

## 4. HPC layer

### 4.1 Settings model — profiles with a per-framework axis

```yaml
hpc:
  active_profile: cx3
  profiles:
    cx3:
      queue: gpu72            # verify explicit -q is accepted (RCS docs use
                              # automatic routing); drop if routing suffices
      gpu: {ngpus: 1, gpu_type: L40S}
      num_cpus: 8
      memory_gb: 64
      modules: [tools/prod]
      frameworks:             # per-framework overlay (two incompatible builds)
        mace:   {binary: "$HOME/lammps-mace/build/lmp",  env_lines: ["..."]}
        deepmd: {binary: "$HOME/deepmd-kit/bin/lmp",     env_lines: ["..."]}
    hx1:                      # VRAM escape hatch ONLY: RCS policy says
      queue: a100             # single-node/small jobs belong on CX3; use HX1
      gpu: {ngpus: 1, gpu_type: A100}   # only if 48GB L40S is exceeded
```

- CX3 is the default GPU target. Documented constraints: L40S 48 GB
  (default), RTX6000 24 GB, A100 40 GB (limited); **12 concurrent GPUs per
  user** on gpu72 — array campaigns serialize beyond that.
- A profile is a partial override merged onto the base `hpc:` block; the
  framework overlay merges last. No profile → today's behavior,
  byte-identical scripts (regression-tested).
- CLI: `FrictionSim2D hpc generate ... --profile cx3`.

### 4.2 Manifest schema + CPU/GPU splitting (new machinery)

Existing manifests are flat path strings and `JobEntry` has no resource
fields, so mixed campaigns need new machinery:

- `JobEntry` gains `needs_gpu: bool`, `framework: Optional[str]`,
  `launcher_flags: str` (recorded at generation time from the registry).
- `HPCScriptGenerator` groups jobs by `(needs_gpu, framework)` and emits one
  array script per group, each with its own resource context (CPU groups
  render exactly as today). `submit_all` lists every script.

### 4.3 Select line + launcher

- GPU select: `select=1:ncpus={n}:mem={m}gb:ngpus={g}:gpu_type={t}` with
  **`mpiprocs={LauncherSpec.mpi_ranks}`** — not the CPU path's
  `mpiprocs=ncpus` (inconsistent with `mpirun -np 1`).
- Templates: `{{ mpi_command }} lmp {{ lmp_flags }}` becomes
  `{{ launch_command }}` (fully resolved: mpi command, per-framework binary,
  merged flags); an `env_lines` block renders after module loads. CPU-only
  rendering is unchanged output.

### 4.4 Local execution (honest scope)

`run.py` does not execute production LAMMPS today (it only builds inputs).
Local MLIP execution on the GPU workstation is delivered as documentation +
a thin helper script (`scripts/run_local.sh <sim_dir> <framework>`) that
resolves the per-framework binary/flags from settings and invokes `lmp` on
`system.in`/`slide.in`. Promoting this into `run.py` is future work.

## 5. Build recipes, benchmark, and handoff documentation

- `scripts/hpc/build_lammps_mace.sh`, `scripts/hpc/build_lammps_deepmd.sh`:
  per-cluster module preambles (CX3, HX1) + local-workstation variant.
  **Two separate builds** (ACEsuit fork + libtorch + Kokkos-CUDA vs pinned
  mainline + deepmd-kit v3 plugin), each with EXTRA-PAIR (D3), versions
  pinned per build. Idempotent and commented. The MACE build additionally
  enables ML-IAP (+ ML-SNAP + PYTHON packages) so the Milestone-0 ML-IAP
  hybrid experiment (§7) needs no rebuild.
- **Foundation models pinned exactly** in `documentation/mlip.md`:
  - MACE: named MACE-MP release (e.g. `mace-mp-0b3-medium`) + the
    `mace_create_lammps_model` conversion command (float64 default noted).
  - DeePMD: exact DPA model + deepmd-kit version + per-head freeze/convert
    commands (`DP_ENABLE_PYTORCH=1` caveats). A published MoS2 DeePMD model
    is the **code-path test fallback only**, not a scientific substitute.
- `scripts/mlip_bilayer_benchmark.py`: bilayer MoS2 binding curve (rigid
  z-displacement single points) for SW+LJ vs MLIP-base vs MLIP+D3 per
  framework. Also reports: MLIP equilibrium lattice constant and interlayer
  spacing (feeds §3.3) and measured VRAM vs atom count (feeds §3.2). The
  go/no-go physics gate before production.
- `documentation/mlip.md`: model acquisition, setups + example configs,
  capability matrix, base-vs-D3 workflow, builder-family screening guide,
  profiles + GPU sizing, local-GPU-PC setup, troubleshooting (MACE hybrid
  limits, ML-IAP status, deepmd multi-instance notes, model-deviation trap).
- `examples/mlip/`: `sheet_mlip_lj.ini`, `sheet_mlip_full.ini`,
  `afm_mlip_lj.ini`, `afm_mlip_full.ini`, plus one screening example per
  additional family (pes-sheet full-cell MACE; hetero full-cell) —
  dispersion toggleable only where the matrix allows.

## 6. Testing

- **Golden-block tests** for every allowed capability-matrix cell across the
  four friction setups, plus one golden config per additional builder
  family (§3.5): pair_style/pair_coeff/mass/group blocks, per-build D3
  tokens, ≥3-instance suffixing, `neigh_modify exclude` emission.
- **Config validation tests**: every §1.1/§3.4 rule (virtual-atom fallback
  mode, missing lat_c, wrong model extension, D3-with-partial-cell,
  MACE-with-partial-cell).
- **HPC rendering tests**: GPU select line (incl. mpiprocs), profile +
  framework-overlay merge, env_lines, CPU/GPU/framework array splitting,
  JobEntry schema; regression: non-MLIP scripts byte-identical.
- **Registry/unit tests**: count_atomtypes short-circuit (both call sites),
  shared model staging, d3 token translation per build.
- **Postprocessing filter test**: existing path-segment parsing unaffected
  by the flavor token (§1.3).
- **Smoke tests**: gated on an actual **GPU probe** (CX3 login nodes have no
  GPUs); runnable on the GPU workstation or as a short gpu72 job.

## 7. Milestone 0 — feasibility gate (first implementation step)

On the GPU workstation (or a short gpu72 job), tiny systems:

1. DeePMD: two, then three, independent `deepmd` sub-style instances in one
   `pair_style hybrid` with NULL maps + `lj/cut` + `sw` (partial-cell
   mechanism for setups 1/3/4).
2. DeePMD under `hybrid/overlay` with `dispersion/d3` (full-cell + D3);
   known segfault reports — genuinely open.
3. MACE: pure `pair_style mace` (full-cell base) and full-cell
   `hybrid/overlay mace + dispersion/d3`.
4. **Virtual-atom compatibility** (§1.4): isolated (neigh-excluded)
   placeholder-mapped atom under pair_mace, deepmd, and dispersion/d3 —
   energy constant, forces zero. Failure → smd/fix_move fallback activates.
5. `smd` and `fix_move` golden + smoke on sheetonsheet and afm (fallback
   path must work regardless).
6. (Stretch) **MACE ML-IAP hybrid experiment**: `pair_style mliap unified`
   as a hybrid sub-style with NULL — undocumented; if it works it unlocks
   MACE partial-cell and revises the matrix.
7. (Stretch) `constraint_mode` × MLIP neighbor-list check (§3.4).

Milestone 0 results update the capability matrix from "Conditional" to
Yes/No before the main build-out proceeds.

## 8. Simulation flow (unchanged shape)

```
.ini (+ mlip keys) + settings.yaml (+ profiles)
  → builders (validation → MLIP routing; pair_style zero surrogate stacking)
  → LAMMPS inputs (hybrid[/overlay] per capability matrix)
  → hpc/ (per-framework GPU scripts)  OR  scripts/run_local.sh (workstation)
  → results → existing postprocessing (schema unchanged; filter test in §6)
```

## 9. Risks (post-review, post-feedback)

| Risk | Impact | Mitigation |
|---|---|---|
| DeePMD multi-instance hybrid unproven (no documented precedent) | Setups 1/3 blocked | Milestone 0 item 1 first; setup 4 (single NULL-partitioned instance, documented) proceeds regardless |
| DeePMD under overlay segfaults (community reports) | Full-cell+D3 loses DeePMD | Milestone 0 item 2; MACE covers full-cell+D3 if its overlay passes |
| Virtual-atom exclusion trick fails for some style | Preferred drive unusable for those runs | Milestone 0 item 4; automatic smd/fix_move fallback validation |
| MACE fork / mainline d3 syntax skew | Broken inputs on one build | Per-build token translation + per-build golden tests |
| Broader family surface (§3.5) grows test/validation scope | Schedule | Family support reuses the same registry/routing; per-family cost is one golden config + validation wiring; hetero Virtual fix is a bounded bug fix |
| lat_c / initial gap vs MLIP equilibrium mismatch ("dome plow") | Unphysical starts | Benchmark reports MLIP equilibrium geometry; documented workflow |
| VRAM under multi-instance + float64 | Job failures | Measured atoms-vs-VRAM gates sizing; HX1 A100-80GB escape hatch |
| DPA-in-LAMMPS conversion friction (v3, per-head freeze, custom OPs) | DeePMD foundation runs delayed | Pinned commands in docs; published-model fallback keeps code-path work unblocked (labeled non-substitute) |
| gpu72 12-GPU user limit | Campaign serialization | Documented; array sizing guidance |
| Foundation-model physics even with D3 | Scientific validity | Benchmark is the explicit go/no-go gate |

## 10. Future work (documented, not implemented)

- **MACE partial-cell**: promote whichever Milestone-0 stretch result works —
  ML-IAP-in-hybrid, or a maintained `pair_mace` fork masking NULL types out
  of the graph (upstream declined; ours to carry).
- **NULL-aware `dispersion/d3` fork** — unlocks D3 for partial-cell runs.
- **Full potential-plugin refactor (approach C)**: promote SW/ReaxFF/… into
  the registry pattern, replacing the `POTENTIALS_*` sets and ReaxFF
  special-cases; do once MLIP golden tests guard the migration.
- Promote `scripts/run_local.sh` into `run.py` proper.
- GUI exposure of the new config surface (gui-p2).
- Multi-GPU scaling / Allegro-class models if system sizes grow.
- MACE fine-tuned on MoS2 sliding configurations (separate research
  project).
