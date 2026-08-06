# MLIP Support for FrictionSim2D — Design Specification (v2)

**Date:** 2026-08-06
**Branch:** `mlip` (worktree off `gui-p2`)
**Status:** v2 — revised after adversarial review + moderation; pending user approval
**Review provenance:** v1 was attacked by an adversarial reviewer (25 findings,
5 blockers) and arbitrated by a moderator (all upheld, 4 additional findings).
This version incorporates every binding disposition. Where v1 claimed
capabilities upstream projects have rejected, v2 states the honest limits.

## Goal

Add machine-learned interatomic potential (MLIP) support to FrictionSim2D so
that four MoS2 setups can be generated, run on Imperial RCS clusters with
correct GPU/CPU resource requests, and reproduced on a local GPU workstation:

1. **Sheet-on-sheet, MLIP intralayer + LJ interlayer** — one MLIP hybrid
   sub-style instance per layer (≥3 layers; the builder enforces
   `MIN_LAYERS = 3`), interlayer pairs on `lj/cut`.
2. **Sheet-on-sheet, full MLIP** — one MLIP describes the whole cell,
   intralayer and interlayer.
3. **AFM tip/sheet/substrate, MLIP sheet intralayer** — per-layer MLIP
   instances for the sheet; SW tip and substrate; LJ between sheet layers and
   between sheet and tip/substrate.
4. **AFM tip/sheet/substrate, MLIP sheet intra+interlayer** — one MLIP
   instance covering all sheet types (NULL-partitioned in hybrid); SW tip and
   substrate; LJ only between sheet and tip/substrate.

Frameworks: **MACE** and **DeePMD**, foundation models (MACE-MP family;
DeePMD DPA family). Training/fine-tuning is out of scope. Dispersion: runs
**without** and **with D3** where possible (base values are the comparison
baseline).

GUI integration is out of scope (backend only); the config surface in §1 is
the interface the GUI will later expose.

## Capability matrix (binding)

Upstream constraints — verified against LAMMPS docs, the ACEsuit maintainer's
own statements, and community reports — make the honest matrix:

| Setup | DeePMD base | DeePMD + D3 | MACE base | MACE + D3 |
|---|---|---|---|---|
| 1. SoS, MLIP intra + LJ inter | Conditional¹ | No² | No³ | No |
| 2. SoS, full MLIP | Yes | Conditional¹ | Yes⁴ | Conditional¹˒⁴ |
| 3. AFM, MLIP sheet intra | Conditional¹ | No² | No³ | No |
| 4. AFM, MLIP sheet intra+inter | Conditional¹ | No² | No³ | No |

¹ Mechanism is documented/community-confirmed but unproven in our stack —
  gated on **Milestone 0** (§7). Nothing is called "verified" until it passes.
² Stock `pair_style dispersion/d3` requires a real element for **every** atom
  type (no NULL): it cannot be restricted to MLIP pairs, so partial-cell D3
  would double-count dispersion over SW/LJ pairs. D3 is therefore valid
  **only** for full-cell-MLIP runs (setup 2). A NULL-aware `dispersion/d3`
  fork is documented future work (§10), not a deliverable.
³ `pair_style mace` does not support NULL type maps — segfaults, and the
  maintainer has rejected selective type exclusion as architecturally
  incompatible (ACEsuit discussion #1314). **MACE serves setup 2 only.**
  The MACE-vs-DeePMD comparison exists only at setup 2; setups 1/3/4 are
  DeePMD-only. "Map tip/sub Si to real Si in MACE-MP" is NOT a workaround:
  MACE would then compute tip/substrate/sheet↔Si interactions that SW and LJ
  also compute (double counting). A pure full-cell-MACE AFM run is a distinct
  optional future experiment (§10), not a substitute for setups 3/4.
⁴ Requires a non-virtual `drive_method` (see §1.4). Full-cell MACE + D3 via
  `hybrid/overlay` is a community-confirmed pattern (matsci #62930) but still
  Milestone-0-gated.

## Non-goals

- Training or fine-tuning MLIPs; GUI changes; multi-GPU/multi-node runs
  (design point: 1 GPU per job at 10³–10⁴ atoms).
- MLIP support for the other builder families (indent, PES scans, hetero,
  flake variants). Config validation **rejects** MLIP `pot_type` outside
  `sheetonsheet` and `afm` dynamic runs in this iteration.
- Patching LAMMPS pair styles (the NULL-aware d3 fork is future work).

## 1. Configuration surface

### 1.1 Per-run `.ini` keys

```ini
[2D]
pot_type = mace            ; or deepmd (alongside existing sw, ...)
pot_path = ../models/mace-mp-0b3-medium.model-lammps.pt
mlip_interlayer = lj       ; 'lj' (setups 1,3) | 'native' (setups 2,4)
dispersion = none          ; 'none' | 'd3'
lat_c = 3.16               ; MANDATORY for MLIP pot_types (see §3.3)
```

Validation rules (config layer, hard errors):
- `mlip_interlayer` / `dispersion` require an MLIP `pot_type`.
- MLIP `pot_type` on tip or substrate is rejected (SW by design).
- `dispersion = d3` requires: sheetonsheet model, `mlip_interlayer = native`,
  and a non-virtual `drive_method` (§1.4).
- `pot_type = mace` requires `mlip_interlayer = native`, sheetonsheet model,
  and a non-virtual `drive_method`.
- MLIP `pot_type` requires explicit `lat_c` and a model file whose extension
  matches the framework (`.model-lammps.pt`/`.pt` for MACE; `.pb`/`.pth` for
  DeePMD).
- MLIP `pot_type` outside sheetonsheet/afm models is rejected (§ Non-goals).
- Supported `constraint_mode` / `drive_method` combinations per setup are
  enumerated in the validation table (see §3.4); everything else rejected.

### 1.2 `settings.yaml` additions

```yaml
potential:
  d3_functional: pbe
  d3_damping: bj        # canonical name; enum-validated
  d3_cutoff: 16.0       # Å
  d3_cn_cutoff: 8.0     # Å
```

`d3_damping` is a **canonical** name that the registry translates into the
token each LAMMPS build actually accepts: the ACEsuit fork ships an older
`dispersion/d3` (pre-rename tokens, different arg conventions) than pinned
mainline (`original|bj|zerom|bjm`). Emitted syntax is pinned and golden-tested
**per framework build**, not globally.

### 1.3 Output directory naming (sweep-loopability rule)

MLIP runs append a flavor token to the **innermost run-parameter segment** of
the output path (the same segment that already encodes temperature/spring
variants, so upstream segments used by postprocessing filters — material,
size — are untouched):

```
<...>/K<temp>_<pot_type>-<interlayer>-<disp>
disp token: 'base' when dispersion=none, 'd3' when dispersion=d3
e.g. .../K0_deepmd-lj-base, .../K0_mace-native-d3
```

Non-MLIP paths are byte-identical to today (regression-tested), and a
postprocessing filter test proves existing segment parsing still works.

### 1.4 Drive method constraint

The default `drive_method = virtual_atom` registers a massless `Virtual` atom
type with no chemical element; it forces hybrid mode and takes
`lj/cut 1e-100` coefficients. `pair_style mace` cannot NULL-map it and
`dispersion/d3` cannot assign it an element — so **all MACE runs and all D3
runs must use a non-virtual drive** (`smd` or `fix_move`), enforced by
validation. DeePMD-base runs may keep `virtual_atom` (NULL covers it).

The non-virtual drive paths are currently under-exercised in the codebase
(e.g. `hetero.py` registers the Virtual type unconditionally — evidence these
branches aren't uniformly wired). Milestone 0 therefore includes golden +
smoke tests of `smd` and `fix_move` on sheetonsheet and afm **before** they
are mandated for MLIP runs.

## 2. `src/core/mlip.py` — MLIP registry

```python
@dataclass(frozen=True)
class MLIPSpec:
    name: str                   # 'mace' | 'deepmd'
    model_in_style_line: bool   # deepmd: True; mace: False (model in pair_coeff)
    style_args: Callable        # (model_path, opts) -> pair_style argument string
    coeff_line: Callable        # (index_str, model_path, null_map) -> pair_coeff line
    supports_null: bool         # deepmd: True; mace: False  → gates setups 1/3/4
    supports_overlay: bool      # per-style flag; overlay layout derived from these
    launcher: LauncherSpec
    file_extensions: tuple      # validation (.pt/.model-lammps.pt vs .pb/.pth)
    d3_tokens: Callable         # canonical damping/functional -> build-specific tokens
    notes: str

@dataclass(frozen=True)
class LauncherSpec:
    binary: str                 # per-framework lmp binary/alias — MACE (ACEsuit
                                # fork) and DeePMD (plugin build) are two
                                # incompatible LAMMPS builds; jobs must resolve
                                # the right one
    modules: list[str]          # per-framework module set (merged onto profile)
    env_lines: list[str]
    mpi_ranks: int              # 1 for MACE (no_domain_decomposition)
    lmp_flags: str              # only doc-verified flags pinned (MACE: '-k on g 1
                                # -sf kk'); anything further is added only after
                                # Milestone 0 verifies it
```

- `MLIP_REGISTRY: dict[str, MLIPSpec]`, `is_mlip(pot_type)` helper.
- `d3_overlay(settings, spec) -> list[str]`: full-cell only; emits the
  build-correct `dispersion/d3` style token + a pair_coeff line listing a real
  element for **every** atom type (guaranteed by the §1.1 validation that D3
  runs are full-cell MLIP + non-virtual drive).

### `PotentialManager` / call-site integration (enumerated, not "surgical")

The v1 claim that `is_sheet_lj(pot_type)` could be routed internally was
wrong: the interlayer decision must reach **five** string-only call sites.
The signature changes to accept the component context (pot_type + interlayer
mode), and every site is updated:

1. `potential_manager.py` `register_component` (`potential_usage` accounting
   — per-layer MLIP counts as n_layers instances, native as 1),
2. `potential_manager.py` `add_self_interaction` routing,
3. `components.py` `stack_multilayer_sheet` (stacking pot file contents),
4. `sheetonsheet.py` `has_internal_lj` template flag (2 sites),
5. `afm.py` `has_internal_lj` template flag.

Other integration points:
- `write_file`: per-style argument assembly generalizes the ReaxFF
  special-case via the registry; `hybrid/overlay` is chosen from the
  `supports_overlay` flags when a D3 overlay is present.
- **`count_atomtypes` fix (was a guaranteed crash):** `utils.py` opens the
  potential file as UTF-8 text; binary model files crash it. MLIP pot_types
  and extensions (`.pt`, `.pth`, `.pb`, `.model-lammps.pt`) are added to the
  single-type short-circuit — at **both** call sites, including the one that
  passes no `pot_type` (extension check is load-bearing there). Unit-tested.
- **Model staging:** `_get_potential_path` currently copies the potential
  into every run directory; a 100-run sweep × a foundation model = gigabytes
  duplicated locally and over rsync. MLIP models are staged **once at the
  simulation root** (shared `potentials/` with a relative prefix); classical
  potentials keep today's behavior.
- Per-layer instances use the numeric-suffix convention (`deepmd 1`,
  `deepmd 2`, …) — ≥3 instances for sheet-on-sheet. **Warning encoded in the
  registry docs:** the documented multi-model form `pair_style deepmd m1 m2`
  is model-deviation mode (only the first model drives forces) — a one-typo
  trap the generator must never emit.

## 3. Builders

### 3.1 Routing

`sheetonsheet.py` / `afm.py`: parse the new keys; `native` → single
self-interaction across all sheet layers, no interlayer LJ; `lj` → existing
per-layer path (≥3 instances for sheet-on-sheet). Cross-component LJ
(sheet↔tip, sheet↔sub) unchanged.

### 3.2 VRAM sizing (measured, not asserted)

MLIP inference memory is dominated by graph activations scaling with
atoms × neighbors (MACE-converted models default to float64), **not** by
weight size. The bilayer benchmark (§5) records measured VRAM vs atom count
per framework and instance count; job sizing guidance in `documentation/`
uses those measurements. No a-priori "it fits" claims.

### 3.3 Stacking without an MLIP-enabled generation machine

`stack_multilayer_sheet` executes LAMMPS **in-process** (python `lammps`
module) for every multilayer build — even when `lat_c` is supplied — and its
rendered script `include`s a potential file derived from the sheet's
`pot_type`. Without intervention, merely *generating* MLIP inputs would
require an MLIP-enabled python-LAMMPS on the generation machine (and CI).

Design: for MLIP sheets, the stacking step receives a **surrogate SW
potential file** (config copy with `pot_type='sw'` + the material's SW path;
type counts and `atom_style` match). Since `lat_c` is mandatory for MLIP
configs (§1.1), no minimization runs during stacking — the surrogate exists
solely to make the `include` loadable; it never influences geometry.

**Real caveat (benchmark-gated):** the user-supplied `lat_c` and the
LJ-derived initial interlayer gap may mismatch the MLIP's equilibrium
geometry. A too-small initial spacing under an attractive foundation model is
the known "dome plow" failure mode. The bilayer benchmark (§5) reports the
MLIP equilibrium lattice constant and interlayer spacing so configs can be
set consistently; `documentation/mlip.md` states this workflow.

### 3.4 Supported-mode table

The spec's validation table (implemented in config validation, documented in
mlip.md):

| Mode | MLIP support |
|---|---|
| `drive_method = virtual_atom` | DeePMD-base only |
| `drive_method = smd`, `fix_move` | all MLIP runs (required for MACE/D3) |
| `constraint_mode = atom_bonds`, `com_spring` | **unsupported** with MLIP in this iteration (molecular conversion + harmonic bonds + special_bonds exclusions interacting with MLIP neighbor lists is unvalidated; Milestone 0 may lift this) |

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
        mace:   {binary: "$HOME/lammps-mace/build/lmp",  env_lines: [...]}
        deepmd: {binary: "$HOME/deepmd-kit/bin/lmp",     env_lines: [...]}
    hx1:                      # VRAM escape hatch ONLY: RCS policy says
      queue: a100             # single-node/small jobs belong on CX3; use HX1
      gpu: {ngpus: 1, gpu_type: A100}   # only if 48GB L40S VRAM is exceeded
      ...
```

- CX3 is the default GPU target. Known constraints recorded in docs: L40S
  48 GB (default), RTX6000 24 GB, A100 40 GB (limited); **12 concurrent GPUs
  per user** on gpu72 — array campaigns serialize beyond that.
- A profile is a partial override merged onto the base `hpc:` block; the
  framework overlay merges last. No profile → today's behavior,
  byte-identical scripts (regression-tested).
- CLI: `FrictionSim2D hpc generate ... --profile cx3`.

### 4.2 Manifest schema + CPU/GPU splitting (new machinery, not reuse)

v1 claimed the existing chunking could split mixed campaigns; it cannot —
manifests are flat path strings and `JobEntry` has no resource fields. Design:

- `JobEntry` gains `needs_gpu: bool`, `framework: Optional[str]`,
  `launcher_flags: str` (recorded at generation time from the registry).
- `HPCScriptGenerator` groups jobs by `(needs_gpu, framework)` and emits one
  array script per group, each rendered with its own resource context
  (CPU groups render exactly as today). `submit_all` lists every script.

### 4.3 Select line + launcher

- GPU select: `select=1:ncpus={n}:mem={m}gb:ngpus={g}:gpu_type={t}` with
  **`mpiprocs={LauncherSpec.mpi_ranks}`** — not the CPU path's
  `mpiprocs=ncpus` (an inconsistent declaration when running
  `mpirun -np 1`).
- Templates: `{{ mpi_command }} lmp {{ lmp_flags }}` becomes
  `{{ launch_command }}` (fully resolved: mpi command, per-framework binary,
  merged flags); an `env_lines` block renders after the module loads.
  CPU-only rendering is unchanged output.

### 4.4 Local execution (honest scope)

`run.py` does not execute production LAMMPS today (it only builds inputs), so
there is no launcher to "override". Local MLIP execution on the GPU
workstation is delivered as **documentation + a thin helper script**
(`scripts/run_local.sh <sim_dir> <framework>`) that resolves the same
per-framework binary/flags from settings and invokes `lmp` on
`system.in`/`slide.in`. Promoting this into `run.py` proper is future work.

## 5. Build recipes, benchmark, and handoff documentation

- `scripts/hpc/build_lammps_mace.sh`, `scripts/hpc/build_lammps_deepmd.sh`:
  per-cluster module preambles (CX3, HX1) + local-workstation variant.
  **Two separate builds** (ACEsuit fork + libtorch + Kokkos-CUDA vs pinned
  mainline + deepmd-kit v3 plugin), each with EXTRA-PAIR (D3) enabled,
  versions pinned per build. Idempotent and commented.
- **Foundation models pinned exactly** in `documentation/mlip.md`:
  - MACE: named MACE-MP release (e.g. `mace-mp-0b3-medium`) + the
    `mace_create_lammps_model` conversion command (note: float64 default).
  - DeePMD: exact DPA model + deepmd-kit version + per-head freeze/convert
    commands (`DP_ENABLE_PYTORCH=1` caveats included). A published MoS2
    DeePMD model is the **code-path test fallback only** — it does not
    substitute for the foundation-model comparison scientifically.
- `scripts/mlip_bilayer_benchmark.py`: bilayer MoS2 binding curve (rigid
  z-displacement single points) for SW+LJ vs MLIP-base vs MLIP+D3 per
  framework. Additionally reports: MLIP equilibrium lattice constant and
  interlayer spacing (feeds §3.3), and measured VRAM vs atom count
  (feeds §3.2). This is the go/no-go physics gate before production.
- `documentation/mlip.md`: model acquisition, the four setups + example
  configs, capability matrix, base-vs-D3 workflow, profiles + GPU sizing,
  local-GPU-PC setup, troubleshooting (MACE hybrid limits, deepmd
  multi-instance notes, model-deviation trap).
- `examples/mlip/`: `sheet_mlip_lj.ini`, `sheet_mlip_full.ini`,
  `afm_mlip_lj.ini`, `afm_mlip_full.ini` (dispersion toggleable only where
  the matrix allows).

## 6. Testing

- **Golden-block tests** for every *allowed* cell of the capability matrix
  (setups × frameworks × dispersion), covering pair_style/pair_coeff/mass/
  group blocks, per-build D3 token emission, and ≥3-instance suffixing.
- **Config validation tests**: every rejection rule in §1.1/§3.4 (including
  MLIP on indent/PES/hetero/flake models, virtual_atom×MACE/D3, missing
  lat_c, wrong model extension).
- **HPC rendering tests**: GPU select line (incl. `mpiprocs=1`), profile +
  framework-overlay merge, env_lines, CPU/GPU/framework array splitting,
  JobEntry schema; regression: non-MLIP scripts byte-identical.
- **Registry/unit tests**: count_atomtypes short-circuit (both call sites),
  shared model staging, d3 token translation per build.
- **Postprocessing filter test**: existing path-segment parsing (material,
  size tokens) unaffected by the flavor token (§1.3).
- **Smoke tests**: gated on an actual **GPU probe** (not just `lmp` presence
  — CX3 login nodes have no GPUs); runnable on the GPU workstation or as a
  short gpu72 job.

## 7. Milestone 0 — feasibility gate (first implementation step)

Before any deep implementation, on the GPU workstation (or a short gpu72
job), with tiny systems:

1. DeePMD: two (then three) independent `deepmd` sub-style instances in one
   `pair_style hybrid` with NULL maps + `lj/cut` + `sw` (setups 1/3/4
   mechanism).
2. DeePMD under `hybrid/overlay` with `dispersion/d3` (setup 2 + D3);
   known segfault reports exist — this is genuinely open.
3. MACE: pure `pair_style mace` (setup 2 base) and full-cell
   `hybrid/overlay mace + dispersion/d3` (setup 2 + D3).
4. `smd` and `fix_move` drive methods on sheetonsheet and afm (golden +
   smoke) — they become load-bearing for MLIP runs (§1.4).
5. (Stretch) `constraint_mode` × MLIP neighbor-list check, to decide whether
   §3.4's restriction can be lifted.

Milestone 0 results update the capability matrix from "Conditional" to
Yes/No before the main build-out proceeds.

## 8. Simulation flow (unchanged shape)

```
.ini (+ mlip keys) + settings.yaml (+ profiles)
  → builders (validation → MLIP routing; SW surrogate for stacking)
  → LAMMPS inputs (hybrid[/overlay] per capability matrix)
  → hpc/ (per-framework GPU scripts)  OR  scripts/run_local.sh (workstation)
  → results → existing postprocessing (schema unchanged; filter test in §6)
```

## 9. Risks (post-review)

| Risk | Impact | Mitigation |
|---|---|---|
| DeePMD multi-instance hybrid unproven (no documented precedent) | Setups 1/3 blocked for both frameworks | Milestone 0 item 1 is the first thing built; setup 4 (single NULL-partitioned instance, documented) is the strongest DeePMD case and proceeds regardless |
| DeePMD under overlay segfaults (community reports) | Setup 2+D3 loses DeePMD | Milestone 0 item 2; MACE covers setup 2+D3 if its overlay path passes |
| MACE fork / mainline d3 syntax skew | Broken inputs on one build | Per-build token translation + per-build golden tests (§1.2, §2) |
| Non-virtual drive paths under-exercised | Silent breakage where MLIP runs are forced onto smd/fix_move | Milestone 0 item 4 before mandating |
| lat_c / initial gap vs MLIP equilibrium mismatch ("dome plow") | Unphysical starts | Benchmark reports MLIP equilibrium geometry; documented config workflow |
| VRAM under multi-instance + float64 | Job failures | Measured atoms-vs-VRAM from benchmark gates job sizing; HX1 A100-80GB escape hatch |
| DPA-in-LAMMPS conversion friction (v3, per-head freeze, custom OPs) | DeePMD foundation runs delayed | Exact pinned commands in docs; published-model fallback keeps code-path work unblocked (labeled non-substitute) |
| gpu72 12-GPU user limit | Campaign serialization | Documented; array sizing guidance |
| Foundation-model physics even with D3 | Scientific validity | Benchmark is the explicit go/no-go gate |

## 10. Future work (documented, not implemented)

- **NULL-aware `dispersion/d3` fork** — would unlock D3 for setups 1/3/4;
  carried as a patch deliverable only if partial-cell D3 becomes scientifically
  necessary.
- **Full-cell MACE AFM** (tip+substrate+sheet all MACE, no SW/LJ) — a
  legitimate *fifth* experiment; explicitly not a substitute for setups 3/4.
- **Full potential-plugin refactor (approach C)**: promote SW/ReaxFF/… into
  the registry pattern, replacing the `POTENTIALS_*` sets and ReaxFF
  special-cases; do it once MLIP golden tests exist to guard the migration.
- Promote `scripts/run_local.sh` into `run.py` proper.
- GUI exposure of the new config surface (gui-p2).
- Multi-GPU scaling / Allegro-class models if system sizes grow.
- MACE fine-tuned on MoS2 sliding configurations (separate research project).
