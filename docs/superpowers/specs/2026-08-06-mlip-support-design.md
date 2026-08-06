# MLIP Support for FrictionSim2D — Design Specification

**Date:** 2026-08-06
**Branch:** `mlip` (worktree off `gui-p2`)
**Status:** Draft — pending adversarial review + user approval

## Goal

Add machine-learned interatomic potential (MLIP) support to FrictionSim2D so that
four MoS2 simulation setups can be generated, run on Imperial RCS clusters
(CX3 / HX1) with correct GPU/CPU resource requests, and reproduced on a local
GPU workstation:

1. **Sheet-on-sheet, MLIP intralayer + LJ interlayer** — each MoS2 layer is
   described by its own MLIP instance (as SW is applied per-layer today);
   interlayer interactions remain `lj/cut`.
2. **Sheet-on-sheet, full MLIP** — one MLIP describes the whole simulation
   cell, intralayer and interlayer.
3. **AFM tip / sheet / substrate, MLIP sheet intralayer** — sheet layers use
   per-layer MLIP instances; tip and substrate stay SW; LJ between sheet
   layers and between sheet and tip/substrate.
4. **AFM tip / sheet / substrate, MLIP sheet intra+interlayer** — the whole
   sheet stack (both layers plus their interlayer interaction) uses one MLIP;
   tip and substrate stay SW; LJ only between the sheet and tip/substrate.

Frameworks: **MACE** and **DeePMD**, implemented in parallel, using
**foundation models** (MACE-MP family; DeePMD DPA family). Training new
models is **out of scope**. Each MLIP setup runs both **without** and
**with D3 dispersion** (base values are needed as the comparison baseline;
D3 is the physically-corrected target).

GUI integration is **out of scope** for this spec (backend only). The GUI
will later expose the new `pot_type` values and switches; the config surface
below is the interface it will build on.

## Non-goals

- Training or fine-tuning MLIPs.
- GUI changes (deferred to gui-p2 work).
- Multi-GPU / multi-node MLIP runs (system sizes here are 10³–10⁴ atoms;
  one GPU per job is the design point).
- Refactoring existing potentials (SW, ReaxFF, …) into the new registry —
  documented as future work (see §9).

## Background constraints (verified)

- **`pair_style dispersion/d3`** exists in stock LAMMPS (EXTRA-PAIR package,
  metal units) and is documented as intended for overlaying D3 on
  ML potentials via `pair_style hybrid/overlay`.
- **MACE + hybrid is risky.** `pair_style mace` lives in the ACEsuit LAMMPS
  fork; community reports show `hybrid/overlay` combinations failing, and
  `no_domain_decomposition` (recommended for small systems) breaks with
  multiple MPI ranks. Mitigation: DeePMD is the verified path for setups
  1 and 3; an early MACE-in-hybrid smoke test is a plan milestone
  (see Risks, §8).
- **DeePMD** (`pair_style deepmd`) has mature hybrid support including
  `NULL` type mapping, and its own GPU backend (no Kokkos suffix needed).
- **CX3 GPU jobs**: PBS `select=1:ncpus=N:mem=Mgb:ngpus=G:gpu_type=T`,
  queue `gpu72`, L40S default (48 GB), A100 limited. **HX1**: PBS,
  4× A100-80GB nodes. Both remain configurable via profiles (§5).
- Existing `PotentialManager` machinery already handles per-layer many-body
  application through hybrid NULL maps (`_add_layer_self_interaction`) —
  setups 1/3 reuse this path; setups 2/4 reuse the single whole-component
  path.

## 1. Configuration surface

### 1.1 Per-run `.ini` keys

Per component (existing keys, new accepted values):

```ini
[2D]
pot_type = mace            ; or deepmd (alongside existing sw, reaxff, ...)
pot_path = ../models/mace-mp-0b3-medium.model-lammps.pt
```

New sheet-section keys (only meaningful when the component's `pot_type` is
an MLIP; validation rejects them otherwise):

```ini
mlip_interlayer = lj       ; 'lj' (setups 1,3) | 'native' (setups 2,4)
dispersion = none          ; 'none' | 'd3'
```

- `mlip_interlayer = lj`: sheet registered with per-layer types (as today for
  SW); one MLIP hybrid sub-style instance per layer via NULL maps; interlayer
  pairs assigned `lj/cut` (existing `add_interlayer_interaction` /
  `add_ghost_lj` path).
- `mlip_interlayer = native`: sheet registered as one MLIP-covered component;
  no interlayer LJ lines are emitted; the MLIP's pair_coeff covers all sheet
  types. Cross-component LJ (sheet↔tip, sheet↔sub) is emitted exactly as
  today.
- `dispersion = d3`: the potential file uses `pair_style hybrid/overlay`
  with a `dispersion/d3` sub-style whose pair_coeff covers **only the
  MLIP-described type pairs** (intralayer for setups 1/3; intra+interlayer
  sheet pairs for setups 2/4). SW/LJ-described pairs get no D3 (their
  parametrizations already include dispersion empirically).

### 1.2 `settings.yaml` additions

Under `potential:` (defaults; overridable per run):

```yaml
potential:
  d3_functional: pbe       # functional the damping params are fitted to
  d3_damping: d3bj         # damping variant supported by pair_style dispersion/d3
  d3_cutoff: 16.0          # Å, dispersion cutoff
  d3_cn_cutoff: 8.0        # Å, coordination-number cutoff
```

Exact argument order/names for `dispersion/d3` are pinned to the LAMMPS
version selected by the build recipes (§6); the registry emits them, tests
golden-check them.

### 1.3 Output directory naming (sweep-loopability rule)

Any parameter that distinguishes runs must appear in the output dir name.
MLIP runs append a potential-flavor token:

```
<existing name>_<pot_type>-<interlayer>-<dispersion>
e.g. ..._mace-native-d3, ..._deepmd-lj-base
```

Non-MLIP runs keep their existing names byte-for-byte (regression-tested).

## 2. `src/core/mlip.py` — MLIP registry

New module, consumed by `PotentialManager`. Contents:

```python
@dataclass(frozen=True)
class MLIPSpec:
    name: str                      # 'mace' | 'deepmd'
    model_in_style_line: bool      # deepmd: True (model path in pair_style args)
                                   # mace:   False (model path in pair_coeff)
    style_args: Callable           # (model_path, opts) -> pair_style argument string
    coeff_line: Callable           # (index_str, model_path, null_map) -> pair_coeff line
    supports_hybrid: bool          # deepmd: True; mace: False-until-verified
    launcher: LauncherSpec         # flags for lmp invocation (see below)
    file_extensions: tuple         # accepted model file suffixes (validation)
    notes: str                     # human-readable caveats (surfaced in docs)

@dataclass(frozen=True)
class LauncherSpec:
    mpi_ranks: int                 # 1 for MACE (no_domain_decomposition)
    lmp_flags: str                 # e.g. '-k on g 1 -sf kk -pk kokkos newton on neigh half'
    env_lines: list[str]           # activation lines injected into job scripts
```

- `MLIP_REGISTRY: dict[str, MLIPSpec]` with `mace` and `deepmd` entries.
- `is_mlip(pot_type) -> bool` helper.
- `d3_overlay(elements, null_map, settings) -> list[str]`: emits the
  `dispersion/d3` style token and pair_coeff lines restricted to the given
  types.

### `PotentialManager` integration (surgical)

- `is_sheet_lj(pot_type)` gains an MLIP branch: for MLIPs the answer depends
  on the sheet's `mlip_interlayer` mode, so the sheet component's registration
  records the mode and `add_self_interaction` routes accordingly (per-layer
  NULL maps vs whole-component map). No change to non-MLIP paths.
- `write_file` style-line assembly generalizes the existing ReaxFF
  special-case: each used style asks the registry (or the ReaxFF branch) for
  its style-line arguments. `hybrid/overlay` is selected instead of `hybrid`
  when a D3 overlay is present. (LAMMPS `hybrid/overlay` accepts the same
  type-partitioned many-body sub-styles; only D3 actually overlays.)
- Model files are staged through the existing `_get_potential_path`
  copy-and-prefix mechanism, unchanged.
- Per-layer MLIP instances (setups 1/3) follow the existing numeric-suffix
  convention (`mace 1`, `mace 2`) exactly as SW multilayer does today.
  Memory note: two model instances on one GPU (~2× model VRAM — fine for
  foundation models on 48 GB L40S).

## 3. Builders

`sheetonsheet.py` and `afm.py`:

- Parse `mlip_interlayer` and `dispersion`; validate (`native`/`d3` require
  an MLIP `pot_type`; MLIP `pot_type` on tip/sub is rejected in this
  iteration — tip and substrate stay SW by design).
- Route sheet registration: `native` → single self-interaction across all
  layers, skip interlayer LJ; `lj` → existing per-layer path.
- Append the potential-flavor token to output dir names (§1.3).
- Everything else (geometry, groups, thermostats, drives) is untouched:
  MLIP types are ordinary LAMMPS types.

## 4. HPC layer — GPU resources and cluster profiles

### 4.1 Settings model

```yaml
hpc:
  active_profile: cx3
  profiles:
    cx3:
      queue: gpu72
      gpu: {ngpus: 1, gpu_type: L40S}
      num_cpus: 8
      memory_gb: 64
      modules: [tools/prod, ...]
      env_lines: ["source $HOME/venvs/lammps-mlip/bin/activate"]
      mpi_command: "mpirun -np 1"
      lmp_flags: ""            # filled per-run from LauncherSpec
    hx1:
      gpu: {ngpus: 1, gpu_type: A100}
      ...
```

- `HPCSettings` gains `gpu: Optional[GPUSettings]` (`ngpus`, `gpu_type`),
  `env_lines: list[str]`, `profiles: dict[str, HPCProfile]`,
  `active_profile: Optional[str]`. A profile is a partial override set
  merged onto the base `hpc:` block; no profile → today's behavior,
  byte-identical scripts (regression-tested).
- CLI: `FrictionSim2D hpc generate ... --profile cx3` overrides
  `active_profile`.

### 4.2 Script generation & templates

- `HPCConfig.to_dict()` renders the select line with
  `:ngpus={n}:gpu_type={t}` appended when GPU is enabled, and passes
  `env_lines` and a fully-resolved `launch_command` (mpi command + `lmp` +
  merged flags: profile flags + the MLIP `LauncherSpec` flags recorded in the
  job manifest at generation time).
- Templates (`pbs_array.j2`, `pbs_combined.j2`, slurm twins): the hardcoded
  `{{ mpi_command }} lmp {{ lmp_flags }}` lines become `{{ launch_command }}`;
  an `env_lines` block renders after `module load`s. CPU-only rendering is
  unchanged output.
- The job manifest records, per job, whether it needs GPU and which launcher
  flags apply, so mixed CPU/GPU campaigns can be split into separate array
  scripts (one CPU array, one GPU array) by the existing chunking machinery.

### 4.3 GPU/CPU balance guidance (documented, defaults encoded in profiles)

- MLIP jobs: 1 MPI rank, 1 GPU, 4–8 CPU cores (host-side hybrid sub-styles
  SW/LJ + D3 run on CPU within the same rank; a few cores cover them at
  these system sizes).
- Pure-CPU jobs (all-SW): unchanged (`num_cpus` ranks, no GPU request).

### 4.4 Local runs (`core/run.py`)

- `run.py` accepts an optional launcher override (binary path + flags)
  resolved from the same profile/LauncherSpec mechanism, so the GPU
  workstation runs the identical inputs with its local `lmp`. No scheduler
  logic locally.

## 5. Build recipes and handoff documentation (first-class deliverables)

The implementation work transfers to another PC with a small GPU; docs must
be complete enough to continue there without this session's context.

- `scripts/hpc/build_lammps_mace.sh` and `scripts/hpc/build_lammps_deepmd.sh`
  with per-cluster module preambles (CX3, HX1) and a local-workstation
  variant: pinned LAMMPS version, EXTRA-PAIR (D3) enabled, Kokkos+CUDA for
  MACE (libtorch), DeePMD-kit v3 plugin build. Idempotent, commented, and
  runnable step-by-step.
- `documentation/mlip.md`:
  - foundation model acquisition (MACE-MP-0b3 medium download + LAMMPS
    conversion via `mace_create_lammps_model`; DeePMD DPA model download +
    freeze/convert),
  - the four setups with their exact example configs,
  - base-vs-D3 workflow,
  - cluster profiles and GPU sizing guidance,
  - local GPU-PC setup section (env, drivers, smoke test),
  - troubleshooting (known MACE hybrid failure modes and the DeePMD
    fallback).
- `examples/mlip/`: four ready-to-run configs —
  `sheet_mlip_lj.ini`, `sheet_mlip_full.ini`, `afm_mlip_lj.ini`,
  `afm_mlip_full.ini` (each dispersion-toggleable).
- `scripts/mlip_bilayer_benchmark.py`: standalone bilayer MoS2 binding-curve
  benchmark (rigid z-displacement single-point evaluations) comparing
  SW+LJ vs MLIP-base vs MLIP+D3 per framework — the go/no-go physics check
  before production friction runs.

## 6. Testing

All input generation is text; no GPU needed in CI:

- **Golden-block tests**: exact pair_style/pair_coeff/mass/group output for
  each of the 4 setups × 2 frameworks × {base, d3} (16 cases) plus
  representative sweeps.
- **Config validation tests**: `mlip_interlayer`/`dispersion` rejected
  without MLIP pot_type; MLIP on tip/sub rejected; model file extension
  checks.
- **HPC rendering tests**: GPU select line, profile merge/override,
  env_lines injection, CPU/GPU array splitting; regression tests asserting
  non-MLIP scripts and non-profile settings render byte-identically to
  current output.
- **Registry unit tests**: style-line vs coeff-line model placement per
  framework; D3 overlay type restriction.
- **Smoke tests** (skip-unless-available): tiny LAMMPS runs gated on an
  `lmp` binary with the required packages, runnable on the GPU PC and
  cluster login/dev nodes.

## 7. Simulation flow (unchanged shape)

```
.ini (+ mlip keys) + settings.yaml (+ profiles)
  → builders (route MLIP registration) → LAMMPS inputs (hybrid[/overlay] blocks)
  → hpc/ (GPU-aware scripts via profile)  OR  run.py (local launcher override)
  → results → existing postprocessing (COF, stick-slip) — no changes needed;
    MLIP runs produce the same output_full_*.json schema.
```

## 8. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| `pair_style mace` unusable as hybrid sub-style (setups 1/3) or under `hybrid/overlay` with D3 | MACE limited to setup 2/4-base | **Milestone 0 smoke test** on the GPU PC before deep MACE work; DeePMD is the verified path for all four setups; spec treats MACE hybrid as conditional |
| `no_domain_decomposition` requires single MPI rank | Caps MACE throughput | Acceptable at 10³–10⁴ atoms; encoded in `LauncherSpec.mpi_ranks = 1` |
| Two MLIP instances per GPU (setups 1/3) exceed VRAM | Job failure | Foundation models are ≲1 GB; 48 GB L40S has ample headroom; documented check in mlip.md |
| `dispersion/d3` argument syntax varies across LAMMPS versions | Broken inputs | LAMMPS version pinned in build scripts; golden tests pin the emitted syntax to that version |
| DeePMD foundation (DPA) model needs conversion / v3-only features | Setup friction | Conversion steps documented; fallback: any published MoS2 DeePMD model exercises the identical code path |
| Foundation-model physics (underbound vdW) even with D3 | Scientific validity | Bilayer benchmark (§5) is the explicit go/no-go gate before production |
| gpu72 walltime/core limits differ from CPU queues | Rejected jobs | Profile carries queue-specific walltime/core defaults; documented |

## 9. Future work (documented, not implemented)

- **Full potential-plugin refactor (approach C)**: promote SW, ReaxFF,
  Tersoff, … into the same registry pattern as `MLIP_REGISTRY`, replacing
  `POTENTIALS_WITH_INTERNAL_LJ` / `POTENTIALS_REQUIRING_LJ` sets and the
  ReaxFF special-cases with per-potential descriptors (possibly entry-point
  based). Do this as a dedicated cleanup once MLIP support is stable and
  golden tests exist to guard the migration.
- GUI exposure of `pot_type` MLIP options, `mlip_interlayer`, `dispersion`
  (gui-p2 work).
- Multi-GPU scaling if system sizes grow (Allegro-class models would then be
  reconsidered).
- MACE fine-tuned on MoS2 sliding configurations (separate research project).
