# MLIP Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement MLIP (MACE + DeePMD) potential support per the approved spec `docs/superpowers/specs/2026-08-06-mlip-support-design.md` (v3): registry, PotentialManager integration, builder routing across all families, GPU-aware HPC profiles, build recipes, benchmark, and docs.

**Architecture:** A new `src/core/mlip.py` descriptor registry drives all MLIP-specific LAMMPS emission; `PotentialManager` delegates to it. HPC gains cluster profiles with a per-framework binary axis. Milestone 0 (hand-written LAMMPS decks + build scripts) verifies every "Conditional" capability-matrix cell before the framework work relies on it.

**Tech Stack:** Python 3.11, Pydantic v2, Jinja2, pytest; LAMMPS (two builds: ACEsuit fork + Kokkos/libtorch for MACE; pinned mainline + deepmd-kit v3 plugin for DeePMD), both with EXTRA-PAIR.

## Global Constraints

- The spec's **capability matrix is binding**: MACE = full-cell only; D3 = full-cell only; partial-cell = DeePMD only. All validation and goldens encode it.
- Non-MLIP output (LAMMPS blocks, HPC scripts, dir names) must stay **byte-identical** — every phase carries a regression test.
- Metal units everywhere; lengths in Å.
- `lat_c` is **mandatory** for MLIP sheet configs.
- Virtual-atom drive is supported via placeholder-element mapping + `neigh_modify exclude` (spec §1.4); `smd`/`fix_move` are the fallback.
- Model files: MACE `.model-lammps.pt`/`.pt`; DeePMD `.pb`/`.pth`. Never emit `pair_style deepmd m1 m2` (model-deviation trap).
- No `Co-Authored-By` trailers in commits.
- Run tests with the `aiida` conda env (`conda activate aiida`); full suite `pytest`, singles `pytest tests/<file> -v`.
- Work on branch `mlip` in this checkout.

---

## Phase 0 — Milestone 0 artifacts (feasibility gate)

These run on the GPU workstation / a short gpu72 job — but the *artifacts* (scripts, decks, runbook) are written and committed here first. Framework tasks (Phase 1+) may proceed in parallel; **production trust** of any "Conditional" matrix cell waits for the runbook results.

### Task 1: LAMMPS build scripts

**Files:**
- Create: `scripts/hpc/build_lammps_deepmd.sh`
- Create: `scripts/hpc/build_lammps_mace.sh`
- Create: `scripts/hpc/README.md`

**Interfaces:**
- Produces: two `lmp` binaries at `$PREFIX/lammps-deepmd/bin/lmp` and `$PREFIX/lammps-mace/build/lmp` (paths later referenced by HPC profile `frameworks:` entries, Task 15, and `scripts/run_local.sh`, Task 19).

- [ ] **Step 1: Write `build_lammps_deepmd.sh`** (pinned mainline + deepmd-kit v3 plugin + EXTRA-PAIR):

```bash
#!/usr/bin/env bash
# Build a DeePMD-enabled LAMMPS. Idempotent: re-running skips completed stages.
# Usage: PREFIX=$HOME/opt TARGET=workstation ./build_lammps_deepmd.sh
#   TARGET: workstation | cx3 | hx1   (selects module preamble)
set -euo pipefail
PREFIX="${PREFIX:-$HOME/opt}"
TARGET="${TARGET:-workstation}"
LAMMPS_TAG="stable_29Aug2024_update3"   # pinned; d3 damping tokens golden-tested against this tag
DEEPMD_VER="3.0.2"                       # deepmd-kit v3 (DPA support)

case "$TARGET" in
  cx3) module purge; module load tools/prod GCC/12.3.0 CUDA/12.4.0 CMake/3.26.3-GCCcore-12.3.0 ;;
  hx1) module purge; module load GCC/12.3.0 CUDA/12.4.0 CMake/3.26.3 ;;
  workstation) : ;;  # assumes system gcc, cuda toolkit, cmake >= 3.20 on PATH
esac

mkdir -p "$PREFIX" && cd "$PREFIX"

# 1. Python env with deepmd-kit (provides libdeepmd + lmp plugin mode)
if [ ! -d "$PREFIX/deepmd-venv" ]; then
  python3 -m venv "$PREFIX/deepmd-venv"
  "$PREFIX/deepmd-venv/bin/pip" install --upgrade pip
  # torch backend build (DPA models are PyTorch); DP_ENABLE_PYTORCH needed for custom OPs
  DP_ENABLE_PYTORCH=1 "$PREFIX/deepmd-venv/bin/pip" install "deepmd-kit[torch]==${DEEPMD_VER}"
fi
source "$PREFIX/deepmd-venv/bin/activate"

# 2. LAMMPS source (pinned)
if [ ! -d "$PREFIX/lammps-deepmd/src" ]; then
  git clone --depth 1 --branch "$LAMMPS_TAG" https://github.com/lammps/lammps.git "$PREFIX/lammps-deepmd/src"
fi

# 3. Configure + build with the deepmd plugin and EXTRA-PAIR (dispersion/d3)
cmake -S "$PREFIX/lammps-deepmd/src/cmake" -B "$PREFIX/lammps-deepmd/build" \
  -D CMAKE_INSTALL_PREFIX="$PREFIX/lammps-deepmd" \
  -D PKG_EXTRA-PAIR=ON -D PKG_MANYBODY=ON -D PKG_KSPACE=ON -D PKG_MOLECULE=ON \
  -D BUILD_MPI=ON -D LAMMPS_INSTALL_RPATH=ON
cmake --build "$PREFIX/lammps-deepmd/build" -j"$(nproc)"
cmake --install "$PREFIX/lammps-deepmd/build"

# 4. Runtime: deepmd is loaded as a plugin — set for every run:
#    export LAMMPS_PLUGIN_PATH=$(python -c "import deepmd.lmp,os;print(os.path.dirname(deepmd.lmp.__file__))")
echo "OK: $PREFIX/lammps-deepmd/bin/lmp"
```

- [ ] **Step 2: Write `build_lammps_mace.sh`** (ACEsuit fork + libtorch + Kokkos-CUDA + EXTRA-PAIR + ML-IAP for the Milestone-0 stretch experiment):

```bash
#!/usr/bin/env bash
# Build a MACE-enabled LAMMPS (ACEsuit fork, Kokkos GPU) with EXTRA-PAIR (d3)
# and ML-IAP (+ML-SNAP+PYTHON) for the mliap hybrid experiment.
# Usage: PREFIX=$HOME/opt TARGET=workstation GPU_ARCH=AMPERE86 ./build_lammps_mace.sh
#   GPU_ARCH: Kokkos arch of the card, e.g. AMPERE86 (RTX30xx), ADA89 (L40S), AMPERE80 (A100)
set -euo pipefail
PREFIX="${PREFIX:-$HOME/opt}"
TARGET="${TARGET:-workstation}"
GPU_ARCH="${GPU_ARCH:-AMPERE86}"

case "$TARGET" in
  cx3) module purge; module load tools/prod GCC/12.3.0 CUDA/12.4.0 CMake/3.26.3-GCCcore-12.3.0 ;;
  hx1) module purge; module load GCC/12.3.0 CUDA/12.4.0 CMake/3.26.3 ;;
  workstation) : ;;
esac

mkdir -p "$PREFIX" && cd "$PREFIX"

# 1. libtorch (CUDA build) — MACE pair style links against it
if [ ! -d "$PREFIX/libtorch" ]; then
  curl -L -o libtorch.zip https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcu124.zip
  unzip -q libtorch.zip -d "$PREFIX" && rm libtorch.zip
fi

# 2. ACEsuit LAMMPS fork (mace branch)
if [ ! -d "$PREFIX/lammps-mace/srcdir" ]; then
  git clone --depth 1 --branch mace https://github.com/ACEsuit/lammps.git "$PREFIX/lammps-mace/srcdir"
fi

# 3. Python env for model conversion + mliappy
if [ ! -d "$PREFIX/mace-venv" ]; then
  python3 -m venv "$PREFIX/mace-venv"
  "$PREFIX/mace-venv/bin/pip" install --upgrade pip mace-torch cython numpy
fi
source "$PREFIX/mace-venv/bin/activate"

cmake -S "$PREFIX/lammps-mace/srcdir/cmake" -B "$PREFIX/lammps-mace/build" \
  -D CMAKE_PREFIX_PATH="$PREFIX/libtorch" \
  -D PKG_ML-MACE=ON -D PKG_EXTRA-PAIR=ON -D PKG_MANYBODY=ON -D PKG_MOLECULE=ON \
  -D PKG_ML-IAP=ON -D PKG_ML-SNAP=ON -D PKG_PYTHON=ON -D MLIAP_ENABLE_PYTHON=ON \
  -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D "Kokkos_ARCH_${GPU_ARCH}=ON" \
  -D BUILD_MPI=ON
cmake --build "$PREFIX/lammps-mace/build" -j"$(nproc)"
echo "OK: $PREFIX/lammps-mace/build/lmp"
```

- [ ] **Step 3: Write `scripts/hpc/README.md`** — one paragraph per script: usage line, TARGET values, where the binary lands, the `LAMMPS_PLUGIN_PATH` runtime export for deepmd, GPU_ARCH table (L40S=ADA89, A100=AMPERE80, RTX30xx=AMPERE86), and the note that d3 syntax is golden-tested per build (spec §1.2).

- [ ] **Step 4: `chmod +x scripts/hpc/*.sh`; commit**

```bash
git add scripts/hpc/ && git commit -m "Add pinned LAMMPS build scripts for DeePMD and MACE stacks"
```

### Task 2: Milestone 0 decks + runbook

**Files:**
- Create: `scripts/milestone0/m0_1_deepmd_multi_hybrid.in`
- Create: `scripts/milestone0/m0_2_deepmd_overlay_d3.in`
- Create: `scripts/milestone0/m0_3_mace_fullcell.in` and `m0_3b_mace_overlay_d3.in`
- Create: `scripts/milestone0/m0_4_virtual_exclude.in`
- Create: `scripts/milestone0/m0_6_mliap_hybrid.in` (stretch)
- Create: `scripts/milestone0/make_geometries.py`
- Create: `documentation/milestone0.md` (runbook)

**Interfaces:**
- Consumes: binaries from Task 1; a MoS2 model file (runbook documents download/conversion).
- Produces: PASS/FAIL results that flip the spec's capability matrix from "Conditional" to Yes/No, and the verified `dispersion/d3` argument order used by Task 4's registry.

- [ ] **Step 1: Write `make_geometries.py`** — a standalone script (ASE only) that writes three tiny data files used by all decks: `bilayer_mos2.data` (2×2×2-layer MoS2, ~24 atoms), `trilayer_mos2.data`, and `bilayer_plus_virtual.data` (bilayer + one massless atom of an extra type 3 Å above). Use `ase.build.mx2` and `ase.io.write(..., format='lammps-data')`. Include exact cell vectors and a `--outdir` flag.

- [ ] **Step 2: Write deck `m0_1_deepmd_multi_hybrid.in`** (the setups-1/3 mechanism — two then three deepmd instances):

```text
units metal
atom_style atomic
read_data trilayer_mos2.data            # types 1-2 layer1, 3-4 layer2, 5-6 layer3 (Mo,S each)
pair_style hybrid deepmd frozen_model.pth deepmd frozen_model.pth deepmd frozen_model.pth lj/cut 11.0
pair_coeff * * deepmd 1 Mo S NULL NULL NULL NULL
pair_coeff * * deepmd 2 NULL NULL Mo S NULL NULL
pair_coeff * * deepmd 3 NULL NULL NULL NULL Mo S
pair_coeff 1*2 3*4 lj/cut 0.0224 3.13   # interlayer L1-L2 (UFF-ish placeholder values)
pair_coeff 3*4 5*6 lj/cut 0.0224 3.13
pair_coeff 1*2 5*6 lj/cut 1e-100 3.13
mass * 50.0
thermo 1
run 0
fix nve all nve
run 20
```

- [ ] **Step 3: Write deck `m0_2_deepmd_overlay_d3.in`** (full-cell + D3; known segfault reports — this is the genuinely open one):

```text
units metal
atom_style atomic
read_data bilayer_mos2.data
# NOTE: dispersion/d3 argument order below is the runbook's verification target —
# adjust to the pinned build's docs (https://docs.lammps.org/pair_dispersion_d3.html)
# and record the working form in documentation/milestone0.md results.
pair_style hybrid/overlay deepmd frozen_model.pth dispersion/d3 bj pbe 16.0 8.0
pair_coeff * * deepmd Mo S Mo S
pair_coeff * * dispersion/d3 Mo S Mo S
mass * 50.0
thermo 1
run 20
```

- [ ] **Step 4: Write decks `m0_3_mace_fullcell.in` / `m0_3b_mace_overlay_d3.in`** — same geometry; `pair_style mace no_domain_decomposition` with `pair_coeff * * mace mace.model-lammps.pt Mo S Mo S`; the `3b` variant wraps it in `hybrid/overlay` with the same `dispersion/d3` line as m0_2. Run via `lmp -k on g 1 -sf kk -in ...`.

- [ ] **Step 5: Write deck `m0_4_virtual_exclude.in`** (spec §1.4 mechanism — the go/no-go for keeping virtual_atom):

```text
units metal
atom_style atomic
read_data bilayer_plus_virtual.data     # type 5 = virtual atom, no neighbors wanted
pair_style mace no_domain_decomposition
pair_coeff * * mace mace.model-lammps.pt Mo S Mo S Mo   # virtual type mapped to placeholder Mo
mass 5 1.0
neigh_modify exclude type 5 1
neigh_modify exclude type 5 2
neigh_modify exclude type 5 3
neigh_modify exclude type 5 4
neigh_modify exclude type 5 5
thermo 1
run 0
# PASS criteria (runbook): potential energy differs from the no-virtual run by a
# CONSTANT (isolated-atom energy); per-atom force on type 5 == 0; repeat deck with
# deepmd and with dispersion/d3 overlay.
run 20
```

- [ ] **Step 6: Write deck `m0_6_mliap_hybrid.in`** (stretch — MACE via ML-IAP as hybrid sub-style): `pair_style hybrid mliap unified mace-mliap_lammps.pt 0 lj/cut 11.0` with NULL-mapped coeff lines mirroring m0_1's layout; header comment: "undocumented territory — any failure mode is a valid FAIL result".

- [ ] **Step 7: Write `documentation/milestone0.md`** — for each item M0.1–M0.7 (numbering matches spec §7): objective, exact commands (binary path, env exports incl. `LAMMPS_PLUGIN_PATH`, `-k on g 1 -sf kk` for MACE), PASS criteria (runs 20 steps, finite energy/forces; M0.4's constant-energy/zero-force criteria; M0.5 = framework-level smd/fix_move goldens, cross-reference Task 10/12 tests), a results table to fill in (`| item | build | result | notes |`), and the instruction: *on completion, update the capability matrix in the spec and the `supports_*` flags in `src/core/mlip.py`, and record the verified `dispersion/d3` argument order in the registry (Task 4)*. Also document model acquisition here: `mace-mp-0b3-medium` download + `mace_create_lammps_model` conversion; DPA model download + `dp --pt freeze` per-head command; `mace_create_lammps_model --format=mliap` for M0.6.

- [ ] **Step 8: Commit**

```bash
git add scripts/milestone0/ documentation/milestone0.md
git commit -m "Add Milestone 0 feasibility decks and runbook"
```

---

## Phase 1 — Registry and crash fixes (TDD, no GPU)

### Task 3: `count_atomtypes` binary-model short-circuit

**Files:**
- Modify: `src/core/utils.py:188-233` (and the extension set used by the second call site ~line 826)
- Test: `tests/test_mlip_registry.py` (new file, first test lands here)

**Interfaces:**
- Produces: `count_atomtypes(path, elements, pot_type=...)` returns `{el: 1 for el in elements}` for MLIP pot_types/extensions without opening the file.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mlip_registry.py
from pathlib import Path
from src.core.utils import count_atomtypes

def test_count_atomtypes_skips_binary_mlip_models(tmp_path):
    model = tmp_path / "mos2.model-lammps.pt"
    model.write_bytes(b"\x80\x02\x00binary-not-utf8\xff")
    for pot_type in ("mace", "deepmd"):
        assert count_atomtypes(str(model), ["Mo", "S"], pot_type=pot_type) == {"Mo": 1, "S": 1}

def test_count_atomtypes_skips_by_extension_without_pot_type(tmp_path):
    for ext in (".pt", ".pth", ".pb"):
        model = tmp_path / f"m{ext}"
        model.write_bytes(b"\xff\xfe")
        assert count_atomtypes(str(model), ["Mo", "S"]) == {"Mo": 1, "S": 1}
```

- [ ] **Step 2: Run to verify failure:** `pytest tests/test_mlip_registry.py -v` → FAIL (`UnicodeDecodeError` or wrong counts).
- [ ] **Step 3: Implement** — in `utils.py`, add `'mace', 'deepmd'` to the `single_type_pots` set and `'.pt', '.pth', '.pb'` to `single_type_exts` (the multi-suffix `.model-lammps.pt` already ends in `.pt`). Both call sites go through these sets — verify the second (≈line 826) uses `single_type_exts`.
- [ ] **Step 4: Run:** `pytest tests/test_mlip_registry.py tests/test_utils.py -v` → PASS (and no regressions).
- [ ] **Step 5: Commit:** `git commit -am "Short-circuit count_atomtypes for binary MLIP model files"`

### Task 4: `src/core/mlip.py` registry

**Files:**
- Create: `src/core/mlip.py`
- Test: `tests/test_mlip_registry.py` (extend)

**Interfaces (produced — used by every later task):**

```python
def is_mlip(pot_type: str) -> bool
def get_mlip(pot_type: str) -> MLIPSpec           # KeyError -> ValueError with hint
MLIP_REGISTRY: Dict[str, MLIPSpec]                # keys: 'mace', 'deepmd'

@dataclass(frozen=True)
class LauncherSpec:
    binary_key: str          # 'mace' | 'deepmd' — resolved to a path by the HPC profile
    mpi_ranks: int           # mace: 1; deepmd: 1 (single-GPU design point)
    lmp_flags: str           # mace: '-k on g 1 -sf kk'; deepmd: ''
    env_lines: tuple         # deepmd: ('export LAMMPS_PLUGIN_PATH=...',)

@dataclass(frozen=True)
class MLIPSpec:
    name: str
    model_in_style_line: bool        # deepmd True, mace False
    supports_null: bool              # deepmd True, mace False
    supports_overlay: bool           # both True until M0 says otherwise
    file_extensions: tuple           # mace: ('.model-lammps.pt', '.pt'); deepmd: ('.pb', '.pth')
    virtual_atom_element: str        # 'Mo' placeholder (any element in the model works)
    launcher: LauncherSpec
    def style_token(self, model_path: str) -> str
        # deepmd -> f"deepmd {model_path}"    mace -> "mace no_domain_decomposition"
    def coeff_line(self, index_str: str, model_path: str, atom_list: list[str]) -> str
        # deepmd -> f"pair_coeff * * deepmd{index_str} {' '.join(atom_list)}"
        # mace   -> f"pair_coeff * * mace {model_path} {' '.join(atom_list)}"

def d3_style_token(settings) -> str
    # f"dispersion/d3 {settings.potential.d3_damping} {settings.potential.d3_functional} "
    # f"{settings.potential.d3_cutoff} {settings.potential.d3_cn_cutoff}"
    # ONE function = one place to fix argument order after Milestone 0 verifies it.
def d3_coeff_line(elements_per_type: list[str]) -> str
    # f"pair_coeff * * dispersion/d3 {' '.join(elements_per_type)}"  — a REAL element for
    # every type (no NULL); caller supplies the virtual placeholder already substituted.
```

- [ ] **Step 1: Write failing tests** — `test_is_mlip`, `test_style_token_per_framework`, `test_coeff_line_mace_places_model_path`, `test_coeff_line_deepmd_places_index`, `test_d3_lines`, `test_get_mlip_unknown_raises`. Example:

```python
from src.core.mlip import is_mlip, get_mlip, d3_coeff_line

def test_style_token_per_framework():
    assert get_mlip("deepmd").style_token("m.pth") == "deepmd m.pth"
    assert get_mlip("mace").style_token("m.pt") == "mace no_domain_decomposition"

def test_coeff_line_mace_places_model_path():
    line = get_mlip("mace").coeff_line("", "m.pt", ["Mo", "S", "NULL"])
    assert line == "pair_coeff * * mace m.pt Mo S NULL"

def test_coeff_line_deepmd_places_index():
    line = get_mlip("deepmd").coeff_line(" 2", "m.pth", ["NULL", "Mo", "S"])
    assert line == "pair_coeff * * deepmd 2 NULL Mo S"
```

- [ ] **Step 2: Run → FAIL** (module missing). **Step 3: Implement `src/core/mlip.py`** exactly per the interface block. **Step 4: Run → PASS.**
- [ ] **Step 5: Commit:** `git commit -am "Add MLIP registry (MACE/DeePMD descriptors, D3 emission)"`

### Task 5: D3 settings model

**Files:**
- Modify: `src/core/config.py:101-113` (`PotentialSettings`)
- Test: `tests/test_config.py` (extend)

**Interfaces:**
- Produces: `settings.potential.d3_functional: str = 'pbe'`, `d3_damping: Literal['original','bj','zerom','bjm'] = 'bj'`, `d3_cutoff: float = 16.0`, `d3_cn_cutoff: float = 8.0`.

- [ ] **Step 1: Failing test** — defaults exist; `d3_damping='d3bj'` raises `ValidationError`.
- [ ] **Step 2–4:** Run/implement (four fields on `PotentialSettings`)/run.
- [ ] **Step 5: Commit:** `git commit -am "Add D3 dispersion settings with damping enum"`

---

## Phase 2 — PotentialManager integration

### Task 6: interlayer-mode plumbing (`is_sheet_lj` signature)

**Files:**
- Modify: `src/core/potential_manager.py` (`is_sheet_lj:568-584`, `register_component:426-476`)
- Modify: `src/builders/components.py:1810`, `src/builders/sheetonsheet.py:193,324`, `src/builders/afm.py:392`
- Test: `tests/test_potential_manager.py` (extend)

**Interfaces:**
- Produces: `is_sheet_lj(pot_type: str, interlayer_mode: Optional[str] = None) -> bool` — MLIP + `'native'` → `False`; MLIP + `'lj'`/`None` → `True`; non-MLIP unchanged. `register_component(..., interlayer_mode: Optional[str] = None)` stores the mode in `self.components[name]['interlayer_mode']` and uses it in the `potential_usage` accounting (`native` counts 1, `lj` counts `n_layers`).
- Consumes: `is_mlip` (Task 4).

- [ ] **Step 1: Failing tests**

```python
def test_is_sheet_lj_mlip_modes(mock_settings):
    pm = PotentialManager(mock_settings, use_langevin=False)
    assert pm.is_sheet_lj("deepmd", interlayer_mode="lj") is True
    assert pm.is_sheet_lj("deepmd", interlayer_mode="native") is False
    assert pm.is_sheet_lj("mace", interlayer_mode="native") is False
    assert pm.is_sheet_lj("sw") is True          # unchanged
    assert pm.is_sheet_lj("rebomos") is False    # unchanged
```

- [ ] **Step 2: Run → FAIL. Step 3: Implement** — add the kwarg; inside: `if is_mlip(pot_lower): return interlayer_mode != 'native'` before the existing set checks. In `register_component`, thread `interlayer_mode` into the usage accounting and the stored component dict. Update the four builder call sites to pass the sheet's mode (builders read it in Task 9; until then they pass `None`, preserving behavior).
- [ ] **Step 4: `pytest tests/test_potential_manager.py -v` → PASS** (all pre-existing tests must pass unmodified — that is the no-regression proof).
- [ ] **Step 5: Commit:** `git commit -am "Thread interlayer mode through is_sheet_lj and registration"`

### Task 7: MLIP emission in PotentialManager

**Files:**
- Modify: `src/core/potential_manager.py` (`add_self_interaction:586-617`, `_add_layer_self_interaction:619-642`, `write_file:999-1068`)
- Test: `tests/test_potential_manager.py` (extend)

**Interfaces:**
- Produces (behavior): for an MLIP component, `add_self_interaction` emits registry `coeff_line`s (per-layer NULL maps for `lj` mode, one whole-component map for `native`); `write_file` builds style tokens via `spec.style_token(model_path)` (generalizing the ReaxFF branch), switches to `pair_style hybrid/overlay` when `self.d3_enabled`, appends the D3 style token + `d3_coeff_line`, substitutes `virtual_atom_element` for the Virtual type in full-cell MLIP/D3 coeff lines, and emits `neigh_modify exclude type {v} *` lines when a virtual atom coexists with MLIP or D3.
- Produces (API): `pm.enable_d3()` sets `self.d3_enabled = True` (builders call it in Task 10).
- Consumes: Tasks 4–6.

- [ ] **Step 1: Failing golden-block tests** (the heart of the test suite — assert exact emitted text). One per mechanism:

```python
def _fake_model(tmp_path, name):
    p = tmp_path / name; p.write_bytes(b"\x00"); return str(p)

def test_deepmd_per_layer_emission(mock_settings, tmp_path):
    """Setup-1 mechanism: 3 layers -> 3 indexed deepmd instances with NULL maps."""
    pm = PotentialManager(mock_settings, use_langevin=False)
    cfg = make_component_config(pot_type="deepmd", pot_path=_fake_model(tmp_path, "m.pth"))
    pm.register_component("sheet", cfg, n_layers=3, interlayer_mode="lj")
    pm.add_self_interaction("sheet")
    pm.add_interlayer_interaction("sheet")
    text = write_and_read(pm, tmp_path)
    assert "pair_style hybrid deepmd" in text and text.count("deepmd") >= 4  # 3 style tokens + lj lines
    assert "pair_coeff * * deepmd 1 Mo S NULL NULL NULL NULL" in text
    assert "pair_coeff * * deepmd 3 NULL NULL NULL NULL Mo S" in text

def test_mace_fullcell_with_d3_and_virtual(mock_settings, tmp_path):
    """Setup-2+D3 mechanism incl. virtual-atom placeholder + exclusion."""
    pm = PotentialManager(mock_settings, use_langevin=False)
    cfg = make_component_config(pot_type="mace", pot_path=_fake_model(tmp_path, "m.model-lammps.pt"))
    pm.register_component("sheet", cfg, n_layers=3, interlayer_mode="native")
    v = pm.register_virtual_atom()
    pm.add_self_interaction("sheet")
    pm.enable_d3()
    text = write_and_read(pm, tmp_path)
    assert "pair_style hybrid/overlay mace no_domain_decomposition dispersion/d3 bj pbe 16.0 8.0" in text
    assert f"neigh_modify exclude type {v} *" in text
    # virtual type gets the placeholder element in BOTH mace and d3 coeff lines
    assert text.count(" Mo S Mo S Mo S Mo") >= 2
```

(`make_component_config` / `write_and_read` are 5-line test helpers defined at the top of the test file — write them, don't import them.)

- [ ] **Step 2: Run → FAIL. Step 3: Implement** in three moves: (a) `add_self_interaction`/`_add_layer_self_interaction` branch on `is_mlip` → build `atom_list` via existing `build_null_map`, emit `spec.coeff_line(index_str, self._get_potential_path(c_conf.pot_path), atom_list)`; for `native` mode treat like the non-layered path (one map covering all layers). (b) `write_file` style-parts loop: MLIP entries contribute `spec.style_token(staged_path)` (deepmd carries the model path in the style line — stage once, reuse the string); keep the ReaxFF branch as-is. (c) D3/virtual: when `d3_enabled`, use `hybrid/overlay`, append `d3_style_token(self.settings)` and a `d3_coeff_line` built from each type's element with `spec.virtual_atom_element` substituted for `Virtual`; same substitution in full-cell MLIP coeff lines; emit exclusion lines after the group section.
- [ ] **Step 4: Run full `pytest tests/test_potential_manager.py -v` → PASS** (existing SW/ReaxFF goldens byte-identical).
- [ ] **Step 5: Commit:** `git commit -am "Emit MLIP hybrid blocks, D3 overlay, and virtual-atom exclusion"`

### Task 8: shared model staging

**Files:**
- Modify: `src/core/potential_manager.py:545-566` (`_get_potential_path`)
- Test: `tests/test_potential_manager.py` (extend)

**Interfaces:**
- Produces: `PotentialManager(..., shared_potentials_dir=..., shared_potentials_prefix=...)` — MLIP model files copy to the shared dir (simulation root) once; classical potentials keep per-run behavior. Builders pass `output_root / 'potentials'` (Task 10/12).

- [ ] **Step 1: Failing test:** register the same fake model in two managers pointed at the same shared dir → exactly one copy on disk; emitted path uses the shared prefix; an SW file still lands in the per-run `provenance/potentials`.
- [ ] **Step 2–4:** Run/implement (branch on `is_mlip` + extension inside `_get_potential_path`)/run.
- [ ] **Step 5: Commit:** `git commit -am "Stage MLIP models once at the simulation root"`

---

## Phase 3 — Config and builders

### Task 9: config surface + validation

**Files:**
- Modify: `src/core/config.py:330-346` (`SheetConfig`) and `ComponentConfig:204-217`
- Test: `tests/test_config.py` (extend)

**Interfaces:**
- Produces: `SheetConfig.mlip_interlayer: Literal['lj','native'] = 'lj'`; `SheetConfig.dispersion: Literal['none','d3'] = 'none'`; a `model_validator` enforcing every §1.1 rule:
  - `mlip_interlayer`/`dispersion` non-default ⇒ `is_mlip(pot_type)`；
  - MLIP ⇒ `lat_c` required; model extension ∈ `spec.file_extensions`;
  - `pot_type='mace'` ⇒ `mlip_interlayer='native'`;
  - `dispersion='d3'` ⇒ `mlip_interlayer='native'`.
- Cell-level rules that need sibling components (D3/MACE forbid SW tip/sub in the same cell) live in the builders' `validate_config` (Task 12), not the Pydantic model.

- [ ] **Step 1: Failing tests** — one per rule, e.g.:

```python
def test_mace_requires_native_interlayer(sheet_kwargs):
    with pytest.raises(ValidationError, match="full-cell"):
        SheetConfig(**{**sheet_kwargs, "pot_type": "mace",
                       "pot_path": "m.model-lammps.pt", "mlip_interlayer": "lj", "lat_c": 3.16})

def test_mlip_requires_lat_c(sheet_kwargs):
    with pytest.raises(ValidationError, match="lat_c"):
        SheetConfig(**{**sheet_kwargs, "pot_type": "deepmd", "pot_path": "m.pth"})
```

- [ ] **Step 2–4:** Run/implement/run. **Step 5: Commit:** `git commit -am "Add MLIP config keys with capability-matrix validation"`

### Task 10: sheetonsheet routing (setups 1 & 2)

**Files:**
- Modify: `src/builders/sheetonsheet.py:163-215` (`_generate_potentials`), `:324` (`has_internal_lj`)
- Test: `tests/test_sheetonsheet_mlip.py` (new)

**Interfaces:**
- Consumes: Tasks 6–9. Produces: end-to-end golden tests through the builder for setup 1 (`deepmd`+`lj`) and setup 2 (`deepmd`/`mace`+`native`, ±d3).

- [ ] **Step 1: Failing tests** — build a full `SheetOnSheetSimulation` with `mock_settings` + a temp `.ini`-equivalent config (follow the existing pattern in `tests/test_sheetonsheet.py` for constructing configs); assert on the written `system.in.settings`: setup 1 contains three indexed `deepmd` maps + interlayer `lj/cut` lines; setup 2 contains ONE un-indexed map spanning all layers and **no** interlayer lj lines; `native`+`dispersion=d3` produces the `hybrid/overlay ... dispersion/d3` header. Also: an all-SW config's settings file is byte-identical to a golden captured before this task (write the capture as the first test).
- [ ] **Step 2: Run → FAIL. Step 3: Implement:** in `_generate_potentials`, pass `interlayer_mode=self.config.sheet.mlip_interlayer` into `register_component` and both `is_sheet_lj` calls; call `pm.enable_d3()` when `self.config.sheet.dispersion == 'd3'`; route `native` to skip the interlayer-LJ branch (it already keys off `has_internal_lj`, which Task 6 made mode-aware — verify, don't assume).
- [ ] **Step 4: Run → PASS. Step 5: Commit:** `git commit -am "Route MLIP interlayer modes through the sheetonsheet builder"`

### Task 11: `pair_style zero` surrogate stacking

**Files:**
- Modify: `src/builders/components.py:1790-1909` (`stack_multilayer_sheet` and the pot-file write around :1805-1813)
- Test: `tests/test_components_mlip.py` (new)

**Interfaces:**
- Produces: when `is_mlip(config.pot_type)`, the stacking potential file contains exactly:

```text
pair_style zero 10.0
pair_coeff * *
```

plus the existing mass lines — never the MLIP style (generation machines have no MLIP-enabled python-lammps).

- [ ] **Step 1: Failing test** — call the pot-file-writing path with a deepmd config and assert the surrogate content; assert an SW config still writes the SW include (regression).
- [ ] **Step 2–4:** Run/implement/run. Note: MLIP configs always have `lat_c` (Task 9), so the minimization branch never runs — add `assert not calculate_lat_c` guard with a clear error for belt-and-braces.
- [ ] **Step 5: Commit:** `git commit -am "Stack MLIP sheets under a pair_style zero surrogate"`

### Task 12: afm routing (setups 3 & 4) + cell-level validation

**Files:**
- Modify: `src/builders/afm.py` (`_generate_potentials` region around :380-430; `validate_config` if present, else the builder's `__init__` validation block)
- Test: `tests/test_afm_mlip.py` (new)

**Interfaces:**
- Produces: setup-3/4 goldens (sheet deepmd `lj`/`native`; tip+sub SW; cross-component `lj/cut` lines unchanged); cell-level validation errors: MLIP on tip/sub in a hybrid cell; `dispersion='d3'` in any cell containing SW components; `pot_type='mace'` in a cell containing SW components; `settings.simulation.constraint_mode in ('atom_bonds','com_spring')` with an MLIP sheet (spec §3.4 — molecular conversion × MLIP neighbor lists is unvalidated). The constraint_mode rule applies in the sheetonsheet builder too — add the same check + test there.

- [ ] **Step 1: Failing tests** — mirror Task 10's pattern with tip/sub present: setup 3 asserts per-layer deepmd maps AND untouched `sw` self-interaction lines for tip/sub AND sheet↔tip `lj/cut` cross lines; setup 4 asserts a single NULL-partitioned deepmd map covering all sheet types with tip/sub types NULL. Validation tests assert the three rejection rules raise `ValueError` with messages naming the offending rule.
- [ ] **Step 2–4:** Run/implement/run (same `interlayer_mode`/`enable_d3` threading as Task 10 + the three explicit checks before registration).
- [ ] **Step 5: Commit:** `git commit -am "Route MLIP sheet modes through the AFM builder with cell-level validation"`

### Task 13: output-dir flavor token

**Files:**
- Modify: `src/core/run.py:1127-1165` (`output_dir_for`)
- Test: `tests/test_run_naming.py` (new or extend existing naming tests)

**Interfaces:**
- Produces: for MLIP configs the innermost run-parameter segment (the `K<temp>...` leaf) gains `_{pot_type}-{mlip_interlayer}-{'base' if dispersion=='none' else 'd3'}`; non-MLIP paths unchanged.

- [ ] **Step 1: Failing tests** — `..._deepmd-lj-base`, `..._mace-native-d3` appear in the leaf segment; an SW config's path is byte-identical to today (capture golden); a postprocessing filter test: feed a tokened path through the existing size/material segment parsing used by `postprocessing/read_data` filters and assert extraction still works.
- [ ] **Step 2–4:** Run/implement/run. **Step 5: Commit:** `git commit -am "Encode MLIP flavor in run directory names"`

### Task 14: builder-family generalization + hetero Virtual fix

**Files:**
- Modify: `src/builders/hetero.py:323,425` (Virtual registration gated on `drive_method == 'virtual_atom'`, matching `sheetonsheet._register_drive_atom:217-226`)
- Test: `tests/test_family_mlip.py` (new)

**Interfaces:**
- Produces: one golden config per family proving MLIP flows through: pes-sheet (full-cell mace — static, no drive atom), pes-tip (deepmd native sheet), indent (deepmd `lj`, setup-3 analogue), hetero (full-cell deepmd across both materials), flake (deepmd native). Since pes/indent/flake subclass the Task-10/12 builders, this is wiring verification, not new mechanism.

- [ ] **Step 1: Failing tests** — for each family, build the smallest valid config with an MLIP sheet and assert (a) generation completes, (b) the settings file contains the expected MLIP style token, (c) for hetero: NO Virtual type is registered when `drive_method='smd'` (this is the bug fix — write this test first and watch it fail against current code).
- [ ] **Step 2–4:** Run/implement/run. The hetero fix is the only code change expected; if a family fails for another reason, fix the minimal wiring (validation added in Task 9/12 must not over-reject full-cell hetero).
- [ ] **Step 5: Commit:** `git commit -am "Enable MLIP across builder families; gate hetero virtual atom on drive method"`

---

## Phase 4 — HPC layer

### Task 15: profile + GPU settings models

**Files:**
- Modify: `src/core/config.py:115-137` (`HPCSettings`)
- Test: `tests/test_config.py` (extend)

**Interfaces:**
- Produces:

```python
class GPUSettings(BaseModel):
    ngpus: int = 1
    gpu_type: Optional[str] = None      # 'L40S' | 'A100' | 'RTX6000' | None

class FrameworkOverlay(BaseModel):
    binary: str                          # path to lmp
    modules: List[str] = []
    env_lines: List[str] = []

class HPCProfile(BaseModel):             # every field optional — partial override
    queue: Optional[str] = None
    gpu: Optional[GPUSettings] = None
    num_cpus: Optional[int] = None
    memory_gb: Optional[int] = None
    walltime_hours: Optional[int] = None
    modules: Optional[List[str]] = None
    env_lines: Optional[List[str]] = None
    mpi_command: Optional[str] = None
    frameworks: Dict[str, FrameworkOverlay] = {}

# on HPCSettings:
profiles: Dict[str, HPCProfile] = {}
active_profile: Optional[str] = None
env_lines: List[str] = []
gpu: Optional[GPUSettings] = None

def resolve_profile(hpc: HPCSettings, name: Optional[str]) -> HPCSettings
    # returns a merged copy: base <- profile (non-None fields win); unknown name -> ValueError
```

- [ ] **Step 1: Failing tests** — merge semantics (profile overrides only what it sets), unknown profile raises, no-profile returns base unchanged (equality).
- [ ] **Step 2–4:** Run/implement (`resolve_profile` in `src/hpc/scripts.py` or `config.py` — put it next to `HPCSettings`)/run.
- [ ] **Step 5: Commit:** `git commit -am "Add HPC profiles with GPU and per-framework settings"`

### Task 16: JobEntry resource fields + array splitting

**Files:**
- Modify: `src/hpc/manifest.py:24-60` (`JobEntry`), `src/hpc/scripts.py` (`generate_combined_scripts:353-404`, `_generate_array_scripts:234-284`)
- Test: `tests/test_hpc_scripts.py` (extend)

**Interfaces:**
- Produces: `JobEntry.needs_gpu: bool = False`, `framework: Optional[str] = None`, `launcher_flags: str = ""` (dataclass defaults ⇒ old manifest JSONs still load). `HPCScriptGenerator` groups sim dirs by `(needs_gpu, framework)` and emits one script + manifest per group, suffixed `run_cpu.pbs`, `run_gpu_deepmd.pbs`, `run_gpu_mace.pbs`; single-group campaigns keep today's filenames (byte-identical regression).

- [ ] **Step 1: Failing tests** — a manifest with 2 CPU jobs + 2 deepmd jobs yields two scripts with correct manifests and select lines; an all-CPU manifest yields exactly today's `run.pbs` (golden capture before implementing).
- [ ] **Step 2–4:** Run/implement/run. **Step 5: Commit:** `git commit -am "Split HPC arrays by GPU need and framework"`

### Task 17: GPU select line + launch command in templates

**Files:**
- Modify: `src/hpc/scripts.py:138-167` (`HPCConfig.to_dict`), all four templates in `src/templates/hpc/`
- Test: `tests/test_hpc_scripts.py` (extend)

**Interfaces:**
- Produces: `to_dict()` gains `launch_command` (resolved `{mpi_command} {binary} {lmp_flags}`; default binary `lmp`) and `env_lines`; GPU-enabled configs render `select=1:ncpus={n}:mem={m}gb:mpiprocs={mpi_ranks}:ngpus={g}[:gpu_type={t}]`. Templates replace `{{ mpi_command }} lmp {{ lmp_flags }}` with `{{ launch_command }}` and render `{% for line in env_lines %}` after module loads.

- [ ] **Step 1: Failing tests** — GPU select string exact match (incl. `mpiprocs=1`); env_lines rendered in order; **CPU regression: render every template with a no-profile config and diff against goldens captured from current `main` — must be byte-identical.**
- [ ] **Step 2–4:** Run/implement/run. **Step 5: Commit:** `git commit -am "Render GPU select lines and resolved launch commands"`

### Task 18: CLI `--profile`

**Files:**
- Modify: `src/cli.py` (`hpc generate` command group)
- Test: `tests/test_cli.py` (extend, CliRunner pattern already used there)

**Interfaces:**
- Produces: `FrictionSim2D hpc generate <dir> --profile cx3` → `resolve_profile` applied before `HPCConfig.from_settings`; `--profile` unknown → clean click error.

- [ ] **Steps 1–4:** failing CliRunner test (`--profile cx3` changes queue in rendered script; unknown profile exits non-zero) → implement (one `click.option`) → pass.
- [ ] **Step 5: Commit:** `git commit -am "Add --profile to hpc generate"`

---

## Phase 5 — Deliverables

### Task 19: `scripts/run_local.sh`

**Files:** Create `scripts/run_local.sh`.
**Interfaces:** `./scripts/run_local.sh <sim_dir> <framework: sw|deepmd|mace>` — reads `settings.yaml` profile `frameworks:` via `python -c` + PyYAML for the binary/env, falls back to `lmp` on PATH for `sw`; runs `system.in` then `slide.in` from the correct working directory (the directory containing `afm/`/`sheetonsheet/` — mirror the prefix-stripping logic from `pbs_combined.j2:38-49`).

- [ ] **Step 1: Write the script** (bash, `set -euo pipefail`, MACE adds `-k on g 1 -sf kk`, deepmd exports `LAMMPS_PLUGIN_PATH`). **Step 2: Shellcheck-by-eye + `bash -n`. Step 3: Commit:** `git commit -am "Add local MLIP run helper"`

### Task 20: bilayer benchmark script

**Files:** Create `scripts/mlip_bilayer_benchmark.py`; Test: `tests/test_benchmark_script.py` (parsing/geometry only — no LAMMPS).
**Interfaces:** `python scripts/mlip_bilayer_benchmark.py --framework deepmd --model m.pth --lmp <binary> [--d3] [--out results.json]` — generates bilayer MoS2 decks over interlayer separations 5.5–9.0 Å (0.1 Å steps, rigid single points), invokes the given binary per point (subprocess), parses `Potential energy` from the log, writes JSON: `{separation: energy}`, equilibrium separation, binding energy/area, in-plane `lat_c` from a 3-point quadratic fit of scaled-cell single points, and `nvidia-smi --query-gpu=memory.used` sampled per run (VRAM vs atoms table for spec §3.2). SW+LJ reference mode: `--framework sw --pot <file.sw>`.

- [ ] **Step 1: Failing tests** for the pure functions: deck text generation (golden), log-energy parsing, quadratic-minimum fit. **Steps 2–4:** implement/run. **Step 5: Commit:** `git commit -am "Add bilayer MoS2 binding benchmark"`

### Task 21: example configs

**Files:** Create `examples/mlip/sheet_mlip_lj.ini`, `sheet_mlip_full.ini`, `afm_mlip_lj.ini`, `afm_mlip_full.ini`, `pes_sheet_mlip.ini`, `hetero_mlip.ini`.
**Interfaces:** each is a minimal working config copied from the corresponding `examples/*.ini` with the sheet section switched to `pot_type = deepmd` (or `mace` in `sheet_mlip_full.ini`), `pot_path = ../models/mace-mp-0b3-medium.model-lammps.pt` (mace) or `../models/dpa3_mos2_frozen.pth` (deepmd — the names pinned in `documentation/mlip.md`), `mlip_interlayer`, `dispersion`, explicit `lat_c = 3.16`, and a header comment naming the setup number and its capability-matrix row.

- [ ] **Step 1: Write the six files. Step 2: Test:** extend `tests/test_config.py` with a loop that parses every `examples/mlip/*.ini` through the config loader and asserts validity (model files stubbed via `tmp_path` copy — follow the existing examples-validation test pattern if present, else write the loop). **Step 3: Run → PASS. Step 4: Commit:** `git commit -am "Add MLIP example configs for all setups"`

### Task 22: documentation

**Files:** Create `documentation/mlip.md`; Modify `documentation/README.md` (index line), `CLAUDE.md` + `AGENTS.md` (one line each: MLIP pot_types exist, spec/plan paths, `mace`/`deepmd` are third potential category).
**Interfaces:** `documentation/mlip.md` sections: capability matrix (copied from spec), model acquisition (exact download URLs + conversion commands, cross-ref `documentation/milestone0.md`), the six example configs and what each demonstrates, base-vs-D3 workflow, HPC profiles + gpu72 constraints (12-GPU user cap, L40S 48 GB default), local GPU-PC quickstart (build scripts → milestone 0 → benchmark → first setup-2 run), troubleshooting (MACE hybrid limits + ML-IAP status, deepmd model-deviation trap, dome-plow/lat_c workflow).

- [ ] **Step 1: Write the docs. Step 2: Verify every referenced path/command exists in the repo (`grep -o` the file for `scripts/\|examples/\|documentation/` and check each). Step 3: Commit:** `git commit -am "Document MLIP support"`

---

## Final verification

- [ ] `pytest` — full suite green.
- [ ] `git diff gui-p2..mlip --stat` — review that no unrelated files changed.
- [ ] Spec self-check: walk the spec's §1–§7 and confirm each maps to a completed task (§1→9/13, §1.4→7+M0.4, §2→4/7/8, §3→10–14, §4→15–18/19, §5→1/20/21/22, §6→tests throughout, §7→Task 2 artifacts).
- [ ] On the GPU machine: run `documentation/milestone0.md` top to bottom; update the spec's capability matrix + `mlip.py` `supports_*` flags + registry d3 argument order with the results; commit as "Record Milestone 0 results".
