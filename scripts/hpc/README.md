# LAMMPS build scripts for the MLIP stacks

MACE and DeePMD need **two separate, mutually incompatible LAMMPS builds**: MACE
requires LAMMPS >= 10 Sep 2025 with [wcwitt/symmetrix](https://github.com/wcwitt/symmetrix)
compiled in (Kokkos, no libtorch), DeePMD requires LAMMPS 29 Aug 2024 plus the
deepmd-kit v3 runtime plugin. The version requirements alone are disjoint, and
symmetrix is *compiled into* LAMMPS rather than loaded as a plugin, so neither
build can be retrofitted into the other. All
builds enable `EXTRA-PAIR`, but that alone is **not** enough for a DeePMD + D3
deck — `dispersion/d3` and the deepmd plugin want incompatible LAMMPS releases,
which is what `build_lammps_deepmd_d3.sh` exists to resolve. Use that binary for
anything combining the two.

Both scripts are idempotent: each stage is guarded by a directory check, so a
re-run after a failure resumes rather than starting over. Neither script needs
root; everything lands under `$PREFIX`.

## `build_lammps_deepmd.sh`

```bash
PREFIX=$HOME/opt TARGET=workstation ./build_lammps_deepmd.sh
```

- `TARGET`: `workstation` (default — assumes system gcc, CUDA toolkit and
  cmake ≥ 3.20 already on `PATH`), `cx3`, or `hx1`. The value only selects the
  `module load` preamble; any other value is rejected.
- Pins `LAMMPS_TAG=stable_22Jul2025_update1` and `DEEPMD_VER=3.0.2`
  (deepmd-kit v3 is the version that carries DPA support). **At that tag the
  deepmd plugin will not load** (ABI lock — see below); the build is useful for
  standalone `dispersion/d3` only. For DeePMD use the conda `mlip` env's LAMMPS,
  and for DeePMD + D3 use `build_lammps_deepmd_d3.sh`.
- Creates a venv at `$PREFIX/deepmd-venv` and installs `deepmd-kit[torch]`
  into it with `DP_ENABLE_PYTORCH=1` (the PyTorch custom OPs are what DPA
  models need).
- **Binary lands at `$PREFIX/lammps-deepmd/bin/lmp`** (via `cmake --install`);
  the build tree is `$PREFIX/lammps-deepmd/build`.

### Runtime export (required for every DeePMD run)

DeePMD is *not* compiled into this binary — it is loaded at runtime as a LAMMPS
plugin. Without the export below, `pair_style deepmd` is simply an unknown
style. Put these two lines in every job script / interactive shell:

```bash
source "$PREFIX/deepmd-venv/bin/activate"
export LAMMPS_PLUGIN_PATH=$(python -c "import deepmd,os;print(os.path.join(os.path.dirname(deepmd.__file__),'lib'))")
```

This is the string that later becomes the DeePMD `LauncherSpec.env_lines`
entry in `src/core/mlip.py`, and the `env_lines:` of the HPC profile's
`frameworks: deepmd:` overlay.

DeePMD runs need no special `lmp` flags (`mpi_ranks = 1` is the design point;
one GPU per job).

## `build_lammps_deepmd_d3.sh` — the backport (use this one for D3)

```bash
PREFIX=$HOME/opt TARGET=workstation ./build_lammps_deepmd_d3.sh
```

`build_lammps_deepmd.sh` above **cannot** give you `pair_style deepmd` and
`pair_style dispersion/d3` at the same time, whatever `LAMMPS_TAG` is set to:

- `dispersion/d3` entered LAMMPS in `stable_22Jul2025`; `stable_29Aug2024_update3`
  has zero d3 sources;
- deepmd-kit is ABI-locked to LAMMPS `2024.08.29` — the pip plugin fails on a
  2025 build with `undefined symbol: _ZN9LAMMPS_NS5utils11getsyserrorEv`, and
  conda-forge `deepmd-kit 3.1.3` hard-constrains `lammps 2024.08.29`.

So this script **backports** instead: it builds the ABI-matching 2024 release
with `pair_dispersion_d3.{cpp,h}` and `d3_parameters.h` copied in from
`stable_22Jul2025_update1`. Verified numerically — under `hybrid/overlay` each
sub-style reproduces its standalone energy bit-for-bit
([`documentation/milestone0.md`](../../documentation/milestone0.md), M0.2).

- **Binary lands at `$PREFIX/lammps-deepmd-d3/bin/lmp`.**
- Builds no venv: it expects an existing deepmd install to supply the runtime
  plugin (the `mlip` conda env on this workstation).
- Reuses `$PREFIX/lammps-deepmd/src` as the donor checkout when it is already
  at the donor tag, rather than cloning ~1 GB again for three files.
- Applies a small `sed` shim for two post-2024 symbols
  (`Error::NOLASTLINE`/`NOPOINTER`, `Info::get_pair_coeff_status`) used only in
  `error->all(...)` abort paths. The script hard-fails if the shim misses a
  site, so a silent half-patch cannot reach the compiler.

### `BUILD_MPI` defaults to **ON** here — deliberately

Opposite of `build_lammps_deepmd.sh`. The deepmd plugin leaves 20 `MPI_*`
symbols undefined and takes them from the host executable; a `BUILD_MPI=OFF`
LAMMPS satisfies MPI from `src/STUBS/mpi.cpp`, which is **C++**, so its symbols
are mangled and do not match the plugin's C ABI. Build against the same MPI the
plugin was built against. On `workstation` the script uses the `mlip` conda
env's MPICH via `MPI_WRAPPER`, with `MPICH_CXX=/usr/bin/g++` so the wrapper
compiles with the system toolchain — this also sidesteps the broken system
MPICH (whose CMake imported target names a deleted include dir) without needing
`CMAKE_DISABLE_FIND_PACKAGE_MPI`.

### Runtime exports

```bash
export LAMMPS_PLUGIN_PATH=/home/mv923/miniconda3/envs/mlip/lib/deepmd_lmp
export LD_LIBRARY_PATH=/home/mv923/miniconda3/envs/mlip/lib
```

Both are required. Miss either — or build without MPI — and the symptom is the
same misleading trio: `Loaded 0 plugins`, then
`ERROR: Unrecognized pair style 'deepmd'`, which reads like a wrong plugin
path. Only `Loaded 1 plugins` proves the plugin is live, so the script ends
with a `plugin list` probe deck that asserts on it (`lmp -h` cannot show a
runtime plugin).

## `build_lammps_mace.sh`

```bash
# login node (needs network): clone, patch, wire symmetrix into LAMMPS
PREFIX=$HOME/opt TARGET=cx3 STAGE=fetch ./build_lammps_mace.sh
# compute node (~30-90 min of nvcc, 959 objects): the actual compile
PREFIX=$HOME/opt TARGET=cx3 GPU_ARCH=ADA89 STAGE=build ./build_lammps_mace.sh
```

`STAGE=all` (the default) does both, which is what you want on a workstation.
The split exists because the two `git clone`s need outbound network — which
compute nodes usually lack — while the compile is far too heavy for a login node.

- `TARGET`: same three values and the same meaning as above. **Note this script
  never `module purge`s**: on CX3 that drops the `/sw-eb` MODULEPATH, GCC and
  CMake become unloadable, and the build silently falls back to g++ 8 (no C++20)
  with no cmake at all → `rc=127`. It loads additively and warns instead.
- Clones LAMMPS `patch_10Sep2025` to `$PREFIX/lammps-mace` and
  `wcwitt/symmetrix` (`--recursive`) to `$PREFIX/symmetrix`.
- **No libtorch, no mace-torch, no Python** — symmetrix reads a plain `.json`
  model. That is the main practical difference from the old ACEsuit build.
- `CONDA_ENV=<prefix>` supplies `mpicxx`/`mpicc` + OpenBLAS and turns MPI on;
  leave it empty to build against modules with `BUILD_MPI=OFF`.
- `GPU=0` builds the CPU/OpenMP variant into `build-omp` instead of `build-gpu`.
- **Binary lands at `$PREFIX/lammps-mace/build-gpu/lmp`** (build tree, not
  `cmake --install`ed).

### The mandatory patch — do not skip this

symmetrix has a **heap overflow that makes MACE silently wrong on CUDA**. In
`pair_symmetrix_mace_kokkos.cpp::coeff()` the `mace_types` view is allocated
with one slot per **model element** but filled with one per **LAMMPS atom
type**. Those agree only when `ntypes == n_elements` — true of upstream's own
examples, false for any multi-layer cell (an h-MoS2 trilayer has 7 types against
a 2-element Mo/S model, so every `pair_coeff` writes 5 ints past a 2-int
allocation).

| build | 24-atom bilayer | 810-atom trilayer |
|---|---|---|
| plain CPU `symmetrix/mace` | −173.40779 eV | −5850.0183 eV |
| Kokkos host (`-k on t N`) | −173.40779 eV | −5850.0183 eV |
| Kokkos CUDA, **unpatched** | **−126.5489 eV** | **−4785.9659 eV** |
| Kokkos CUDA, patched | −173.40779 eV | −5850.0183 eV |

The unpatched CUDA number is wrong by 27 % **with no error raised**. The same
corruption also kills `hybrid/overlay symmetrix/mace dispersion/d3` in ~4.6 s
inside `malloc_consolidate()`. It is **not upstream** (as of `origin/main`
0d86e1e), so a fresh clone reintroduces it — which is why
`symmetrix-mace_types-overflow.patch` ships here and the script **refuses to
build without it**. Always confirm **−173.40779 eV** on the reference bilayer,
on a GPU node, before trusting any number.

### The model is not what you download

symmetrix does not read a MACE `.model` or `.model-lammps.pt` checkpoint. Extract
a `.json` on a machine that has `mace-torch` + the symmetrix python package
(`STAGE=model` creates `$PREFIX/mace-venv` for exactly this):

```bash
symmetrix_extract_mace -m mace-mpa-0-medium.model --chemical-symbols Mo S
```

The element list is **baked into the `.json`**, so one file serves one element
set — a different material needs a different extraction. (A DeePMD `.pth` is the
opposite: it carries the full periodic table.) The cluster that *runs* LAMMPS
needs neither `mace-torch` nor Python; ship it the `.json`.

### Runtime flags

The registry (`src/core/mlip.py`, MACE `LauncherSpec.lmp_flags`) emits the
**Kokkos host (OpenMP)** backend, not CUDA:

```bash
"$PREFIX/lammps-mace/build-gpu/lmp" -k on t ${OMP_NUM_THREADS:-8} -sf kk -pk kokkos neigh half -in deck.in
```

This default is chosen for **safety, not speed**: the host backend is correct
whether or not symmetrix is patched, and is still ~4× faster than the plain CPU
pair style (0.64 vs 2.58 s/step at 810 atoms). CUDA is a further ~2.3× (0.108 vs
0.251 s/step) but is silently wrong on a stock checkout — opt in only after
applying the patch **and** verifying the reference energy.

`neigh half` is **mandatory and paired with the flags above**:
`symmetrix/mace/kk` requires `newton pair on`, while KOKKOS defaults to
`neigh full`, which demands newton off (`kokkos.cpp:713`). Change the two
together or runs abort. symmetrix is OpenMP-threaded rather than rank-parallel,
so give it the whole node and one MPI rank.

### Known non-fatal teardown abort

MACE runs exit `rc=134` ("corrupted size vs. prev_size") **after** results are
written — it even happens on a bare `lmp -h`. Benign for the physics, but it
breaks PBS job dependencies. Tolerate it by checking the run completed, never by
blanket-ignoring the exit code.

### `GPU_ARCH` table

`GPU_ARCH` is the Kokkos architecture macro for the card you are building for
(it becomes `-D Kokkos_ARCH_<GPU_ARCH>=ON`). Building for the wrong arch either
fails to compile or silently falls back to slow paths.

| Card | `GPU_ARCH` |
|---|---|
| NVIDIA L40S (gpu72 default) | `ADA89` |
| NVIDIA A100 | `AMPERE80` |
| NVIDIA RTX 30xx (local workstation, default) | `AMPERE86` |

## D3 syntax is golden-tested **per build**

`pair_style dispersion/d3` does not have one stable argument spelling across
LAMMPS versions — the damping keyword was renamed at one point, and the two
builds here reach `dispersion/d3` by different routes:

- **MACE** is LAMMPS `patch_10Sep2025`, so `dispersion/d3` is native mainline.
- **DeePMD + D3** is a *backport* into a 29 Aug 2024 tree (deepmd-kit is ABI
  locked to it), which is why `build_lammps_deepmd_d3.sh` exists at all. Its d3
  term was verified **bit-identical** to the same quantity measured standalone
  on an upstream 2025 build — that equivalence is what certifies the backport as
  numerically faithful rather than merely compiling.

Both therefore accept `<damping> <functional> <rthr> <cn_thr>` with
`original|zero|zerom|bj|bjm` today. `settings.potential.d3_damping` is
nonetheless a **canonical** name that `src/core/mlip.py::d3_style_token`
translates into the token the target build accepts, and the emitted line is
pinned by a golden test **for each build separately**, so a future divergence
surfaces as a failing test rather than as wrong physics.

Consequence: if you change `LAMMPS_TAG` or `LAMMPS_BRANCH` here, the D3 golden
tests must be re-run against the new binary and the registry's token table
updated. The verified argument order is recorded in
[`documentation/milestone0.md`](../../documentation/milestone0.md) (item M0.2)
and flows from there into the registry.

## Where these paths are consumed

The two binary paths above are what the HPC profile `frameworks:` entries point
at (`FrameworkOverlay.binary`), and what `scripts/run_local.sh` resolves for a
local run. Keep `$PREFIX` consistent between the build and the profile, or set
the profile's `binary:` to the absolute path printed by the `OK:` line at the
end of each script.
