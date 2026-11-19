#!/usr/bin/env bash
# Build a LAMMPS that has BOTH `pair_style deepmd` (via the deepmd-kit runtime
# plugin) AND `pair_style dispersion/d3` -- a combination no prebuilt binary
# offers. Idempotent: re-running skips completed stages.
#
# Usage: PREFIX=$HOME/opt TARGET=workstation ./build_lammps_deepmd_d3.sh
#   TARGET: workstation | cx3 | hx1   (selects the module preamble)
#
# ---------------------------------------------------------------------------
# WHY THIS SCRIPT EXISTS (the ABI lock) -- see
# docs/superpowers/specs/2026-08-11-mlip-post-m0-design.md §4
# ---------------------------------------------------------------------------
# `pair_style dispersion/d3` entered LAMMPS in `stable_22Jul2025`; the
# EXTRA-PAIR directory of `stable_29Aug2024_update3` contains ZERO d3 sources.
# But deepmd-kit is ABI-locked to LAMMPS 2024.08.29 -- verified twice:
#   * the pip wheel's plugin refuses to load into a 22Jul2025 build with
#     `undefined symbol: _ZN9LAMMPS_NS5utils11getsyserrorEv`;
#   * conda-forge `deepmd-kit 3.1.3` carries the hard constraint
#     `lammps 2024.08.29`, so the solver will not even consider 2025.07.22.
# So neither release alone can host both styles.
#
# `pair_dispersion_d3` is, however, a self-contained pair style: it includes
# only atom.h, comm.h, error.h, force.h, info.h, memory.h, neigh_list.h,
# neighbor.h, update.h and pair.h -- all long-stable LAMMPS internals. So the
# fix is a BACKPORT: build the ABI-matching 2024 release with the 2025 d3
# sources dropped into its EXTRA-PAIR package.
#
# PROVENANCE (record this whenever the binary's numbers are quoted):
#   host tree   : lammps/lammps @ stable_29Aug2024_update3   (ABI for deepmd)
#   donor tree  : lammps/lammps @ stable_22Jul2025_update1   (d3 sources)
#   files copied into <host>/src/EXTRA-PAIR/ :
#       pair_dispersion_d3.cpp
#       pair_dispersion_d3.h
#       d3_parameters.h
#   plus the compat shim in step 3b below. The host tree is otherwise pristine.
#
# COMPAT SHIM (step 3b) -- the ONE place the "long-stable internals" claim of
# spec §4 is not quite true. The copied source does not compile as-is against
# 2024 headers; it fails at exactly 11 sites, ALL of them error-message
# decoration, none of them physics:
#   * `Error::NOLASTLINE` / `Error::NOPOINTER` (10 sites) -- these constants
#     were added to error.h AFTER 2024. They tell the 2025 error printer which
#     input argument to blame. `NOPOINTER` is precisely what the 2024
#     `all(FLERR, fmt, ...)` overload already implies, so dropping the argument
#     restores stock 2024 behaviour rather than changing anything.
#   * `Info::get_pair_coeff_status(lmp)` (1 site) -- a 2025-only static helper
#     that appends a diagnostic table to the "All pair coeffs are not set"
#     message. Replaced with a literal.
# Every patched site is inside an `error->all(...)` abort path, so no executed
# code path, energy, or force is touched. The D3 maths and the C6/parameter
# tables are byte-for-byte the 2025 originals.
# ---------------------------------------------------------------------------
set -euo pipefail

PREFIX="${PREFIX:-$HOME/opt}"
TARGET="${TARGET:-workstation}"

HOST_TAG="stable_29Aug2024_update3"     # ABI-matched to the deepmd plugin
DONOR_TAG="stable_22Jul2025_update1"    # first release carrying dispersion/d3

HOST_SRC="$PREFIX/lammps-deepmd-d3/src"
HOST_BUILD="$PREFIX/lammps-deepmd-d3/build"
# Reuse an existing donor checkout if one is already on disk at the same tag --
# the clone is ~1 GB and the only thing wanted from it is three files.
DONOR_SRC="${DONOR_SRC:-$PREFIX/lammps-deepmd/src}"

D3_FILES=(pair_dispersion_d3.cpp pair_dispersion_d3.h d3_parameters.h)

# MPI must be ON even on a single-GPU workstation -- this is the opposite of
# build_lammps_deepmd.sh and it is NOT a preference. The deepmd plugin (.so)
# leaves 20 MPI_* symbols undefined and expects the host executable to supply
# them. A `-D BUILD_MPI=OFF` LAMMPS supplies MPI from src/STUBS/mpi.cpp, which
# is C++ -- its symbols are mangled (`_Z13MPI_AllgatherPviiS_iii`), so they do
# NOT satisfy the plugin's C-ABI `MPI_Allgather`. The failure is at dlopen:
#   Open of .../dpplugin.so failed: undefined symbol: MPI_Allgather
#   Loaded 0 plugins ...
#   ERROR: Unrecognized pair style 'deepmd'
# i.e. it looks exactly like a missing LAMMPS_PLUGIN_PATH. Build with the SAME
# MPI the plugin was built against (conda-forge deepmd-kit => mpi_mpich).
BUILD_MPI="${BUILD_MPI:-ON}"

# MPI wrapper used on the `workstation` target. The system MPICH on this host
# is broken (its CMake imported target names a deleted include dir), so we use
# the MPICH inside the conda env that ships deepmd-kit -- which is also the
# exact MPI the plugin was linked against. Its mpicxx defaults to a
# conda-only compiler (x86_64-conda-linux-gnu-c++) that need not exist, but
# MPICH honours $MPICH_CXX, so we point it back at the system g++ and get
# conda MPICH headers/libs with a system toolchain.
MPI_WRAPPER="${MPI_WRAPPER:-/home/mv923/miniconda3/envs/mlip/bin/mpicxx}"
export MPICH_CXX="${MPICH_CXX:-/usr/bin/g++}"

# NEVER `module purge` on cx3: it drops the /sw-eb MODULEPATH, after which GCC
# and CMake cannot be loaded AT ALL. The build then silently falls back to the
# RHEL8 system g++ 8 (which cannot compile the C++ standard LAMMPS needs) with
# no cmake on PATH, and dies rc=127 about 19 s in -- a failure that looks like a
# missing dependency rather than a module problem. Load additively and warn
# rather than dying, so a site with different module names can still get through
# and the guards below report what actually happened.
MODS=""
load_modules() {
  for m in "$@"; do
    if module load "$m" 2>/dev/null; then MODS="$MODS $m"
    else echo "WARN: cannot load module $m" >&2; fi
  done
  MODS="${MODS# }"
}
die() { echo "FATAL: $*" >&2; echo "       modules loaded: ${MODS:-<none>}" >&2; exit 1; }

# The module VERSIONS are pinned deliberately: this binary is ABI-matched to the
# deepmd plugin, so do not drift them to match build_lammps_mace.sh. Override
# individually if a site retires one.
case "$TARGET" in
  cx3) load_modules tools/prod "${GCC_MODULE:-GCC/12.3.0}" \
                    "${CUDA_MODULE:-CUDA/12.4.0}" \
                    "${CMAKE_MODULE:-CMake/3.26.3-GCCcore-12.3.0}"; BUILD_MPI=ON ;;
  hx1) load_modules "${GCC_MODULE:-GCC/12.3.0}" "${CUDA_MODULE:-CUDA/12.4.0}" \
                    "${CMAKE_MODULE:-CMake/3.26.3}"; BUILD_MPI=ON ;;
  workstation)
    # Keep conda off PATH: a conda env's MPI/compiler wrappers leak into the
    # configure and produce a binary linked against libraries the plugin's
    # runtime does not have.
    export PATH="/usr/bin:/usr/local/cuda-12.5/bin:$PATH" ;;
  *) echo "ERROR: unknown TARGET '$TARGET' (expected workstation|cx3|hx1)" >&2; exit 2 ;;
esac

mkdir -p "$PREFIX"

# 1. Host tree: the ABI-matching 2024 release.
if [ ! -d "$HOST_SRC" ]; then
  git clone --depth 1 --branch "$HOST_TAG" https://github.com/lammps/lammps.git "$HOST_SRC"
fi

# 2. Donor tree: the 2025 release that first shipped dispersion/d3.
if [ ! -f "$DONOR_SRC/src/EXTRA-PAIR/pair_dispersion_d3.cpp" ]; then
  DONOR_SRC="$PREFIX/lammps-d3-donor/src"
  if [ ! -d "$DONOR_SRC" ]; then
    git clone --depth 1 --branch "$DONOR_TAG" https://github.com/lammps/lammps.git "$DONOR_SRC"
  fi
fi

# 3. Backport the three d3 files. Copy every time (cheap, and keeps a re-run
#    after a donor update honest).
for f in "${D3_FILES[@]}"; do
  cp -f "$DONOR_SRC/src/EXTRA-PAIR/$f" "$HOST_SRC/src/EXTRA-PAIR/$f"
done
echo "backported ${D3_FILES[*]} from $DONOR_TAG into $HOST_TAG"

# 3b. Compat shim -- see the COMPAT SHIM block in the header for why each of
#     these is semantics-preserving. Applied to the freshly-copied file, so it
#     is idempotent by construction.
D3_CPP="$HOST_SRC/src/EXTRA-PAIR/pair_dispersion_d3.cpp"
sed -i \
  -e 's/Error::NOLASTLINE, *//g' \
  -e 's/Error::NOPOINTER, *//g' \
  -e 's/Info::get_pair_coeff_status(lmp)/std::string("(per-pair status unavailable in this LAMMPS version)")/g' \
  "$D3_CPP"
if grep -q 'Error::NOLASTLINE\|Error::NOPOINTER\|get_pair_coeff_status' "$D3_CPP"; then
  echo "ERROR: compat shim did not remove every post-2024 Error/Info symbol" >&2
  exit 1
fi
echo "applied 2024-header compat shim to pair_dispersion_d3.cpp"

# 4. Configure + build.
#
# PKG_PLUGIN=ON IS LOAD-BEARING, not optional. deepmd-kit v3 ships `pair_style
# deepmd` as a runtime shared object that LAMMPS discovers through
# LAMMPS_PLUGIN_PATH (step 5) -- that machinery lives in LAMMPS's PLUGIN
# package. Without it the binary builds fine and then reports "Unrecognized
# pair style 'deepmd'", which looks like a model/path problem rather than a
# missing build flag.
#
# LAMMPS runs `find_package(MPI QUIET COMPONENTS CXX)` UNCONDITIONALLY, purely
# to pick the default for BUILD_MPI -- so `-D BUILD_MPI=OFF` alone does NOT
# stop it. On a host with a broken/partial MPI (e.g. an MPICH wrapper whose
# imported target names a deleted include dir) that probe is a hard CMake
# error and the configure dies even though we asked for no MPI. Suppressing
# the find_package outright is the only thing that actually skips it.
MPI_ARGS=(-D BUILD_MPI="$BUILD_MPI")
if [ "$BUILD_MPI" = "OFF" ]; then
  # Retained for completeness, but such a binary CANNOT load the deepmd
  # plugin -- see the BUILD_MPI block near the top.
  MPI_ARGS+=(-D CMAKE_DISABLE_FIND_PACKAGE_MPI=ON)
elif [ "$TARGET" = "workstation" ]; then
  # Naming a working wrapper makes FindMPI use it and skip the broken
  # system-MPI probe entirely, so no CMAKE_DISABLE_FIND_PACKAGE_MPI is needed.
  [ -x "$MPI_WRAPPER" ] || { echo "ERROR: MPI_WRAPPER '$MPI_WRAPPER' not executable" >&2; exit 2; }
  MPI_ARGS+=(-D MPI_CXX_COMPILER="$MPI_WRAPPER")
fi

# Toolchain guards. Without these a module that failed to load surfaces as a
# bare rc=127 from cmake; with them it is named. NB no hard nvcc check here
# (unlike build_lammps_mace.sh): this build compiles no CUDA itself, so a
# missing nvcc is a warning, not fatal.
command -v cmake >/dev/null || die "no cmake on PATH"
gxx=$(g++ -dumpversion 2>/dev/null | cut -d. -f1)
[ -n "$gxx" ] || die "no g++ on PATH"
[ "$gxx" -ge 10 ] || die "g++ $gxx is too old for LAMMPS -- the GCC module did not load"
if [ "$TARGET" != "workstation" ] && ! command -v nvcc >/dev/null; then
  echo "WARN: no nvcc on PATH -- the CUDA module may not have loaded" >&2
fi

# Parallelism. $NCPUS is unreliable on CX3 -- job 3972247 had ncpus=8 in both
# Resource_List and resources_used, but $NCPUS was 1 INSIDE the job, so a
# -j$NCPUS build ran single-threaded: ~5 h instead of ~45 min (cpupercent=99 in
# `qstat -f` is the quickest way to spot it). `nproc` has the opposite failure:
# on a shared node it reports the whole machine, not the cgroup allocation.
# Neither is safe unattended, so warn on the known trap and take NJOBS=<n>.
NJOBS="${NJOBS:-${NCPUS:-$(nproc)}}"
if [ "${NCPUS:-}" = "1" ] && [ "$(nproc)" -gt 1 ]; then
  echo "WARN: \$NCPUS=1 but nproc=$(nproc) -- \$NCPUS is unreliable on CX3." >&2
  echo "      Building with -j$NJOBS. If \$NCPUS is wrong this build is" >&2
  echo "      SINGLE-THREADED (~5 h instead of ~45 min). Set NJOBS=<n> from" >&2
  echo "      your own '#PBS -l select=...ncpus=N' to fix it. Deliberately not" >&2
  echo "      guessing nproc here: that oversubscribes a shared node." >&2
fi

cmake -S "$HOST_SRC/cmake" -B "$HOST_BUILD" \
  -D CMAKE_INSTALL_PREFIX="$PREFIX/lammps-deepmd-d3" \
  -D PKG_PLUGIN=ON \
  -D PKG_EXTRA-PAIR=ON -D PKG_MANYBODY=ON -D PKG_MOLECULE=ON \
  -D PKG_RIGID=ON -D PKG_KSPACE=ON -D PKG_EXTRA-FIX=ON \
  "${MPI_ARGS[@]}" -D LAMMPS_INSTALL_RPATH=ON
cmake --build "$HOST_BUILD" -j"$NJOBS"
cmake --install "$HOST_BUILD"

# 5. Runtime. deepmd is a *plugin*, not compiled in. The path differs by
#    install method -- there is no `deepmd.lmp` python module in deepmd-kit v3:
#      conda: $CONDA_PREFIX/lib/deepmd_lmp/dpplugin.so
#      pip  : <site-packages>/deepmd/lib/libdeepmd_lmpplugin.so
#    e.g. for the `mlip` conda env on this workstation:
#      export LAMMPS_PLUGIN_PATH=/home/mv923/miniconda3/envs/mlip/lib/deepmd_lmp
#      export LD_LIBRARY_PATH=/home/mv923/miniconda3/envs/mlip/lib
#    LD_LIBRARY_PATH is required as well: the plugin pulls in the conda env's
#    own libabsl/libtensorflow-framework, and without it dlopen fails with
#    `libabsl_log_flags.so.2601.0.0: cannot open shared object file` -- which,
#    like the MPI case above, surfaces only as "Loaded 0 plugins".
LMP="$PREFIX/lammps-deepmd-d3/bin/lmp"
DEEPMD_PLUGIN_DIR="${DEEPMD_PLUGIN_DIR:-/home/mv923/miniconda3/envs/mlip/lib/deepmd_lmp}"
DEEPMD_LIB_DIR="${DEEPMD_LIB_DIR:-/home/mv923/miniconda3/envs/mlip/lib}"

# NOTE: capture into a variable rather than piping into `grep -q`. Under
# `set -o pipefail`, `grep -q` exits at the first match, SIGPIPEs `lmp`, and
# the pipeline then reports lmp's 141 -- so the check fails intermittently
# depending on whether grep wins the race. It did exactly that on a re-run.
help_txt="$("$LMP" -h 2>/dev/null || true)"
case "$help_txt" in
  *dispersion/d3*) ;;
  *) echo "ERROR: built binary does not list pair_style dispersion/d3" >&2; exit 1 ;;
esac

# Both capabilities must live in THIS one binary, so verify the plugin too --
# `-h` cannot show it (it is a runtime plugin), hence the trivial deck.
if [ -d "$DEEPMD_PLUGIN_DIR" ]; then
  probe="$(mktemp -d)"
  printf 'plugin list\n' > "$probe/probe.in"
  probe_out="$(cd "$probe" && env -u PYTHONPATH \
      LAMMPS_PLUGIN_PATH="$DEEPMD_PLUGIN_DIR" \
      LD_LIBRARY_PATH="$DEEPMD_LIB_DIR" \
      "$LMP" -in probe.in 2>&1 || true)"
  rm -rf "$probe"
  case "$probe_out" in
    *"Loaded 1 plugins"*) echo "OK: deepmd plugin loads into this binary" ;;
    *) echo "ERROR: deepmd plugin failed to load from $DEEPMD_PLUGIN_DIR" >&2
       echo "$probe_out" | grep -i 'plugin\|undefined\|cannot open' >&2 || true
       exit 1 ;;
  esac
fi
echo "OK: $LMP  (dispersion/d3 + deepmd plugin in one binary)"
