#!/usr/bin/env bash
# Build a DeePMD-enabled LAMMPS. Idempotent: re-running skips completed stages.
# Usage: PREFIX=$HOME/opt TARGET=workstation ./build_lammps_deepmd.sh
#   TARGET: workstation | cx3 | hx1   (selects module preamble)
set -euo pipefail
PREFIX="${PREFIX:-$HOME/opt}"
TARGET="${TARGET:-workstation}"
# Interpreter used to create the venv. Override when the system python3 lacks
# ensurepip -- Debian/Ubuntu split it into a separate python3.x-venv package,
# which needs root to install, so `python3 -m venv` fails there with
# "ensurepip is not available". Any python >=3.9 with ensurepip works, e.g. a
# conda base python: PYTHON_BIN=$HOME/miniconda3/bin/python3
PYTHON_BIN="${PYTHON_BIN:-python3}"
# Pinned LAMMPS. MUST be >= stable_22Jul2025: `pair_style dispersion/d3` (the
# D3 column of the capability matrix) entered EXTRA-PAIR in that release.
# Verified empirically -- src/EXTRA-PAIR in stable_29Aug2024_update3, the
# original pin, contains NO d3 sources at all, so every D3 deck would fail with
# "Unrecognized pair style" on a build that otherwise looked correct.
LAMMPS_TAG="stable_22Jul2025_update1"
DEEPMD_VER="3.0.2"                       # deepmd-kit v3 (DPA support)
# MPI is OFF by default on a workstation: both frameworks are single-rank by
# design (LauncherSpec.mpi_ranks == 1 -- MACE forbids domain decomposition,
# DeePMD gives one process the GPU), so MPI buys nothing at the 1-GPU design
# point and a broken/partial system MPI (e.g. an MPICH wrapper pointing at a
# missing include dir) fails the LAMMPS configure step for no benefit.
# Cluster targets set it back ON, where MPI comes from the module system.
BUILD_MPI="${BUILD_MPI:-OFF}"

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

# The module VERSIONS are pinned deliberately: deepmd-kit is ABI-locked (see the
# LAMMPS_TAG note above), so do not drift these to match build_lammps_mace.sh.
# Override individually if a site retires one.
case "$TARGET" in
  cx3) load_modules tools/prod "${GCC_MODULE:-GCC/12.3.0}" \
                    "${CUDA_MODULE:-CUDA/12.4.0}" \
                    "${CMAKE_MODULE:-CMake/3.26.3-GCCcore-12.3.0}"; BUILD_MPI=ON ;;
  hx1) load_modules "${GCC_MODULE:-GCC/12.3.0}" "${CUDA_MODULE:-CUDA/12.4.0}" \
                    "${CMAKE_MODULE:-CMake/3.26.3}"; BUILD_MPI=ON ;;
  workstation) : ;;  # assumes system gcc, cuda toolkit, cmake >= 3.20 on PATH
  *) echo "ERROR: unknown TARGET '$TARGET' (expected workstation|cx3|hx1)" >&2; exit 2 ;;
esac

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

mkdir -p "$PREFIX" && cd "$PREFIX"

# 1. Python env with deepmd-kit (provides libdeepmd + lmp plugin mode)
if [ ! -d "$PREFIX/deepmd-venv" ]; then
  "$PYTHON_BIN" -m venv "$PREFIX/deepmd-venv"
  "$PREFIX/deepmd-venv/bin/pip" install --upgrade pip
  # torch backend build (DPA models are PyTorch); DP_ENABLE_PYTORCH needed for custom OPs
  DP_ENABLE_PYTORCH=1 "$PREFIX/deepmd-venv/bin/pip" install "deepmd-kit[torch]==${DEEPMD_VER}"
fi
# shellcheck disable=SC1091
source "$PREFIX/deepmd-venv/bin/activate"

# 2. LAMMPS source (pinned)
if [ ! -d "$PREFIX/lammps-deepmd/src" ]; then
  git clone --depth 1 --branch "$LAMMPS_TAG" https://github.com/lammps/lammps.git "$PREFIX/lammps-deepmd/src"
fi

# 3. Configure + build with the deepmd plugin and EXTRA-PAIR (dispersion/d3)
#
# PKG_PLUGIN=ON IS LOAD-BEARING, not optional. deepmd-kit v3 ships `pair_style
# deepmd` as a runtime shared object that LAMMPS discovers through
# LAMMPS_PLUGIN_PATH (see step 4) -- that machinery lives in LAMMPS's PLUGIN
# package. Without it the binary builds fine and then reports
# "Unrecognized pair style 'deepmd'" at run time, which looks like a model or
# path problem rather than a missing build flag.
# LAMMPS runs `find_package(MPI QUIET COMPONENTS CXX)` UNCONDITIONALLY, purely
# to pick the default for BUILD_MPI -- so `-D BUILD_MPI=OFF` alone does not stop
# it. On a host with a broken/partial MPI (e.g. an MPICH wrapper whose imported
# target names a deleted include dir) that probe is a hard CMake error, and the
# configure dies even though we asked for no MPI. Suppressing the find_package
# outright is the only thing that actually skips it.
MPI_ARGS=(-D BUILD_MPI="$BUILD_MPI")
if [ "$BUILD_MPI" = "OFF" ]; then
  MPI_ARGS+=(-D CMAKE_DISABLE_FIND_PACKAGE_MPI=ON)
fi

# Toolchain guards. Without these a module that failed to load surfaces as a
# bare rc=127 from cmake; with them it is named. NB no hard nvcc check here
# (unlike build_lammps_mace.sh): this build compiles no CUDA itself -- the GPU
# work lives in the venv's torch -- so a missing nvcc is a warning, not fatal.
command -v cmake >/dev/null || die "no cmake on PATH"
gxx=$(g++ -dumpversion 2>/dev/null | cut -d. -f1)
[ -n "$gxx" ] || die "no g++ on PATH"
[ "$gxx" -ge 10 ] || die "g++ $gxx is too old for LAMMPS -- the GCC module did not load"
if [ "$TARGET" != "workstation" ] && ! command -v nvcc >/dev/null; then
  echo "WARN: no nvcc on PATH -- the CUDA module may not have loaded" >&2
fi

cmake -S "$PREFIX/lammps-deepmd/src/cmake" -B "$PREFIX/lammps-deepmd/build" \
  -D CMAKE_INSTALL_PREFIX="$PREFIX/lammps-deepmd" \
  -D PKG_PLUGIN=ON \
  -D PKG_EXTRA-PAIR=ON -D PKG_MANYBODY=ON -D PKG_KSPACE=ON -D PKG_MOLECULE=ON \
  "${MPI_ARGS[@]}" -D LAMMPS_INSTALL_RPATH=ON
cmake --build "$PREFIX/lammps-deepmd/build" -j"$NJOBS"
cmake --install "$PREFIX/lammps-deepmd/build"

# 4. Runtime: deepmd is loaded as a plugin — set for every run:
#    source $PREFIX/deepmd-venv/bin/activate
#    export LAMMPS_PLUGIN_PATH=$(python -c "import deepmd,os;print(os.path.join(os.path.dirname(deepmd.__file__),'lib'))")
echo "OK: $PREFIX/lammps-deepmd/bin/lmp"
