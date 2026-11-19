#!/usr/bin/env bash
#
# run_local.sh — run one generated FrictionSim2D case locally (spec §4.4).
#
#   ./scripts/run_local.sh <sim_dir> <framework: sw|deepmd|mace>
#
# `run.py` only *builds* LAMMPS inputs; it never executes production LAMMPS.
# On the GPU workstation the missing piece is small and entirely mechanical:
# pick the right binary for the framework (MACE and DeePMD are two mutually
# incompatible LAMMPS builds), add its launcher flags and environment exports,
# and invoke `system.in` then `slide.in` from the directory the generated
# scripts' relative paths are written against. That is all this script does.
# Promoting it into `run.py` is future work.
#
# WHERE THE PIECES COME FROM
#   binary     settings.yaml -> hpc.profiles.<active>.frameworks.<framework>.binary
#              falling back to $LMP_<FRAMEWORK>, then hpc.lmp_binary, then `lmp`
#              on PATH (which is all the classical `sw` path ever needs).
#   flags/env  src/core/mlip.py -> get_mlip(framework).launcher  (MACE's
#              `-k on g 1 -sf kk`, DeePMD's LAMMPS_PLUGIN_PATH export). Never
#              re-spelled here: the registry is the single source of truth.
#              See scripts/hpc/README.md and documentation/milestone0.md.
#   workdir    the same prefix-stripping the PBS template does
#              (src/templates/hpc/pbs_combined.j2, lines 38-49).
#
# ENVIRONMENT
#   FRICTIONSIM_SETTINGS   explicit settings.yaml (else ./settings.yaml, else
#                          ~/.config/FrictionSim2D/settings.yaml)
#   FRICTIONSIM_PROFILE    HPC profile name (else hpc.active_profile)
#   LMP_SW / LMP_DEEPMD / LMP_MACE   binary override per framework
#   LMP_EXTRA_FLAGS        appended to the launcher flags
#   RUN_LOCAL_DRY_RUN=1    print the resolved commands, run nothing
#
set -euo pipefail

usage() {
    cat <<'EOF'
usage: run_local.sh <sim_dir> <framework>

  sim_dir     a generated case directory (the one containing lammps/system.in),
              e.g. out/K50_zeta0p1/afm/L2/MoS2_x50_y50/F5nN_v10
  framework   sw | deepmd | mace

  Runs lammps/system.in and then lammps/slide.in (or slide_*.in) for that case.
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

if [ "$#" -ne 2 ]; then
    echo "run_local.sh: expected 2 arguments, got $#" >&2
    usage >&2
    exit 2
fi

SIM_PATH="${1%/}"
FRAMEWORK="$2"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

case "$FRAMEWORK" in
    sw|deepmd|mace) ;;
    *)
        echo "run_local.sh: unknown framework '$FRAMEWORK' (expected sw, deepmd or mace)" >&2
        exit 2
        ;;
esac

if [ ! -d "$SIM_PATH" ]; then
    echo "run_local.sh: no such simulation directory: $SIM_PATH" >&2
    exit 2
fi
if [ ! -f "$SIM_PATH/lammps/system.in" ]; then
    echo "run_local.sh: $SIM_PATH/lammps/system.in not found — is this a generated case directory?" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Working directory: mirror pbs_combined.j2:38-49.
#
# The generated LAMMPS scripts use paths relative to the directory that
# CONTAINS afm/ or sheetonsheet/, not to the case directory, so a sim root
# above that level (K50_zeta0p1/ in a multi-sim campaign) must be stripped off
# and cd'd into, with the remainder passed to `lmp -in`.
# ---------------------------------------------------------------------------
LAMMPS_PREFIX=$(echo "$SIM_PATH" | sed 's|/afm/.*||; s|/sheetonsheet/.*||')
if [ "$LAMMPS_PREFIX" = "$SIM_PATH" ]; then
    case "$SIM_PATH" in
        afm/*|sheetonsheet/*)
            # Already relative to the containing directory — stay in $PWD,
            # exactly as the PBS template stays in $PBS_O_WORKDIR.
            LAMMPS_PREFIX=""
            LAMMPS_SIM_PATH="$SIM_PATH"
            ;;
        *)
            # No afm/ or sheetonsheet/ segment at all (a hand-placed case dir).
            # The PBS template cannot hit this — it always has a campaign root
            # — so run from the parent and address the case by its basename.
            LAMMPS_PREFIX="$(dirname "$SIM_PATH")"
            LAMMPS_SIM_PATH="$(basename "$SIM_PATH")"
            ;;
    esac
else
    LAMMPS_SIM_PATH="${SIM_PATH#${LAMMPS_PREFIX}/}"
fi

# ---------------------------------------------------------------------------
# Resolve binary, flags and environment.
# ---------------------------------------------------------------------------
SETTINGS_SNIPPET=$(
    "$PYTHON" - "$REPO_ROOT" "$FRAMEWORK" <<'PY'
"""Emit shell assignments for the framework's binary, flags and env lines."""
import os
import shlex
import sys

repo_root, framework = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo_root)

lines = []


def note(msg):
    lines.append("echo " + shlex.quote("run_local.sh: " + msg) + " >&2")


# --- launcher flags / env exports come from the MLIP registry ---------------
flags, env_lines = "", []
try:
    from src.core.mlip import get_mlip, is_mlip
except Exception as exc:                      # pragma: no cover - defensive
    note("could not import src/core/mlip.py (%s); using no launcher flags" % exc)
    is_mlip = lambda _f: False                # noqa: E731
if is_mlip(framework):
    launcher = get_mlip(framework).launcher
    flags = launcher.lmp_flags
    if framework == "deepmd" and os.environ.get("LAMMPS_PLUGIN_PATH"):
        note("LAMMPS_PLUGIN_PATH already set; keeping it")
    else:
        env_lines.extend(launcher.env_lines)

# --- binary + extra env from settings.yaml's profile framework overlay ------
binary = None
candidates = [
    os.environ.get("FRICTIONSIM_SETTINGS"),
    "settings.yaml",
    os.path.expanduser("~/.config/FrictionSim2D/settings.yaml"),
]
settings_path = next(
    (c for c in candidates if c and os.path.isfile(os.path.expanduser(c))), None
)
if settings_path is None:
    note("no settings.yaml found; falling back to environment / PATH")
else:
    settings_path = os.path.expanduser(settings_path)
    try:
        import yaml
        with open(settings_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception as exc:
        note("could not read %s (%s); falling back to environment / PATH"
             % (settings_path, exc))
        data = {}
    hpc = data.get("hpc") or {}
    profile_name = os.environ.get("FRICTIONSIM_PROFILE") or hpc.get("active_profile")
    profile = (hpc.get("profiles") or {}).get(profile_name) or {}
    if profile_name and not profile:
        note("profile %r is not defined in %s" % (profile_name, settings_path))
    overlay = (profile.get("frameworks") or {}).get(framework) or {}
    binary = overlay.get("binary") or hpc.get("lmp_binary")
    if binary:
        note("binary from %s (profile %s)" % (settings_path, profile_name))
    for mod in overlay.get("modules") or profile.get("modules") or []:
        lines.append(
            "if command -v module >/dev/null 2>&1; then module load %s; fi"
            % shlex.quote(mod)
        )
    env_lines.extend(overlay.get("env_lines") or [])

# --- per-framework environment override, then PATH -------------------------
env_binary = os.environ.get("LMP_%s" % framework.upper())
if env_binary:
    binary = env_binary
    note("binary from $LMP_%s" % framework.upper())
if not binary:
    binary = "lmp"
    note("no configured binary for %r; using 'lmp' from PATH" % framework)

extra = os.environ.get("LMP_EXTRA_FLAGS", "").strip()
if extra:
    flags = (flags + " " + extra).strip()

lines.append("LMP_BINARY=%s" % shlex.quote(os.path.expandvars(binary)))
lines.append("LMP_FLAGS=%s" % shlex.quote(flags))
# env_lines are *shell* lines by design (see mlip.LauncherSpec / the HPC
# profile overlay), so they are emitted verbatim and eval'd by the caller.
lines.extend(env_lines)
print("\n".join(lines))
PY
)

if ! eval "$SETTINGS_SNIPPET"; then
    echo "run_local.sh: failed to apply the resolved environment." >&2
    echo "  For DeePMD, activate the deepmd venv first (see scripts/hpc/README.md):" >&2
    echo "    source \$PREFIX/deepmd-venv/bin/activate" >&2
    exit 1
fi

# `export X=$(cmd)` returns the status of `export`, not of the substitution, so
# a failed plugin-path lookup above is silent. Check the result, not the status:
# without this, LAMMPS would start and only then report "unknown pair style".
if [ "$FRAMEWORK" = "deepmd" ] && [ -z "${LAMMPS_PLUGIN_PATH:-}" ]; then
    echo "run_local.sh: LAMMPS_PLUGIN_PATH is empty — pair_style deepmd would be" >&2
    echo "  an unknown style. Activate the deepmd venv and export it first:" >&2
    echo "    source \$PREFIX/deepmd-venv/bin/activate" >&2
    echo "    export LAMMPS_PLUGIN_PATH=\$(python -c \"import deepmd,os;print(os.path.join(os.path.dirname(deepmd.__file__),'lib'))\")" >&2
    exit 1
fi

if ! command -v "$LMP_BINARY" >/dev/null 2>&1 && [ ! -x "$LMP_BINARY" ]; then
    echo "run_local.sh: LAMMPS binary '$LMP_BINARY' not found or not executable." >&2
    echo "  Set it in settings.yaml (hpc.profiles.<p>.frameworks.$FRAMEWORK.binary)" >&2
    echo "  or export LMP_$(echo "$FRAMEWORK" | tr '[:lower:]' '[:upper:]')=/path/to/lmp" >&2
    exit 1
fi

run_deck() {
    # shellcheck disable=SC2086  # $LMP_FLAGS must word-split (-k on g 1 -sf kk)
    local deck="$1"
    echo "==> $LMP_BINARY $LMP_FLAGS -in $deck"
    if [ "${RUN_LOCAL_DRY_RUN:-0}" = "1" ]; then
        return 0
    fi
    "$LMP_BINARY" $LMP_FLAGS -in "$deck"
}

if [ -n "$LAMMPS_PREFIX" ]; then
    echo "run_local.sh: cd $LAMMPS_PREFIX  (case: $LAMMPS_SIM_PATH)"
    cd "$LAMMPS_PREFIX"
else
    echo "run_local.sh: running from $PWD  (case: $LAMMPS_SIM_PATH)"
fi

# Phase 1: system initialisation (writes load_*N.data)
run_deck "${LAMMPS_SIM_PATH}/lammps/system.in"

# Phase 2: slide. Static models (pes-sheet, pes-tip) and indent have no
# sliding phase, and some AFM cases split the slide into slide_<loop>.in
# variants, so a missing slide.in is normal — never a hard failure.
shopt -s nullglob
SLIDE_DECKS=()
if [ -f "${LAMMPS_SIM_PATH}/lammps/slide.in" ]; then
    SLIDE_DECKS=("${LAMMPS_SIM_PATH}/lammps/slide.in")
else
    SLIDE_DECKS=("${LAMMPS_SIM_PATH}"/lammps/slide_*.in)
fi
shopt -u nullglob

if [ "${#SLIDE_DECKS[@]}" -eq 0 ]; then
    echo "run_local.sh: no slide.in / slide_*.in in ${LAMMPS_SIM_PATH}/lammps —" \
         "static or indent-only case, nothing further to run."
else
    for deck in "${SLIDE_DECKS[@]}"; do
        run_deck "$deck"
    done
fi

echo "run_local.sh: completed ${SIM_PATH}"
