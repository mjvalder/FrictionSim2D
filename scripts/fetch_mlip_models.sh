#!/usr/bin/env bash
# Fetch and prepare an MLIP foundation model for FrictionSim2D.
#
#   ./scripts/fetch_mlip_models.sh deepmd
#   ./scripts/fetch_mlip_models.sh mace --elements Mo S
#   ./scripts/fetch_mlip_models.sh deepmd --list-heads
#
# Neither model ships in the repository: they are third-party weights of 18-120
# MB with their own licences and release channels, and the DeePMD one is a
# *derived* artefact (a freeze of a checkpoint at a chosen head) with no
# canonical download URL. This script is the recipe instead. See
# examples/potentials/mlip/README.md.
#
# Environment:
#   MODELS_DIR   where the finished models land (default: $HOME/opt/models)
#   MACE_HEAD    head to keep if the MACE checkpoint is multihead (default: the
#                tool's own -- the first non-PT head)
#   SYMMETRIX_SRC  path to an existing symmetrix clone, to avoid re-cloning
#   PYTHON_BIN   interpreter used to create the venvs (default: python3)
#   VENV_DIR     where the tool venvs live      (default: $HOME/opt)
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-$HOME/opt/models}"
VENV_DIR="${VENV_DIR:-$HOME/opt}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# DPA-3.1-3M + the Alex2D head are the measured defaults -- every number in
# documentation/mlip.md section 5 rests on them, and `lat_c = 6.112` in the
# shipped configs is that model's equilibrium. Overriding either is a
# measurement exercise, not a config edit: re-run scripts/mlip_bilayer_benchmark.py
# and update the configs. See examples/potentials/mlip/README.md.
DP_MODEL="${DP_MODEL:-DPA-3.1-3M}"
DP_HEAD="${DP_HEAD:-Alex2D}"
MACE_MODEL="${MACE_MODEL:-mace-mpa-0-medium}"

usage() { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
die() { echo "FATAL: $*" >&2; exit 1; }

FRAMEWORK="${1:-}"; shift || true
ELEMENTS=()
LIST_HEADS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --elements) shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do ELEMENTS+=("$1"); shift; done ;;
    --model)    DP_MODEL="$2"; MACE_MODEL="$2"; shift 2 ;;
    --head)     DP_HEAD="$2"; shift 2 ;;
    --list-heads) LIST_HEADS=1; shift ;;
    -h|--help)  usage 0 ;;
    *) die "unknown option '$1' (try --help)" ;;
  esac
done

mkdir -p "$MODELS_DIR"

# ---------------------------------------------------------------- deepmd ----
fetch_deepmd() {
  local venv="$VENV_DIR/deepmd-venv"
  if [ ! -d "$venv" ]; then
    echo ">> creating $venv"
    # DP_ENABLE_PYTORCH=1 is load-bearing: without it the PyTorch custom OPs are
    # missing and the freeze below silently produces a model LAMMPS cannot run.
    "$PYTHON_BIN" -m venv "$venv"
    "$venv/bin/pip" install --quiet --upgrade pip
    DP_ENABLE_PYTORCH=1 "$venv/bin/pip" install --quiet "deepmd-kit[torch]"
  fi

  local ckpt="$MODELS_DIR/${DP_MODEL}.pt"
  if [ ! -f "$ckpt" ]; then
    echo ">> downloading $DP_MODEL (~47 MB)"
    ( cd "$MODELS_DIR" && "$venv/bin/dp" pretrained download "$DP_MODEL" )
  fi
  [ -f "$ckpt" ] || die "expected $ckpt after download -- did the model name change?"

  if [ "$LIST_HEADS" = 1 ]; then
    echo ">> heads in $DP_MODEL:"
    "$venv/bin/python" - "$ckpt" <<'PYEOF'
import sys, torch
m = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
for h in sorted({k.split('.')[1] for k in m["model"] if k.startswith("model.")}):
    print("   ", h)
PYEOF
    return
  fi

  # DPA models are MULTITASK: a bare `dp freeze` fails with "Head must be
  # specified in multitask mode". --head picks the training domain, and for
  # layered 2D materials that is Alexandria-2D.
  local out="$MODELS_DIR/dpa3_$(echo "$DP_HEAD" | tr '[:upper:]' '[:lower:]').pth"
  if [ -f "$out" ]; then
    echo ">> already present: $out"
  else
    echo ">> freezing $DP_MODEL at head '$DP_HEAD'"
    "$venv/bin/dp" --pt freeze -c "$ckpt" -o "$out" --head "$DP_HEAD"
  fi
  echo
  echo "DONE. Point a config at it:"
  echo "    pot_type = deepmd"
  echo "    pot_path = $out"
  echo "This model carries all 118 elements -- one file for every material."
}

# ------------------------------------------------------------------ mace ----
fetch_mace() {
  [ "${#ELEMENTS[@]}" -gt 0 ] || die \
    "mace needs --elements: the element list is BAKED INTO the extracted .json.
       e.g. ./scripts/fetch_mlip_models.sh mace --elements Mo S"

  local venv="$VENV_DIR/mace-venv"
  if [ ! -d "$venv" ]; then
    echo ">> creating $venv"
    "$PYTHON_BIN" -m venv "$venv"
    "$venv/bin/pip" install --quiet --upgrade pip
    # symmetrix supplies symmetrix_extract_mace; mace-torch loads the checkpoint.
    "$venv/bin/pip" install --quiet mace-torch
    # NOTE the `#subdirectory=symmetrix`: the repository root has NO pyproject.toml
    # (it holds libsymmetrix/, pair_symmetrix/ and symmetrix/ side by side), so a
    # plain `pip install git+...` fails. This is a scikit-build-core package with
    # a pybind11 extension, so it COMPILES on install -- it needs cmake >= 3.27
    # and a C++ compiler on PATH. If you already ran
    # `scripts/hpc/build_lammps_mace.sh` there is a clone at $VENV_DIR/symmetrix;
    # set SYMMETRIX_SRC to it to install from disk instead of re-cloning.
    if [ -n "${SYMMETRIX_SRC:-}" ]; then
      "$venv/bin/pip" install --quiet "$SYMMETRIX_SRC/symmetrix"
    elif [ -d "$VENV_DIR/symmetrix/symmetrix" ]; then
      "$venv/bin/pip" install --quiet "$VENV_DIR/symmetrix/symmetrix"
    else
      "$venv/bin/pip" install --quiet \
        "git+https://github.com/wcwitt/symmetrix.git#subdirectory=symmetrix"
    fi
  fi

  local ckpt="$MODELS_DIR/${MACE_MODEL}.model"
  if [ ! -f "$ckpt" ]; then
    echo ">> downloading $MACE_MODEL (~79 MB, 89 elements)"
    "$venv/bin/python" - "$ckpt" <<'PYEOF'
import shutil, sys
from mace.calculators.foundations_models import mace_mp
calc = mace_mp(model="medium", default_dtype="float64", device="cpu")
src = getattr(calc, "model_path", None) or getattr(calc, "models", [None])[0]
if src is None:
    raise SystemExit("could not locate the downloaded checkpoint in ~/.cache/mace")
shutil.copy2(str(src), sys.argv[1])
print("staged", sys.argv[1])
PYEOF
  fi
  [ -f "$ckpt" ] || die "expected $ckpt after download"

  # ONE download, MANY extractions: the .json is derived locally from the
  # checkpoint above, per element set. A new material needs a re-extraction,
  # never another download. Keep the element suffix in the filename -- the
  # campaign's shared potentials/ store is keyed by basename, so two element
  # sets both called mace.json would collide.
  # symmetrix_extract_mace's own default is
  # "<model stem>-<sorted elements joined by ->.json"; we pass -o explicitly so
  # the name is deterministic regardless of the order given on the command line.
  local suffix; suffix=$(printf '%s\n' "${ELEMENTS[@]}" | sort | paste -sd-)
  local out="$MODELS_DIR/${MACE_MODEL}-${suffix}.json"
  if [ -f "$out" ]; then
    echo ">> already present: $out"
  else
    echo ">> extracting symmetrix model for elements: ${ELEMENTS[*]}"
    "$venv/bin/symmetrix_extract_mace" -m "$ckpt" \
      --chemical-symbols "${ELEMENTS[@]}" \
      ${MACE_HEAD:+--head "$MACE_HEAD"} -o "$out"
  fi
  [ -f "$out" ] || die "extraction produced no $out"
  echo
  echo "DONE. Point a config at it:"
  echo "    pot_type = mace"
  echo "    pot_path = $out"
  echo "This model knows ONLY ${ELEMENTS[*]}. Re-run with different --elements"
  echo "for another material; no new download is needed."
}

case "$FRAMEWORK" in
  deepmd) fetch_deepmd ;;
  mace)   fetch_mace ;;
  ""|-h|--help) usage 0 ;;
  *) die "unknown framework '$FRAMEWORK' (expected deepmd|mace)" ;;
esac
