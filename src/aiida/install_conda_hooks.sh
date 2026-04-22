#!/usr/bin/env bash
# Install AiiDA service activate/deactivate hooks into the current conda env.

set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "No active conda environment detected."
    echo "Activate your AiiDA env first, e.g.: conda activate frictionsim2d-aiida"
    exit 1
fi

HOOK_SRC_DIR="${CONDA_PREFIX}/libexec/frictionsim2d/hooks"
ACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
DEACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/deactivate.d"

mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"

install -m 0755 "${HOOK_SRC_DIR}/activate_services.sh" "${ACTIVATE_DIR}/frictionsim2d_aiida_activate.sh"
install -m 0755 "${HOOK_SRC_DIR}/deactivate_services.sh" "${DEACTIVATE_DIR}/frictionsim2d_aiida_deactivate.sh"

echo "Installed conda hooks:"
echo "  ${ACTIVATE_DIR}/frictionsim2d_aiida_activate.sh"
echo "  ${DEACTIVATE_DIR}/frictionsim2d_aiida_deactivate.sh"
echo
echo "Re-activate the environment to apply:"
echo "  conda deactivate && conda activate ${CONDA_DEFAULT_ENV:-frictionsim2d-aiida}"
