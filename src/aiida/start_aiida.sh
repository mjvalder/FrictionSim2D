#!/usr/bin/env bash
# =============================================================================
# start_aiida.sh – One-time AiiDA setup / startup script for FrictionSim2D
#
# Run this script after activating the frictionsim2d-aiida conda environment
# to start RabbitMQ and configure the AiiDA profile used by FrictionSim2D.
#
# Usage:
#   conda activate frictionsim2d-aiida
#   export PYTHONPATH=/path/to/FrictionSim2D
#   bash src/aiida/start_aiida.sh [--profile PROFILE_NAME]
#
# Options:
#   --profile NAME   AiiDA profile name (default: friction2d)
#   --no-daemon      Skip starting the AiiDA daemon
#   --help           Show this help message
# =============================================================================

set -euo pipefail

# --- Defaults ----------------------------------------------------------------
PROFILE_NAME="friction2d"
START_DAEMON=true

# --- Argument parsing --------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            PROFILE_NAME="$2"
            shift 2
            ;;
        --no-daemon)
            START_DAEMON=false
            shift
            ;;
        --help|-h)
            sed -n '2,20p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "  FrictionSim2D – AiiDA environment setup"
echo "  Profile: ${PROFILE_NAME}"
echo "============================================================"

# --- 1. Verify conda environment ---------------------------------------------
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "⚠  WARNING: No active conda environment detected."
    echo "   Run: conda activate frictionsim2d-aiida"
fi

# --- 2. Start RabbitMQ -------------------------------------------------------
echo ""
echo "▶ Starting RabbitMQ broker..."
if rabbitmqctl status &>/dev/null 2>&1; then
    echo "  ✓ RabbitMQ is already running"
else
    rabbitmq-server -detached
    echo "  ✓ RabbitMQ started in detached mode"
    # Give RabbitMQ time to initialise
    sleep 3
fi

# --- 3. Create / load AiiDA profile ------------------------------------------
echo ""
echo "▶ Configuring AiiDA profile '${PROFILE_NAME}'..."

if verdi profile list | grep -q "^${PROFILE_NAME}$" 2>/dev/null; then
    echo "  ✓ Profile '${PROFILE_NAME}' already exists"
    verdi profile setdefault "${PROFILE_NAME}"
else
    verdi presto \
        --profile-name "${PROFILE_NAME}" \
        --use-postgres
    echo "  ✓ Profile '${PROFILE_NAME}' created"
fi

# --- 4. Start AiiDA daemon ---------------------------------------------------
if [[ "${START_DAEMON}" == "true" ]]; then
    echo ""
    echo "▶ Starting AiiDA daemon..."
    verdi -p "${PROFILE_NAME}" daemon start 1
    echo "  ✓ Daemon started"
fi

# --- 5. Summary --------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Setup complete."
echo ""
echo "  Next steps:"
echo "   • Register your HPC computer and LAMMPS code:"
echo "       FrictionSim2D aiida setup"
echo ""
echo "   • Run a simulation with AiiDA tracking:"
echo "       FrictionSim2D run afm examples/afm_config.ini --aiida"
echo ""
echo "   • Check the daemon status at any time:"
echo "       verdi daemon status"
echo "============================================================"
