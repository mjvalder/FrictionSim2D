#!/usr/bin/env bash
# Configure RabbitMQ consumer timeout for AiiDA in the active conda env.
#
# Newer RabbitMQ versions (>=3.8.15 / v4.x) default to a 30-minute consumer
# timeout, which causes long-running AiiDA workflows to crash. This script
# writes a safe value to $CONDA_PREFIX/etc/rabbitmq/rabbitmq.conf.
#
# Run once after activating your conda environment, then restart RabbitMQ:
#
#   conda activate fs2d_test
#   bash src/aiida/configure_rabbitmq.sh
#   rabbitmqctl shutdown   # if already running
#   rabbitmq-server -detached
#
# The timeout can be overridden:
#   RABBITMQ_CONSUMER_TIMEOUT_MS=7200000 bash configure_rabbitmq.sh
#
# See: https://github.com/aiidateam/aiida-core/wiki/RabbitMQ-version-to-use

set -euo pipefail

TIMEOUT_MS="${RABBITMQ_CONSUMER_TIMEOUT_MS:-36000000}"  # default: 10 hours

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "ERROR: No active conda environment detected."
    echo "Activate your environment first, e.g.: conda activate fs2d_test"
    exit 1
fi

CONF_DIR="${CONDA_PREFIX}/etc/rabbitmq"
CONF_FILE="${CONF_DIR}/rabbitmq.conf"

mkdir -p "${CONF_DIR}"

ENTRY="consumer_timeout = ${TIMEOUT_MS}"

if [[ -f "${CONF_FILE}" ]]; then
    if grep -Eq '^\s*consumer_timeout\s*=' "${CONF_FILE}"; then
        sed -i -E "s|^\s*consumer_timeout\s*=.*$|${ENTRY}|" "${CONF_FILE}"
        echo "Updated existing consumer_timeout in ${CONF_FILE}"
    else
        printf '\n%s\n' "${ENTRY}" >> "${CONF_FILE}"
        echo "Appended consumer_timeout to ${CONF_FILE}"
    fi
else
    cat > "${CONF_FILE}" <<EOF
# FrictionSim2D AiiDA RabbitMQ settings
${ENTRY}
EOF
    echo "Created ${CONF_FILE}"
fi

echo "  consumer_timeout = ${TIMEOUT_MS} ms ($(( TIMEOUT_MS / 3600000 )) hour(s))"
echo ""
echo "If RabbitMQ is already running, restart it to apply:"
echo "  rabbitmqctl shutdown && rabbitmq-server -detached"
