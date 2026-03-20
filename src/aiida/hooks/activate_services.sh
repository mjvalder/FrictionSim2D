#!/usr/bin/env bash
# Start local PostgreSQL and RabbitMQ services for the active conda env.

set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "No active conda environment detected; skipping AiiDA service startup."
    exit 0
fi

# PostgreSQL -----------------------------------------------------------------
if [[ ! -d "${CONDA_PREFIX}/var/postgres" ]]; then
    echo "Initializing PostgreSQL data directory..."
    mkdir -p "${CONDA_PREFIX}/var"
    "${CONDA_PREFIX}/bin/initdb" -D "${CONDA_PREFIX}/var/postgres"
fi

if ! "${CONDA_PREFIX}/bin/pg_isready" -q; then
    echo "Starting PostgreSQL..."
    "${CONDA_PREFIX}/bin/pg_ctl" \
        -D "${CONDA_PREFIX}/var/postgres" \
        -l "${CONDA_PREFIX}/var/postgres/server.log" start
fi

# RabbitMQ -------------------------------------------------------------------
RABBITMQ_CONSUMER_TIMEOUT_MS="${RABBITMQ_CONSUMER_TIMEOUT_MS:-36000000}"
RABBITMQ_CONF_DIR="${CONDA_PREFIX}/etc/rabbitmq"
RABBITMQ_CONF_FILE="${RABBITMQ_CONF_DIR}/rabbitmq.conf"

mkdir -p "${RABBITMQ_CONF_DIR}"
if [[ -f "${RABBITMQ_CONF_FILE}" ]]; then
    if grep -Eq '^\s*consumer_timeout\s*=' "${RABBITMQ_CONF_FILE}"; then
        sed -i -E "s|^\s*consumer_timeout\s*=.*$|consumer_timeout = ${RABBITMQ_CONSUMER_TIMEOUT_MS}|" "${RABBITMQ_CONF_FILE}"
    else
        printf "\nconsumer_timeout = %s\n" "${RABBITMQ_CONSUMER_TIMEOUT_MS}" >> "${RABBITMQ_CONF_FILE}"
    fi
else
    cat > "${RABBITMQ_CONF_FILE}" <<EOF
# FrictionSim2D AiiDA RabbitMQ settings
consumer_timeout = ${RABBITMQ_CONSUMER_TIMEOUT_MS}
EOF
fi

if ! rabbitmqctl status >/dev/null 2>&1; then
    echo "Starting RabbitMQ..."
    rabbitmq-server -detached
    sleep 3
fi
