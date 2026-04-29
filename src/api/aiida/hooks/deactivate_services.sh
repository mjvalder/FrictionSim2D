#!/usr/bin/env bash
# Stop local PostgreSQL and RabbitMQ services for the active conda env.

set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    exit 0
fi

if command -v rabbitmqctl >/dev/null 2>&1 && rabbitmqctl status >/dev/null 2>&1; then
    echo "Stopping RabbitMQ..."
    rabbitmqctl stop >/dev/null 2>&1 || true
fi

if [[ -f "${CONDA_PREFIX}/bin/pg_isready" ]] && "${CONDA_PREFIX}/bin/pg_isready" -q; then
    echo "Stopping PostgreSQL..."
    "${CONDA_PREFIX}/bin/pg_ctl" -D "${CONDA_PREFIX}/var/postgres" stop
fi
