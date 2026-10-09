#!/bin/bash
# Runner used for the MathDial annotation runs.
# Credentials are read from the environment; nothing secret is stored in this file.
#   export GATEWAY_BASE_URL=...   GATEWAY_API_KEY=...
#   ./run_mathdial.sh --protocol prefix --split test --run-dir RUN --max-calls 4000
set -euo pipefail
cd "$(dirname "$0")"
if [ -z "${GATEWAY_BASE_URL:-}" ] || [ -z "${GATEWAY_API_KEY:-}" ]; then
  echo "GATEWAY_BASE_URL and GATEWAY_API_KEY must be set in the environment" >&2
  exit 2
fi
# Drop inherited ANTHROPIC_* variables so no unrelated credential can be sent.
exec env -u ANTHROPIC_AUTH_TOKEN -u ANTHROPIC_API_KEY -u ANTHROPIC_BASE_URL \
  python3 annotate_mathdial.py "$@"
