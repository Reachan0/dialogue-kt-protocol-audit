#!/bin/bash
# Runner used for the second-annotator (QATD2k) runs.
# Credentials are read from the environment; nothing secret is stored in this file.
set -euo pipefail
cd "$(dirname "$0")"
if [ -z "${GATEWAY_BASE_URL:-}" ] || [ -z "${GATEWAY_API_KEY:-}" ]; then
  echo "GATEWAY_BASE_URL and GATEWAY_API_KEY must be set in the environment" >&2
  exit 2
fi
exec env -u ANTHROPIC_AUTH_TOKEN -u ANTHROPIC_API_KEY -u ANTHROPIC_BASE_URL \
  python3 annotate_second_annotator.py "$@"
