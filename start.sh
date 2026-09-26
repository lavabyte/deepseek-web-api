#!/usr/bin/env bash
# Out-of-the-box launcher for Linux / macOS.
#
# There is nothing to install: the server has zero npm dependencies and Node
# strips the TypeScript types natively. This script only checks that a new
# enough Node.js is present and then starts the server.
#
# Usage:
#   ./start.sh            # start on the default port (8787)
#   PORT=9000 ./start.sh  # start on a custom port
set -euo pipefail
cd "$(dirname "$0")"

MIN_MAJOR=22
MIN_MINOR=6

if ! command -v node >/dev/null 2>&1; then
  echo "ERROR: Node.js is not installed." >&2
  echo "This server needs Node.js ${MIN_MAJOR}.${MIN_MINOR} or newer (no npm packages required)." >&2
  echo "Download: https://nodejs.org/" >&2
  exit 1
fi

NODE_VERSION="$(node -p 'process.versions.node')"
MAJOR="${NODE_VERSION%%.*}"
REST="${NODE_VERSION#*.}"
MINOR="${REST%%.*}"

if [ "$MAJOR" -lt "$MIN_MAJOR" ] || { [ "$MAJOR" -eq "$MIN_MAJOR" ] && [ "$MINOR" -lt "$MIN_MINOR" ]; }; then
  echo "ERROR: Node.js ${NODE_VERSION} is too old." >&2
  echo "This server needs Node.js ${MIN_MAJOR}.${MIN_MINOR} or newer (it strips TypeScript types natively)." >&2
  echo "Download: https://nodejs.org/" >&2
  exit 1
fi

# Optional: bootstrap .env from the example on first run. The server works without
# it (every setting has a default), so this is purely informational.
if [ ! -f .env ] && [ -f .env.example ]; then
  echo "Note: no .env found. Using built-in defaults (port 8787, host 127.0.0.1)."
  echo "      Copy .env.example to .env to change settings."
fi

echo "Starting deepseek-web-api on Node.js ${NODE_VERSION}..."
echo "OpenAI-compatible endpoint: http://127.0.0.1:${PORT:-8787}/v1"
echo
exec node src/server.mjs
