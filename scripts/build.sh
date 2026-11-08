#!/usr/bin/env bash
# Build the DeepSeek V4.1 Flash server for one NVIDIA DGX Spark.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
command -v make >/dev/null 2>&1 || { echo '[ds41] make is required' >&2; exit 1; }

cd "$ROOT"
make cuda-spark
[[ -x "$ROOT/ds4-server" ]] || { echo '[ds41] build did not produce ./ds4-server' >&2; exit 1; }
printf '[ds41] built: %s\n' "$ROOT/ds4-server"
