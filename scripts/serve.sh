#!/usr/bin/env bash
# Start the single-Spark DeepSeek V4.1 Flash service (foreground).
# Requests Vision, MTP and C4 together; does not silently disable features.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
MODEL_DIR="${DS41_MODEL_DIR:-$HOME/AI/models/ds41-1017-ivq}"
MODEL="${DS41_MODEL:-$MODEL_DIR/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf}"
ENGRAM="${DS41_ENGRAM_DIR:-$MODEL_DIR/deepseek-engram}"
VISION="${DS41_VISION_GGUF:-$MODEL_DIR/DeepSeek-V4.1-Flash.vision.gguf}"
BIN="${DS41_SERVER_BIN:-$ROOT/ds4-server}"
MODEL_ID="${DS41_MODEL_ID:-big}"
HOST="${DS41_HOST:-0.0.0.0}"
PORT="${DS41_PORT:-8000}"
MAX_SEQS="${DS41_MAX_SEQS:-4}"
MTP_MODE="${DS41_MTP_MODE:-on}"
PREFIX_REUSE="${DS41_PREFIX_REUSE:-off}"
MEM_BUDGET_MB="${DS41_MEM_BUDGET_MB:-110000}"
MEM_FLOOR_GB="${DS41_MEM_FLOOR_GB:-2}"
ZCHAIN="${DS41_ZCHAIN_DIR:-}"

die() { printf '[ds41] ERROR: %s\n' "$*" >&2; exit 1; }
[[ -x "$BIN" ]] || die "server not built: $BIN (run scripts/build.sh)"
[[ -s "$MODEL" ]] || die "model missing: $MODEL (run scripts/download-model.sh)"
[[ -d "$ENGRAM" ]] || die "Engram directory missing: $ENGRAM"
for name in model-00047-of-00048.safetensors model-00048-of-00048.safetensors; do
  [[ -s "$ENGRAM/$name" ]] || die "Engram shard missing: $ENGRAM/$name"
done
[[ -s "$VISION" ]] || die "Vision model missing: $VISION"
[[ "$MAX_SEQS" =~ ^[1-8]$ ]] || die 'DS41_MAX_SEQS must be 1..8'
[[ "$PORT" =~ ^[0-9]+$ ]] && (( 10#$PORT >= 1 && 10#$PORT <= 65535 )) || die 'DS41_PORT must be 1..65535'
[[ -n "$MODEL_ID" ]] || die 'DS41_MODEL_ID must not be empty'

args=(
  --cuda -m "$MODEL" --engram-dir "$ENGRAM" --vision-gguf "$VISION"
  --mem-budget-mb "$MEM_BUDGET_MB" --mem-floor-gb "$MEM_FLOOR_GB"
  --max-seqs "$MAX_SEQS" --mtp-mode "$MTP_MODE" --prefix-reuse "$PREFIX_REUSE"
  --host "$HOST" --port "$PORT" --model-id "$MODEL_ID" --no-update-check
)
if [[ -n "$ZCHAIN" ]]; then
  [[ -d "$ZCHAIN" ]] || die "GR/RB sidecar missing: $ZCHAIN"
  args+=(--zchain "$ZCHAIN")
fi

printf '[ds41] starting model=%s port=%s max_seqs=%s mtp=%s vision=on\n' "$MODEL_ID" "$PORT" "$MAX_SEQS" "$MTP_MODE"
exec "$BIN" "${args[@]}"
