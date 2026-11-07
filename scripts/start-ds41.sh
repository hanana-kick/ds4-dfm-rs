#!/usr/bin/env bash
# DeepSeek-V4.1-Flash-1017 iVQ — single DGX Spark server launcher.
# Defaults: model big, :8000, four active sequences, embedded MTP ON, Vision ON.
# Does not download weights, modify sudoers, stop other servers, or downgrade
# unsupported settings. The engine must implement the requested combination.
set -Eeuo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "$HERE/.." && pwd -P)"
MODEL_DIR="${DS41_MODEL_DIR:-$HOME/AI/models/ds41-1017-ivq}"
MODEL="${DS41_MODEL:-$MODEL_DIR/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf}"
ENGRAM="${DS41_ENGRAM_DIR:-$MODEL_DIR/deepseek-engram}"
VISION="${DS41_VISION_GGUF:-$MODEL_DIR/DeepSeek-V4.1-Flash.vision.gguf}"
BIN="${DS41_SERVER_BIN:-$ROOT/ds4-server}"
HOST="${DS41_HOST:-0.0.0.0}"
PORT="${DS41_PORT:-8000}"
MODEL_ID="${DS41_MODEL_ID:-big}"
MAX_SEQS="${DS41_MAX_SEQS:-4}"
MTP_MODE="${DS41_MTP_MODE:-on}"
PREFIX_REUSE="${DS41_PREFIX_REUSE:-off}"
MEM_BUDGET_MB="${DS41_MEM_BUDGET_MB:-110000}"
MEM_FLOOR_GB="${DS41_MEM_FLOOR_GB:-2}"
ZCHAIN="${DS41_ZCHAIN_DIR:-}"

die() { printf '[ds41] ERROR: %s\n' "$*" >&2; exit 1; }
usage() {
  cat <<'HELP'
Usage: bash scripts/start-ds41.sh [start|check|print]
  start    Check server capability, then run in the foreground (default)
  check    Run the engine's --check-config without starting a server
  print    Print the requested server command

Env: DS41_MODEL_DIR, DS41_MODEL, DS41_ENGRAM_DIR, DS41_VISION_GGUF,
     DS41_SERVER_BIN, DS41_HOST, DS41_PORT, DS41_MODEL_ID,
     DS41_MAX_SEQS (1..8; default 4), DS41_MTP_MODE (on by default),
     DS41_PREFIX_REUSE, DS41_MEM_BUDGET_MB, DS41_MEM_FLOOR_GB,
     DS41_ZCHAIN_DIR (optional, must match this exact model)
HELP
}
ACTION="${1:-start}"
[[ $# -le 1 ]] || { usage >&2; exit 2; }
case "$ACTION" in
  start|check|print) ;;
  -h|--help|help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac

[[ "$PORT" =~ ^[0-9]+$ ]] && (( 10#$PORT >= 1 && 10#$PORT <= 65535 )) \
  || die "invalid DS41_PORT: $PORT"
[[ "$MAX_SEQS" =~ ^[1-8]$ ]] || die "DS41_MAX_SEQS must be 1..8"
[[ "$MEM_BUDGET_MB" =~ ^[0-9]+$ ]] && (( MEM_BUDGET_MB > 0 )) \
  || die "invalid DS41_MEM_BUDGET_MB"
[[ "$MEM_FLOOR_GB" =~ ^[0-9]+$ ]] || die "invalid DS41_MEM_FLOOR_GB"
case "$MTP_MODE" in on|auto|off) ;; *) die "invalid DS41_MTP_MODE: $MTP_MODE" ;; esac
case "$PREFIX_REUSE" in off|auto|exact|partial) ;;
  *) die "invalid DS41_PREFIX_REUSE: $PREFIX_REUSE" ;;
esac
[[ -n "$MODEL_ID" ]] || die "DS41_MODEL_ID cannot be empty"

if [[ "$ACTION" != print ]]; then
  [[ -x "$BIN" ]] || die "server binary not found: $BIN (build: cd '$ROOT' && make cuda-spark)"
  [[ -s "$MODEL" ]] || die "GGUF not found: $MODEL (run bash '$HERE/prepare-ds41-model.sh')"
  [[ -s "$VISION" ]] || die "Vision GGUF not found: $VISION"
  [[ -d "$ENGRAM" ]] || die "Engram directory not found: $ENGRAM"
  for shard in model-00047-of-00048.safetensors model-00048-of-00048.safetensors; do
    [[ -s "$ENGRAM/$shard" ]] || die "missing Engram shard: $ENGRAM/$shard"
  done
fi

cmd=(
  "$BIN" --cuda
  -m "$MODEL"
  --engram-dir "$ENGRAM"
  --vision-gguf "$VISION"
  --mem-budget-mb "$MEM_BUDGET_MB"
  --mem-floor-gb "$MEM_FLOOR_GB"
  --max-seqs "$MAX_SEQS"
  --mtp-mode "$MTP_MODE"
  --prefix-reuse "$PREFIX_REUSE"
  --host "$HOST"
  --port "$PORT"
  --model-id "$MODEL_ID"
  --no-update-check
)
if [[ -n "$ZCHAIN" ]]; then
  [[ "$ACTION" == print || -d "$ZCHAIN" ]] || die "GR/RB sidecar missing: $ZCHAIN"
  cmd+=(--zchain "$ZCHAIN")
fi

if [[ "$ACTION" == print ]]; then
  printf '[ds41] requested command:\n'
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

printf '[ds41] model=%s port=%s max_seqs=%s mtp=%s vision=on prefix_reuse=%s\n' \
  "$MODEL_ID" "$PORT" "$MAX_SEQS" "$MTP_MODE" "$PREFIX_REUSE"

if [[ "$ACTION" == check ]]; then
  exec "${cmd[@]}" --check-config
fi

# No silent C1 fallback or MTP disable. Current feature-branch binaries without
# concurrent DSpark will reject C4 + MTP here; fix the engine, not these flags.
if ! "${cmd[@]}" --check-config >/dev/null; then
  die "engine rejected C${MAX_SEQS} + MTP=${MTP_MODE}; update the concurrent DSpark implementation. Refusing to silently turn off MTP."
fi
exec "${cmd[@]}"
