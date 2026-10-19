#!/usr/bin/env bash
set -Eeuo pipefail
# Explicit opt-in; never kill or interfere with any production inference process.
MODEL_DIR=$(printenv MODEL_DIR || printf '/home/hana/AI/ds4-dfm-rs/gguf/DeepSeek-V4.1-Flash-1017')
RUN_MODE=$(printenv RUN_MODE || printf 'gpu-baseline')
ROOT=$PWD; OUT=$ROOT/.artifacts/v41; PORT=18080
BASELINE=$HOME/.cache/dfm-v41-continuous/serial-baseline.json
mkdir -p "$OUT" "$(dirname "$BASELINE")"
case "$RUN_MODE" in
  gpu-baseline) N=1; MODE=record; unset DS41_EXPERIMENTAL_BATCH;;
  gpu-batch-2|gpu-batch-4|gpu-batch-8|gpu-batch-16)
    N=$(printf %s "$RUN_MODE"|cut -d- -f3); MODE=compare; export DS41_EXPERIMENTAL_BATCH=1;;
  *) echo "Unsupported RUN_MODE=$RUN_MODE" >&2; exit 2;;
esac
MODEL=$MODEL_DIR/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf
VISION=$MODEL_DIR/DeepSeek-V4.1-Flash.vision.gguf
ENGRAM=$MODEL_DIR/deepseek-engram
for p in "$MODEL" "$VISION" "$ENGRAM/model-00047-of-00048.safetensors" "$ENGRAM/model-00048-of-00048.safetensors"; do
  test -r "$p" || { echo "Missing $p" >&2; exit 2; }
done
if ss -ltnH|grep -Eq ':(8000|18080)[[:space:]]'; then
  echo "An inference port is active; refusing GPU gate" >&2; exit 2
fi
if pgrep -x ds4-server >/dev/null || pgrep -x vllm >/dev/null; then
  echo "Another inference engine is active; refusing GPU gate" >&2; exit 2
fi
SERVER_PID=
cleanup() {
  if test -n "$SERVER_PID" && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID"; wait "$SERVER_PID" || true
  fi
}
trap cleanup EXIT
./ds4-server --cuda -m "$MODEL" --vision-gguf "$VISION" \
  --engram-dir "$ENGRAM" --mem-budget-mb 110000 \
  --mtp-mode off --prefix-reuse off --max-seqs "$N" \
  --model-id big --host 127.0.0.1 --port "$PORT" >"$OUT/server.log" 2>&1 &
SERVER_PID=$!
for i in $(seq 1 240); do
  kill -0 "$SERVER_PID" 2>/dev/null || { tail -100 "$OUT/server.log"; exit 1; }
  if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/v1/models" >"$OUT/models.json" 2>/dev/null; then break; fi
  sleep 3
done
test -s "$OUT/models.json" || exit 1
curl -fsS "http://127.0.0.1:$PORT/v1/stats" >"$OUT/stats.json"
python3 - "$OUT/stats.json" "$N" <<'PY'
import json,sys
a=json.load(open(sys.argv[1]))
n=int(sys.argv[2])
assert ((a.get("serving") or {}).get("effective") or {}).get("max_seqs")==n, "not running requested max_seqs"
PY
if test "$MODE" = record; then
  python3 scripts/ci/v41_api_gate.py --base-url "http://127.0.0.1:$PORT" --record --out "$OUT/serial-baseline.json"
  cp "$OUT/serial-baseline.json" "$BASELINE"
else
  test -r "$BASELINE" || { echo "Run gpu-baseline first" >&2; exit 2; }
  python3 scripts/ci/v41_api_gate.py --base-url "http://127.0.0.1:$PORT" --concurrency "$N" --compare "$BASELINE" --out "$OUT/batch-$N.json"
  # Proof is actual multi-row GPU dispatch, not merely max-seqs=N in the config.
  grep -Eq '\[v41-batch\] forward sequences=([2-9]|1[0-6])([[:space:]]|$)' "$OUT/server.log" || {
    echo "Native multi-sequence Forward was not observed" >&2; exit 1;
  }
fi
echo "PASS $RUN_MODE" | tee "$OUT/result.txt"
