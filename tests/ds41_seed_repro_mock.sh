#!/usr/bin/env bash
# End-to-end seeded retry + reference replay against a tiny Node HTTP double.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
TMP="$(mktemp -d)"
REF_PID=''
cleanup() {
  if [[ -n "$REF_PID" ]]; then
    kill -TERM "$REF_PID" 2>/dev/null || true
    wait "$REF_PID" 2>/dev/null || true
  fi
  rm -rf -- "$TMP"
}
trap cleanup EXIT

for cmd in node curl jq; do
  command -v "$cmd" >/dev/null || { echo "missing $cmd" >&2; exit 2; }
done
mkdir -p "$TMP/engram"
printf 'mock model\n' > "$TMP/model.gguf"
printf 'mock vision\n' > "$TMP/vision.gguf"
printf 'mock shard 47\n' > "$TMP/engram/model-00047-of-00048.safetensors"
printf 'mock shard 48\n' > "$TMP/engram/model-00048-of-00048.safetensors"
cat > "$TMP/launcher.sh" <<'LAUNCH'
#!/usr/bin/env bash
exec node "$DS41_MOCK_SERVER"
LAUNCH
chmod +x "$TMP/launcher.sh"

free_port() {
  node -e 'const s=require("net").createServer();
  s.listen(0,"127.0.0.1",()=>{console.log(s.address().port);s.close()})'
}
PORT="$(free_port)"
MOCK="$ROOT/tests/fixtures/ds41-repro/mock-server.cjs"

local_status=0
DS41_MOCK_SERVER="$MOCK" \
DS41_SERVER_BIN=/bin/true \
DS41_REPRO_LAUNCHER="$TMP/launcher.sh" \
DS41_MODEL="$TMP/model.gguf" \
DS41_VISION_GGUF="$TMP/vision.gguf" \
DS41_ENGRAM_DIR="$TMP/engram" \
DS41_REPRO_SEEDS='17 1' \
DS41_REPRO_PORT="$PORT" \
DS41_REPRO_OUT="$TMP/run" \
bash "$ROOT/tests/ds41_seed_repro_gate.sh" || local_status=$?

[[ "$local_status" == 1 ]] || {
  echo "expected one reproduced local failure, got exit $local_status" >&2
  exit 1
}
jq -e '.failed==1 and .passed==1 and
  (.rows | any(.seed==17 and .status=="reproducible_failure" and
               .replay=="completed_at_16k")) and
  (.rows | any(.seed==1 and .status=="pass"))' "$TMP/run/report.json" >/dev/null
jq -e '.messages[0].content[1].image_url.url |
  startswith("data:image/png;base64,")' \
  "$TMP/run/payloads/seed-17-a.json" >/dev/null

REF_PORT="$(free_port)"
MOCK_PORT="$REF_PORT" node "$MOCK" > "$TMP/reference.log" 2>&1 & REF_PID=$!
for ((i=0;i<50;i++)); do
  if curl --silent --fail --max-time 1 \
    "http://127.0.0.1:$REF_PORT/v1/models" >/dev/null; then
    break
  fi
  kill -0 "$REF_PID" || { cat "$TMP/reference.log" >&2; exit 1; }
  sleep 0.1
done
DS41_REPRO_REFERENCE_URL="http://127.0.0.1:$REF_PORT" \
  bash "$ROOT/tests/ds41_seed_repro_gate.sh" --reference "$TMP/run"

jq -e '.reference_runs==1 and
  .greedy_reference_exact_match==true and
  (.rows | any(.seed==17 and .reference=="both_failed"))' \
  "$TMP/run/report.json" >/dev/null
# Reject seed 0 before starting a server: native V4.1 turns it into
# wall-clock entropy and it must never enter fixed-seed reproducibility.
invalid=0
DS41_REPRO_SEEDS='0 1' DS41_SERVER_BIN=/bin/true \
DS41_REPRO_LAUNCHER="$TMP/launcher.sh" \
DS41_MODEL="$TMP/model.gguf" DS41_VISION_GGUF="$TMP/vision.gguf" \
DS41_ENGRAM_DIR="$TMP/engram" \
bash "$ROOT/tests/ds41_seed_repro_gate.sh" > "$TMP/invalid.log" 2>&1 || invalid=$?
[[ "$invalid" == 2 ]] && grep -q 'seed=0 selects clock entropy' "$TMP/invalid.log" || {
  echo 'seed 0 was not rejected' >&2; cat "$TMP/invalid.log" >&2; exit 1;
}

echo 'seed-repro-mock: local replay, reference comparison and seed-zero guard PASS'
