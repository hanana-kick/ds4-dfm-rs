#!/usr/bin/env bash
# No GPU; validate all request waves, output parity, and rejection of fake MTP.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
TMP=$(mktemp -d)
trap 'rm -rf -- "$TMP"' EXIT
for cmd in node jq curl ss; do command -v "$cmd" >/dev/null; done
mkdir -p "$TMP/engram"
printf 'fake model\n' > "$TMP/model.gguf"
printf 'fake vision\n' > "$TMP/vision.gguf"
printf 'fake engram\n' > "$TMP/engram/model-00047-of-00048.safetensors"
printf 'fake engram\n' > "$TMP/engram/model-00048-of-00048.safetensors"
cat > "$TMP/launcher.sh" <<'LAUNCH'
#!/usr/bin/env bash
exec node "$DS41_C8_MOCK_SERVER"
LAUNCH
chmod +x "$TMP/launcher.sh"
port() {
  node -e 'let s=require("net").createServer();s.listen(0,"127.0.0.1",()=>{console.log(s.address().port);s.close()})'
}
PORT=$(port)
export DS41_C8_LAUNCHER="$TMP/launcher.sh"
export DS41_C8_MOCK_SERVER="$ROOT/tests/fixtures/ds41-c8/mock-server.cjs"
export DS41_SERVER_BIN=/bin/true
export DS41_MODEL="$TMP/model.gguf"
export DS41_VISION_GGUF="$TMP/vision.gguf"
export DS41_ENGRAM_DIR="$TMP/engram"
export DS41_C8_PORT="$PORT"
export DS41_C8_BOOT_TIMEOUT=20
export DS41_C8_REQUEST_TIMEOUT=30
export DS41_C8_MODES='c1-off c4-off c4-on c8-off c8-on'
export DS41_C8_OUT="$TMP/positive"
bash "$ROOT/tests/ds41_c8_mtp_gate.sh"
awk '/c8-on/ && /rows=8/ && /MTP=on/ {ok=1} END{exit !ok}' "$TMP/positive/proofs.tsv"
grep -q $'c8-on\ttext-1\tMATCH' "$TMP/positive/parity.tsv"
jq -e -s 'any(.[]; .mode=="c8-on" and .case=="thinkmix-think" and .status=="ok")' "$TMP/positive/summary.jsonl" >/dev/null

export DS41_C8_MODES='c1-off c8-on'
export DS41_C8_THINK=off
export DS41_C8_MEDIA=off
export DS41_C8_MOCK_OMIT_MTP=on
export DS41_C8_OUT="$TMP/no-mtp"
rc=0
bash "$ROOT/tests/ds41_c8_mtp_gate.sh" > "$TMP/no-mtp.txt" 2>&1 || rc=$?
[[ "$rc" == 1 ]] && grep -q 'MTP verify_rows>8' "$TMP/no-mtp.txt" || {
  echo 'FAIL: missing MTP was not detected' >&2; cat "$TMP/no-mtp.txt" >&2; exit 1;
}
unset DS41_C8_MOCK_OMIT_MTP
export DS41_C8_MOCK_OMIT_BATCH=on
export DS41_C8_OUT="$TMP/no-batch"
rc=0
bash "$ROOT/tests/ds41_c8_mtp_gate.sh" > "$TMP/no-batch.txt" 2>&1 || rc=$?
[[ "$rc" == 1 ]] && grep -q 'missing real C8 packed GPU step' "$TMP/no-batch.txt" || {
  echo 'FAIL: missing packed batch was not detected' >&2; cat "$TMP/no-batch.txt" >&2; exit 1;
}
unset DS41_C8_MOCK_OMIT_BATCH
export DS41_C8_SEED=0
rc=0
bash "$ROOT/tests/ds41_c8_mtp_gate.sh" > "$TMP/bad-seed.txt" 2>&1 || rc=$?
[[ "$rc" == 2 ]] && grep -q 'nonzero fixed seed' "$TMP/bad-seed.txt" || {
  echo 'FAIL: clock seed was not rejected' >&2; exit 1;
}
# Cold and warm C4 tool probes should work on a real HTTP double.
export DS41_C8_SEED=424242
export DS41_C8_MEDIA=on
export DS41_C8_OUT="$TMP/tool-probe"
bash "$ROOT/tests/ds41_c8_mtp_gate.sh" --probe-tool > "$TMP/probe-ok.log" 2>&1
[[ "$(cat "$TMP/tool-probe/c4-off/tool-cold.http")" == 200 ]]
[[ "$(cat "$TMP/tool-probe/c4-off/tool-warm.http")" == 200 ]]

# A 503 must fail with HTTP status, JSON error, and saved server evidence.
export DS41_C8_MOCK_TOOL_ERROR=on
export DS41_C8_OUT="$TMP/tool-error"
rc=0
bash "$ROOT/tests/ds41_c8_mtp_gate.sh" --probe-tool > "$TMP/probe-error.log" 2>&1 || rc=$?
[[ "$rc" == 1 ]] || { echo "expected probe failure (exit 1), got $rc" >&2; exit 1; }
grep -q 'HTTP=503' "$TMP/probe-error.log"
grep -q 'synthetic concurrent tool capacity refusal' "$TMP/probe-error.log"
bash "$ROOT/tests/ds41_c8_mtp_gate.sh" --inspect "$TMP/tool-error" > "$TMP/inspect.log"
grep -q 'HTTP=503' "$TMP/inspect.log"
grep -q 'synthetic concurrent tool capacity refusal' "$TMP/inspect.log"

echo 'ds41-c8-mtp mock: parity, eight-way MTP, seed guard and C4 tool diagnostics PASS'
