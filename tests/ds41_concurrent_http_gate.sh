#!/usr/bin/env bash
# Live GB10 qualification gate; NEVER reported as passing by host-only CI.
# Usage: bash tests/ds41_concurrent_http_gate.sh BIN MODEL ENGRAM VISION IMAGE_A IMAGE_B [ZCHAIN|-]
# Caller must stop other GPU workloads. Uses an unused loopback port, and only
# terminates its own server PID. Keeps requests/responses/logs on any failure.
set -euo pipefail
[[ $# -ge 6 && $# -le 7 ]] || { echo "usage: $0 BIN MODEL ENGRAM VISION IMAGE_A IMAGE_B [ZCHAIN|-]" >&2; exit 2; }
BIN=$(realpath "$1"); MODEL=$(realpath "$2"); ENGRAM=$(realpath "$3"); VISION=$(realpath "$4")
A=$(realpath "$5"); B=$(realpath "$6"); ZCHAIN=${7:--}
for cmd in curl jq sha256sum base64; do command -v "$cmd" >/dev/null; done
[[ -x "$BIN" && -f "$MODEL" && -d "$ENGRAM" && -f "$VISION" && -f "$A" && -f "$B" ]] || exit 2
[[ "$(sha256sum "$A" | cut -d' ' -f1)" != "$(sha256sum "$B" | cut -d' ' -f1)" ]] || { echo 'Images must differ' >&2; exit 2; }
PORT=${DS41_GATE_PORT:-18041}; URL="http://127.0.0.1:$PORT"
[[ "$PORT" =~ ^[0-9]+$ && $PORT -ge 1024 && $PORT -le 65535 ]] || exit 2
OUT=${DS41_GATE_OUT:-"$(pwd)/runs/ds41-concurrency-$(date +%Y%m%d-%H%M%S)"}
mkdir -p "$OUT"; OUT=$(realpath "$OUT")
[[ ! -e "$OUT/start.txt" ]] || { echo 'Use a fresh output directory' >&2; exit 2; }
PID=''; cleanup(){ if [[ -n "$PID" ]]; then kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; PID=''; fi; }
trap cleanup EXIT INT TERM
fail(){ echo "FAIL: $*; evidence: $OUT" >&2; exit 1; }
printf '%s\n' "$(date -Is)" > "$OUT/start.txt"
sha256sum "$BIN" "$MODEL" "$VISION" "$A" "$B" > "$OUT/inputs.sha256"
git rev-parse HEAD > "$OUT/source.txt" 2>/dev/null || true
common=(--cuda -m "$MODEL" --engram-dir "$ENGRAM" --vision-gguf "$VISION" --mem-budget-mb 110000
        --mem-floor-gb 2 --mtp-mode off --prefix-reuse off --host 127.0.0.1 --port "$PORT" --model-id big --no-update-check)
[[ "$ZCHAIN" == '-' ]] || common+=(--zchain "$(realpath "$ZCHAIN")")
boot(){
    local width=$1
    if curl -s --max-time 1 "$URL/v1/models" >/dev/null; then fail "port $PORT already serves a process"; fi
    if command -v ss >/dev/null && ss -ltnH | awk '{print $4}' | grep -qE ":${PORT}$"; then fail "port $PORT is occupied"; fi
    "$BIN" "${common[@]}" --max-seqs "$width" > "$OUT/server-c${width}.log" 2>&1 & PID=$!
    for ((i=0;i<600;i++)); do
        kill -0 "$PID" 2>/dev/null || fail "C$width server exited"
        if curl -fsS --max-time 2 "$URL/v1/models" > "$OUT/models-c${width}.json" 2>/dev/null; then return; fi
        sleep 1
    done
    fail "C$width readiness timed out"
}
request(){
    local input=$1 output=$2
    curl -fsS --max-time 600 "$URL/v1/chat/completions" -H 'Content-Type: application/json' --data-binary "@$input" > "$output"
    jq -e '(.error == null) and (.choices|length == 1) and (.choices[0].finish_reason != "error")' "$output" >/dev/null
}
normalize(){ jq -S '.choices[0] | {finish_reason,message:(.message|del(.tool_calls[]?.id))}' "$1"; }
for k in 1 2 3 4; do
    jq -n --arg marker "REQUEST_${k}_ISOLATED" '{model:"big",temperature:0,reasoning_effort:"none",max_tokens:256,messages:[{role:"user",content:("Begin with "+$marker+". Then write a detailed numbered explanation of how a hash table works. Do not mention other request markers.")}]}' > "$OUT/text-$k.json"
done
# Encode by file, not --arg, to avoid ARG_MAX failures on real screenshots.
for pair in "a:$A" "b:$B"; do
    tag=${pair%%:*}; image=${pair#*:}; mime=$(file -b --mime-type "$image")
    [[ "$mime" == image/png || "$mime" == image/jpeg ]] || fail 'use PNG/JPEG fixtures'
    { printf 'data:%s;base64,' "$mime"; base64 -w0 "$image"; } > "$OUT/image-$tag.data"
    jq -n --rawfile image "$OUT/image-$tag.data" '{model:"big",temperature:0,reasoning_effort:"none",max_tokens:256,messages:[{role:"user",content:[{type:"text",text:"Describe this image, its objects, colors, and visible text in detail."},{type:"image_url",image_url:{url:$image}}]}]}' > "$OUT/image-$tag.json"
done
jq -n '{model:"big",temperature:0,reasoning_effort:"low",max_tokens:2048,messages:[{role:"user",content:"Call test_echo with value ping. Do not execute any other tool."}],tools:[{type:"function",function:{name:"test_echo",description:"Return the value",parameters:{type:"object",properties:{value:{type:"string"}},required:["value"]}}}],tool_choice:{type:"function",function:{name:"test_echo"}}}' > "$OUT/tool.json"
boot 1
for input in text-{1,2,3,4} image-{a,b} tool; do request "$OUT/$input.json" "$OUT/c1-$input.json" || fail "C1 $input"; done
cleanup
boot 4
waiters=(); for input in text-{1,2,3,4}; do request "$OUT/$input.json" "$OUT/c4-$input.json" & waiters+=("$!"); done
for p in "${waiters[@]}"; do wait "$p" || fail 'parallel text request'; done
# Four different active streams, including two distinct image histories.
waiters=(); for input in image-{a,b} text-{1,2}; do request "$OUT/$input.json" "$OUT/c4-media-$input.json" & waiters+=("$!"); done
for p in "${waiters[@]}"; do wait "$p" || fail 'parallel vision/text request'; done
for input in text-{1,2,3,4}; do
    normalize "$OUT/c1-$input.json" > "$OUT/reference.json"
    normalize "$OUT/c4-$input.json" > "$OUT/actual.json"
    diff -u "$OUT/reference.json" "$OUT/actual.json" > "$OUT/$input.diff" || fail "greedy C1/C4 drift: $input"
done
for input in image-{a,b}; do
    normalize "$OUT/c1-$input.json" > "$OUT/reference.json"
    normalize "$OUT/c4-media-$input.json" > "$OUT/actual.json"
    diff -u "$OUT/reference.json" "$OUT/actual.json" > "$OUT/$input.diff" || fail "image isolation/C1-C4 drift: $input"
done
# Do not pass on C4 configuration alone: prove a genuine four-row GPU step.
grep -q '\[v41-batch\] packed_decode rows=4 ' "$OUT/server-c4.log" || fail 'no actual four-row decode observed'
request "$OUT/tool.json" "$OUT/c4-tool.json" || fail 'tool request'
jq -e '.choices[0].finish_reason=="tool_calls" and .choices[0].message.tool_calls[0].function.name=="test_echo" and ((.choices[0].message.tool_calls[0].function.arguments|fromjson).value=="ping")' "$OUT/c4-tool.json" >/dev/null || fail 'DSML tool-call parse'
# Tool return stays in the normal HTTP history path, not a fabricated success.
jq --slurpfile result "$OUT/c4-tool.json" '.messages += [$result[0].choices[0].message,{role:"tool",tool_call_id:$result[0].choices[0].message.tool_calls[0].id,content:"ping"}] | .tool_choice="none"' "$OUT/tool.json" > "$OUT/tool-followup.json"
request "$OUT/tool-followup.json" "$OUT/c4-tool-followup.json" || fail 'tool followup'
jq -e '.choices[0].finish_reason=="stop" and (.choices[0].message.content|ascii_downcase|contains("ping"))' "$OUT/c4-tool-followup.json" >/dev/null || fail 'tool result not consumed'
# Disconnect one long stream while another request remains live.
jq '.stream=true | .max_tokens=4096' "$OUT/text-1.json" > "$OUT/cancel.json"
curl -sS --max-time 3 "$URL/v1/chat/completions" -H 'Content-Type: application/json' --data-binary "@$OUT/cancel.json" > "$OUT/cancel.sse" 2> "$OUT/cancel.err" & CANCEL=$!
request "$OUT/text-2.json" "$OUT/after-cancel.json" || fail 'peer request lost on disconnect'
wait "$CANCEL" || true
request "$OUT/text-3.json" "$OUT/after-cancel-2.json" || fail 'slot was not reusable'
kill -0 "$PID" || fail 'server died'
printf '%s\n' 'PASS: bounded C1/C4 greedy parity, image isolation, four-row execution, tool roundtrip and disconnect recovery' | tee "$OUT/result.txt"
printf 'Evidence: %s\n' "$OUT"
