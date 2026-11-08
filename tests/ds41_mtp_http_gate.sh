#!/usr/bin/env bash
# Real-model C1 plain vs C4 MTP gate. Runs only on the model host.
# No download, sudo, clock changes, or stopping unrelated processes.
set -Eeuo pipefail
[[ $# -ge 6 && $# -le 7 ]] || {
  echo "usage: $0 BIN GGUF ENGRAM_DIR VISION_GGUF IMAGE_A IMAGE_B [ZCHAIN|-]" >&2; exit 2;
}
for c in curl jq realpath sha256sum base64 file ss; do command -v "$c" >/dev/null; done
BIN=$(realpath "$1"); MODEL=$(realpath "$2"); ENGRAM=$(realpath "$3"); VISION=$(realpath "$4")
IMAGE_A=$(realpath "$5"); IMAGE_B=$(realpath "$6"); ZCHAIN=${7:--}
[[ -x "$BIN" && -f "$MODEL" && -d "$ENGRAM" && -f "$VISION" && -f "$IMAGE_A" && -f "$IMAGE_B" ]]
[[ "$(sha256sum "$IMAGE_A" | cut -d' ' -f1)" != "$(sha256sum "$IMAGE_B" | cut -d' ' -f1)" ]] || { echo 'use different images' >&2; exit 2; }
PORT=${DS41_GATE_PORT:-18042}
[[ "$PORT" =~ ^[0-9]+$ && ${#PORT} -le 5 ]] && ((10#$PORT>1023 && 10#$PORT<65536)) || exit 2
OUT=${DS41_GATE_OUT:-"$PWD/runs/ds41-mtp-$(date +%Y%m%d-%H%M%S)-$$"}
mkdir -p "$OUT"; OUT=$(realpath "$OUT")
[[ ! -e "$OUT/start.txt" ]] || { echo 'use a fresh output directory' >&2; exit 2; }
URL="http://127.0.0.1:$PORT"; PID=''
stop() { if [[ -n "$PID" ]]; then kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true; PID=''; fi; }
trap stop EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
fail() { printf 'FAIL: %s; evidence: %s\n' "$*" "$OUT" >&2; exit 1; }
date -Is > "$OUT/start.txt"
sha256sum "$BIN" "$VISION" "$IMAGE_A" "$IMAGE_B" > "$OUT/inputs.sha256"
git rev-parse HEAD > "$OUT/source.txt" 2>/dev/null || true
common=(--cuda -m "$MODEL" --engram-dir "$ENGRAM" --vision-gguf "$VISION"
  --mem-budget-mb 110000 --mem-floor-gb 2 --prefix-reuse off
  --host 127.0.0.1 --port "$PORT" --model-id big --no-update-check)
[[ "$ZCHAIN" == '-' ]] || common+=(--zchain "$(realpath "$ZCHAIN")")
boot() {
  local width=$1 mtp=$2 tag=$3
  if ss -ltnH | awk '{print $4}' | grep -qE ":${PORT}$"; then fail "port $PORT is occupied"; fi
  DS41_ROLLBACK_CHECK=1 "$BIN" "${common[@]}" --max-seqs "$width" --mtp-mode "$mtp" > "$OUT/$tag.log" 2>&1 & PID=$!
  for ((i=0;i<600;i++)); do
    kill -0 "$PID" 2>/dev/null || fail "$tag server exited"
    if curl -fsS --max-time 2 "$URL/v1/models" > "$OUT/$tag-models.json" 2>/dev/null; then return; fi
    sleep 1
  done
  fail "$tag startup timed out"
}
request() {
  curl -fsS --max-time 900 "$URL/v1/chat/completions" -H 'Content-Type: application/json' --data-binary "@$1" > "$2"
  jq -e '(.error==null) and (.choices|length==1) and (.choices[0].finish_reason!="error")' "$2" >/dev/null
}
normal() { jq -S '.choices[0] | {finish_reason,message:(.message | del(.tool_calls[]?.id))}' "$1"; }
for i in 1 2 3 4; do
  jq -n --arg marker "REQUEST_${i}" '{model:"big",temperature:0,reasoning_effort:"none",max_tokens:512,messages:[{role:"user",content:("Start with "+$marker+" and print the integers from 1 through 500 in order, each followed by a one-sentence definition of an algorithm. Do not stop early.")}]}' > "$OUT/text-$i.json"
done
for pair in "a:$IMAGE_A" "b:$IMAGE_B"; do
  tag=${pair%%:*}; image=${pair#*:}; mime=$(file -b --mime-type "$image")
  [[ "$mime" == image/png || "$mime" == image/jpeg ]] || fail 'images must be PNG/JPEG'
  { printf 'data:%s;base64,' "$mime"; base64 -w0 "$image"; } > "$OUT/image-$tag.data"
  jq -n --rawfile image "$OUT/image-$tag.data" '{model:"big",temperature:0,reasoning_effort:"none",max_tokens:256,messages:[{role:"user",content:[{type:"text",text:"Describe the objects, colors and all visible text in this image in detail."},{type:"image_url",image_url:{url:$image}}]}]}' > "$OUT/image-$tag.json"
done
jq -n '{model:"big",temperature:0,reasoning_effort:"low",max_tokens:2048,messages:[{role:"user",content:"Call test_echo with value ping exactly once."}],tools:[{type:"function",function:{name:"test_echo",description:"Return value",parameters:{type:"object",properties:{value:{type:"string"}},required:["value"]}}}],tool_choice:{type:"function",function:{name:"test_echo"}}}' > "$OUT/tool.json"

boot 1 off c1-plain
for name in text-{1,2,3,4} image-{a,b} tool; do request "$OUT/$name.json" "$OUT/c1-$name.json" || fail "C1 $name"; done
stop
boot 4 on c4-mtp
pids=(); for name in text-{1,2,3,4}; do request "$OUT/$name.json" "$OUT/c4-$name.json" & pids+=("$!"); done
for p in "${pids[@]}"; do wait "$p" || fail 'C4 text request'; done
for name in text-{1,2,3,4}; do diff -u <(normal "$OUT/c1-$name.json") <(normal "$OUT/c4-$name.json") > "$OUT/$name.diff" || fail "greedy C1/C4 mismatch: $name"; done

# The same active group includes both real images and independent text histories.
pids=(); for name in image-{a,b} text-{1,2}; do request "$OUT/$name.json" "$OUT/c4-media-$name.json" & pids+=("$!"); done
for p in "${pids[@]}"; do wait "$p" || fail 'C4 image/text request'; done
for name in image-{a,b} text-{1,2}; do diff -u <(normal "$OUT/c1-$name.json") <(normal "$OUT/c4-media-$name.json") > "$OUT/media-$name.diff" || fail "C1/C4 media mismatch: $name"; done
request "$OUT/tool.json" "$OUT/c4-tool.json" || fail 'tool request'
jq -e '.choices[0] | (.finish_reason=="tool_calls") and (.message.tool_calls[0].function.name=="test_echo") and ((.message.tool_calls[0].function.arguments|fromjson).value=="ping")' "$OUT/c4-tool.json" >/dev/null || fail 'tool not parsed'
jq -s '{model:"big",temperature:0,reasoning_effort:"none",max_tokens:256,messages:(.[0].messages + [.[1].choices[0].message,{role:"tool",tool_call_id:.[1].choices[0].message.tool_calls[0].id,content:"ping"},{role:"user",content:"Repeat the exact tool result. Do not call another tool."}])}' "$OUT/tool.json" "$OUT/c4-tool.json" > "$OUT/roundtrip.json"
request "$OUT/roundtrip.json" "$OUT/roundtrip-response.json" || fail 'tool continuation'
jq -e '(.choices[0].message.content//"")|contains("ping")' "$OUT/roundtrip-response.json" >/dev/null || fail 'tool result lost'

# A configured width is not proof: require a real C4 step with draft rows.
awk '/\[v41-batch\] packed_decode rows=4 /{for(i=1;i<=NF;i++)if($i~/^verify_rows=/){split($i,a,"=");if(a[2]>4)ok=1}} END{exit !ok}' "$OUT/c4-mtp.log" || fail 'no actual C4+MTP verification step'
grep -Eq '\[v41-mtp\] request complete rounds=[1-9][0-9]* offered=[1-9][0-9]*' "$OUT/c4-mtp.log" || fail 'MTP never drafted'
if grep -Eq 'rollback check FAIL|CUDA error|illegal memory access' "$OUT/c4-mtp.log"; then fail 'native/CUDA error'; fi
printf 'PASS: real-model C4 MTP + text/images/tools; evidence: %s\n' "$OUT"
