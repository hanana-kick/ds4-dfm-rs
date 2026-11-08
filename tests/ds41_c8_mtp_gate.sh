#!/usr/bin/env bash
# Live C1/C4/C8 x MTP OFF/ON parity, eight-way execution and reasoning gate.
# Starts one task-owned server at a time. No Python/downloads/sudo/global kills.
set -Eeuo pipefail
umask 077

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
LAUNCHER="${DS41_C8_LAUNCHER:-$ROOT/scripts/serve.sh}"
ANALYZER="$ROOT/tests/ds41_repetition_gate.sh"
MODEL_DIR="${DS41_MODEL_DIR:-$HOME/AI/models/ds41-1017-ivq}"
MODEL="${DS41_MODEL:-$MODEL_DIR/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf}"
ENGRAM="${DS41_ENGRAM_DIR:-$MODEL_DIR/deepseek-engram}"
VISION="${DS41_VISION_GGUF:-$MODEL_DIR/DeepSeek-V4.1-Flash.vision.gguf}"
BIN="${DS41_SERVER_BIN:-$ROOT/ds4-server}"
MODEL_ID="${DS41_MODEL_ID:-big}"
PORT="${DS41_C8_PORT:-18045}"
BOOT_TIMEOUT="${DS41_C8_BOOT_TIMEOUT:-600}"
TIMEOUT="${DS41_C8_REQUEST_TIMEOUT:-2400}"
TEXT_TOKENS="${DS41_C8_TEXT_TOKENS:-512}"
THINK_TOKENS="${DS41_C8_THINK_TOKENS:-8192}"
SEED="${DS41_C8_SEED:-424242}"
IMAGE_A="${DS41_C8_IMAGE_A:-$ROOT/tests/fixtures/ds41-repetition/yellow-face.png}"
IMAGE_B="${DS41_C8_IMAGE_B:-$ROOT/tests/fixtures/chat-template/images/blue.png}"
MEDIA="${DS41_C8_MEDIA:-on}"
THINK="${DS41_C8_THINK:-on}"
TOOL_POLICY="${DS41_C8_TOOL_POLICY:-record}"
WIDTHS="${DS41_C8_MODES:-c1-off c4-off c4-on c8-off c8-on}"
OUT=''
PID=''

usage() {
  cat <<'HELP'
Usage: bash tests/ds41_c8_mtp_gate.sh [--self-test|--help|--inspect DIR|--probe-tool]

Actual DGX Spark model gate (defaults: C1 plain, C4 plain/MTP, C8 plain/MTP).
Runs eight concurrent distinct requests for C8, then mixed Vision+text,
DSML tool roundtrip, and seeded high-thinking. Requires actual GPU logs:
  [v41-batch] packed_decode rows=8 ... verify_rows>8 (C8 MTP)
  [v41-mtp] request complete rounds>0 offered>0 (C8 MTP)

DS41_C8_MODES='c1-off c8-off c8-on' (requires c1-off; optional c1-on)
DS41_C8_IMAGE_A=/path/a.png  DS41_C8_IMAGE_B=/path/b.png
DS41_C8_MEDIA=on|off  DS41_C8_THINK=on|off
DS41_C8_TOOL_POLICY=record|strict (record: continue after tool failure,
  preserve failure evidence and finish with FAIL; strict: stop immediately)
DS41_C8_TEXT_TOKENS=512  DS41_C8_THINK_TOKENS=8192
DS41_C8_SEED=424242 (nonzero fixed; seed=0 uses native clock entropy)
DS41_C8_PORT=18045  DS41_C8_REQUEST_TIMEOUT=2400
DS41_C8_OUT=/fresh/nonexistent/directory
DS41_C8_LAUNCHER=/path/to/launcher  DS41_SERVER_BIN=/path/to/server
DS41_MODEL, DS41_MODEL_DIR, DS41_ENGRAM_DIR, DS41_VISION_GGUF,
DS41_MEM_BUDGET_MB, DS41_MEM_FLOOR_GB, DS41_ZCHAIN_DIR

Pass means observed eight independent active rows, real MTP target
verification, non-thinking greedy C1 parity, valid DSML tool result and
completed/repetition-free high thinking alongside 7 requests in C8.
Results/logs are kept. No GPU or model is run by GitHub host-only CI.
--inspect DIR explains existing HTTP failures, without loading a model.
--probe-tool tests C4/MTP OFF forced tool before/after text+Vision waves.
HELP
}
die() { printf '[ds41-c8] ERROR: %s\n' "$*" >&2; exit 2; }
fail() { printf '[ds41-c8] FAIL: %s; evidence: %s\n' "$*" "$OUT" >&2; exit 1; }
inspect_tool() {
  local dir=$1 mode=$2 name=$3 path="$1/$2/$3"
  [[ -f "$path.http" || -f "$path.response.json" || -f "$path.curl.err" ]] || return 0
  printf '[ds41-c8] %s/%s HTTP=%s\n' "$mode" "$name" "$(cat "$path.http" 2>/dev/null || printf unavailable)"
  if [[ -s "$path.curl.err" ]]; then
    printf 'curl error:\n'; tail -n 12 "$path.curl.err"
  fi
  if [[ -s "$path.response.json" ]]; then
    if ! jq -c '{error:(.error//null),finish:(.choices[0].finish_reason//null),tool_calls:((.choices[0].message.tool_calls//[])|length),usage:(.usage//null)}' "$path.response.json"; then
      printf 'Response is not valid JSON. First 400 bytes:\n'
      head -c 400 "$path.response.json" || true
      printf '\n'
    fi
  else
    printf 'No HTTP response body saved\n'
  fi
  if [[ -s "$dir/$mode/server.log" ]]; then
    printf 'Last 35 server log lines:\n'
    tail -n 35 "$dir/$mode/server.log"
  fi
}
request_failed() {
  local mode=$1 name=$2
  inspect_tool "$OUT" "$mode" "$name" >&2
  fail "$mode $name request rejected or invalid HTTP response"
}
stop_server() {
  if [[ -z "$PID" ]]; then return; fi
  kill -TERM "$PID" 2>/dev/null || true
  for ((i=0;i<40;i++)); do
    if ! kill -0 "$PID" 2>/dev/null; then break; fi
    sleep 0.2
  done
  if kill -0 "$PID" 2>/dev/null; then kill -KILL "$PID" 2>/dev/null || true; fi
  wait "$PID" 2>/dev/null || true
  PID=''
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

TOOL_PROBE=off
case "${1:-}" in
  --inspect)
    [[ $# -eq 2 && -d "$2" ]] || die 'usage: --inspect EXISTING_RUN_DIRECTORY'
    command -v jq >/dev/null || die 'jq required for inspection'
    for mode in c1-off c4-off c4-on c8-off c8-on; do
      for case_id in tool tool-cold tool-warm tool-followup; do
        inspect_tool "$2" "$mode" "$case_id"
      done
    done
    exit 0 ;;
  --probe-tool)
    [[ $# -eq 1 ]] || die 'unexpected arguments'
    TOOL_PROBE=on ;;

  --help|-h) usage; exit 0 ;;
  --self-test)
    [[ $# -eq 1 ]] || die 'unexpected arguments'
    bash -n "$ROOT/tests/ds41_c8_mtp_gate.sh"
    perl -x "$ANALYZER" --self-test
    exit 0 ;;
  '') [[ $# -eq 0 ]] || die 'unexpected arguments' ;;
  *) usage >&2; exit 2 ;;
esac
for cmd in curl jq perl ss base64 file mktemp sha256sum stat diff awk grep; do
  command -v "$cmd" >/dev/null || die "missing command: $cmd"
done
[[ "$PORT" =~ ^[0-9]{4,5}$ ]] && (( 10#$PORT <= 65535 )) || die 'port must be 1000..65535'
for val in "$BOOT_TIMEOUT" "$TIMEOUT" "$TEXT_TOKENS" "$THINK_TOKENS"; do
  [[ "$val" =~ ^[1-9][0-9]*$ ]] || die 'timeouts/budgets must be positive integers'
done
[[ "$SEED" =~ ^[1-9][0-9]{0,8}$ ]] || die 'nonzero fixed seed required (1..999999999)'
[[ "$MEDIA" == on || "$MEDIA" == off ]] || die 'DS41_C8_MEDIA must be on|off'
[[ "$THINK" == on || "$THINK" == off ]] || die 'DS41_C8_THINK must be on|off'
[[ "$TOOL_POLICY" == record || "$TOOL_POLICY" == strict ]] || die 'DS41_C8_TOOL_POLICY must be record|strict'
[[ -x "$LAUNCHER" && -x "$BIN" ]] || die 'build first: bash scripts/build.sh'
[[ -s "$MODEL" && -s "$VISION" && -d "$ENGRAM" ]] || die 'model/Engram/Vision missing'
for shard in model-00047-of-00048.safetensors model-00048-of-00048.safetensors; do
  [[ -s "$ENGRAM/$shard" ]] || die "Engram shard missing: $shard"
done
if [[ "$MEDIA" == on || "$THINK" == on ]]; then
  [[ -s "$IMAGE_A" ]] || die "image A missing: $IMAGE_A"
  [[ -s "$IMAGE_B" ]] || die "image B missing: $IMAGE_B"
  [[ "$(sha256sum "$IMAGE_A" | cut -d' ' -f1)" != "$(sha256sum "$IMAGE_B" | cut -d' ' -f1)" ]] || die 'image fixtures must differ'
  for img in "$IMAGE_A" "$IMAGE_B"; do
    mime=$(file -b --mime-type "$img")
    [[ "$mime" == image/png || "$mime" == image/jpeg ]] || die "PNG/JPEG required: $img"
  done
fi
read -r -a MODES <<< "$WIDTHS"
if [[ "$TOOL_PROBE" == on ]]; then MODES=(c4-off); fi
if [[ "$TOOL_PROBE" == off ]]; then
  (( ${#MODES[@]} > 1 )) || die 'at least c1-off + a concurrent mode required'
fi
if [[ "$TOOL_PROBE" == off ]]; then
  [[ "${MODES[0]}" == c1-off ]] || die 'c1-off must be the first mode'
fi
if [[ "$TOOL_PROBE" == off ]]; then
  [[ " ${MODES[*]} " == *' c8-on '* ]] || die 'c8-on must be selected to qualify C8 MTP'
fi
declare -A seen
for mode in "${MODES[@]}"; do
  case "$mode" in c1-off|c1-on|c4-off|c4-on|c8-off|c8-on) ;; *) die "invalid mode: $mode" ;; esac
  [[ ! -v seen[$mode] ]] || die "duplicate mode: $mode"
  seen[$mode]=1
done
[[ "$MODEL_ID" != '' ]] || die 'model ID is empty'

mkdir -p "$ROOT/runs"
if [[ -n "${DS41_C8_OUT:-}" ]]; then
  OUT="$DS41_C8_OUT"
  mkdir -- "$OUT" || die 'DS41_C8_OUT must be a fresh directory'
else
  OUT=$(mktemp -d "$ROOT/runs/ds41-c8-$(date +%Y%m%d-%H%M%S)-XXXXXXXX")
fi
OUT="$(cd -- "$OUT" && pwd -P)"
mkdir -p "$OUT/payloads"
printf '%s\n' "$(date -Is)" > "$OUT/start.txt"
printf '%s\n' "${MODES[*]}" > "$OUT/modes.txt"
printf 'seed=%s text_tokens=%s think_tokens=%s media=%s think=%s\n' "$SEED" "$TEXT_TOKENS" "$THINK_TOKENS" "$MEDIA" "$THINK" > "$OUT/settings.txt"
git -C "$ROOT" rev-parse HEAD > "$OUT/source.txt" 2>/dev/null || true
sha256sum "$BIN" > "$OUT/binary.sha256"
stat -c '%n size=%s mtime=%y' "$MODEL" "$VISION" "$BIN" > "$OUT/artifacts.txt"

# Eight long-running unique prompts ensure a chance for eight live GPU rows.
for ((i=1;i<=8;i++)); do
  jq -n --arg model "$MODEL_ID" --arg marker "REQUEST_${i}_ISOLATED" \
    --argjson tokens "$TEXT_TOKENS" \
    '{model:$model,stream:false,temperature:0,top_p:1,min_p:0,seed:424242,
      reasoning_effort:"none",max_tokens:$tokens,
      messages:[{role:"user",content:("Begin with "+$marker+". List and explain 150 distinct hash-table operations in numbered order, using a full sentence for each. Do not mention another request marker.")}]}' > "$OUT/payloads/text-$i.json"
done
if [[ "$MEDIA" == on ]]; then
  for pair in "a:$IMAGE_A" "b:$IMAGE_B"; do
    tag=${pair%%:*}; image=${pair#*:}; mime=$(file -b --mime-type "$image")
    { printf 'data:%s;base64,' "$mime"; base64 -w0 "$image"; } > "$OUT/payloads/image-$tag.data"
    jq -n --arg model "$MODEL_ID" --rawfile data "$OUT/payloads/image-$tag.data" \
      '{model:$model,stream:false,temperature:0,top_p:1,min_p:0,seed:424242,
        reasoning_effort:"none",max_tokens:512,
        messages:[{role:"user",content:[
          {type:"text",text:"Describe the objects and colors in this image. Give at least 15 detailed observations."},
          {type:"image_url",image_url:{url:$data}}]}]}' > "$OUT/payloads/image-$tag.json"
  done
fi
jq -n --arg model "$MODEL_ID" \
  '{model:$model,stream:false,temperature:0,seed:424242,reasoning_effort:"low",max_tokens:2048,
   messages:[{role:"user",content:"Call test_echo with value ping exactly once."}],
   tools:[{type:"function",function:{name:"test_echo",description:"Return value",parameters:{type:"object",properties:{value:{type:"string"}},required:["value"]}}}],
   tool_choice:{type:"function",function:{name:"test_echo"}}}' > "$OUT/payloads/tool.json"
if [[ "$THINK" == on ]]; then
  { printf 'data:%s;base64,' "$(file -b --mime-type "$IMAGE_A")"; base64 -w0 "$IMAGE_A"; } > "$OUT/payloads/think-image.data"
  jq -n --arg model "$MODEL_ID" --argjson seed "$SEED" \
    --rawfile image "$OUT/payloads/think-image.data" \
    '{model:$model,stream:false,temperature:1,top_p:1,min_p:0,seed:$seed,
      reasoning_effort:"high",max_tokens:8192,
      messages:[{role:"user",content:[
        {type:"text",text:"이게 뭔가요? 보이는 특징을 기준으로 판단하고 최종 답변을 간결하게 제시하세요."},
        {type:"image_url",image_url:{url:$image}}]}]}' > "$OUT/payloads/think.json"
  jq --argjson limit "$THINK_TOKENS" '.max_tokens=$limit' "$OUT/payloads/think.json" > "$OUT/payloads/think.tmp"
  mv "$OUT/payloads/think.tmp" "$OUT/payloads/think.json"
fi

URL="http://127.0.0.1:$PORT"
port_busy() { ss -ltnH | awk '{print $4}' | grep -Eq ":${PORT}$"; }
boot() {
  local mode=$1 width="${1:1:1}" mtp="${1##*-}" dir="$OUT/$1"
  mkdir -p "$dir"
  if port_busy; then fail "port $PORT occupied; other processes untouched"; fi
  printf '[ds41-c8] starting %s width=%s MTP=%s\n' "$mode" "$width" "$mtp"
  DS41_MODEL_DIR="$MODEL_DIR" DS41_MODEL="$MODEL" DS41_ENGRAM_DIR="$ENGRAM" \
    DS41_VISION_GGUF="$VISION" DS41_SERVER_BIN="$BIN" \
    DS41_HOST=127.0.0.1 DS41_PORT="$PORT" DS41_MODEL_ID="$MODEL_ID" \
    DS41_MAX_SEQS="$width" DS41_MTP_MODE="$mtp" DS41_PREFIX_REUSE=off \
    "$LAUNCHER" > "$dir/server.log" 2>&1 & PID=$!
  for ((second=0;second<BOOT_TIMEOUT;second++)); do
    kill -0 "$PID" 2>/dev/null || fail "$mode server exited before readiness"
    if curl -fsS --max-time 2 "$URL/v1/models" > "$dir/models.json" 2>/dev/null &&
       jq -e --arg id "$MODEL_ID" '(.data//[]) | any(.id==$id)' "$dir/models.json" >/dev/null; then
      return 0
    fi
    sleep 1
  done
  fail "$mode readiness timeout"
}
request() {
  local payload=$1 target=$2 http=000
  if ! http=$(curl --silent --show-error --connect-timeout 4 --max-time "$TIMEOUT" \
      -o "$target.response.json" -w '%{http_code}' \
      -H 'Content-Type: application/json' \
      --data-binary "@$OUT/payloads/$payload.json" \
      "$URL/v1/chat/completions" 2> "$target.curl.err"); then
    printf '000\n' > "$target.http"; return 1
  fi
  printf '%s\n' "$http" > "$target.http"
  [[ "$http" == 200 ]] && jq -e '(.error == null) and ((.choices//[])|length==1) and (.choices[0].finish_reason=="stop" or .choices[0].finish_reason=="length" or .choices[0].finish_reason=="tool_calls")' "$target.response.json" >/dev/null
}
normal() {
  jq -S '{finish_reason:(.choices[0].finish_reason//null),
    reasoning:(.choices[0].message.reasoning_content // .choices[0].message.reasoning // ""),
    content:(.choices[0].message.content // ""),
    tool_calls:[.choices[0].message.tool_calls[]? | {type,function}]}' "$1"
}
record() {
  local mode=$1 case_id=$2 effort=$3
  local path="$OUT/$mode/$case_id"
  [[ -s "$path.response.json" ]] || fail "no response for $mode/$case_id"
  local test_case="$case_id"
  if [[ "$case_id" == tool-followup ]]; then test_case=roundtrip; fi
  perl -x "$ANALYZER" "$path.response.json" "$mode" "$test_case" "$effort" > "$path.analysis.json"
  jq -e '(.reasoning_loop==false) and (.answer_loop==false)' "$path.analysis.json" >/dev/null || fail "$mode/$case_id repeated text detected"
  jq -nc --arg mode "$mode" --arg case "$case_id" --slurpfile v "$path.analysis.json" \
    '{mode:$mode,case:$case,status:$v[0].status,
      finish:$v[0].finish_reason,reasoning_chars:$v[0].reasoning_chars,
      answer_chars:$v[0].answer_chars,completion_tokens:$v[0].completion_tokens,
      repeated:$v[0].reasoning_loop}' >> "$OUT/summary.jsonl"
}
compare() {
  local name=$1 actual=$2 actual_case=${3:-$1}
  local ref="$OUT/c1-off/$name.response.json" path="$OUT/$actual/$actual_case.response.json"
  [[ -s "$ref" && -s "$path" ]] || fail "missing parity input $name $actual"
  if ! diff -u <(normal "$ref") <(normal "$path") > "$OUT/parity-$actual-$name.diff"; then
    printf '%s\t%s\tMISMATCH\n' "$actual" "$name" >> "$OUT/parity.tsv"
    fail "greedy parity mismatch C1/$actual $name"
  fi
  printf '%s\t%s\tMATCH\n' "$actual" "$name" >> "$OUT/parity.tsv"
  rm -f "$OUT/parity-$actual-$name.diff"
}
wave() {
  local mode=$1 prefix=$2; shift 2
  local pids=() names=() name
  for name in "$@"; do
    request "$name" "$OUT/$mode/$prefix$name" & pids+=("$!")
    names+=("$name")
  done
  local bad=0 i
  for ((i=0;i<${#pids[@]};i++)); do
    wait "${pids[i]}" || bad=$((bad+1))
  done
  (( bad==0 )) || fail "$mode $prefix wave: $bad failed HTTP request(s)"
  for name in "${names[@]}"; do
    effort=none
    if [[ "$name" == think ]]; then effort=high; fi
    record "$mode" "$prefix$name" "$effort"
  done
}
prove_batch() {
  local mode=$1 width=${1:1:1} mtp=${1##*-}
  if [[ "$width" == 1 ]]; then
    if [[ "$mtp" == on ]]; then
      grep -q 'DSpark drafter armed' "$OUT/$mode/server.log" || fail "missing C1 MTP activation log"
    fi
    return 0
  fi
  local log="$OUT/$mode/server.log"
  awk -v width="$width" -v mtp="$mtp" '
    /\[v41-batch\] packed_decode rows=/ {
      rows=-1; verify=-1
      for(i=1;i<=NF;i++) {
        if($i ~ /^rows=/) { split($i,a,"="); rows=a[2]+0 }
        if($i ~ /^verify_rows=/) { split($i,b,"="); verify=b[2]+0 }
      }
      if(rows==width) { packed=1; if(verify>width) draft=1 }
    }
    END { if(!packed || (mtp=="on" && !draft)) exit 1 }
  ' "$log" || fail "missing real C$width packed GPU step or MTP verify_rows>$width"
  if [[ "$mtp" == on ]]; then
    grep -Eq '\[v41-mtp\] request complete rounds=[1-9][0-9]* offered=[1-9][0-9]*' "$log" || fail "$mode MTP offered-token counters absent"
  fi
  if grep -Eqi 'rollback check FAIL|CUDA error|illegal memory access' "$log"; then fail "$mode native CUDA failure"; fi
  printf '%s\trows=%s\tMTP=%s\tobserved\n' "$mode" "$width" "$mtp" >> "$OUT/proofs.tsv"
}

: > "$OUT/summary.jsonl"
printf 'mode\tcase\tparity\n' > "$OUT/parity.tsv"
printf 'mode\twidth\tmtp\tresult\n' > "$OUT/proofs.tsv"
tool_failures=0
printf 'mode\tcase\treason\n' > "$OUT/tool-failures.tsv"
tool_failure() {
  local mode=$1 case_id=$2 reason=$3
  tool_failures=$((tool_failures + 1))
  printf '%s\t%s\t%s\n' "$mode" "$case_id" "$reason" >> "$OUT/tool-failures.tsv"
  inspect_tool "$OUT" "$mode" "$case_id" >&2
  printf '[ds41-c8] RECORDED FAIL: %s/%s (%s); continuing to later modes\n' \
    "$mode" "$case_id" "$reason" >&2
  if [[ "$TOOL_POLICY" == strict ]]; then
    fail "$mode/$case_id: $reason"
  fi
}
if [[ "$TOOL_PROBE" == on ]]; then
  boot c4-off
  for round in cold warm; do
    if [[ "$round" == warm ]]; then
      wave c4-off '' text-1 text-2 text-3 text-4
      wave c4-off '' text-5 text-6 text-7 text-8
      if [[ "$MEDIA" == on ]]; then
        wave c4-off media- image-a image-b text-1 text-2
      fi
    fi
    request tool "$OUT/c4-off/tool-$round" || request_failed c4-off "tool-$round"
    record c4-off "tool-$round" low
    jq -e '.choices[0].finish_reason=="tool_calls" and
      (.choices[0].message.tool_calls|length)==1 and
      .choices[0].message.tool_calls[0].function.name=="test_echo" and
      ((.choices[0].message.tool_calls[0].function.arguments|fromjson).value=="ping")' \
      "$OUT/c4-off/tool-$round.response.json" >/dev/null || fail "c4-off tool-$round malformed"
    printf '[ds41-c8] C4 tool %s: HTTP 200, one valid test_echo call\n' "$round"
  done
  stop_server
  printf '[ds41-c8] PROBE PASS (not C8 qualification); evidence: %s\n' "$OUT"
  exit 0
fi
for mode in "${MODES[@]}"; do
  boot "$mode"
  width=${mode:1:1}
  if [[ "$width" == 1 ]]; then
    for ((i=1;i<=8;i++)); do
      request "text-$i" "$OUT/$mode/text-$i" || fail "$mode text-$i"
      record "$mode" "text-$i" none
    done
  elif [[ "$width" == 4 ]]; then
    wave "$mode" '' text-1 text-2 text-3 text-4
    wave "$mode" '' text-5 text-6 text-7 text-8
  else
    wave "$mode" '' text-1 text-2 text-3 text-4 text-5 text-6 text-7 text-8
  fi
  if [[ "$mode" != c1-off ]]; then
    for ((i=1;i<=8;i++)); do compare "text-$i" "$mode"; done
  fi
  if [[ "$MEDIA" == on ]]; then
    if [[ "$width" == 1 ]]; then
      for name in image-a image-b; do
        request "$name" "$OUT/$mode/$name" || fail "$mode $name"
        record "$mode" "$name" none
      done
    elif [[ "$width" == 4 ]]; then
      wave "$mode" media- image-a image-b text-1 text-2
    else
      wave "$mode" media- image-a image-b text-1 text-2 text-3 text-4 text-5 text-6
    fi
    if [[ "$mode" != c1-off ]]; then
      for name in image-a image-b; do compare "$name" "$mode" "media-$name"; done
      for ((i=1;i<=width-2;i++)); do compare "text-$i" "$mode" "media-text-$i"; done
    fi
  fi
  # Tool failures are evidence, not permission to claim a pass. In 'record'
  # mode they must not hide C8 packed-MTP, Vision, or Thinking results.
  tool_valid=0
  if request tool "$OUT/$mode/tool"; then
    record "$mode" tool low
    if jq -e '.choices[0].finish_reason=="tool_calls" and
        (.choices[0].message.tool_calls|length)==1 and
        .choices[0].message.tool_calls[0].function.name=="test_echo" and
        ((.choices[0].message.tool_calls[0].function.arguments|fromjson).value=="ping")' \
      "$OUT/$mode/tool.response.json" >/dev/null; then
      tool_valid=1
    else
      tool_failure "$mode" tool 'no valid test_echo tool call'
    fi
  else
    if [[ -s "$OUT/$mode/tool.response.json" ]] &&
       jq -e '(.choices//[]|length)>0' "$OUT/$mode/tool.response.json" >/dev/null 2>&1; then
      record "$mode" tool low
    fi
    tool_failure "$mode" tool 'HTTP transport, response shape, or finish_reason error'
  fi
  if (( tool_valid == 1 )); then
    if jq -s --arg model "$MODEL_ID" '{model:$model,stream:false,temperature:0,
       reasoning_effort:"none",max_tokens:256,
       messages:(.[0].messages + [.[1].choices[0].message,
         {role:"tool",tool_call_id:.[1].choices[0].message.tool_calls[0].id,content:"ping"},
         {role:"user",content:"Repeat the tool result exactly, do not call a tool."}])}' \
      "$OUT/payloads/tool.json" "$OUT/$mode/tool.response.json" \
      > "$OUT/payloads/tool-followup.json"; then
      if request tool-followup "$OUT/$mode/tool-followup"; then
        record "$mode" tool-followup none
        if ! jq -e '(.choices[0].message.content // "") | contains("ping")' \
          "$OUT/$mode/tool-followup.response.json" >/dev/null; then
          tool_failure "$mode" tool-followup 'tool result not reflected'
        fi
      else
        if [[ -s "$OUT/$mode/tool-followup.response.json" ]] &&
           jq -e '(.choices//[]|length)>0' "$OUT/$mode/tool-followup.response.json" >/dev/null 2>&1; then
          record "$mode" tool-followup none
        fi
        tool_failure "$mode" tool-followup 'HTTP transport, response shape, or finish_reason error'
      fi
    else
      tool_failure "$mode" tool 'could not construct tool continuation'
    fi
  fi
  if [[ "$THINK" == on ]]; then
    if [[ "$width" == 1 ]]; then
      request think "$OUT/$mode/think" || fail "$mode high-thinking request"
      record "$mode" think high
      quality="$OUT/$mode/think.analysis.json"
    elif [[ "$width" == 4 ]]; then
      wave "$mode" thinkmix- think text-1 text-2 text-3
      quality="$OUT/$mode/thinkmix-think.analysis.json"
    else
      wave "$mode" thinkmix- think text-1 text-2 text-3 text-4 text-5 text-6 text-7
      quality="$OUT/$mode/thinkmix-think.analysis.json"
    fi
    jq -e '.status=="ok" and .finish_reason=="stop" and
      .final_answer_present==true and .reasoning_exposed==true and
      .reasoning_loop==false and .answer_loop==false' \
      "$quality" >/dev/null || fail "$mode high-thinking failed/truncated/repeated"
  fi
  prove_batch "$mode"
  stop_server
  printf '[ds41-c8] completed %s\n' "$mode"
done
jq -r -s '(["mode","case","status","finish","reasoning_chars","answer_chars","tokens"]|@tsv),
  (.[]|[.mode,.case,.status,.finish,(.reasoning_chars|tostring),(.answer_chars|tostring),(.completion_tokens|tostring)]|@tsv)' \
  "$OUT/summary.jsonl" > "$OUT/summary.tsv"
cat "$OUT/proofs.tsv" "$OUT/parity.tsv" "$OUT/tool-failures.tsv"
if (( tool_failures > 0 )); then
  printf '[ds41-c8] FAIL: %s tool failure(s) recorded; packed/MTP evidence collected. evidence: %s\n' \
    "$tool_failures" "$OUT" >&2
  exit 1
fi
printf '[ds41-c8] PASS: C8 packed decode/MTP verified with greedy parity, tools and reasoning; evidence: %s\n' "$OUT"
