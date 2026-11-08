#!/usr/bin/env bash
# Start a real DeepSeek-V4.1 server in each C1/C4 x MTP mode and detect
# repeated reasoning, missing termination, broken tool calls, and C1 drift.
# No sudo, downloads, Python, global kills, or changes to the normal server.
set -Eeuo pipefail
umask 077

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
ANALYZER="$ROOT/tests/ds41_repetition_gate.sh"
LAUNCHER="${DS41_TEST_LAUNCHER:-$ROOT/scripts/serve.sh}"
MODEL_DIR="${DS41_MODEL_DIR:-$HOME/AI/models/ds41-1017-ivq}"
MODEL="${DS41_MODEL:-$MODEL_DIR/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf}"
ENGRAM="${DS41_ENGRAM_DIR:-$MODEL_DIR/deepseek-engram}"
VISION="${DS41_VISION_GGUF:-$MODEL_DIR/DeepSeek-V4.1-Flash.vision.gguf}"
BIN="${DS41_SERVER_BIN:-$ROOT/ds4-server}"
PORT="${DS41_TEST_PORT:-18043}"
BOOT_TIMEOUT="${DS41_TEST_BOOT_TIMEOUT:-600}"
REQUEST_TIMEOUT="${DS41_TEST_REQUEST_TIMEOUT:-900}"
TOKENS="${DS41_TEST_MAX_TOKENS:-2048}"
EFFORT="${DS41_TEST_REASONING_EFFORT:-high}"
DIAGNOSE="${DS41_TEST_DIAGNOSE:-off}"
VERIFY="${DS41_TEST_VERIFY:-off}"
VERIFY_HIGH_TOKENS="${DS41_VERIFY_HIGH_TOKENS:-16384}"
VERIFY_SEED="${DS41_VERIFY_SEED:-424242}"
VERIFIER="$ROOT/tests/ds41_thinking_verdict.pl"
MODEL_ID="${DS41_MODEL_ID:-big}"
MEM_BUDGET="${DS41_MEM_BUDGET_MB:-110000}"
MEM_FLOOR="${DS41_MEM_FLOOR_GB:-2}"
TEXT_PROMPT="${DS41_TEST_TEXT_PROMPT:-이게 뭔가요? 노란 배경의 아이콘에 검은 타원형 눈 두 개, 쉼표 같은 코, 곡선형 입이 있습니다. 확실하지 않으면 추측이라고 명시하고 간결하게 답하세요.}"
IMAGE_PROMPT="${DS41_TEST_IMAGE_PROMPT:-이게 뭔가요? 보이는 특징을 기준으로 판단하고 최종 답변을 간결하게 제시하세요.}"
FIXTURE_IMAGE="$ROOT/tests/fixtures/ds41-repetition/yellow-face.png"
IMAGE="${DS41_TEST_IMAGE:-${1:-}}"
if [[ ( "$DIAGNOSE" == on || "$VERIFY" == on ) && -z "$IMAGE" && "${DS41_TEST_TEXT_ONLY:-off}" != on ]]; then
  IMAGE="$FIXTURE_IMAGE"
fi
SERVER_PID=''

usage() {
  cat <<'HELP'
Usage: bash tests/ds41_repetition_gate.sh [IMAGE.png|IMAGE.jpg]
  Starts independent local servers for c1-off c1-on c4-off c4-on.
  A supplied image reproduces the vision+thinking path (recommended).
  DS41_TEST_DIAGNOSE=on runs a sequential C1/MTP-off A-E sampler matrix.
  It defaults to tests/fixtures/ds41-repetition/yellow-face.png.
  DS41_TEST_VERIFY=on runs seeded C1 high-budget termination checks.

Requires: built ./ds4-server, downloaded iVQ/Engram/Vision, curl, jq,
          Perl JSON::PP, ss, file, base64. Stop an existing model server first
          if it holds the DGX Spark model lock/memory; this test never kills it.

Environment:
  DS41_TEST_IMAGE=path/to/original.png (overrides bundled image)
  DS41_TEST_MODES='c1-off c1-on c4-off c4-on' (space-separated subset)
  DS41_TEST_PORT=18043
  DS41_TEST_MAX_TOKENS=2048
  DS41_TEST_REASONING_EFFORT=high (none|low|high|max)
  DS41_TEST_DIAGNOSE=on (C1, MTP off; A-E matrix; returns 0 on HTTP success)
  DS41_TEST_VERIFY=on (C1, MTP off; high 8K/extended vs low 8K/none)
  DS41_VERIFY_HIGH_TOKENS=16384 (extended high budget, must exceed 8192)
  DS41_VERIFY_SEED=424242 (same seed across all verification requests)
  DS41_TEST_TEXT_ONLY=on (diagnostic mode: skip bundled image when no image given)
  DS41_TEST_TEXT_PROMPT=... / DS41_TEST_IMAGE_PROMPT=...
  DS41_TEST_OUT=/fresh/results/dir
  DS41_TEST_BOOT_TIMEOUT=600 / DS41_TEST_REQUEST_TIMEOUT=900
  DS41_MODEL_DIR, DS41_MODEL, DS41_ENGRAM_DIR, DS41_VISION_GGUF,
  DS41_SERVER_BIN, DS41_MEM_BUDGET_MB, DS41_MEM_FLOOR_GB, DS41_ZCHAIN_DIR

Output: per-mode server.log, raw HTTP JSON, reasoning/answer metrics,
        summary.jsonl, summary.tsv, parity.tsv; exit 1 on any non-OK result.
HELP
}
die() { printf '[ds41-repeat] ERROR: %s\n' "$*" >&2; exit 2; }
stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for ((i=0; i<30; i++)); do
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
      sleep 0.2
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then kill -KILL "$SERVER_PID" 2>/dev/null || true; fi
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=''
  fi
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
[[ $# -le 1 ]] || { usage >&2; exit 2; }
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then usage; exit 0; fi
for cmd in curl jq perl ss file base64 mktemp stat; do
  command -v "$cmd" >/dev/null 2>&1 || die "missing required command: $cmd"
done
[[ -x "$LAUNCHER" ]] || die "launcher not executable: $LAUNCHER"
[[ -f "$ANALYZER" ]] || die "analyzer not found: $ANALYZER"
[[ "$PORT" =~ ^[0-9]{1,5}$ ]] && (( 10#$PORT >= 1024 && 10#$PORT <= 65535 )) || die 'DS41_TEST_PORT must be 1024..65535'
for v in "$BOOT_TIMEOUT" "$REQUEST_TIMEOUT" "$TOKENS"; do
  [[ "$v" =~ ^[1-9][0-9]*$ ]] || die 'timeouts and max_tokens must be positive integers'
done
case "$EFFORT" in none|low|high|max) ;; *) die 'DS41_TEST_REASONING_EFFORT must be none|low|high|max' ;; esac
case "$DIAGNOSE" in on|off) ;; *) die 'DS41_TEST_DIAGNOSE must be on|off' ;; esac
case "$VERIFY" in on|off) ;; *) die 'DS41_TEST_VERIFY must be on|off' ;; esac
if [[ "$VERIFY" == on ]]; then
  [[ "$DIAGNOSE" == off ]] || die 'choose DS41_TEST_VERIFY or DS41_TEST_DIAGNOSE'
  [[ -f "$VERIFIER" ]] || die "verification classifier missing: $VERIFIER"
  [[ "$VERIFY_HIGH_TOKENS" =~ ^[1-9][0-9]*$ ]] && (( VERIFY_HIGH_TOKENS > 8192 )) ||
    die 'DS41_VERIFY_HIGH_TOKENS must be an integer greater than 8192'
  [[ "$VERIFY_SEED" =~ ^[0-9]{1,9}$ ]] || die 'DS41_VERIFY_SEED must be 0..999999999'
  if [[ -z "${DS41_TEST_REQUEST_TIMEOUT:-}" ]]; then REQUEST_TIMEOUT=2400; fi
fi
case "${DS41_TEST_TEXT_ONLY:-off}" in on|off) ;; *) die 'DS41_TEST_TEXT_ONLY must be on|off' ;; esac
[[ -n "$MODEL_ID" ]] || die 'DS41_MODEL_ID may not be empty'
[[ -x "$BIN" ]] || die "build the server first: ./scripts/build.sh (missing $BIN)"
[[ -s "$MODEL" && -d "$ENGRAM" && -s "$VISION" ]] || die 'model/Engram/Vision missing: ./scripts/download-model.sh'
for name in model-00047-of-00048.safetensors model-00048-of-00048.safetensors; do
  [[ -s "$ENGRAM/$name" ]] || die "missing Engram shard: $name"
done
if [[ -n "$IMAGE" ]]; then
  [[ -s "$IMAGE" ]] || die "image not found: $IMAGE"
  MIME=$(file -b --mime-type "$IMAGE")
  [[ "$MIME" == image/png || "$MIME" == image/jpeg ]] || die 'image must be PNG or JPEG'
fi
read -r -a MODES <<< "${DS41_TEST_MODES:-c1-off c1-on c4-off c4-on}"
if [[ "$DIAGNOSE" == on || "$VERIFY" == on ]]; then MODES=(c1-off); fi
(( ${#MODES[@]} > 0 )) || die 'DS41_TEST_MODES is empty'
declare -A SEEN=()
for mode in "${MODES[@]}"; do
  case "$mode" in c1-off|c1-on|c4-off|c4-on) ;; *) die "unknown test mode: $mode" ;; esac
  [[ ! -v SEEN[$mode] ]] || die "duplicate mode: $mode"
  SEEN[$mode]=1
done

RUN_ROOT="$ROOT/runs"; mkdir -p "$RUN_ROOT"
if [[ -n "${DS41_TEST_OUT:-}" ]]; then
  OUT="$DS41_TEST_OUT"
  mkdir -- "$OUT" || die 'DS41_TEST_OUT must be a fresh nonexistent directory'
  OUT="$(cd -- "$OUT" && pwd -P)"
else
  OUT="$(mktemp -d "$RUN_ROOT/ds41-repetition-$(date +%Y%m%d-%H%M%S)-XXXXXXXX")"
fi
mkdir -p "$OUT/payloads"
printf '%s\n' "$(date -Is)" > "$OUT/start.txt"
printf '%s\n' "${MODES[*]}" > "$OUT/modes.txt"
git -C "$ROOT" rev-parse HEAD > "$OUT/source.txt" 2>/dev/null || true
stat -c '%n size=%s mtime=%y' "$BIN" "$MODEL" "$VISION" > "$OUT/artifacts.txt"
if [[ -n "$IMAGE" ]]; then stat -c '%n size=%s mtime=%y' "$IMAGE" >> "$OUT/artifacts.txt"; fi

jq -n --arg model "$MODEL_ID" --arg prompt "$TEXT_PROMPT" --arg effort "$EFFORT" --argjson limit "$TOKENS" '{model:$model,stream:false,temperature:0,max_tokens:$limit,reasoning_effort:$effort,messages:[{role:"user",content:$prompt}]}' > "$OUT/payloads/text.json"
jq -n --arg model "$MODEL_ID" --arg effort "$EFFORT" --argjson limit "$TOKENS" '{model:$model,stream:false,temperature:0,max_tokens:$limit,reasoning_effort:$effort,messages:[{role:"user",content:"test_echo 도구를 value=ping으로 정확히 한 번 호출하세요."}],tools:[{type:"function",function:{name:"test_echo",description:"Return the provided value unchanged",parameters:{type:"object",properties:{value:{type:"string"}},required:["value"]}}}],tool_choice:"auto"}' > "$OUT/payloads/tool.json"
if [[ -n "$IMAGE" ]]; then
  { printf 'data:%s;base64,' "$MIME"; base64 -w0 "$IMAGE"; } > "$OUT/payloads/image.data"
  jq -n --arg model "$MODEL_ID" --arg prompt "$IMAGE_PROMPT" --arg effort "$EFFORT" --argjson limit "$TOKENS" --rawfile data "$OUT/payloads/image.data" '{model:$model,stream:false,temperature:0,max_tokens:$limit,reasoning_effort:$effort,messages:[{role:"user",content:[{type:"text",text:$prompt},{type:"image_url",image_url:{url:$data}}]}]}' > "$OUT/payloads/image.json"
fi

URL="http://127.0.0.1:$PORT"
port_busy() { ss -ltnH | awk '{print $4}' | grep -Eq ":${PORT}$"; }
boot() {
  local mode=$1 width mtp dir="$OUT/$1"
  width=${mode:1:1}; mtp=${mode##*-}
  mkdir -p "$dir"
  if port_busy; then printf '[ds41-repeat] Port %s already occupied; not stopping another service\n' "$PORT" | tee "$dir/boot-error.txt" >&2; return 1; fi
  printf '[ds41-repeat] start %-7s C%s MTP=%s\n' "$mode" "$width" "$mtp"
  DS41_MODEL_DIR="$MODEL_DIR" DS41_MODEL="$MODEL" DS41_ENGRAM_DIR="$ENGRAM" \
    DS41_VISION_GGUF="$VISION" DS41_SERVER_BIN="$BIN" \
    DS41_HOST=127.0.0.1 DS41_PORT="$PORT" DS41_MODEL_ID="$MODEL_ID" \
    DS41_MAX_SEQS="$width" DS41_MTP_MODE="$mtp" DS41_PREFIX_REUSE=off \
    DS41_MEM_BUDGET_MB="$MEM_BUDGET" DS41_MEM_FLOOR_GB="$MEM_FLOOR" \
    "$LAUNCHER" > "$dir/server.log" 2>&1 & SERVER_PID=$!
  for ((second=0;second<BOOT_TIMEOUT;second++)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then printf 'server exited before readiness\n' > "$dir/boot-error.txt"; return 1; fi
    if curl --silent --show-error --fail --max-time 2 "$URL/v1/models" > "$dir/models.json" 2>/dev/null && \
       jq -e --arg id "$MODEL_ID" '(.data // []) | any(.id == $id)' "$dir/models.json" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  printf 'readiness timeout after %s seconds\n' "$BOOT_TIMEOUT" > "$dir/boot-error.txt"
  return 1
}

request() {
  local kind=$1 target=$2 http=000
  : > "$target.response.json"
  if ! http=$(curl --silent --show-error --max-time "$REQUEST_TIMEOUT" --connect-timeout 4 \
      -o "$target.response.json" -w '%{http_code}' \
      -H 'Content-Type: application/json' --data-binary "@$OUT/payloads/$kind.json" \
      "$URL/v1/chat/completions" 2> "$target.curl.err"); then
    printf '%s\n' "$http" > "$target.http"
    return 1
  fi
  printf '%s\n' "$http" > "$target.http"
  [[ "$http" == '200' ]] && jq -e '(.error == null) and ((.choices // []) | length > 0)' "$target.response.json" >/dev/null 2>&1
}

analyse() {
  local mode=$1 case_id=$2 effort=${3:-$EFFORT}
  local path="$OUT/$mode/$case_id"
  if [[ ! -s "$path.response.json" ]]; then
    printf '%s\n' '{"error":{"message":"transport failure or empty response"}}' > "$path.response.json"
  fi
  perl -x "$ANALYZER" "$path.response.json" "$mode" "$case_id" "$effort" > "$path.analysis.json"
  jq -r '.choices[0].message.reasoning_content // .choices[0].message.reasoning // empty' \
    "$path.response.json" > "$path.thinking.txt" 2>/dev/null || true
  jq -r '.choices[0].message.content // empty' \
    "$path.response.json" > "$path.answer.txt" 2>/dev/null || true
  cat "$path.analysis.json" >> "$OUT/summary.jsonl"
}

# Controlled sampler diagnosis. All profiles share one C1/MTP-off server.
# A is the original failure; B isolates temperature; C/D vary the
# reasoning budget/effort; E disables thinking. No stochastic parity test.
if [[ "$DIAGNOSE" == on ]]; then
  : > "$OUT/summary.jsonl"
  printf 'profile\ttemperature\tthinking\trequested_tokens\tstatus\tfinish\tthinking_chars\tanswer_chars\tduplicate_windows\ttruncated\treasoning_exposed\tfinal_answer_present\n' > "$OUT/diagnostics.tsv"

  if ! boot c1-off; then
    die "serial diagnostic server failed: $OUT/c1-off/server.log"
  fi

  diag_kind=text
  if [[ -n "$IMAGE" ]]; then diag_kind=image; fi
  diag_transport=0
  for profile in A B C D E; do
    case "$profile" in
      A) temp=0; effort=high; limit=2048 ;;
      B) temp=1.0; effort=high; limit=2048 ;;
      C) temp=1.0; effort=high; limit=8192 ;;
      D) temp=1.0; effort=low; limit=8192 ;;
      E) temp=1.0; effort=none; limit=512 ;;
    esac

    jq --argjson temp "$temp" --arg effort "$effort" --argjson limit "$limit" \
      '.temperature=$temp | .top_p=1.0 | .min_p=0 | .reasoning_effort=$effort | .max_tokens=$limit' \
      "$OUT/payloads/$diag_kind.json" > "$OUT/payloads/diag-$profile.json"
    if ! request "diag-$profile" "$OUT/c1-off/diag-$profile"; then
      diag_transport=$((diag_transport + 1))
    fi
    analyse c1-off "diag-$profile" "$effort"

    jq -r --arg profile "$profile" --arg temp "$temp" --arg effort "$effort" --argjson limit "$limit" \
      '[$profile,$temp,$effort,($limit|tostring),.status,.finish_reason,
        (.reasoning_chars|tostring),(.answer_chars|tostring),
        (.reasoning_duplicate_windows|tostring),(.truncated|tostring),
        (.reasoning_exposed|tostring),(.final_answer_present|tostring)] | @tsv' \
      "$OUT/c1-off/diag-$profile.analysis.json" >> "$OUT/diagnostics.tsv"
  done

  stop_server
  cat "$OUT/diagnostics.tsv"
  printf '[ds41-repeat] diagnostic evidence: %s\n' "$OUT"
  if (( diag_transport > 0 )); then
    printf '[ds41-repeat] FAIL: %s HTTP request(s) failed\n' "$diag_transport" >&2
    exit 1
  fi
  printf '[ds41-repeat] DIAGNOSTIC COMPLETE: classifications are not a quality PASS\n'
  exit 0
fi

# Bound the high-thinking question with fixed sampling and real HTTP responses.
# A PASS requires both controls and the extended high request to terminate.
# A truncated extended request is bounded evidence, not proof of an EOS bug.
if [[ "$VERIFY" == on ]]; then
  : > "$OUT/summary.jsonl"
  printf 'profile\ttemperature\treasoning_effort\tmax_tokens\tseed\tstatus\tfinish\tcompletion_tokens\tthinking_chars\tanswer_chars\ttruncated\tfinal_answer\tduplicate_windows\n' > "$OUT/verification.tsv"
  printf 'same prompt/image, C1, MTP off, temperature=1, top_p=1, min_p=0, seed=%s\n' \
    "$VERIFY_SEED" > "$OUT/verification-settings.txt"

  if ! boot c1-off; then
    die "verification server failed: $OUT/c1-off/server.log"
  fi

  verify_kind=text
  if [[ -n "$IMAGE" ]]; then verify_kind=image; fi
  transport_failures=0
  for profile in high-base high-extended low-control no-thinking; do
    case "$profile" in
      high-base) effort=high; limit=8192 ;;
      high-extended) effort=high; limit=$VERIFY_HIGH_TOKENS ;;
      low-control) effort=low; limit=8192 ;;
      no-thinking) effort=none; limit=512 ;;
    esac

    jq --arg effort "$effort" --argjson limit "$limit" \
       --argjson seed "$VERIFY_SEED" \
      '.temperature=1.0 | .top_p=1.0 | .min_p=0 |
       .reasoning_effort=$effort | .max_tokens=$limit | .seed=$seed' \
      "$OUT/payloads/$verify_kind.json" > "$OUT/payloads/verify-$profile.json"
    if ! request "verify-$profile" "$OUT/c1-off/$profile"; then
      transport_failures=$((transport_failures + 1))
    fi
    analyse c1-off "$profile" "$effort"
    jq -r --arg profile "$profile" --arg effort "$effort" \
       --argjson limit "$limit" --arg seed "$VERIFY_SEED" \
      '[$profile,"1.0",$effort,($limit|tostring),$seed,.status,
       .finish_reason,(.completion_tokens|tostring),
       (.reasoning_chars|tostring),(.answer_chars|tostring),
       (.truncated|tostring),(.final_answer_present|tostring),
       (.reasoning_duplicate_windows|tostring)] | @tsv' \
      "$OUT/c1-off/$profile.analysis.json" >> "$OUT/verification.tsv"
  done

  stop_server
  verdict_exit=0
  perl "$VERIFIER" "$OUT" > "$OUT/verification.json" || verdict_exit=$?
  cat "$OUT/verification.tsv"
  jq -r '[.verdict,.category,.reason] | @tsv' "$OUT/verification.json"
  printf '[ds41-repeat] verification evidence: %s\n' "$OUT"
  if (( transport_failures > 0 )); then
    printf '[ds41-repeat] INCONCLUSIVE: %s HTTP transport error(s)\n' \
      "$transport_failures" >&2
    exit 2
  fi
  exit "$verdict_exit"
fi

: > "$OUT/summary.jsonl"
transport_failures=0
boot_failures=0
for mode in "${MODES[@]}"; do
  if ! boot "$mode"; then
    printf '[ds41-repeat] boot failed: %s; see %s/server.log\n' "$mode" "$OUT/$mode" >&2
    jq -nc --arg m "$mode" '{mode:$m,case:"boot",status:"boot_failed",finish_reason:"",reasoning_chars:0,answer_chars:0,reasoning_repeat_copies:0,reasoning_duplicate_windows:0,answer_repeat_copies:0,completion_tokens:0,error_message:"server startup failed"}' >> "$OUT/summary.jsonl"
    boot_failures=$((boot_failures + 1))
    stop_server
    continue
  fi
  if [[ "$mode" == c1-* ]]; then
    cases=(text-1 tool-1)
    if [[ -n "$IMAGE" ]]; then cases+=(image-1); fi
  else
    cases=(text-1 text-2 tool-1)
    if [[ -n "$IMAGE" ]]; then cases+=(image-1); else cases+=(text-3); fi
  fi
  if [[ "$mode" == c1-* ]]; then
    for item in "${cases[@]}"; do
      kind=${item%-*}
      if ! request "$kind" "$OUT/$mode/$item"; then transport_failures=$((transport_failures + 1)); fi
    done
  else
    pids=()
    for item in "${cases[@]}"; do
      kind=${item%-*}
      request "$kind" "$OUT/$mode/$item" & pids+=("$!")
    done
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then transport_failures=$((transport_failures + 1)); fi
    done
  fi
  for item in "${cases[@]}"; do analyse "$mode" "$item"; done
  if [[ "$mode" == c1-on ]] && ! grep -q 'DSpark drafter armed' "$OUT/$mode/server.log"; then
    jq -nc --arg m "$mode" '{mode:$m,case:"mtp-proof",status:"mtp_not_observed",finish_reason:"",reasoning_chars:0,answer_chars:0,reasoning_repeat_copies:0,reasoning_duplicate_windows:0,answer_repeat_copies:0,completion_tokens:0,error_message:"serial MTP drafter initialization not logged"}' >> "$OUT/summary.jsonl"
  fi
  if [[ "$mode" == c4-* ]]; then
    # Require real four-request execution, not merely a configured capacity.
    if ! grep -Eq '\[v41-batch\] packed_decode rows=4 ' "$OUT/$mode/server.log"; then
      jq -nc --arg m "$mode" '{mode:$m,case:"batch-proof",status:"batch_not_observed",finish_reason:"",reasoning_chars:0,answer_chars:0,reasoning_repeat_copies:0,reasoning_duplicate_windows:0,answer_repeat_copies:0,completion_tokens:0,error_message:"no logged packed_decode rows=4"}' >> "$OUT/summary.jsonl"
    fi
    if [[ "$mode" == c4-on ]] && \
       ! grep -Eq '\[v41-batch\] packed_decode rows=4 .*verify_rows=([5-9]|[1-9][0-9]+)' "$OUT/$mode/server.log"; then
      jq -nc --arg m "$mode" '{mode:$m,case:"mtp-proof",status:"mtp_not_observed",finish_reason:"",reasoning_chars:0,answer_chars:0,reasoning_repeat_copies:0,reasoning_duplicate_windows:0,answer_repeat_copies:0,completion_tokens:0,error_message:"no logged 4-request target verification with >4 rows"}' >> "$OUT/summary.jsonl"
    fi
  fi
  stop_server
  printf '[ds41-repeat] completed %s: %s case(s)\n' "$mode" "${#cases[@]}"
done

# Greedy temperature-0 output should not change with MTP or batch width.
# Ignore only dynamically allocated tool-call IDs and usage/timing metadata.
normalise() {
  jq -S '{finish_reason:(.choices[0].finish_reason // null), message: {content:(.choices[0].message.content // null), reasoning_content:(.choices[0].message.reasoning_content // .choices[0].message.reasoning // null), tool_calls:([.choices[0].message.tool_calls[]? | {type,function}])}}' "$1"
}
parity_failures=0
printf 'baseline\tcomparison\tcase\tstatus\n' > "$OUT/parity.tsv"
compare() {
  local base=$1 actual=$2 left=$3 right=${4:-$3}
  [[ -s "$OUT/$base/$left.response.json" && -s "$OUT/$actual/$right.response.json" ]] || return 0
  local filename="$OUT/parity-$base-$actual-$left-$right.diff"
  if diff -u <(normalise "$OUT/$base/$left.response.json") <(normalise "$OUT/$actual/$right.response.json") > "$filename"; then
    printf '%s\t%s\t%s\tMATCH\n' "$base" "$actual" "$left/$right" >> "$OUT/parity.tsv"
    rm -f "$filename"
  else
    printf '%s\t%s\t%s\tMISMATCH\n' "$base" "$actual" "$left/$right" >> "$OUT/parity.tsv"
    parity_failures=$((parity_failures + 1))
  fi
}
for kind in text-1 tool-1 image-1; do
  for pair in 'c1-off c1-on' 'c1-off c4-off' 'c1-on c4-on' 'c4-off c4-on'; do
    read -r from to <<< "$pair"
    [[ -n "${SEEN[$from]:-}" && -n "${SEEN[$to]:-}" ]] && compare "$from" "$to" "$kind"
  done
done
for mode in c4-off c4-on; do
  if [[ -n "${SEEN[$mode]:-}" ]]; then
    for kind in text-2 text-3; do compare "$mode" "$mode" text-1 "$kind"; done
  fi
done

jq -r -s '(["mode","case","status","finish","think_chars","answer_chars","max_repeat","duplicate_windows","tokens"] | @tsv), (.[] | [.mode,.case,.status,.finish_reason,.reasoning_chars,.answer_chars,.reasoning_repeat_copies,.reasoning_duplicate_windows,.completion_tokens] | @tsv)' "$OUT/summary.jsonl" > "$OUT/summary.tsv"
cat "$OUT/summary.tsv"
printf '\n[ds41-repeat] Parity:\n'; cat "$OUT/parity.tsv"
printf '[ds41-repeat] evidence: %s\n' "$OUT"
if ! jq -e -s 'all(.[]; .status == "ok")' "$OUT/summary.jsonl" >/dev/null || \
   (( transport_failures > 0 || boot_failures > 0 || parity_failures > 0 )); then
  printf '[ds41-repeat] FAIL: repetition/truncation/tool failure, missing reasoning, transport error or greedy mismatch\n' >&2
  exit 1
fi
printf '[ds41-repeat] PASS: every observed response terminated without detected repetition, tools parsed, greedy parity held\n'
exit 0

: <<'__DS41_PERL__'
#!/usr/bin/env perl
# Inspect one real OpenAI chat completion. This is a diagnostic classifier,
# not a replacement for target-vs-draft token/logit parity checks.
use strict;
use warnings;
use utf8;
use JSON::PP;
use Encode qw(decode);

sub loop_metrics {
    my ($source) = @_;
    my $text = lc($source // '');
    $text =~ s/\s+/ /g;
    $text =~ s/^\s+|\s+$//g;
    my $len = length($text);
    my $window = 100; # unicode characters, not UTF-8 bytes
    my (%seen, %copies);
    my ($max_copies, $duplicate_positions) = (0, 0);
    for (my $i = 0; $i + $window <= $len; $i++) {
        my $key = substr($text, $i, $window);
        if (exists $seen{$key}) {
            if ($i - $seen{$key} >= $window) {
                $seen{$key} = $i;
                my $copies = ++$copies{$key};
                $max_copies = $copies + 1 if $copies + 1 > $max_copies;
                $duplicate_positions++;
            }
        } else {
            $seen{$key} = $i;
        }
    }
    # Three separated identical 100-character passages, or two unusually
    # long overlapping stretches. The latter is deliberately only suspect.
    my $loop = $len >= 300 && ($max_copies >= 3 || $duplicate_positions >= 220);
    return {
        characters => $len,
        repeated_window_positions => $duplicate_positions,
        maximum_nonoverlapping_copies => $max_copies,
        loop => $loop ? JSON::PP::true : JSON::PP::false,
    };
}
sub classify {
    my ($payload, $mode, $case, $effort) = @_;
    $effort //= 'high';
    my $choice = ref($payload->{choices}) eq 'ARRAY' ? $payload->{choices}[0] : undef;
    my $message = ref($choice) eq 'HASH' ? $choice->{message} : undef;
    my $reasoning = ref($message) eq 'HASH' ? ($message->{reasoning_content} // $message->{reasoning} // '') : '';
    my $answer = ref($message) eq 'HASH' ? ($message->{content} // '') : '';
    $reasoning = '' unless defined($reasoning) && !ref($reasoning);
    $answer = '' if ref($answer); # multimodal outgoing structures not treated as prose
    my $reason = loop_metrics($reasoning);
    my $content = loop_metrics($answer);
    my $finish = ref($choice) eq 'HASH' ? ($choice->{finish_reason} // '') : '';
    my $call_ok = JSON::PP::false;
    if ($case =~ /^tool/ && ref($message) eq 'HASH' && ref($message->{tool_calls}) eq 'ARRAY' && @{$message->{tool_calls}} == 1) {
        my $call = $message->{tool_calls}[0];
        if (ref($call) eq 'HASH' && ref($call->{function}) eq 'HASH' && ($call->{function}{name} // '') eq 'test_echo') {
            my $args = eval { JSON::PP->new->decode($call->{function}{arguments} // '{}') };
            $call_ok = JSON::PP::true if ref($args) eq 'HASH' && ($args->{value} // '') eq 'ping';
        }
    }
    my $status = 'ok';
    if (ref($payload->{error}) eq 'HASH' || !ref($choice) || !ref($message)) {
        $status = 'response_error';
    } elsif ($reason->{loop} || $content->{loop}) {
        $status = 'repetition';
    } elsif ($finish eq 'length') {
        $status = 'truncated';
    } elsif ($finish ne 'stop' && $finish ne 'tool_calls') {
        $status = 'unexpected_finish';
    } elsif ($case =~ /^tool/ && ($finish ne 'tool_calls' || !$call_ok)) {
        $status = 'tool_failure';
    } elsif ($case !~ /^tool/ && $answer !~ /\S/) {
        $status = 'no_final_answer';
    } elsif ($reasoning eq '' && $effort ne 'none' && $case !~ /^tool/) {
        $status = 'reasoning_not_exposed';
    }
    my $usage = $payload->{usage};
    return {
        mode => $mode, case => $case, status => $status,
        finish_reason => $finish,
        reasoning_chars => $reason->{characters}, answer_chars => $content->{characters},
        reasoning_repeat_copies => $reason->{maximum_nonoverlapping_copies},
        reasoning_duplicate_windows => $reason->{repeated_window_positions},
        answer_repeat_copies => $content->{maximum_nonoverlapping_copies},
        reasoning_loop => $reason->{loop}, answer_loop => $content->{loop},
        reasoning_exposed => $reason->{characters} > 0 ? JSON::PP::true : JSON::PP::false,
        final_answer_present => $answer =~ /\S/ ? JSON::PP::true : JSON::PP::false,
        truncated => $finish eq 'length' ? JSON::PP::true : JSON::PP::false,
        tool_ok => $call_ok,
        completion_tokens => ref($usage) eq 'HASH' ? ($usage->{completion_tokens} // 0) : 0,
        error_message => ref($payload->{error}) eq 'HASH' ? ($payload->{error}{message} // 'server error') : '',
    };
}

if (@ARGV == 1 && $ARGV[0] eq '--self-test') {
    my $segment = 'This reasoning is circling through the same set of candidates without using any new facts, and ends in a repeated assumption. ';
    my $cycle = classify({choices=>[{finish_reason=>'length',message=>{reasoning_content=>$segment x 5,content=>''}}]},'c1-on','text-1');
    die "repetition not detected\n" unless $cycle->{status} eq 'repetition' && $cycle->{reasoning_repeat_copies} >= 3;
    my $normal = classify({choices=>[{finish_reason=>'stop',message=>{reasoning_content=>'Brief reasoning based on the available evidence.',content=>'An icon.'}}]},'c1-off','text-1');
    die "normal misclassified\n" unless $normal->{status} eq 'ok';
    my $trunc = classify({choices=>[{finish_reason=>'length',message=>{reasoning_content=>'Still considering this.',content=>''}}]},'c1-off','text-1');
    die "truncation not detected\n" unless $trunc->{status} eq 'truncated';
    my $valid = classify({choices=>[{finish_reason=>'tool_calls',message=>{reasoning_content=>'The correct tool is obvious.',content=>'',tool_calls=>[{function=>{name=>'test_echo',arguments=>'{"value":"ping"}'}}]}}]},'c4-on','tool-1');
    die "tool not detected\n" unless $valid->{status} eq 'ok' && $valid->{tool_ok};
    my $invalid = classify({choices=>[{finish_reason=>'stop',message=>{reasoning_content=>'I will call a tool.',content=>'No call.'}}]},'c4-on','tool-1');
    die "tool missing not detected\n" unless $invalid->{status} eq 'tool_failure';
    my $unfinished = classify({choices=>[{finish_reason=>'stop',message=>{reasoning_content=>'Still thinking.',content=>''}}]},'c1-on','image-1');
    die "thinking without final answer not detected\n" unless $unfinished->{status} eq 'no_final_answer';
    my $nonascii = classify({choices=>[{finish_reason=>'length',message=>{reasoning_content=>('노란색 바탕에서 눈과 입의 모양만 다시 검토합니다. ' x 30),content=>''}}]},'c1-on','image-1');
    die "unicode repetition not detected\n" unless $nonascii->{status} eq 'repetition';
    my $no_thinking = classify({choices=>[{finish_reason=>'stop',message=>{content=>'An icon.'}}]},'c1-off','diag-E','none');
    die "thinking-off falsely rejected\n" unless $no_thinking->{status} eq 'ok' && $no_thinking->{final_answer_present};
    my $missing_thinking = classify({choices=>[{finish_reason=>'stop',message=>{content=>'An icon.'}}]},'c1-off','diag-B','high');
    die "missing thinking not detected\n" unless $missing_thinking->{status} eq 'reasoning_not_exposed';
    my $tool_no_reason = classify({choices=>[{finish_reason=>'tool_calls',message=>{content=>'',tool_calls=>[{function=>{name=>'test_echo',arguments=>'{"value":"ping"}'}}]}}]},'c1-off','tool-1');
    die "valid tool call without exposed thinking rejected\n" unless $tool_no_reason->{status} eq 'ok';
    my $double_call = classify({choices=>[{finish_reason=>'tool_calls',message=>{content=>'',tool_calls=>[{function=>{name=>'test_echo',arguments=>'{"value":"ping"}'}},{function=>{name=>'test_echo',arguments=>'{"value":"ping"}'}}]}}]},'c1-off','tool-1');
    die "duplicate tool calls not detected\n" unless $double_call->{status} eq 'tool_failure';
    print "repetition-analyzer: 11 self-tests PASS\n";
    exit 0;
}
@ARGV == 3 || @ARGV == 4 or die "Usage: $0 RESPONSE.json MODE CASE [EFFORT] | --self-test\n";
my ($path, $mode, $case, $effort) = @ARGV;
open my $fh, '<:raw', $path or die "open $path: $!\n";
local $/;
my $bytes = <$fh>;
my $payload = eval { JSON::PP->new->utf8->decode($bytes) };
if ($@ || ref($payload) ne 'HASH') {
    $payload = {error => {message => 'invalid/empty JSON response'}};
}
my $report = classify($payload, $mode, $case, $effort);
print JSON::PP->new->canonical->utf8->encode($report), "\n";

__END__
__DS41_PERL__