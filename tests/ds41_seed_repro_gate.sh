#!/usr/bin/env bash
# One serial C1/MTP-off seed sweep; separate reference replay after shutdown.
set -Eeuo pipefail
umask 077

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
ANALYZER="$ROOT/tests/ds41_repetition_gate.sh"
REPORT="$ROOT/tests/ds41_seed_repro_report.pl"
LAUNCHER="${DS41_REPRO_LAUNCHER:-$ROOT/scripts/serve.sh}"
BIN="${DS41_SERVER_BIN:-$ROOT/ds4-server}"
DIR="${DS41_MODEL_DIR:-$HOME/AI/models/ds41-1017-ivq}"
MODEL="${DS41_MODEL:-$DIR/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf}"
VISION="${DS41_VISION_GGUF:-$DIR/DeepSeek-V4.1-Flash.vision.gguf}"
ENGRAM="${DS41_ENGRAM_DIR:-$DIR/deepseek-engram}"
MODEL_ID="${DS41_MODEL_ID:-big}"
PORT="${DS41_REPRO_PORT:-18043}"
TIMEOUT="${DS41_REPRO_REQUEST_TIMEOUT:-2400}"
BOOT_TIMEOUT="${DS41_REPRO_BOOT_TIMEOUT:-600}"
SEEDS="${DS41_REPRO_SEEDS:-1 2 3 4 5 6 7 8 9 10 424242}"
TEMP="${DS41_REPRO_TEMPERATURE:-1.0}"
BASE="${DS41_REPRO_BASE_TOKENS:-8192}"
EXT="${DS41_REPRO_EXT_TOKENS:-16384}"
LONG="${DS41_REPRO_LONG_TOKENS:-32768}"
IMAGE="${DS41_REPRO_IMAGE:-$ROOT/tests/fixtures/ds41-repetition/yellow-face.png}"
REF_URL="${DS41_REPRO_REFERENCE_URL:-}"
REF_MODEL="${DS41_REPRO_REFERENCE_MODEL_ID:-$MODEL_ID}"
SERVER_PID=''

usage() {
  cat <<'HELP'
Usage:
  bash tests/ds41_seed_repro_gate.sh
  DS41_REPRO_REFERENCE_URL=http://127.0.0.1:8001 \
    bash tests/ds41_seed_repro_gate.sh --reference /path/to/run
  bash tests/ds41_seed_repro_gate.sh --report /path/to/run
  bash tests/ds41_seed_repro_gate.sh --self-test

Local run: serial C1 MTP off; seeded high-thinking 8192-token
requests twice per seed. Failures/unstable outputs retry at 16384
and, if necessary, 32768. A greedy probe is also saved for
cross-engine comparison. No concurrent model copies are started.

DS41_REPRO_SEEDS='1 2 3 ... 10 424242'  (1..64 unique positive seeds)
Seed 0 is NOT deterministic: YoungAi V4.1 replaces 0 with clock entropy.
DS41_REPRO_TEMPERATURE=1.0
DS41_REPRO_IMAGE=/path/to/original.png
DS41_REPRO_TEXT_ONLY=on
DS41_REPRO_REQUEST_TIMEOUT=2400
DS41_REPRO_BASE_TOKENS=8192
DS41_REPRO_EXT_TOKENS=16384
DS41_REPRO_LONG_TOKENS=32768
DS41_REPRO_HASH_WEIGHTS=on           (optional, expensive)
DS41_REPRO_OUT=/new/nonexistent/path
DS41_REPRO_REFERENCE_MODEL_ID=big

Reference mode sends SAVED payloads to an already running localhost
OpenAI-compatible endpoint. Start it only after the local run stops;
it is never started or killed by this script.
HELP
}
die() { printf '[ds41-seeds] ERROR: %s\n' "$*" >&2; exit 2; }
stop_server() {
  if [[ -z "$SERVER_PID" ]]; then return; fi
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  for ((i=0;i<40;i++)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
    sleep 0.2
  done
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -KILL "$SERVER_PID" 2>/dev/null || true
  fi
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=''
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ACTION="${1:-run}"
case "$ACTION" in
  run) [[ $# -le 1 ]] || die 'unexpected args' ;;
  --reference|--report) [[ $# -eq 2 ]] || die 'provide run directory' ;;
  --self-test)
    bash -n "$ROOT/tests/ds41_seed_repro_gate.sh"
    perl "$REPORT" --self-test
    perl -x "$ANALYZER" --self-test
    exit 0 ;;
  --help|-h) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
for cmd in jq perl curl ss file base64 stat mktemp sha256sum; do
  command -v "$cmd" >/dev/null || die "missing command: $cmd"
done
[[ -f "$ANALYZER" && -f "$REPORT" ]] || die 'analyzers missing'
[[ "$PORT" =~ ^[0-9]{4,5}$ ]] && (( PORT <= 65535 )) ||
  die 'port must be 1000..65535'
for n in "$TIMEOUT" "$BOOT_TIMEOUT" "$BASE" "$EXT" "$LONG"; do
  [[ "$n" =~ ^[1-9][0-9]*$ ]] || die 'timeouts/tokens must be positive integers'
done
(( BASE < EXT && EXT < LONG )) || die 'require base < extended < long'
jq -en --arg temp "$TEMP" '($temp|tonumber) as $t | $t>=0 and $t<=2' \
  >/dev/null || die 'temperature must be numeric, 0..2'

if [[ "$ACTION" == --report ]]; then exec perl "$REPORT" "$2"; fi

if [[ "$ACTION" == --reference ]]; then
  OUT="$2"
  [[ -s "$OUT/cases.jsonl" && -s "$OUT/to-replay.txt" ]] ||
    die 'not a completed local sweep'
  [[ "$REF_URL" =~ ^http://(127\.0\.0\.1|localhost):[0-9]{2,5}$ ]] ||
    die 'reference URL must be explicit localhost HTTP; tunnel remote first'
  mkdir -p "$OUT/reference" "$OUT/reference-payloads"
  curl -fsS --max-time 5 "$REF_URL/v1/models" > "$OUT/reference-models.json" ||
    die 'reference model endpoint unavailable'
  jq -e --arg id "$REF_MODEL" '(.data // []) | any(.id==$id)' \
    "$OUT/reference-models.json" >/dev/null || die 'reference model not listed'
  printf 'reference_url=%s\nmodel=%s\nweight_equivalence=UNVERIFIED\n' \
    "$REF_URL" "$REF_MODEL" > "$OUT/reference-settings.txt"
  errors=0
  while IFS= read -r id; do
    [[ "$id" =~ ^(greedy|seed-[0-9]+-a|seed-[0-9]+-(16k|32k))$ ]] ||
      die "invalid case: $id"
    [[ -s "$OUT/payloads/$id.json" ]] || die "missing payload: $id"
    jq --arg model "$REF_MODEL" '.model=$model' \
      "$OUT/payloads/$id.json" > "$OUT/reference-payloads/$id.json"
    target="$OUT/reference/$id"
    http=000
    if ! http=$(curl -sS --connect-timeout 4 --max-time "$TIMEOUT" \
      -o "$target.response.json" -w '%{http_code}' \
      -H 'Content-Type: application/json' \
      --data-binary "@$OUT/reference-payloads/$id.json" \
      "$REF_URL/v1/chat/completions" 2>"$target.curl.err"); then
      errors=$((errors+1))
    fi
    printf '%s\n' "$http" > "$target.http"
    if [[ ! -s "$target.response.json" ]]; then
      printf '%s\n' '{"error":{"message":"reference transport failure"}}' \
        > "$target.response.json"
    fi
    perl -x "$ANALYZER" "$target.response.json" c1-off text-1 high \
      > "$target.analysis.json"
    if [[ "$http" != 200 ]]; then errors=$((errors+1)); fi
  done < "$OUT/to-replay.txt"
  perl "$REPORT" "$OUT" || true
  (( errors == 0 )) || die "$errors reference HTTP error(s)"
  printf '[ds41-seeds] comparison: %s/report.json\n' "$OUT"
  exit 0
fi

[[ -x "$LAUNCHER" && -x "$BIN" ]] || die 'build first: bash scripts/build.sh'
[[ -s "$MODEL" && -s "$VISION" && -d "$ENGRAM" ]] ||
  die 'model/Vision/Engram missing'
for name in model-00047-of-00048.safetensors model-00048-of-00048.safetensors; do
  [[ -s "$ENGRAM/$name" ]] || die "missing Engram shard: $name"
done
if [[ "${DS41_REPRO_TEXT_ONLY:-off}" == on ]]; then IMAGE=''; fi
if [[ -n "$IMAGE" ]]; then
  [[ -s "$IMAGE" ]] || die "image missing: $IMAGE"
  MIME=$(file -b --mime-type "$IMAGE")
  [[ "$MIME" == image/png || "$MIME" == image/jpeg ]] ||
    die 'image must be PNG or JPEG'
fi
read -r -a LIST <<< "$SEEDS"
(( ${#LIST[@]} > 0 && ${#LIST[@]} <= 64 )) || die 'need 1..64 seeds'
declare -A SEEN=()
for seed in "${LIST[@]}"; do
  [[ "$seed" =~ ^[1-9][0-9]{0,8}$ ]] ||
    die "invalid fixed seed: $seed (V4.1 seed=0 selects clock entropy; use 1..999999999)"
  [[ ! -v SEEN[$seed] ]] || die "duplicate seed: $seed"
  SEEN[$seed]=1
done

mkdir -p "$ROOT/runs"
if [[ -n "${DS41_REPRO_OUT:-}" ]]; then
  OUT="$DS41_REPRO_OUT"
  mkdir -- "$OUT" || die 'DS41_REPRO_OUT must be new'
else
  OUT=$(mktemp -d "$ROOT/runs/ds41-seeds-$(date +%Y%m%d-%H%M%S)-XXXXXXXX")
fi
OUT=$(cd "$OUT" && pwd -P)
mkdir -p "$OUT/local" "$OUT/payloads"
: > "$OUT/cases.jsonl"
: > "$OUT/to-replay.txt"
date -Is > "$OUT/start.txt"
git -C "$ROOT" rev-parse HEAD > "$OUT/source.txt" 2>/dev/null || true
printf 'seeds=%s\ntemperature=%s\nbase=%s\nextended=%s\nlong=%s\nimage=%s\n' \
  "${LIST[*]}" "$TEMP" "$BASE" "$EXT" "$LONG" "$IMAGE" > "$OUT/settings.txt"
stat -c '%n size=%s mtime=%y' "$BIN" "$MODEL" "$VISION" \
  "$ENGRAM/model-00047-of-00048.safetensors" \
  "$ENGRAM/model-00048-of-00048.safetensors" > "$OUT/artifacts.txt"
sha256sum "$BIN" > "$OUT/binary.sha256"
if [[ "${DS41_REPRO_HASH_WEIGHTS:-off}" == on ]]; then
  sha256sum "$MODEL" "$VISION" \
    "$ENGRAM/model-00047-of-00048.safetensors" \
    "$ENGRAM/model-00048-of-00048.safetensors" > "$OUT/weights.sha256"
else
  echo 'Model and sidecar SHA256 not reverified' > "$OUT/weights-unverified.txt"
fi

if [[ -n "$IMAGE" ]]; then
  { printf 'data:%s;base64,' "$MIME"; base64 -w0 "$IMAGE"; } > "$OUT/image.data"
  jq -n --arg model "$MODEL_ID" \
    --arg prompt "${DS41_REPRO_IMAGE_PROMPT:-이게 뭔가요? 보이는 특징을 기준으로 판단하고 최종 답변을 간결하게 제시하세요.}" \
    --rawfile data "$OUT/image.data" \
    '{model:$model,stream:false,messages:[{role:"user",content:[
       {type:"text",text:$prompt},{type:"image_url",image_url:{url:$data}}]}]}' \
    > "$OUT/template.json"
else
  jq -n --arg model "$MODEL_ID" \
    --arg prompt "${DS41_REPRO_TEXT_PROMPT:-이게 뭔가요? 노란 배경의 아이콘에 검은 타원형 눈 두 개, 쉼표 같은 코, 곡선형 입이 있습니다. 확실하지 않으면 추측이라고 명시하고 간결하게 답하세요.}" \
    '{model:$model,stream:false,messages:[{role:"user",content:$prompt}]}' \
    > "$OUT/template.json"
fi

URL="http://127.0.0.1:$PORT"
if ss -ltnH | awk '{print $4}' | grep -Eq ":$PORT\$"; then
  die "port $PORT occupied; will not stop another server"
fi
printf '[ds41-seeds] C1 MTP=off; seeds=%s\n' "$SEEDS"
DS41_MODEL_DIR="$DIR" DS41_MODEL="$MODEL" DS41_VISION_GGUF="$VISION" \
  DS41_ENGRAM_DIR="$ENGRAM" DS41_SERVER_BIN="$BIN" \
  DS41_HOST=127.0.0.1 DS41_PORT="$PORT" DS41_MODEL_ID="$MODEL_ID" \
  DS41_MAX_SEQS=1 DS41_MTP_MODE=off DS41_PREFIX_REUSE=off \
  DS41_MEM_BUDGET_MB="${DS41_MEM_BUDGET_MB:-110000}" \
  DS41_MEM_FLOOR_GB="${DS41_MEM_FLOOR_GB:-2}" \
  "$LAUNCHER" > "$OUT/server.log" 2>&1 & SERVER_PID=$!
ready=0
for ((i=0;i<BOOT_TIMEOUT;i++)); do
  kill -0 "$SERVER_PID" 2>/dev/null || die 'server died before readiness'
  if curl -fsS --max-time 2 "$URL/v1/models" > "$OUT/models.json" 2>/dev/null &&
      jq -e --arg id "$MODEL_ID" '(.data // []) | any(.id==$id)' \
      "$OUT/models.json" >/dev/null; then
    ready=1; break
  fi
  sleep 1
done
(( ready == 1 )) || die "readiness timed out: $OUT/server.log"

write_payload() {
  local id=$1 seed=$2 temp=$3 max=$4
  jq --argjson seed "$seed" --argjson temp "$temp" --argjson max "$max" \
    '.temperature=$temp | .top_p=1 | .min_p=0 | .seed=$seed |
     .max_tokens=$max | .reasoning_effort="high"' \
    "$OUT/template.json" > "$OUT/payloads/$id.json"
}
good() {
  jq -e '.status=="ok" and .finish_reason=="stop" and
    .final_answer_present == true and .reasoning_loop == false and
    .answer_loop == false' "$1" >/dev/null
}
request() {
  local id=$1 seed=$2 temp=$3 max=$4 target="$OUT/local/$1"
  write_payload "$id" "$seed" "$temp" "$max"
  jq -nc --arg id "$id" --argjson seed "$seed" --argjson max "$max" \
    --argjson temp "$temp" \
    '{id:$id,seed:$seed,max_tokens:$max,temperature:$temp}' >> "$OUT/cases.jsonl"
  local http=000
  if ! http=$(curl -sS --connect-timeout 4 --max-time "$TIMEOUT" \
    -o "$target.response.json" -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    --data-binary "@$OUT/payloads/$id.json" \
    "$URL/v1/chat/completions" 2>"$target.curl.err"); then
    transport=$((transport+1))
  fi
  printf '%s\n' "$http" > "$target.http"
  if [[ ! -s "$target.response.json" ]]; then
    echo '{"error":{"message":"empty response"}}' > "$target.response.json"
  fi
  if [[ "$http" != 200 ]]; then transport=$((transport+1)); fi
  perl -x "$ANALYZER" "$target.response.json" c1-off text-1 high \
    > "$target.analysis.json"
  printf '[ds41-seeds] %s: HTTP %s, %s\n' "$id" "$http" \
    "$(jq -r .status "$target.analysis.json")"
}
normalize() {
  jq -S '{finish_reason:.choices[0].finish_reason,
    reasoning:(.choices[0].message.reasoning_content //
       .choices[0].message.reasoning // ""),
    content:(.choices[0].message.content // ""),
    tool_calls:(.choices[0].message.tool_calls // [])}' "$1"
}
same_output() {
  diff -q <(normalize "$OUT/local/$1.response.json") \
          <(normalize "$OUT/local/$2.response.json") >/dev/null
}

transport=0
request greedy 424242 0 2048
echo greedy >> "$OUT/to-replay.txt"
for seed in "${LIST[@]}"; do
  a="seed-$seed-a"; b="seed-$seed-b"
  request "$a" "$seed" "$TEMP" "$BASE"
  request "$b" "$seed" "$TEMP" "$BASE"
  if good "$OUT/local/$a.analysis.json" &&
     good "$OUT/local/$b.analysis.json" &&
     same_output "$a" "$b"; then
    continue
  fi
  echo "$seed" >> "$OUT/failure-seeds.txt"
  echo "$a" >> "$OUT/to-replay.txt"
  ext="seed-$seed-16k"
  request "$ext" "$seed" "$TEMP" "$EXT"
  echo "$ext" >> "$OUT/to-replay.txt"
  if ! good "$OUT/local/$ext.analysis.json"; then
    long="seed-$seed-32k"
    request "$long" "$seed" "$TEMP" "$LONG"
    echo "$long" >> "$OUT/to-replay.txt"
  fi
done
stop_server
report_exit=0
perl "$REPORT" "$OUT" || report_exit=$?
(( transport == 0 )) || die "$transport HTTP transport failure(s)"
printf '[ds41-seeds] evidence: %s\n' "$OUT"
exit "$report_exit"
