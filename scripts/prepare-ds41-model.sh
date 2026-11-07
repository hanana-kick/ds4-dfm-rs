#!/usr/bin/env bash
# Download and verify the official DeepSeek-V4.1-Flash-1017 iVQ
# GGUF, Engram and Vision artifacts for ds4-dfm-rs.
# This does not start the server, requires no sudo/Python/quantization.
set -Eeuo pipefail

REPO='Baekpica/DeepSeek-V4.1-Flash-1017-iVQ-GGUF'
MODEL_DIR="${DS41_MODEL_DIR:-$HOME/AI/models/ds41-1017-ivq}"
BASE='DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf'
BASE_SHA='746fb52dac7de7b4ef9ffb32dbc103e2439a89f52aad3757a2ed595b75a8f4c0'
BASE_SIZE=113556639424
VISION='DeepSeek-V4.1-Flash.vision.gguf'
E47='deepseek-engram/model-00047-of-00048.safetensors'
E48='deepseek-engram/model-00048-of-00048.safetensors'

say() { printf '[ds41-model] %s\n' "$*"; }
die() { printf '[ds41-model] ERROR: %s\n' "$*" >&2; exit 1; }
for command in hf sha256sum stat df awk cat mv; do
  command -v "$command" >/dev/null 2>&1 || die "missing '$command'; install the Hugging Face CLI (https://hf.co/cli) and GNU coreutils"
done
mkdir -p "$MODEL_DIR"
MODEL_DIR=$(cd "$MODEL_DIR" && pwd -P)
cd "$MODEL_DIR"

if [[ -f "$BASE" ]]; then
  say 'Assembled GGUF exists; verifying checksum'
  printf '%s  %s\n' "$BASE_SHA" "$BASE" | sha256sum -c --status - || die "existing $BASE is corrupt; refusing to overwrite"
  say 'Downloading/checking Vision, 1017 Engram and checksum metadata'
  hf download "$REPO" --local-dir "$MODEL_DIR" \
    --include "$VISION" 'deepseek-engram/*.safetensors' 'SHA256SUMS' 'RECIPE.md' 'SOURCE.md'
else
  say "Downloading 1017 iVQ checkpoint (about 318 GB) -> $MODEL_DIR"
  hf download "$REPO" --local-dir "$MODEL_DIR"
fi

[[ -s SHA256SUMS ]] || die 'missing SHA256SUMS'
expected=$(awk -v file="$BASE" '$2 == file {print $1}' SHA256SUMS)
[[ "$expected" == "$BASE_SHA" ]] || die 'wrong iVQ revision; refusing to mix checkpoints'
for file in "$VISION" "$E47" "$E48"; do
  [[ -s "$file" ]] || die "required file missing: $file"
  grep -Fq " $file" SHA256SUMS || die "checksum entry missing: $file"
done
say 'Verifying Vision and both 1017 Engram shards'
awk -v base="$BASE" '$2 != base && $2 !~ /\.part[0-9][0-9]-of-40$/ {print}' SHA256SUMS | sha256sum -c --quiet - \
  || die 'Vision/Engram checksum mismatch'

if [[ ! -f "$BASE" ]]; then
  parts=()
  for ((i=1; i<=40; i++)); do
    printf -v part '%s.part%02d-of-40' "$BASE" "$i"
    [[ -s "$part" ]] || die "missing GGUF part: $part"
    grep -Fq " $part" SHA256SUMS || die "checksum entry missing: $part"
    parts+=("$part")
  done
  say 'Verifying all 40 GGUF parts'
  awk '$2 ~ /\.part[0-9][0-9]-of-40$/ {print}' SHA256SUMS | sha256sum -c --quiet - \
    || die 'GGUF part checksum mismatch'
  available=$(df -B1 --output=avail . | tail -n1 | tr -d '[:space:]')
  required=$(( BASE_SIZE + 2*1024*1024*1024 ))
  ((available >= required)) || die "assembly requires another 116GB free; only $((available/1000000000))GB available; originals preserved"

  staging="${BASE}.assembling"
  if [[ -e "$staging" ]]; then
    say "Removing incomplete previous assembly: $staging"
    rm -f -- "$staging"
  fi
  trap 'rm -f -- "${staging:-}"' EXIT
  say 'Assembling GGUF without deleting source parts until checksum passes'
  cat -- "${parts[@]}" > "$staging" || die 'GGUF assembly failed; inputs preserved'
  actual=$(stat -c %s "$staging")
  [[ "$actual" == "$BASE_SIZE" ]] || die "GGUF size mismatch: $actual != $BASE_SIZE"
  printf '%s  %s\n' "$BASE_SHA" "$staging" | sha256sum -c --status - \
    || die 'assembled GGUF checksum mismatch'
  mv -n -- "$staging" "$BASE" || die 'final rename failed'
  trap - EXIT
  say 'Verified; removing the 40 redundant source parts'
  rm -f -- "${parts[@]}"
fi

say 'Verified model ready:'
printf '  MODEL=%s/%s\n  ENGRAM=%s/deepseek-engram\n  VISION=%s/%s\n' "$MODEL_DIR" "$BASE" "$MODEL_DIR" "$MODEL_DIR" "$VISION"
say 'MTP draft weights are already embedded in the GGUF.'
