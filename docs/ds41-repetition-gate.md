# DeepSeek V4.1 repetition and parity gate

The test starts its own servers via `scripts/serve.sh`; no additional operator startup script is needed. Run it on the single DGX Spark after downloading the model and building the engine.

```bash
cd ~/AI/ds4-dfm-rs-c4
git switch feat/ds41-entrpi-concurrency
git pull --ff-only
./scripts/build.sh
DS41_TEST_IMAGE=/absolute/path/to/reproducing-logo.png bash tests/ds41_repetition_gate.sh
```

Without an image, the script tests text reasoning and `test_echo` tool calling. It runs C1 MTP OFF/ON and C4 MTP OFF/ON sequentially; C4 generates four concurrent HTTP requests (one image, two text and one tool when an image was provided). Input requests are non-streaming, `temperature=0`, `reasoning_effort=high`, and `max_tokens=2048` by default.

An unused `127.0.0.1:18043` listener is required. The gate shuts down **only** the server PID it launches and does not kill existing servers, edit sudoers, download models, or alter `scripts/download-model.sh`, `scripts/build.sh`, `scripts/serve.sh`. Stop any other model server before testing if it holds the native instance lock or unified memory.

## Reports

Each run creates a new `runs/ds41-repetition-*/` folder containing `summary.tsv`, `summary.jsonl`, `parity.tsv`, original requests/responses, per-request `.thinking.txt` and `.answer.txt`, HTTP statuses and server logs. Those local files may contain private image/prompt data.

Classification: `repetition` detects repeating 100-character Unicode windows (a heuristic, not semantic equivalence); `truncated` means finish_reason=length; `no_final_answer` means the request stopped while leaving no final response; `reasoning_not_exposed` means no usable reasoning field was returned; `tool_failure` indicates a missing/malformed `test_echo(value=ping)` invocation; `batch_not_observed` and `mtp_not_observed` flag missing real four-request execution or draft verification. The gate also compares the deterministic output across modes and reports `MISMATCH` in `parity.tsv`, ignoring generated tool-call IDs and usage/timing.

A successful test requires usable reasoning, appropriate EOS/tool finish, no detected repetition, real C4 and MTP activity in server logs, and no observed temperature-0 parity drift. It is **not** a proof of model accuracy or long-term CUDA stability. Exit 1 on failure, retaining all evidence.

## C1 sampler/thinking diagnosis (A-E)

This mode runs five **sequential** requests on one C1 server with MTP
disabled. It uses the committed
[`yellow-face.png`](../tests/fixtures/ds41-repetition/yellow-face.png) by
default. The 256×256 PNG is a **synthetic approximation**, not the original
image that triggered the failure. Its editable source is
[`yellow-face.svg`](../tests/fixtures/ds41-repetition/yellow-face.svg).

```bash
# Reproducible repository fixture; no external image path required
DS41_TEST_DIAGNOSE=on bash tests/ds41_repetition_gate.sh

# Reproduce the actual failing image when available
DS41_TEST_DIAGNOSE=on DS41_TEST_IMAGE=/path/to/original.png \
  bash tests/ds41_repetition_gate.sh

# Keep text-only diagnosis available
DS41_TEST_DIAGNOSE=on DS41_TEST_TEXT_ONLY=on \
  bash tests/ds41_repetition_gate.sh
```

The script uses `top_p=1.0` and `min_p=0` in diagnostic mode.
The standard C1/C4 gate remains text-only by default; pass
`DS41_TEST_IMAGE=tests/fixtures/ds41-repetition/yellow-face.png`
to test its Vision path.

| Profile | Temperature | Reasoning effort | Max tokens | Purpose |
|---|---:|---|---:|---|
| A | 0 | high | 2048 | Reproduce greedy baseline |
| B | 1.0 | high | 2048 | Isolate sampling effect |
| C | 1.0 | high | 8192 | Check output-budget limitation |
| D | 1.0 | low | 8192 | Compare reasoning effort |
| E | 1.0 | none | 512 | Check non-thinking path |

Produces `diagnostics.tsv`, `summary.jsonl`, original requests and
responses, per-profile thinking/answer text, and one server log. Results
separately report repeated **character** windows, truncation, exposed
reasoning and final-answer presence; they do not measure repeated token IDs.
No parity is asserted across stochastic profiles. Exit 0 indicates completed
HTTP requests, **not** model-quality success; an HTTP failure returns exit 1.

In the normal C1/C4 gate, a correctly framed tool call no longer fails
merely because its reasoning field is absent. The tool test requires
**exactly one** valid `test_echo(value=ping)` call. Requests explicitly
using `reasoning_effort=none` do not require a reasoning field.

## Focused runs

```bash
# Test the problematic path only
DS41_TEST_MODES='c4-on' DS41_TEST_IMAGE=/path/to/logo.png bash tests/ds41_repetition_gate.sh

# Increase the Thinking budget and reproduce a longer loop
DS41_TEST_MAX_TOKENS=4096 DS41_TEST_REASONING_EFFORT=max DS41_TEST_IMAGE=/path/to/logo.png bash tests/ds41_repetition_gate.sh
```

Use `bash tests/ds41_repetition_gate.sh --help` for all environment variables. The check requires `bash`, `curl`, `jq`, `perl` with core `JSON::PP`, `ss`, `file`, `base64`, and standard Linux tools; **no Python**.

## Model-free validation

```bash
bash -n tests/ds41_repetition_gate.sh
perl -x tests/ds41_repetition_gate.sh --self-test
```

Real-model GPU evidence must be collected on the DGX Spark; the GitHub Actions step covers only syntax/classifier tests.
