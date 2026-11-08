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

## High Thinking termination verification

The earlier A–E diagnosis is observational. This verification mode has
**machine-checked results**, a fixed sampling seed and explicit exit codes.
It starts one C1 / MTP-OFF server, then sends **sequential** requests with the
same prompt, image, temperature 1.0, top_p 1.0 and min_p 0:

| Profile | Effort | Max tokens | Condition checked |
|---|---|---:|---|
| high-base | high | 8192 | Reproduce high-budget cutoff |
| high-extended | high | 16384 | Test whether extra budget completes |
| low-control | low | 8192 | Control: reasoning completion |
| no-thinking | none | 512 | Control: final answer without thinking |

```bash
git pull --ff-only
bash scripts/build.sh
DS41_TEST_VERIFY=on bash tests/ds41_repetition_gate.sh

# Optional: original failing image, or a larger high budget
DS41_TEST_VERIFY=on DS41_TEST_IMAGE=/path/to/original.png \
  DS41_VERIFY_HIGH_TOKENS=32768 \
  DS41_TEST_REQUEST_TIMEOUT=3600 bash tests/ds41_repetition_gate.sh

# Text-only controls
DS41_TEST_VERIFY=on DS41_TEST_TEXT_ONLY=on \
  bash tests/ds41_repetition_gate.sh
```

With no explicit image, uses the committed synthetic yellow-face fixture.
`DS41_VERIFY_SEED` defaults to 424242 and is sent in each request. The
extended budget must exceed 8192; request timeout defaults to 2400 seconds
in this mode unless `DS41_TEST_REQUEST_TIMEOUT` is set.

- **PASS (exit 0):** Low and no-thinking controls finish; high-extended produces a final answer without detected repetition. If high-base was truncated, its exposed reasoning must prefix the high-extended reasoning exactly under the identical seed.
- **FAIL (exit 1):** High-extended remains truncated, stops without a final answer, or any profile has detected repeated windows.
- **INCONCLUSIVE (exit 2):** Missing/error response, failed control, absent reasoning, or differing high-output prefixes despite the identical seed.

Evidence: `verification.json`, `verification.tsv`, `summary.jsonl`, serialized
request payloads, full responses, thinking/answer text and server log, under
`runs/ds41-repetition-*/`. The verification script requires a built
server; it does not download, rebuild or kill unrelated processes.

This is a **bounded HTTP termination test**. It does not expose raw generated
DSML token IDs or prove a sampler, quantization or EOS-parser defect. If 16K
still truncates, the correct result is *unresolved within 16K*, not
*EOS is broken*. `perl tests/ds41_thinking_verdict.pl --self-test` validates
eight verdict scenarios without GPU or weights.

## Reproduce intermittent failures across seeds

Use `tests/ds41_seed_repro_gate.sh` for a seeded, repeatable
**C1/MTP-off** experiment. This is separate from the fixed-seed 8K/16K
termination check above. The default fixed seeds are 1–10 and 424242
(**never 0**): native YoungAi V4.1 treats `seed=0` as an instruction to
draw a wall-clock seed. Explicit `DS41_REPRO_SEEDS='0 ...'` is rejected
before starting the model. If an older run contains seed 0, recomputing
its report marks that row `clock_seed_ungraded`, excluded from the
fixed-seed pass/fail rate. Its 8K/16K/32K generations are independent
random draws, **not** evidence that a larger budget fixed the output.

```bash
git pull --ff-only
bash scripts/build.sh

# 11 default seeds x 2, plus a temperature=0 reference probe
bash tests/ds41_seed_repro_gate.sh

# Reproduce using the actual original failing image
DS41_REPRO_IMAGE=/absolute/path/to/failing-image.png \
  bash tests/ds41_seed_repro_gate.sh

# Faster focused run
DS41_REPRO_SEEDS='0 1 424242' \
  bash tests/ds41_seed_repro_gate.sh
```

The script starts **one** server on loopback (default port 18043),
sends the same high-thinking prompt twice with each explicit seed and
temperature 1.0, and compares normalized output (reasoning + final text +
finish status). It checks the same 100-character repetition heuristic used
by the existing gate; **token-ID repetition is not directly measured**.
For failed/unstable seeds it replays at 16,384 tokens and, if that still
fails, at 32,768 tokens, preserving the seed and exact input.

A successful run writes `runs/ds41-seeds-*/` with `report.tsv`,
`report.json`, `failure-seeds.txt` (only if any), original JSON
payloads, individual HTTP responses and status codes, analyzer outputs,
a separate 2,048-token temperature-0 greedy probe, model/binary details,
and the server log. `DS41_REPRO_HASH_WEIGHTS=on` performs an optional full
SHA256 pass over the ~113 GB GGUF plus Vision and Engram input shards;
without it weight integrity is explicitly unverified.

### Compare against a reference engine

The local sweep **stops its own server** before returning. Start a separate
reference backend serving OpenAI-compatible `/v1/chat/completions`
(YoungAi or another engine only if it really implements that API and the
same model assets); no assumption is made about its interface or file
compatibility. On one DGX Spark, do **not** run two large models at once.

```bash
# After the local run has exited, launch the reference separately.
# Use an SSH loopback tunnel if the reference is on another host.
DS41_REPRO_REFERENCE_URL=http://127.0.0.1:8001 \
  bash tests/ds41_seed_repro_gate.sh --reference \
  /absolute/path/to/runs/ds41-seeds-...

# Recompute report without starting a server
bash tests/ds41_seed_repro_gate.sh --report \
  /absolute/path/to/runs/ds41-seeds-...
```

For previous runs containing seed 0, use `--report` to reclassify
the existing evidence without another costly model run.

Reference replay uses the **saved exact payloads**, changing only
`model` when `DS41_REPRO_REFERENCE_MODEL_ID` is set. It targets
failed seeds and the deterministic greedy probe, writing separate reference
responses and comparisons in the same run folder. It does not start, stop,
or reconfigure the reference server. It accepts localhost HTTP only to
avoid inadvertently sending the original image or prompt to a third party.

Results:

- `PASS`: all tested seeds completed twice with identical output and
  no detected string repetition.
- `FAIL`: at least one seed failed, truncated, or produced inconsistent
  results.
- `INCONCLUSIVE`: incomplete/error HTTP evidence.

`local_only_failure` versus `both_failed` is a **lead**, not a
causal verdict. Even identical seeds can yield different samples under
different RNG/sampler algorithms. The grep-safe greedy comparison is more
informative, but different weight quantizations, prompt templates, Vision
processing and numerical kernels still confound results. Verify weight
hashes, prompt tokens and token-level logits independently before calling
an engine or quantization defect. This test does not require Python.

Model-free CI covers Bash syntax, Perl classification and a simulated HTTP
endpoint; it does **not** certify actual DGX Spark inference.

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
