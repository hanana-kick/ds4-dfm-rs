# Experimental V4.1 concurrent serving (fork-only)

Status: **implementation + model-free checks; not GPU-qualified**. Do not
replace a production endpoint on the strength of host CI. This does not claim
to solve reasoning repetition, improve model quality, or preserve the original
MTP throughput. `main`'s serial defaults remain unchanged.

## Sources and scope

- Baseline: Baekpica/ds4-dfm-rs `502019da159b86c26b0885c59fb8466000d3d863`.
- Scheduler ownership reference: Entrpi/ds4 v0.6.5,
  `addc0c4bd6c664f8eed556df5c41f17990dcfcb6`: one GPU owner, per-request
  mutable state, rolling admission/retirement, shared weights. The old V4
  session layout cannot simply be assigned to V4.1.
- V4.1 arithmetic: fodelf/YoungAi
  `7fb26cc332d740416792204a29c7d13d2986caf5`,
  `src/core/core_v41_multi.c`, `core_v41_req.c`, `core_v41_state.c` (MIT).
  The plain packed-row formulation is adapted to this tree's existing
  `ds4_ds41_forward.inc`. C HTTP, global sampler state, speculative batching
  and multiple CUDA attention streams are deliberately not imported.

`--max-seqs 2..8` selects ephemeral request states. **C4** is the intended
initial gate. A decode step packs one token from each ready request into
shared embedding/dense/HC/MoE/head operations, with independent attention
caches. This is not a lock around whole-request generation. Attention kernels
are issued on one owner stream because the inherited backend has shared
workspace. That is not a throughput guarantee.

The request's prefill progresses one existing native chunk per scheduler turn.
Ready requests also decode in that turn. A new image's tower pass can still
pause other rows while that pass executes; tower scheduling is not preemptive.

## Contract

| Item | Serial/default | Explicit concurrent lane |
|---|---|---|
| Request slots | 1 | 2..8, memory-dependent admission |
| Weights | existing VQ/Engram/sidecars | same weights, one model load |
| KV/Engram history | existing checkpoint path | request-local state/feed/worker |
| Vision | existing `--vision-gguf` | same preprocessing/tower, per-request image state |
| MTP | existing embedded DSpark | **must set `--mtp-mode off`** |
| Prefix/disk cache | existing qualified serial surface | **must set `--prefix-reuse off`; no disk cache** |
| Sampling | existing serial sampler | request-local Rust RNG and generated-token penalties |
| DSML tools/reasoning | existing HTTP machinery | shared Rust stepper, greedy DSML syntax positions |
| CUDA graphs | existing serial path | eager packed decode only |
| GPU qualification | original recorded C1 gates | **pending** |

`ephemeral` in capability discovery is not persistent bank admission,
fork/rewind, snapshots or disk restoration. The existing `--serving-check`
continues to test the serial checkpoint contracts, and explicitly lists the
packed concurrency category as not checked. The historical qualified bank
count stays **1**; a successful build/listener does not promote it to 4.

A request exceeding current memory headroom receives 503/Retry-After when no
active request can free space. With live requests it waits for retirement.
Admission prices actual prompt + output, native KV/scratch, image buffers and
an explicit host reserve, with the existing conservative 2x band. It does not
reserve four complete 1M contexts. No swap/offload or expert pruning is added.

Each request opens an independent Engram feed. The serial asset cache cannot
be reused here: it shares a mutable history/image-mask/IO owner. Teardown drains
asynchronous readers before pinned destinations are freed. Immutable file data
may still be shared by the operating system's page cache.

Disconnects retire only that request. A failed native GPU step poisons its
cohort; there is no reuse of partially written KV. Restart the server after a
native error. No stale global prefix checkpoint is imported into this lane.

The primary concurrent wire gate is **OpenAI Chat Completions**, including
images and DSML tools. Other protocol shapes retain the shared router's
eligibility/fallback rules; do not assume every Anthropic/Responses request
is batched. `return_token_ids` is refused on the packed lane rather than
silently dropped. Non-greedy seeded output is not promised identical to the
old CUDA sampler; independent request RNG is tested model-free. Greedy C1/C4
output parity is a mandatory real-GPU gate.

## Build and run on the Spark

The model, build and server startup each have one script. From the root of
the fork checkout:

```bash
bash scripts/download-model.sh
bash scripts/build.sh
bash scripts/serve.sh
```

The model downloader installs the iVQ GGUF, the matching 1017 Engram shards,
and Vision sidecar under `~/AI/models/ds41-1017-ivq`. The build script runs
`make cuda-spark`. The server script uses model ID `big`, port `8000`,
Vision enabled, `--max-seqs 4`, and `--mtp-mode on` by default. Paths and
server settings may be overridden through `DS41_*` environment variables.
No `--ctx` flag is passed because this family reads context from model metadata.

**Engine limitation:** the current concurrent kernel supports C4 without MTP,
but rejects C4 + MTP. The launcher deliberately requests both and therefore
will fail until batched speculative verification is implemented. It does not
silently fall back to C1 or switch MTP off. It does not stop existing servers;
the requested port must be free.

## Validation

Model-free checks cover Rust contracts, independent sampler histories/RNG,
overflow-safe admission, native/bridge syntax and the actual bridge lifetime
with GPU entry-point doubles (including cancellation, failed allocation,
cohort dispatch and image release). ASan/UBSan can run the same bridge gate.
They cannot prove CUDA arithmetic, Vision answers or long-session stability.

The separate live gate requires the model and two different PNG/JPEG images:

```bash
bash tests/ds41_concurrent_http_gate.sh \
  ./ds4-server "$MODEL" "$ENGRAM" "$VISION" \
  /path/to/image-a.png /path/to/image-b.png "$CODE_GR"
```

It starts only task-owned servers on loopback port 18041, compares serial
plain-decode against C4, sends mixed image/text cohorts, checks actual four-row
execution and a tool-result roundtrip, then disconnect/reuse behavior. A
mismatch or absence of an actual four-row step fails. Set `DS41_GATE_OUT` to a
new evidence directory and `DS41_GATE_PORT` to another unused port as needed.
The gate preserves raw requests, responses, source/artifact hashes and logs;
its cleanup only stops the PID it launched. Model hash verification reads the
GGUF once. It never downloads weights or changes clocks/swap/sudoers.

Before promotion, extend GPU qualification to: request-order permutations,
short/long independent histories, C1/C2/C4 numeric logits, 32K/64K pressure,
late image admission, streaming tool arguments, invalid-image rejection,
failed Engram reads, host-memory pressure, and the user's actual looping agent
request. A passing bounded HTTP gate is not a general reasoning-quality claim.
