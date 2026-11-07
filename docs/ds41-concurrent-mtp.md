# DeepSeek V4.1 concurrent MTP execution

The concurrent lane executes embedded DSpark together with `--max-seqs 2..8`.
The existing service launcher requests C4 + MTP on + Vision on. No new service
script or opt-in experimental flag is required. The three operator scripts
remain `download-model.sh`, `build.sh` and `serve.sh`.

## Request and batch ownership

A request owns target attention/KV, image-mask/Engram history, target-layer
hidden ring, three draft towers, draft window/graph/Markov cache, verification
snapshot, Rust RNG, emitted-token penalty history, and its acceptance cursor.
Weights are shared. The owner thread invokes GPU work; it never runs whole
requests to completion under a serialization lock.

The round is:

1. Each ready request drafts from its confirmed target-layer hidden ring.
2. Concatenate `[pending token, draft_1, ..., draft_k]` from all ready requests.
3. Snapshot each request's SWA ring and compressor remainder, then verify the
   concatenated target rows. Dense, HC, MoE and vocabulary projections use
   row tiles of at most the existing GEMV bound (8). C4/K5 has 24 logical
   target rows, not a 4-row plain decode in disguise.
4. The host samples each request's target logits in order. The target sample
   wins. Matching a deterministic draft advances to the next verified row;
   mismatch becomes the next round's pending token. A deterministic proposal
   is a point-mass distribution, so this preserves target sampling rather
   than accepting based on a heuristic confidence threshold.
5. Stop strings, EOS, output budget and tool syntax are evaluated after every
   target token and before consuming another draft. Never emit provisional
   draft bytes or advance RNG/penalty state for unaccepted future rows.
6. Commit the confirmed input plus accepted draft inputs. The existing native
   rollback restores rejected SWA cells and compressor pending rows and
   rewinds compressed-cache counts and the valid hidden-ring frontier.

## Why row tiles

The inherited Q4_K head changes arithmetic beyond the small decode-row bound:
its prefill GEMM rounds weights differently from its decode GEMV. Sending 24
rows directly through that branch would change the target being sampled.
The new code caches non-owning row views and tiles shared projections while
keeping each request's attention history and absolute positions independent.
The extra launches are a performance tradeoff, not a measured speedup claim.

`--mtp-mode on` requests the model's trained block, bounded by remaining
context. `auto` uses request-local draft confidence/cost observations; it does
not disable MTP merely because more than two requests are active. `off`
retains packed ordinary decode. Missing/failed draft state is an error when
MTP is requested, never an automatic switch to plain decode or C1.

## Vision and lifecycle

Existing image preprocessing and the tower remain connected. Prefill writes
image-aware target state and captures the hidden ring; decode drops image
scratch, not that ring. Engram providers are request-local and are drained
before native pinned destinations are freed. Cancelling one response frees
only that request. A failed native target step poisons its entire cohort;
partially written CUDA state is never reused.

A new image tower pass is still non-preemptive. This patch does not add
cross-request prefix reuse, persistent banks, disk KV, or a whole-batch target
CUDA graph. Keep `--prefix-reuse off` in the concurrent launcher. Draft graphs
are request-local. The historical serial MTP path is unchanged.

## Tests and their scope

`make test-ds41-mtp-control` compiles the production native include against
checked in-memory tensor doubles. It covers C4/K5 packing, row tiling, C2..C8,
every rejection depth, mixed verification widths, near-end contexts, repeated
rounds, missing drafts, cancellation/reuse, duplicate requests, target/feed
failure, short prompts, partial allocation failure and cleanup. ASan/UBSan:

```bash
make test-ds41-mtp-control \
  DS41_TEST_SANITIZE='-fsanitize=address,undefined -fno-omit-frame-pointer'
cargo test -p ds4-core v41_run::batch -- --test-threads=1
cargo test -p ds4-core v41_serving_contract -- --test-threads=1
```

The native control fixture and sanitizers were run in the authoring
container. Rust workspace compilation, CUDA compilation, real model logits,
Vision output and GB10 throughput were **not run in that container**. The
mock target is not a model-quality oracle and does not establish numeric
parity of the real kernels.

The real-GPU test is separate and runs the actual binary/model:

```bash
bash tests/ds41_mtp_http_gate.sh \
  ./ds4-server "$MODEL" "$ENGRAM" "$VISION" \
  /path/to/image-a.png /path/to/image-b.png
```

It compares greedy C1 plain against C4 MTP, including mixed image/text
requests, a tool call and tool-result continuation. It requires a logged
four-request target step with more than four verification rows, and actual
MTP offered-token counters. Starting the listener alone cannot pass it.
All requests/responses/logs are retained. It runs on an unused loopback port
and stops only the process it created. This gate has not been run here.

## Thinking repetition and parity

Run `bash tests/ds41_repetition_gate.sh /path/to/reproducing-image.png` on the model host. It starts the existing server launcher for C1/C4 × MTP ON/OFF and records raw reasoning, repeated spans, token-limit exhaustion, tools, and temperature-0 parity. See [repetition gate](ds41-repetition-gate.md). It never stops unrelated processes.
