# V4.1 continuous batching: isolated port

Base: Baekpica/ds4-dfm-rs at 94d776ace4a8fc7b09230f83152606adb6a7e150.
Reference: JLFN/YoungAi at a4e6d3177f310cabf3e8d5d214d437a7068ddf23.
Branch: hanana-kick/ds4-dfm-rs:feat/v41-continuous.

## Constraints

- DFM remains the engine, loader, tokenizer, image encoder, and HTTP host.
- Port only per-request native KV, shared-row Forward, multi-sequence CUDA Graph
  and Rust rolling scheduling from YoungAi. Entrpi is an architecture reference.
- Keep the existing single-request V4.1 path and existing other model paths.
- Do not remove the --max-seqs>1 admission gate before native batch proof.
- Preserve per-request vision spans, tool-call streaming, stop/EOS and cancellation.

## File-by-file implementation order

P0: Assert byte/semantic parity serial vs multi text+image requests, save timings.
P1: core_v41_state.c/core_v41_req.c -> ds4_ds41_batch.inc, opaque native request
handles in native/bridge/ds4_bridge.{h,c} and crates/ds4-sys/src/lib.rs.
P2: core_v41_multi.c -> shared-row Forward; independent attention/KV.
core_v41_mgraph.c -> CUDA Graph (with state generation identity, safe teardown).
P3: server_sched_v41.c -> Rust crates/ds4-server/src/v41_cont.rs, implement
ContExec/ContSource rolling admission without changing other families.
P4: Use DFM v41_media.rs and Vision encoder during per-request prefill; the
image participates in the same decode batch as text requests.
P5: Per-request MTP verify/rollback and mixed-modal 2/4/8/16 sequence gates.
MTP may fall back to normal batching for active requests >=3.

## Safety prerequisites

- DFM V41Feed's Arc<Mutex<FeedCore>> shares mutable hist/image_mask and clears
  them on new request: split immutable shard assets from per-request Engram
  history before admitting parallel requests.
- ds4_ds41_kvreuse.inc's global checkpoint K is serial: initially disable
  reuse for batched requests; later implement per-bank image-bound checkpoints.
- Per-request sampler, RNG, progress/greedy callbacks and image pointers must
  not remain process-global. Drain pinned IO on cancellation.
- Prefix identity must include actual image bytes/geometry/position, never
  text placeholder tokens alone.
- GPU memory admission must account for prefill peak, per-request KV, and Vision.
  A max-seqs=16 configuration does not imply 16 full 1M contexts fit GB10.
- Compare greedy serial output and committed KV frontier, detect stale graph
  pointers and CUDA errors; never count serial calls as a batch pass.

## CI

Push on feature branch runs model-free tests on GitHub-hosted Ubuntu.
Explicit workflow_dispatch GPU modes need a trusted Linux ARM64 self-hosted
runner labeled dgx-spark-v41, and a maintenance window with no active model
service. The GPU test binds 127.0.0.1:18080 only and NEVER kills production.
