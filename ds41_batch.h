/* SPDX-License-Identifier: MIT
 * DeepSeek V4.1 request-local state and packed plain decode.
 * Arithmetic adapted from fodelf/YoungAi @ 7fb26cc332d740416792204a29c7d13d2986caf5,
 * src/core/core_v41_{multi,req,state}.c. Scheduling/sampling belong to Rust.
 */
#ifndef DS41_BATCH_H
#define DS41_BATCH_H
#include "ds41_forward.h"
#include "ds41_round.h"
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
#define DS41_BATCH_MAX 8
struct ds4_v41_request;
struct ds4_v41_batch;
/* Request owns its prompt and all mutable CUDA state. Feed and image patches
 * are borrowed until close; the bridge drains providers before destroying them.
 * No global checkpoint, sampling face, or progress hook is read or written.
 * Model weights/GR/RB must remain unchanged throughout every active request. */
struct ds4_v41_request *ds4_v41_request_open(void *engine,
    const int32_t *prompt, int np, int n_predict, int no_engram,
    struct ds4_v41_images *images, const ds41_engram_feed *feed);
void ds4_v41_request_close(struct ds4_v41_request *r);
/* Run one prefill chunk; 0 = more chunks, 1 = complete, -1 = error.
 * On complete, copy exactly vocab floats to logits and shrink prefill scratch.
 * A failed request is poisoned; it cannot be re-entered. */
int ds4_v41_request_prefill(struct ds4_v41_request *r, float *logits, size_t count);
uint32_t ds4_v41_request_position(const struct ds4_v41_request *r);
uint64_t ds4_v41_request_peak(void *engine, int np, int n_predict, int no_engram);
struct ds4_v41_batch *ds4_v41_batch_open(void *engine, int width);
void ds4_v41_batch_close(struct ds4_v41_batch *b);
/* One token per ready request, row-packed projection/MoE, isolated attention.
 * Read row i into host logits + i*vocab. No sampling/accept/EOS decisions here.
 * Any execution failure poisons the batch and ALL supplied requests. */
int ds4_v41_batch_step(struct ds4_v41_batch *b, struct ds4_v41_request **requests,
    const int32_t *tokens, int count, float *logits, size_t floats);
/* Set before prefill: 0=off, 1=adaptive, 2=on. Missing MTP is an error. */
int ds4_v41_request_set_mtp(struct ds4_v41_request *r, int mode, uint32_t verify, int graph);
/* Fixed-stride proposals (DS41_VERIFY_MAX per request), dense packed logits.
 * Target rows remain provisional until request_commit(1+accepted drafts). */
int ds4_v41_batch_verify(struct ds4_v41_batch *b, struct ds4_v41_request **requests,
    const int32_t *tokens, int count, float *logits, size_t floats,
    int32_t *proposals, size_t proposal_slots, uint32_t *widths);
int ds4_v41_request_commit(struct ds4_v41_request *r, uint32_t keep);
#ifdef __cplusplus
}
#endif
#endif
