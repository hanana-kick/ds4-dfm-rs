/* SPDX-License-Identifier: MIT
 * Opaque, owner-thread-affine V4.1 requests. No HTTP or scheduling policy.
 * Provider/patch pointers are retained until request_close; Rust owns stable
 * allocations for both. The provider drains before native pinned buffers die.
 */
#include "ds41_batch.h"
struct ds4_bridge_v41_request {
    ds4_bridge_model *model;
    struct ds4_v41_request *native;
    ds4_v41_feed *feed;
    struct ds4_v41_images *images;
    ds4_bridge_v41_drain_fn drain;
    void *provider;
};
struct ds4_bridge_v41_batch {
    ds4_bridge_model *model;
    struct ds4_v41_batch *native;
};
uint64_t ds4_bridge_v41_request_peak(ds4_bridge_model *m, int np, int predict, int no_engram) {
    return m ? ds4_v41_request_peak(m->engine, np, predict, no_engram) : UINT64_MAX;
}
void ds4_bridge_v41_request_close(ds4_bridge_v41_request *r) {
    if (!r) return;
    /* drain's contract includes joining failed/cancelled async reads, even
     * when its return code reports the original I/O failure. */
    if (r->feed && r->drain) (void)r->drain(r->provider);
    ds4_v41_request_close(r->native);
    ds4_v41_images_free(r->images);
    ds4_v41_feed_close(r->feed);
    free(r);
}
ds4_bridge_v41_request *ds4_bridge_v41_request_open(ds4_bridge_model *m,
        const int32_t *prompt, int np, int predict, int no_engram,
        const ds4_bridge_v41_images *images,
        ds4_bridge_v41_rows_fn rows, ds4_bridge_v41_wait_fn wait,
        ds4_bridge_v41_next_fn next, ds4_bridge_v41_drain_fn drain, void *provider,
        char *err, size_t errlen) {
    if (!m || !m->engine || !prompt || np < 1 || predict < 1 ||
        ((rows || wait || next) && (!drain || !provider)) ||
        (images && (images->n <= 0 || images->np != (uint32_t)np))) {
        set_err(err, errlen, "invalid V4.1 request or provider lifetime contract"); return NULL;
    }
    ds4_bridge_v41_request *r = calloc(1, sizeof *r);
    if (!r) { set_err(err, errlen, "V4.1 request allocation failed"); return NULL; }
    r->model = m; r->provider = provider; r->drain = drain;
    if (!no_engram && ds4_v41_engram_count(m->engine) > 0) {
        if (!rows || !drain) goto fail;
        r->feed = ds4_v41_feed_open(m->engine, np, rows, provider, wait, provider, next, provider);
        if (!r->feed) goto fail;
    }
    if (images) {
        r->images = v41_bridge_images_build(images);
        if (!r->images) goto fail;
    }
    r->native = ds4_v41_request_open(m->engine, prompt, np, predict, no_engram,
                                    r->images, ds4_v41_feed_handle(r->feed));
    if (r->native) return r;
fail:
    set_err(err, errlen, "V4.1 request state/feed/image allocation refused");
    ds4_bridge_v41_request_close(r);
    return NULL;
}
int ds4_bridge_v41_request_prefill(ds4_bridge_v41_request *r, float *logits, size_t count,
                                    char *err, size_t errlen) {
    int rc = r ? ds4_v41_request_prefill(r->native, logits, count) : -1;
    if (rc < 0) set_err(err, errlen, "V4.1 request prefill failed; request is poisoned");
    if (rc == 1 && r->images) {
        /* native shrink detached vimg; no graph or CUDA op retains it. */
        ds4_v41_images_free(r->images); r->images = NULL;
    }
    return rc;
}
uint32_t ds4_bridge_v41_request_position(const ds4_bridge_v41_request *r) {
    return r ? ds4_v41_request_position(r->native) : 0;
}
ds4_bridge_v41_batch *ds4_bridge_v41_batch_open(ds4_bridge_model *m, int width, char *err, size_t errlen) {
    if (!m || !m->engine || width < 1 || width > DS41_BATCH_MAX) {
        set_err(err, errlen, "V4.1 batch width must be 1..8 on CUDA"); return NULL;
    }
    ds4_bridge_v41_batch *b = calloc(1, sizeof *b);
    if (!b) { set_err(err, errlen, "V4.1 batch allocation failed"); return NULL; }
    b->model = m; b->native = ds4_v41_batch_open(m->engine, width);
    if (!b->native) { free(b); set_err(err, errlen, "V4.1 packed decode requires a CUDA V4.1 model"); return NULL; }
    return b;
}
void ds4_bridge_v41_batch_close(ds4_bridge_v41_batch *b) {
    if (!b) return;
    ds4_v41_batch_close(b->native); free(b);
}
int ds4_bridge_v41_batch_step(ds4_bridge_v41_batch *b, ds4_bridge_v41_request *const *requests,
        const int32_t *tokens, int count, float *logits, size_t floats, char *err, size_t errlen) {
    struct ds4_v41_request *native[DS41_BATCH_MAX];
    if (!b || !requests || count < 1 || count > DS41_BATCH_MAX) {
        set_err(err, errlen, "invalid V4.1 packed decode batch"); return 1;
    }
    for (int i = 0; i < count; i++) {
        if (!requests[i] || requests[i]->model != b->model) {
            set_err(err, errlen, "V4.1 batch cannot mix models"); return 1;
        }
        native[i] = requests[i]->native;
    }
    int rc = ds4_v41_batch_step(b->native, native, tokens, count, logits, floats);
    if (rc) set_err(err, errlen, "V4.1 packed decode failed; discard the active cohort");
    return rc;
}
