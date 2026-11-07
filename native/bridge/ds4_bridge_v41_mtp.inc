/* SPDX-License-Identifier: MIT
 * Owner-thread bridge for concurrent DSpark verification transactions.
 * Included after ds4_bridge_v41_batch.inc; opaque request structs stay private.
 */
uint64_t ds4_bridge_v41_request_mtp_bytes(ds4_bridge_model *m) {
    return m ? ds4_v41_request_mtp_bytes(m->engine) : UINT64_MAX;
}
int ds4_bridge_v41_request_mtp(ds4_bridge_v41_request *r, int on, int k, char *err, size_t len) {
    int rc = r ? ds4_v41_request_mtp(r->native, on, k) : 1;
    if (rc) set_err(err, len, "V4.1 request could not initialize embedded MTP");
    return rc;
}
int ds4_bridge_v41_request_commit(ds4_bridge_v41_request *r, uint32_t keep, char *err, size_t len) {
    if (!r || (r->feed && (!r->drain || r->drain(r->provider)))) {
        set_err(err, len, "V4.1 Engram drain failed before speculative commit"); return 1;
    }
    int rc = ds4_v41_request_commit(r->native, keep);
    if (rc) set_err(err, len, "V4.1 speculative rollback/commit failed; discard cohort");
    return rc;
}
int ds4_bridge_v41_batch_verify(ds4_bridge_v41_batch *b, ds4_bridge_v41_request *const *requests,
        const int32_t *tokens, const uint32_t *remaining, int count, int32_t *candidates,
        uint32_t *rows, float *logits, size_t floats, char *err, size_t errlen) {
    struct ds4_v41_request *native[DS41_BATCH_MAX];
    if (!b || !requests || !tokens || !remaining || !candidates || !rows || !logits ||
        count < 1 || count > DS41_BATCH_MAX) {
        set_err(err, errlen, "invalid V4.1 speculative batch"); return 1;
    }
    for (int i = 0; i < count; i++) {
        if (!requests[i] || requests[i]->model != b->model) {
            set_err(err, errlen, "V4.1 verification cannot mix models"); return 1;
        }
        for (int j = 0; j < i; j++) if (requests[j] == requests[i]) {
            set_err(err, errlen, "V4.1 verification contains a duplicate request"); return 1;
        }
        native[i] = requests[i]->native;
    }
    int rc = ds4_v41_batch_verify(b->native, native, tokens, remaining, count,
                                  candidates, rows, logits, floats);
    if (rc) set_err(err, errlen, "V4.1 packed MTP verification failed; discard cohort");
    return rc;
}
