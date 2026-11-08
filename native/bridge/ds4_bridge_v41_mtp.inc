/* SPDX-License-Identifier: MIT
 * Explicit request-local draft/verify/commit ABI. Included AFTER the packed
 * bridge: its opaque handles bind native state to one immutable model.
 */
int ds4_bridge_v41_request_set_mtp(ds4_bridge_v41_request *r, int mode,
        uint32_t verify, int graph, char *err, size_t errlen) {
    int rc = r ? ds4_v41_request_set_mtp(r->native, mode, verify, graph) : 1;
    if (rc) set_err(err, errlen, "V4.1 requested MTP could not be armed; no plain-decode fallback");
    return rc;
}
int ds4_bridge_v41_batch_verify(ds4_bridge_v41_batch *b,
        ds4_bridge_v41_request *const *requests, const int32_t *tokens, int count,
        float *logits, size_t floats, int32_t *proposals, size_t proposal_slots,
        uint32_t *widths, char *err, size_t errlen) {
    struct ds4_v41_request *native[DS41_BATCH_MAX];
    if (!b || !requests || !tokens || !logits || !proposals || !widths ||
        count < 1 || count > DS41_BATCH_MAX ||
        proposal_slots < (size_t)count * DS41_VERIFY_MAX) {
        set_err(err, errlen, "invalid V4.1 packed MTP batch/output buffers"); return 1;
    }
    for (int i = 0; i < count; i++) {
        if (!requests[i] || requests[i]->model != b->model) {
            set_err(err, errlen, "V4.1 verification cannot mix models"); return 1;
        }
        native[i] = requests[i]->native;
    }
    int rc = ds4_v41_batch_verify(b->native, native, tokens, count, logits, floats,
                                 proposals, proposal_slots, widths);
    if (rc) set_err(err, errlen, "V4.1 packed MTP verification failed; discard the active cohort");
    return rc;
}
int ds4_bridge_v41_request_commit(ds4_bridge_v41_request *r, uint32_t keep,
        char *err, size_t errlen) {
    int rc = r ? ds4_v41_request_commit(r->native, keep) : 1;
    if (rc) set_err(err, errlen, "V4.1 speculative rollback/commit failed; discard the active cohort");
    return rc;
}
