/* DeepSeek V4.1 independent sequence state and shared dense Forward.
 * Port target: JLFN/YoungAi src/core/{core_v41_req,core_v41_multi,core_v41_state}.c
 * DFM retains its own VQ, Vision, Engram and sampling kernels.
 * Included after ds4_ds41_draft.inc in the V4.1 native translation unit.
 *
 * Batch MTP and CUDA Graph are separate gates. This entry runs target-model
 * decode without speculation and cannot be used as evidence for those gates. */
#ifndef DS4_DS41_BATCH_INC
#define DS4_DS41_BATCH_INC
#define DS4_V41_MULTI_MAX 16u

typedef struct ds4_v41_request {
    ds4_engine *engine;
    ds4_v41_state st;
    const int32_t *prompt;          /* caller-owned until prefill completes */
    uint32_t np, done, cap;
    int n_predict, prefilled, failed;
    struct ds4_v41_images *images;  /* borrowed, prefill only */
    ds4_decode_sampling sampling;   /* per-request; no global mutation */
    ds4_gpu_tensor *am;
    float *rowbuf;
    v41_hist hist;
    uint64_t rng;
    int32_t next;
} ds4_v41_request;

typedef struct ds4_v41_batch {
    ds4_engine *engine;
    ds4_v41_state rows;         /* shared dense + attention projection rows */
    uint32_t cap;               /* row capacity, not a model context cap */
    float *onehot;
} ds4_v41_batch;

typedef struct {
    ds4_gpu_tensor *tok, *pos, *xn, *erows, *qrn, *q, *kvn, *o, *logits;
} v41_batch_views;

static ds4_gpu_tensor *v41_batch_view(const ds4_gpu_tensor *t, uint64_t r0, uint64_t rows, uint64_t bytes_per_row) {
    return t ? ds4_gpu_tensor_view(t, r0 * bytes_per_row, rows * bytes_per_row) : NULL;
}
static void v41_batch_detach(ds4_v41_state *st, v41_batch_views *v) {
    /* The base request state no longer owns dense rows after shrink. */
    if (v->tok) ds4_gpu_tensor_free(v->tok);
    if (v->pos) ds4_gpu_tensor_free(v->pos);
    if (v->xn) ds4_gpu_tensor_free(v->xn);
    if (v->erows) ds4_gpu_tensor_free(v->erows);
    if (v->qrn) ds4_gpu_tensor_free(v->qrn);
    if (v->q) ds4_gpu_tensor_free(v->q);
    if (v->kvn) ds4_gpu_tensor_free(v->kvn);
    if (v->o) ds4_gpu_tensor_free(v->o);
    if (v->logits) ds4_gpu_tensor_free(v->logits);
    st->tok = st->pos = st->xn = st->erows = st->qrn = st->q = st->kvn = st->o = st->logits = NULL;
    memset(v, 0, sizeof *v);
}
static bool v41_batch_attach(ds4_v41_state *st, ds4_v41_state *b, uint32_t row, v41_batch_views *v) {
    const uint64_t E = DS4_N_EMBD, Q = DS4_N_LORA_Q;
    const uint64_t HD = DS4_N_HEAD_DIM, NH = DS4_N_HEAD;
    const uint64_t er = (uint64_t)(g_ds4_v41.engram_max_ngram - 1u) *
                        g_ds4_v41.engram_heads * g_ds4_v41.engram_head_dim;
    memset(v, 0, sizeof *v);
    v->tok = v41_batch_view(b->tok, row, 1u, 4u);
    v->pos = v41_batch_view(b->pos, row, 1u, 4u);
    v->xn = v41_batch_view(b->xn, row, 1u, E * 4u);
    v->qrn = v41_batch_view(b->qrn, row, 1u, Q * 4u);
    v->q = v41_batch_view(b->q, row, 1u, NH * HD * 4u);
    v->kvn = v41_batch_view(b->kvn, row, 1u, HD * 4u);
    v->o = v41_batch_view(b->o, row, 1u, NH * HD * 4u);
    v->logits = v41_batch_view(b->logits, row, 1u, (uint64_t)DS4_N_VOCAB * 4u);
    if (er) v->erows = v41_batch_view(b->erows, row, 1u, er * 4u);
    if (!v->tok || !v->pos || !v->xn || !v->qrn || !v->q || !v->kvn ||
        !v->o || !v->logits || (g_ds4_v41.n_engram && !v->erows)) {
        v41_batch_detach(st, v);
        return false;
    }
    st->tok = v->tok; st->pos = v->pos; st->xn = v->xn;
    st->qrn = v->qrn; st->q = v->q; st->kvn = v->kvn;
    st->o = v->o; st->logits = v->logits; st->erows = v->erows;
    st->n = 1u; st->pos0 = st->n_past; st->graph = 0; st->egraph_uploaded = 0;
    st->idx_owner = st->cand_owner = -1; st->idx_topk = st->idx_ratio = 0u;
    st->ced_skip = 0; st->vrows_on = 0; st->mainh_wrote = 0u;
    return true;
}

/* Relinquish the 2K-wide prefill row workspace while retaining request KV,
 * compact cache, pending compressor rows, and main_hidden draft history.
 * A fresh batch-owned workspace supplies every shared Forward row later. */
static bool v41_batch_shrink(ds4_v41_state *st, uint32_t rcap) {
    if (!st || rcap == 0u || rcap > st->cap_tok) return false;
    const uint64_t rowb = (uint64_t)DS4_N_HEAD_DIM * 4u;
    const uint64_t swa = DS4_N_SWA;
    v41_graph_free(st);
    v41_free_dense(st);
    v41_free_attn(st);
    V41_FREE(&st->erows, &st->ekv, &st->iscore, &st->cand);
    for (uint32_t i = 0; i < DS4_V41_MAX_ENGRAM; ++i) V41_FREE(&st->eraw[i]);
    st->iscap = st->isrows = st->icrows = 0u;
    st->iscap_gen++;
    bool ok = true;
    v41_alloc_attn_cache(st, rcap, &ok);
    if (g_ds4_v41.n_engram) {
        const uint64_t cols = (uint64_t)(g_ds4_v41.engram_max_ngram - 1u) * g_ds4_v41.engram_heads;
        const uint64_t stride = g_ds4_v41.engram_head_dim + g_ds4_v41.engram_head_dim / 32u;
        for (uint32_t i = 0; i < g_ds4_v41.n_engram; i++) st->eraw[i] = v41_alloc(rcap * cols * stride, &ok);
    }
    /* Preserve the full 128-row history exactly, not just the current tail.
     * Discarded rows belong to previous requests only after their GPU writes
     * have completed. The owner drains the engine on any failure. */
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor *new_window = v41_alloc((swa + rcap) * rowb, &ok);
        if (ok && !new_window) ok = false;
        if (ok && !ds4_gpu_tensor_fill_f32(new_window, 0.f, (swa + rcap) * DS4_N_HEAD_DIM)) ok = false;
        if (ok && !ds4_gpu_tensor_copy(new_window, 0, st->win[il], 0, swa * rowb)) ok = false;
        if (!ok) {
            if (new_window) ds4_gpu_tensor_free(new_window);
            break;
        }
        if (!ds4_gpu_synchronize()) { ds4_gpu_tensor_free(new_window); ok = false; break; }
        ds4_gpu_tensor_free(st->win[il]); st->win[il] = new_window;
        V41_FREE(&st->snap_win[il], &st->snap_cpre_kv[il], &st->snap_cpre_sc[il]);
        if (!g_ds4_v41.is_kv_source[il] || !st->cpre_kv[il]) continue;
        const uint32_t ratio = ds4_layer_compress_ratio(il);
        ds4_gpu_tensor *ck = v41_alloc(((uint64_t)ratio + rcap) * rowb, &ok);
        ds4_gpu_tensor *cs = v41_alloc(((uint64_t)ratio + rcap) * rowb, &ok);
        if (ok && (!ds4_gpu_tensor_copy(ck, 0, st->cpre_kv[il], 0, (uint64_t)ratio * rowb) ||
                   !ds4_gpu_tensor_copy(cs, 0, st->cpre_sc[il], 0, (uint64_t)ratio * rowb))) ok = false;
        if (!ok || !ds4_gpu_synchronize()) {
            if (ck) ds4_gpu_tensor_free(ck);
            if (cs) ds4_gpu_tensor_free(cs);
            ok = false; break;
        }
        ds4_gpu_tensor_free(st->cpre_kv[il]); ds4_gpu_tensor_free(st->cpre_sc[il]);
        st->cpre_kv[il] = ck; st->cpre_sc[il] = cs;
    }
    if (!ok) return false;
    st->cap_tok = rcap; st->logits_rows = 0u;
    st->n_direct1 = 0u; memset(st->n_direct_n, 0, sizeof st->n_direct_n);
    return true;
}

struct ds4_v41_request *ds4_v41_req_create(void *engine, const int32_t *prompt,
                                           int n_prompt, int n_predict,
                                           const ds4_decode_sampling *sampling,
                                           const ds41_engram_feed *feed,
                                           struct ds4_v41_images *images) {
    ds4_engine *e = (ds4_engine *)engine;
    if (!e || DS4_MODEL_VARIANT != DS4_VARIANT_V41 || !prompt ||
        n_prompt < 1 || n_predict < 0 || (uint32_t)n_prompt >= g_ds4_v41.ctx) return NULL;
    const uint32_t np = (uint32_t)n_prompt;
    const uint64_t want = (uint64_t)np + (uint32_t)n_predict + DS4_MTP_MAX_BLOCK + 2u;
    const uint32_t ctx = want < g_ds4_v41.ctx ? (uint32_t)want : g_ds4_v41.ctx;
    const uint32_t cap = v41_gen_cap(e, np);
    ds4_v41_request *r = (ds4_v41_request *)calloc(1, sizeof *r);
    if (!r) return NULL;
    r->engine = e; r->prompt = prompt; r->np = np; r->cap = cap;
    r->n_predict = n_predict; r->images = images;
    if (sampling) r->sampling = *sampling;
    r->rng = r->sampling.seed ? r->sampling.seed :
             ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    if (!v41_state_alloc(&r->st, cap, ctx, DS4_MTP_MAX_BLOCK + 2u)) { free(r); return NULL; }
    r->st.head_last_only = 1;
    r->st.efeed = feed; r->st.vimg = images; r->st.psamp = &r->sampling;
    r->st.dev_sample = r->sampling.temperature > 0.f &&
        !(r->sampling.freq_penalty != 0.f || r->sampling.presence_penalty != 0.f || r->sampling.dry_multiplier > 0.f);
    r->st.samp = (ds4_gpu_sample_params){
        .temperature = r->sampling.temperature, .top_p = r->sampling.top_p,
        .min_p = r->sampling.min_p, .top_k = r->sampling.top_k, .seed = r->rng
    };
    const bool penalty = r->sampling.freq_penalty != 0.f ||
        r->sampling.presence_penalty != 0.f || r->sampling.dry_multiplier > 0.f;
    if (penalty) {
        r->rowbuf = (float *)malloc((size_t)DS4_N_VOCAB * sizeof(float));
        r->hist.cap = (uint32_t)n_predict + 2u;
        r->hist.tok = (int32_t *)malloc((size_t)r->hist.cap * sizeof(int32_t));
        if (r->sampling.dry_multiplier > 0.f)
            r->hist.brk = ds4_decode_breakers(e, DS4_N_VOCAB);
    }
    r->am = ds4_gpu_tensor_alloc((DS4_MTP_MAX_BLOCK + 2u) * 16u);
    if (!r->am || (penalty && (!r->rowbuf || !r->hist.tok))) {
        ds4_gpu_tensor_free(r->am);
        free(r->rowbuf); free(r->hist.tok); free((void *)r->hist.brk);
        v41_state_free(&r->st); free(r); return NULL;
    }
    return r;
}

void ds4_v41_req_destroy(struct ds4_v41_request *r) {
    if (!r) return;
    (void)ds4_gpu_synchronize();
    if (r->am) ds4_gpu_tensor_free(r->am);
    /* Never free an attached tensor view as part of the request's ownership. */
    v41_state_free(&r->st);
    free(r->rowbuf); free(r->hist.tok); free((void *)r->hist.brk);
    free(r);
}
int ds4_v41_req_prefill(struct ds4_v41_request *r) {
    if (!r || r->failed) return -1;
    if (r->prefilled) return 1;
    if (!v41_prefill_chunk(r->engine, &r->st, r->prompt, r->np, &r->done, r->cap)) {
        r->failed = 1; return -1;
    }
    if (r->done < r->np) return 0;
    r->st.vimg = NULL;
    if (!v41_next_token(&r->st, r->am, r->st.last_logit_row,
                        r->rowbuf, &r->rng, &r->hist, &r->next) ||
        !v41_batch_shrink(&r->st, 1u)) {
        r->failed = 1; return -1;
    }
    r->prefilled = 1;
    return 1;
}
int ds4_v41_req_next(const struct ds4_v41_request *r, int *out) {
    if (!r || !out || !r->prefilled || r->failed) return 0;
    *out = r->next; return 1;
}
int ds4_v41_req_position(const struct ds4_v41_request *r) {
    return r && r->prefilled && !r->failed ? (int)r->st.n_past : -1;
}
int ds4_v41_req_prefill_progress(const struct ds4_v41_request *r, int *completed, int *total) {
    if (!r || !completed || !total) return 0;
    *completed = (int)r->done; *total = (int)r->np; return 1;
}
void ds4_v41_batch_destroy(struct ds4_v41_batch *b);
struct ds4_v41_batch *ds4_v41_batch_create(void *engine, int rows) {
    ds4_engine *e = (ds4_engine *)engine;
    if (!e || DS4_MODEL_VARIANT != DS4_VARIANT_V41 || rows < 2 || rows > (int)DS4_V41_MULTI_MAX) return NULL;
    ds4_v41_batch *b = (ds4_v41_batch *)calloc(1, sizeof *b);
    if (!b) return NULL;
    b->engine = e; b->cap = (uint32_t)rows;
    b->rows.cap_tok = b->cap; b->rows.ctx = g_ds4_v41.ctx;
    b->rows.idx_owner = b->rows.cand_owner = -1;
    bool ok = true;
    v41_alloc_dense(&b->rows, b->cap, b->cap, &ok);
    v41_alloc_attn_rows(&b->rows, b->cap, &ok);
    if (g_ds4_v41.n_engram) {
        const uint64_t cols = (uint64_t)(g_ds4_v41.engram_max_ngram - 1u) * g_ds4_v41.engram_heads;
        const uint64_t len = cols * g_ds4_v41.engram_head_dim;
        b->rows.erows = v41_alloc((uint64_t)b->cap * len * 4u, &ok);
        b->rows.ekv = v41_alloc((uint64_t)b->cap * (DS4_N_HC + 1u) * DS4_N_EMBD * 4u, &ok);
    }
    b->onehot = (float *)calloc((size_t)b->cap * DS4_N_HC, sizeof(float));
    if (!b->onehot) ok = false;
    if (b->onehot) for (uint32_t i = 0; i < b->cap; i++) b->onehot[i * DS4_N_HC] = 1.0f;
    if (!ok) { ds4_v41_batch_destroy(b); return NULL; }
    return b;
}
void ds4_v41_batch_destroy(struct ds4_v41_batch *b) {
    if (!b) return;
    (void)ds4_gpu_synchronize();
    v41_state_free(&b->rows);
    free(b->onehot); free(b);
}
static bool v41_batch_body(ds4_v41_batch *b, ds4_v41_request *const *req, int n) {
    ds4_engine *e = b->engine;
    ds4_v41_state *B = &b->rows;
    const ds4_model *m = &e->model;
    const uint32_t E = DS4_N_EMBD, HC = DS4_N_HC;
    B->n = (uint32_t)n; B->pos0 = 0u; B->graph = 0; B->ced_skip = 0;
    if (!v41_embed(m, B->x, B->tok, e->weights.token_embd, DS4_N_VOCAB, n, E) ||
        !ds4_gpu_v41_expand_hc_tensor(B->hc, B->x, E, HC, n)) return false;
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &e->weights.layer[il];
        if (g_ds4_v41.engram_index_of[il] >= 0) {
            for (int i = 0; i < n; i++) {
                if (!req[i]->st.no_engram && !v41_engram_rows(e, &req[i]->st, il)) return false;
            }
            if (!v41_engram_apply(e, B, il)) return false;
        }
        if (g_ds4_v41.mtp_target_slot[il] >= 0) for (int i = 0; i < n; i++) {
            ds4_v41_state *s = &req[i]->st;
            if (!s->mainh) continue;
            if (!ds4_gpu_v41_hc_mean_tensor(s->mainh, B->hc, E, HC,
                                            1u, (uint32_t)i,
                                            (uint32_t)g_ds4_v41.mtp_target_slot[il],
                                            g_ds4_v41.n_mtp_target, s->n_past, s->mainh_cap, NULL))
                return false;
            s->mainh_wrote = 1u;
        }
        if (!v41_hc_half(e, B, l, true) || !v41_attn_in(e, B, il)) return false;
        for (int i = 0; i < n; i++)
            if (!v41_attn_cache(e, &req[i]->st, il)) return false;
        if (!v41_attn_out(e, B, il)) return false;
        if (!ds4_gpu_v41_hc_post_tensor(B->hc2, B->attn_out, B->hc,
                                        B->post, B->comb, E, HC, n)) return false;
        { ds4_gpu_tensor *temp = B->hc; B->hc = B->hc2; B->hc2 = temp; }
        if (!v41_hc_half(e, B, l, false) || !v41_moe(m, l, B, il)) return false;
        if (!ds4_gpu_v41_hc_post_tensor(B->hc2, B->y, B->hc,
                                        B->post, B->comb, E, HC, n)) return false;
        { ds4_gpu_tensor *temp = B->hc; B->hc = B->hc2; B->hc2 = temp; }
    }
    if (!ds4_gpu_v41_hc_pre_tensor(B->x, B->hc, B->pre_mix, E, HC, n) ||
        !ds4_gpu_v41_rms_norm_tensor(B->xn, B->x, m->map, m->size,
                                    e->weights.output_norm->abs_offset, E, n, DS4_RMS_EPS) ||
        !v41_tproj(m, B->logits, e->weights.output, E, DS4_N_VOCAB, B->xn, n, 0)) return false;
    return true;
}
int ds4_v41_batch_step(struct ds4_v41_batch *b, struct ds4_v41_request *const *r, int n,
                       int32_t *next) {
    if (!b || !r || !next || n < 2 || (uint32_t)n > b->cap) return 1;
    int32_t *ids = (int32_t *)malloc((size_t)n * 4u);
    int32_t *pos = (int32_t *)malloc((size_t)n * 4u);
    v41_batch_views *views = (v41_batch_views *)calloc((size_t)n, sizeof *views);
    if (!ids || !pos || !views) { free(ids); free(pos); free(views); return 1; }
    bool ok = true;
    for (int i = 0; i < n; i++) {
        ds4_v41_request *q = r[i];
        if (!q || q->engine != b->engine || !q->prefilled || q->failed ||
            q->st.n_past >= q->st.ctx) { ok = false; break; }
        for (int j = 0; j < i; j++) if (r[j] == q) ok = false;
        if (!ok) break;
        ids[i] = q->next; pos[i] = (int32_t)q->st.n_past;
        const v41_engram_feed *fd = q->st.efeed;
        if (g_ds4_v41.n_engram && (!fd || !fd->prepare ||
            fd->prepare(q->st.n_past, &ids[i], 1, fd->ud))) { ok = false; break; }
    }
    if (ok) ok = ds4_gpu_tensor_write(b->rows.tok, 0, ids, (uint64_t)n * 4u) &&
                 ds4_gpu_tensor_write(b->rows.pos, 0, pos, (uint64_t)n * 4u) &&
                 ds4_gpu_tensor_write(b->rows.pre_mix, 0, b->onehot, (uint64_t)n * DS4_N_HC * 4u);
    int attached = 0;
    for (int i = 0; ok && i < n; i++) {
        if (!v41_batch_attach(&r[i]->st, &b->rows, (uint32_t)i, &views[i])) { ok = false; break; }
        attached++;
    }
    if (ok) {
        if (!ds4_gpu_begin_commands()) ok = false;
        else {
            ok = v41_batch_body(b, r, n);
            if (!ds4_gpu_end_commands() || !ds4_gpu_synchronize()) ok = false;
        }
    }
    /* Read logits through per-row views before freeing them. */
    for (int i = 0; ok && i < n; i++)
        if (!v41_next_token(&r[i]->st, r[i]->am, 0u, r[i]->rowbuf,
                            &r[i]->rng, &r[i]->hist, &next[i])) ok = false;
    for (int i = 0; i < attached; i++) {
        ds4_v41_state *st = &r[i]->st;
        if (ok) {
            if (st->mainh && st->mainh_wrote) {
                const bool contiguous = st->mainh_n && st->mainh_end == (int64_t)st->pos0 - 1;
                const uint32_t count = contiguous ? st->mainh_n + st->mainh_wrote : st->mainh_wrote;
                st->mainh_n = count > st->mainh_cap ? st->mainh_cap : count;
                st->mainh_end = (int64_t)st->pos0;
            }
            st->n_past++;
            r[i]->next = next[i];
        } else r[i]->failed = 1;
        v41_batch_detach(st, &views[i]);
    }
    /* A failed batch may have partly mutated KV. Fail closed, never resume. */
    if (!ok) for (int i = 0; i < n; i++) if (r[i]) r[i]->failed = 1;
    if (ok) fprintf(stderr, "[v41-batch] forward sequences=%d rows=%d\n", n, n);
    free(ids); free(pos); free(views);
    return ok ? 0 : 1;
}
#endif /* DS4_DS41_BATCH_INC */
