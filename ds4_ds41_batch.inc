/* SPDX-License-Identifier: MIT
 * Request-local V4.1 packed decode. Derived from YoungAi's core_v41_multi.c,
 * core_v41_req.c and core_v41_state.c at 7fb26cc332d740416792204a29c7d13d2986caf5.
 * The Entrpi rolling-admission model is driven by the Rust host, NOT C HTTP.
 * This first lane is plain decode: no shared MTP graphs, checkpoints or sampler.
 */
#include "ds41_batch.h"

#ifndef DS4_NO_GPU
struct ds4_v41_request {
    ds4_engine *engine;
    ds4_v41_state state;
    int32_t *prompt;
    uint32_t np, cursor, cap;
    bool ready, poisoned;
};
struct ds4_v41_batch {
    ds4_engine *engine;
    ds4_v41_state rows;
    uint32_t width;
    float *onehot;
    bool poisoned;
};
typedef struct {
    ds4_gpu_tensor *tok, *pos, *xn, *erows, *qrn, *q, *kvn, *o;
} v41_batch_views;

/* The native allocation formula, priced for the actual request rather than
 * the metadata's 1M maximum. Includes pinned feeds/positions and host logits.
 * CUDA backend workspace is not exactly priced: the host applies the existing
 * V4.1 2x admission band and its explicit available-memory floor. */
uint64_t ds4_v41_request_peak(void *engine, int np, int n_predict, int no_engram) {
    ds4_engine *e = engine;
    if (!e || DS4_MODEL_VARIANT != DS4_VARIANT_V41 || np < 1 || n_predict < 1 ||
        (uint64_t)np + (uint64_t)n_predict > g_ds4_v41.ctx) return UINT64_MAX;
    uint32_t cap = (uint32_t)np < DS4_V41_CHUNK ? (uint32_t)np : DS4_V41_CHUNK;
    uint32_t ctx = (uint32_t)np + (uint32_t)n_predict;
    uint64_t b = v41_bytes_rows(cap, 1, 1) + v41_bytes_kv(ctx, cap);
    b += (uint64_t)ctx * sizeof(int32_t) + (uint64_t)DS4_N_VOCAB * sizeof(float);
    b += (uint64_t)cap * DS4_N_LAYER * sizeof(int32_t);
    if (!no_engram && g_ds4_v41.n_engram) {
        const ds4_v41_cfg *v = &g_ds4_v41;
        uint64_t cols = (uint64_t)(v->engram_max_ngram - 1) * v->engram_heads;
        uint64_t hd = v->engram_head_dim;
        /* Device raw + pinned host raw, row-id scratch, decoded rows and wkv. */
        b += (uint64_t)cap * (2 * v->n_engram * cols * (hd + hd/32) +
             v->n_engram * cols * 8 + cols * hd * 4 + (DS4_N_HC+1) * DS4_N_EMBD * 4);
    }
    uint64_t groups = 1;
    while (groups < ctx) groups <<= 1;
    uint64_t bs = g_ds4_v41.candidate_block_size > 0 ? (uint64_t)g_ds4_v41.candidate_block_size : 1;
    b += (uint64_t)(cap/32 + 2) * groups * 4 + (uint64_t)cap * ((groups+bs-1)/bs);
    return b;
}

/* Only the compact attention/cache state stays resident for a decode request.
 * Dense projection/MoE rows are the batch's, attached with non-owning views.
 * Preserve every live SWA ring and compressor remainder before dropping its
 * larger allocation. Index scratch is rebuilt on the next decode step. */
static bool v41_batch_shrink(ds4_v41_state *st) {
    const uint64_t hd = DS4_N_HEAD_DIM, swa = DS4_N_SWA, rowb = hd * 4;
    v41_graph_free(st);
    V41_FREE(&st->erows, &st->ekv, &st->iscore, &st->cand, &st->mainh);
    for (uint32_t k = 0; k < DS4_V41_MAX_ENGRAM; k++) V41_FREE(&st->eraw[k]);
    st->iscap = st->isrows = st->icrows = 0;
    st->iscap_gen++;
    v41_free_dense(st);
    v41_free_attn(st);
    bool ok = true;
    v41_alloc_attn_cache(st, 1, &ok);
    if (!st->no_engram && g_ds4_v41.n_engram) {
        const ds4_v41_cfg *v = &g_ds4_v41;
        uint64_t cols = (uint64_t)(v->engram_max_ngram-1) * v->engram_heads;
        uint64_t stride = v->engram_head_dim + v->engram_head_dim/32;
        for (uint32_t k = 0; k < v->n_engram; k++) st->eraw[k] = v41_alloc(cols*stride, &ok);
    }
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor *w = v41_alloc((swa+1)*rowb, &ok);
        if (ok) ok = ds4_gpu_tensor_fill_f32(w, 0.f, (swa+1)*hd) &&
                     ds4_gpu_tensor_copy(w, 0, st->win[il], 0, swa*rowb);
        ds4_gpu_tensor_free(st->win[il]); st->win[il] = w;
        if (st->cpre_kv[il]) {
            uint32_t ratio = ds4_layer_compress_ratio(il);
            ds4_gpu_tensor *ck = v41_alloc((ratio+1u)*rowb, &ok);
            ds4_gpu_tensor *cs = v41_alloc((ratio+1u)*rowb, &ok);
            if (ok) ok = ds4_gpu_tensor_copy(ck, 0, st->cpre_kv[il], 0, ratio*rowb) &&
                         ds4_gpu_tensor_copy(cs, 0, st->cpre_sc[il], 0, ratio*rowb);
            ds4_gpu_tensor_free(st->cpre_kv[il]); st->cpre_kv[il] = ck;
            ds4_gpu_tensor_free(st->cpre_sc[il]); st->cpre_sc[il] = cs;
        }
    }
    st->cap_tok = 1; st->logits_rows = 0; st->n_direct1 = 0;
    memset(st->n_direct_n, 0, sizeof st->n_direct_n);
    st->vimg = NULL; st->vrows_on = 0;
    return ok && ds4_gpu_synchronize();
}

struct ds4_v41_request *ds4_v41_request_open(void *engine,
        const int32_t *prompt, int np, int n_predict, int no_engram,
        struct ds4_v41_images *images, const ds41_engram_feed *feed) {
    ds4_engine *e = engine;
    if (!e || DS4_MODEL_VARIANT != DS4_VARIANT_V41 || !e->metal_ready || !prompt || np < 1 || n_predict < 1 ||
        (uint64_t)np + (uint64_t)n_predict > g_ds4_v41.ctx ||
        (!no_engram && g_ds4_v41.n_engram && !feed) ||
        (images && images->np != (uint32_t)np)) return NULL;
    for (int i = 0; i < np; i++) if (prompt[i] < 0 || (uint32_t)prompt[i] >= DS4_N_VOCAB) return NULL;
    struct ds4_v41_request *r = calloc(1, sizeof *r);
    if (!r) return NULL;
    r->engine = e; r->np = (uint32_t)np;
    r->cap = r->np < DS4_V41_CHUNK ? r->np : DS4_V41_CHUNK;
    r->prompt = malloc((size_t)np * sizeof *prompt);
    if (!r->prompt) { free(r); return NULL; }
    memcpy(r->prompt, prompt, (size_t)np*sizeof *prompt);
    if (!v41_state_alloc(&r->state, r->cap, r->np+(uint32_t)n_predict, 1)) {
        free(r->prompt); free(r); return NULL;
    }
    r->state.head_last_only = 1;
    r->state.no_engram = !!no_engram;
    r->state.efeed = feed;
    r->state.vimg = images;
    /* Concurrent speculation is deliberately not enabled by this API. */
    V41_FREE(&r->state.mainh);
    r->state.mainh_cap = 0;
    return r;
}
void ds4_v41_request_close(struct ds4_v41_request *r) {
    if (!r) return;
    (void)ds4_gpu_synchronize();
    v41_state_free(&r->state);
    free(r->prompt); free(r);
}
uint32_t ds4_v41_request_position(const struct ds4_v41_request *r) {
    return r ? r->state.n_past : 0;
}
int ds4_v41_request_prefill(struct ds4_v41_request *r, float *logits, size_t count) {
    if (!r || r->poisoned || r->ready || !logits || count != DS4_N_VOCAB) return -1;
    if (!v41_prefill_chunk(r->engine, &r->state, r->prompt, r->np, &r->cursor, r->cap)) goto fail;
    if (r->cursor < r->np) return 0;
    if (!ds4_gpu_tensor_read(r->state.logits, (uint64_t)r->state.last_logit_row*count*sizeof(float),
                            logits, count*sizeof(float)) || !v41_batch_shrink(&r->state)) goto fail;
    r->ready = true;
    free(r->prompt); r->prompt = NULL;
    return 1;
fail:
    r->poisoned = true;
    return -1;
}

static void v41_batch_detach(ds4_v41_state *s, v41_batch_views *v) {
    V41_FREE(&v->tok, &v->pos, &v->xn, &v->erows, &v->qrn, &v->q, &v->kvn, &v->o);
    s->tok = s->pos = s->xn = s->erows = s->qrn = s->q = s->kvn = s->o = NULL;
    s->blk_ids = NULL;
}
static bool v41_batch_attach(ds4_v41_state *s, ds4_v41_state *b, uint32_t row, v41_batch_views *v) {
    uint64_t e = DS4_N_EMBD, hd = DS4_N_HEAD_DIM, nh = DS4_N_HEAD, q = DS4_N_LORA_Q;
    uint64_t erow = (uint64_t)(g_ds4_v41.engram_max_ngram-1)*g_ds4_v41.engram_heads*g_ds4_v41.engram_head_dim*4;
#define ROW_VIEW(field, bytes) v->field = ds4_gpu_tensor_view(b->field, (uint64_t)row*(bytes), (bytes))
    ROW_VIEW(tok, 4); ROW_VIEW(pos, 4); ROW_VIEW(xn, e*4); ROW_VIEW(qrn, q*4);
    ROW_VIEW(q, nh*hd*4); ROW_VIEW(kvn, hd*4); ROW_VIEW(o, nh*hd*4);
    if (!s->no_engram && g_ds4_v41.n_engram) { ROW_VIEW(erows, erow); }
#undef ROW_VIEW
    if (!v->tok || !v->pos || !v->xn || !v->qrn || !v->q || !v->kvn || !v->o ||
        (!s->no_engram && g_ds4_v41.n_engram && !v->erows)) return false;
    s->tok = v->tok; s->pos = v->pos; s->xn = v->xn; s->erows = v->erows;
    s->qrn = v->qrn; s->q = v->q; s->kvn = v->kvn; s->o = v->o;
    s->n = 1; s->pos0 = s->n_past;
    s->idx_owner = -1; s->cand_owner = -1; s->idx_topk = s->idx_ratio = 0;
    s->graph = 0; s->ced_skip = s->ced_done = 0; s->vrows_on = 0;
    return true;
}
struct ds4_v41_batch *ds4_v41_batch_open(void *engine, int width) {
    ds4_engine *e = engine;
    if (!e || DS4_MODEL_VARIANT != DS4_VARIANT_V41 || !e->metal_ready || width < 1 || width > DS41_BATCH_MAX) return NULL;
    struct ds4_v41_batch *b = calloc(1, sizeof *b);
    if (!b) return NULL;
    b->engine = e; b->width = (uint32_t)width; b->rows.cap_tok = (uint32_t)width;
    bool ok = true;
    v41_alloc_dense(&b->rows, (uint32_t)width, (uint32_t)width, &ok);
    v41_alloc_attn_rows(&b->rows, (uint32_t)width, &ok);
    if (g_ds4_v41.n_engram) {
        const ds4_v41_cfg *v = &g_ds4_v41;
        uint64_t in_dim = (uint64_t)(v->engram_max_ngram-1)*v->engram_heads*v->engram_head_dim;
        b->rows.erows = v41_alloc((uint64_t)width*in_dim*4, &ok);
        b->rows.ekv = v41_alloc((uint64_t)width*(DS4_N_HC+1)*DS4_N_EMBD*4, &ok);
    }
    b->onehot = calloc((size_t)width*DS4_N_HC, sizeof(float));
    if (!b->onehot) { ds4_v41_batch_close(b); return NULL; }
    for (uint32_t i = 0; i < (uint32_t)width*DS4_N_HC; i++) b->onehot[i] = i%DS4_N_HC == 0 ? 1.f : 0.f;
    if (!ok) { ds4_v41_batch_close(b); return NULL; }
    return b;
}
void ds4_v41_batch_close(struct ds4_v41_batch *b) {
    if (!b) return;
    (void)ds4_gpu_synchronize();
    v41_state_free(&b->rows);
    free(b->onehot); free(b);
}

static bool v41_batch_body(struct ds4_v41_batch *b, struct ds4_v41_request **req, uint32_t n) {
    ds4_engine *e = b->engine;
    ds4_v41_state *s = &b->rows;
    const ds4_model *m = &e->model;
    uint32_t emb = DS4_N_EMBD, hc = DS4_N_HC;
    s->n = n; s->pos0 = 0; s->graph = 0; s->ced_skip = s->ced_done = 0;
    s->no_engram = req[0]->state.no_engram;
    s->idx_owner = s->cand_owner = -1;
    bool ok = v41_embed(m, s->x, s->tok, e->weights.token_embd, DS4_N_VOCAB, n, emb) &&
              ds4_gpu_v41_expand_hc_tensor(s->hc, s->x, emb, hc, n);
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &e->weights.layer[il];
        if (!s->no_engram && g_ds4_v41.engram_index_of[il] >= 0) {
            for (uint32_t i = 0; ok && i < n; i++) ok = v41_engram_rows(e, &req[i]->state, il);
            if (ok) ok = v41_engram_apply(e, s, il);
        }
        if (ok) ok = v41_hc_half(e, s, l, true) && v41_attn_in(e, s, il);
        /* Shared scratch on the CUDA backend has one owner. Per-request
         * attention is deliberately issued on that owner stream, not threads. */
        for (uint32_t i = 0; ok && i < n; i++) ok = v41_attn_cache(e, &req[i]->state, il);
        if (ok) ok = v41_attn_out(e, s, il) && ds4_gpu_v41_hc_post_tensor(s->hc2, s->attn_out, s->hc, s->post, s->comb, emb, hc, n);
        { ds4_gpu_tensor *t = s->hc; s->hc = s->hc2; s->hc2 = t; }
        if (ok) ok = v41_hc_half(e, s, l, false) && v41_moe(m, l, s, il) &&
                     ds4_gpu_v41_hc_post_tensor(s->hc2, s->y, s->hc, s->post, s->comb, emb, hc, n);
        { ds4_gpu_tensor *t = s->hc; s->hc = s->hc2; s->hc2 = t; }
    }
    return ok && ds4_gpu_v41_hc_pre_tensor(s->x, s->hc, s->pre_mix, emb, hc, n) &&
           ds4_gpu_v41_rms_norm_tensor(s->xn, s->x, m->map, m->size, e->weights.output_norm->abs_offset, emb, n, DS4_RMS_EPS) &&
           v41_tproj(m, s->logits, e->weights.output, emb, DS4_N_VOCAB, s->xn, n, 0);
}
int ds4_v41_batch_step(struct ds4_v41_batch *b, struct ds4_v41_request **req,
        const int32_t *tokens, int count, float *logits, size_t floats) {
    if (!b || b->poisoned || !req || !tokens || !logits || count < 1 ||
        (uint32_t)count > b->width || floats != (size_t)count*DS4_N_VOCAB) return 1;
    int32_t pos[DS41_BATCH_MAX];
    v41_batch_views views[DS41_BATCH_MAX] = {{0}};
    /* Validate the whole set before writing any state, including duplicates. */
    for (int i = 0; i < count; i++) {
        struct ds4_v41_request *r = req[i];
        if (!r || r->engine != b->engine || !r->ready || r->poisoned ||
            r->state.n_past >= r->state.ctx || r->state.hc || r->state.cap_tok != 1 ||
            r->state.no_engram != req[0]->state.no_engram || tokens[i] < 0 ||
            (uint32_t)tokens[i] >= DS4_N_VOCAB) return 1;
        for (int j = 0; j < i; j++) if (req[j] == r) return 1;
        pos[i] = (int32_t)r->state.n_past;
    }
    bool ok = ds4_gpu_tensor_write(b->rows.tok, 0, tokens, (uint64_t)count*4) &&
              ds4_gpu_tensor_write(b->rows.pos, 0, pos, (uint64_t)count*4) &&
              ds4_gpu_tensor_write(b->rows.pre_mix, 0, b->onehot, (uint64_t)count*DS4_N_HC*4);
    int attached = 0;
    for (int i = 0; ok && i < count; i++) {
        ds4_v41_state *s = &req[i]->state;
        attached = i+1; /* also release partial views if attach itself fails */
        ok = v41_batch_attach(s, &b->rows, (uint32_t)i, &views[i]);
        s->blk_ids = tokens+i;
        if (s->himg) s->himg[s->n_past] = 0;
        if (ok && s->efeed && s->efeed->prepare)
            ok = s->efeed->prepare(s->pos0, tokens+i, 1, s->efeed->ud) == 0;
    }
    bool begun = false;
    if (ok) { begun = ds4_gpu_begin_commands() != 0; ok = begun; }
    if (ok) ok = v41_batch_body(b, req, (uint32_t)count);
    /* Balance the command scope even after an intermediate kernel failure. */
    if (begun) {
        bool ended = ds4_gpu_end_commands() != 0;
        bool synced = ds4_gpu_synchronize() != 0;
        ok = ok && ended && synced;
    }
    if (ok) ok = ds4_gpu_tensor_read(b->rows.logits, 0, logits, floats*sizeof(float)) != 0;
    for (int i = 0; i < attached; i++) v41_batch_detach(&req[i]->state, &views[i]);
    for (int i = 0; i < count; i++) {
        if (ok) req[i]->state.n_past++;
        else req[i]->poisoned = true;
    }
    if (!ok) b->poisoned = true;
    return ok ? 0 : 1;
}
#else
struct ds4_v41_request *ds4_v41_request_open(void *e, const int32_t *p, int n, int g, int no,
        struct ds4_v41_images *i, const ds41_engram_feed *f) {
    (void)e; (void)p; (void)n; (void)g; (void)no; (void)i; (void)f; return NULL;
}
void ds4_v41_request_close(struct ds4_v41_request *r) { (void)r; }
int ds4_v41_request_prefill(struct ds4_v41_request *r, float *l, size_t n) { (void)r; (void)l; (void)n; return -1; }
uint32_t ds4_v41_request_position(const struct ds4_v41_request *r) { (void)r; return 0; }
uint64_t ds4_v41_request_peak(void *e, int n, int g, int no) { (void)e; (void)n; (void)g; (void)no; return UINT64_MAX; }
struct ds4_v41_batch *ds4_v41_batch_open(void *e, int n) { (void)e; (void)n; return NULL; }
void ds4_v41_batch_close(struct ds4_v41_batch *b) { (void)b; }
int ds4_v41_batch_step(struct ds4_v41_batch *b, struct ds4_v41_request **r, const int32_t *t, int n, float *l, size_t f) {
    (void)b; (void)r; (void)t; (void)n; (void)l; (void)f; return 1;
}
#endif
