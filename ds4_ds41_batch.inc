/* SPDX-License-Identifier: MIT
 * Request-local V4.1 state and packed multi-request DSpark verification.
 * Arithmetic/rollback: YoungAi 7fb26cc core_v41_{multi,req,state}.c and this
 * tree's existing ds4_ds41_{forward,draft}.inc. Rust owns target sampling,
 * DSML/EOS decisions and commits only the accepted input prefix of each row.
 */
#include "ds41_batch.h"
#include "ds41_spec_plan.h"

#ifndef DS4_NO_GPU
_Static_assert(DS41_VERIFY_MAX == DS4_MTP_MAX_BLOCK + 1u, "verification ABI drift");
struct ds4_v41_request {
    ds4_engine *engine;
    ds4_v41_state state;
    ds4_v41_draft draft;
    int32_t *prompt;
    uint32_t np, cursor, cap, verify_k, round_rows, round_start;
    bool ready, poisoned, mtp;
};
struct ds4_v41_batch {
    ds4_engine *engine;
    ds4_v41_state rows;
    uint32_t width, capacity;
    float *onehot;
    bool poisoned;
};
typedef struct {
    ds4_gpu_tensor *tok, *pos, *xn, *erows, *qrn, *q, *kvn, *o;
} v41_batch_views;

/* Short prompts still need the complete draft verification scratch. */
static uint32_t v41_request_cap(uint32_t np, uint32_t ctx) {
    uint32_t cap = np < DS4_V41_CHUNK ? np : DS4_V41_CHUNK;
    if (cap < DS41_VERIFY_MAX) cap = DS41_VERIFY_MAX;
    return cap < ctx ? cap : ctx;
}
uint64_t ds4_v41_request_peak(void *engine, int np, int n_predict, int no_engram) {
    ds4_engine *e = engine;
    if (!e || DS4_MODEL_VARIANT != DS4_VARIANT_V41 || np < 1 || n_predict < 1 ||
        (uint64_t)np + (uint64_t)n_predict > g_ds4_v41.ctx) return UINT64_MAX;
    uint32_t ctx = (uint32_t)np + (uint32_t)n_predict;
    uint32_t cap = v41_request_cap((uint32_t)np, ctx);
    uint32_t lr = cap < DS41_VERIFY_MAX ? cap : DS41_VERIFY_MAX;
    uint64_t b = v41_bytes_rows(cap, lr, 1) + v41_bytes_kv(ctx, cap);
    b += (uint64_t)ctx * (sizeof(int32_t) + sizeof(uint8_t));
    b += (uint64_t)DS41_VERIFY_MAX * DS4_N_VOCAB * sizeof(float);
    b += (uint64_t)cap * DS4_N_LAYER * sizeof(int32_t);
    if (!no_engram && g_ds4_v41.n_engram) {
        const ds4_v41_cfg *v = &g_ds4_v41;
        uint64_t cols = (uint64_t)(v->engram_max_ngram - 1) * v->engram_heads;
        uint64_t hd = v->engram_head_dim;
        b += (uint64_t)cap * (2 * v->n_engram * cols * (hd + hd/32) +
             v->n_engram * cols * 8 + cols * hd * 4 + (DS4_N_HC+1) * DS4_N_EMBD * 4);
    }
    uint64_t groups = 1;
    while (groups < ctx) groups <<= 1;
    uint64_t bs = g_ds4_v41.candidate_block_size > 0 ? (uint64_t)g_ds4_v41.candidate_block_size : 1;
    b += (uint64_t)(cap/32 + DS41_VERIFY_MAX + 1u) * groups * 4;
    b += (uint64_t)cap * ((groups+bs-1)/bs);
    return b;
}

/* Conservative per-request extra allocation bound. Immutable model weights
 * are shared. The host applies its existing 2x admission band to this quote. */
uint64_t ds4_v41_request_mtp_bytes(void *engine) {
    ds4_engine *e = engine;
    const ds4_v41_cfg *v = &g_ds4_v41;
    if (!e || DS4_MODEL_VARIANT != DS4_VARIANT_V41 || !v->mtp_block ||
        v->mtp_block >= DS41_VERIFY_MAX || !v->mtp_towers || !v->n_mtp_target ||
        !e->weights.v41_towers.main_proj || !e->weights.v41_towers.markov_embd ||
        !e->weights.v41_towers.confidence) return UINT64_MAX;
    const uint64_t cap = v->mtp_block+1u, sw = DS4_N_SWA, ebd = DS4_N_EMBD;
    const uint64_t hd = DS4_N_HEAD_DIM, rank = v->mtp_markov_rank, voc = DS4_N_VOCAB;
    const uint64_t pc = cap > sw ? cap : sw;
    uint64_t b = v41_bytes_rows((uint32_t)cap, (uint32_t)cap, 1);
    b += (uint64_t)v->mtp_towers*(sw+pc)*hd*4 + (2*pc+sw*v->n_mtp_target)*ebd*4;
    b += (2*pc+sw)*hd*4 + (pc+sw)*4;
    b += (DS4_V41_MKCACHE_SLOTS+1ull)*voc*4 + DS4_V41_MKCACHE_SLOTS*4ull;
    b += (2*cap*rank + rank + 2*cap*ebd + 4*cap + cap*DS4_N_HC)*4;
    b += (uint64_t)v->mtp_towers*v->mtp_experts*3*8 + 256;
    b += v41_bytes_rows((uint32_t)cap, (uint32_t)cap, 1);
    b += (uint64_t)DS4_N_LAYER*(DS41_VERIFY_MAX+2ull)*hd*4*3;
    return b;
}

/* Keep the MTP main-hidden ring, the drafter's sole history source. */
static bool v41_batch_shrink(ds4_v41_state *st, uint32_t cap, bool mtp) {
    const uint64_t hd = DS4_N_HEAD_DIM, swa = DS4_N_SWA, rowb = hd * 4;
    v41_graph_free(st);
    V41_FREE(&st->erows, &st->ekv, &st->iscore, &st->cand);
    if (!mtp) { V41_FREE(&st->mainh); st->mainh_cap = st->mainh_n = 0; }
    for (uint32_t k = 0; k < DS4_V41_MAX_ENGRAM; k++) V41_FREE(&st->eraw[k]);
    st->iscap = st->isrows = st->icrows = 0;
    st->iscap_gen++;
    v41_free_dense(st);
    v41_free_attn(st);
    bool ok = true;
    v41_alloc_attn_cache(st, cap, &ok);
    if (!st->no_engram && g_ds4_v41.n_engram) {
        const ds4_v41_cfg *v = &g_ds4_v41;
        uint64_t cols = (uint64_t)(v->engram_max_ngram-1) * v->engram_heads;
        uint64_t stride = v->engram_head_dim + v->engram_head_dim/32;
        for (uint32_t k = 0; k < v->n_engram; k++) st->eraw[k] = v41_alloc(cap*cols*stride, &ok);
    }
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor *w = v41_alloc((swa+cap)*rowb, &ok);
        if (ok) ok = ds4_gpu_tensor_fill_f32(w, 0.f, (swa+cap)*hd) &&
                     ds4_gpu_tensor_copy(w, 0, st->win[il], 0, swa*rowb);
        ds4_gpu_tensor_free(st->win[il]); st->win[il] = w;
        if (st->cpre_kv[il]) {
            uint32_t ratio = ds4_layer_compress_ratio(il);
            ds4_gpu_tensor *ck = v41_alloc((ratio+cap)*rowb, &ok);
            ds4_gpu_tensor *cs = v41_alloc((ratio+cap)*rowb, &ok);
            if (ok) ok = ds4_gpu_tensor_copy(ck, 0, st->cpre_kv[il], 0, ratio*rowb) &&
                         ds4_gpu_tensor_copy(cs, 0, st->cpre_sc[il], 0, ratio*rowb);
            ds4_gpu_tensor_free(st->cpre_kv[il]); st->cpre_kv[il] = ck;
            ds4_gpu_tensor_free(st->cpre_sc[il]); st->cpre_sc[il] = cs;
        }
    }
    st->cap_tok = cap; st->logits_rows = 0; st->n_direct1 = 0;
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
    r->cap = v41_request_cap(r->np, r->np+(uint32_t)n_predict);
    r->prompt = malloc((size_t)np * sizeof *prompt);
    if (!r->prompt) { free(r); return NULL; }
    memcpy(r->prompt, prompt, (size_t)np*sizeof *prompt);
    if (!v41_state_alloc(&r->state, r->cap, r->np+(uint32_t)n_predict, DS41_VERIFY_MAX)) {
        free(r->prompt); free(r); return NULL;
    }
    r->state.head_last_only = 1;
    r->state.no_engram = !!no_engram;
    r->state.efeed = feed;
    r->state.vimg = images;
    return r;
}
int ds4_v41_request_mtp(struct ds4_v41_request *r, int on, int verify_k) {
    if (!r || r->poisoned || r->ready || r->cursor || r->mtp ||
        verify_k < 0 || (uint32_t)verify_k >= DS41_VERIFY_MAX) return 1;
    if (!on) return 0;
    if (ds4_v41_request_mtp_bytes(r->engine) == UINT64_MAX ||
        !r->state.mainh || !v41_draft_alloc(r->engine, &r->draft)) {
        r->poisoned = true; return 1;
    }
    r->mtp = true;
    r->verify_k = (uint32_t)verify_k;
    /* Deterministic drafts are verified by actual request-local target draws
     * in Rust, including temperature, penalties and DSML constraints. */
    r->draft.dev_sample = 0;
    return 0;
}
void ds4_v41_request_close(struct ds4_v41_request *r) {
    if (!r) return;
    (void)ds4_gpu_synchronize();
    v41_draft_free(&r->draft);
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
    uint32_t cap = r->mtp ? r->draft.block+1u : 1u;
    if (cap > r->state.ctx) cap = r->state.ctx;
    if (!ds4_gpu_tensor_read(r->state.logits, (uint64_t)r->state.last_logit_row*count*sizeof(float),
                            logits, count*sizeof(float)) || !v41_batch_shrink(&r->state, cap, r->mtp)) goto fail;
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
static bool v41_batch_attach(ds4_v41_state *s, ds4_v41_state *b, uint32_t row,
                              uint32_t n, v41_batch_views *v) {
    uint64_t e = DS4_N_EMBD, hd = DS4_N_HEAD_DIM, nh = DS4_N_HEAD, q = DS4_N_LORA_Q;
    uint64_t erow = g_ds4_v41.n_engram ?
        (uint64_t)(g_ds4_v41.engram_max_ngram-1)*g_ds4_v41.engram_heads*g_ds4_v41.engram_head_dim*4 : 0;
#define ROW_VIEW(field, bytes) v->field = ds4_gpu_tensor_view(b->field, (uint64_t)row*(bytes), (uint64_t)n*(bytes))
    ROW_VIEW(tok, 4); ROW_VIEW(pos, 4); ROW_VIEW(xn, e*4); ROW_VIEW(qrn, q*4);
    ROW_VIEW(q, nh*hd*4); ROW_VIEW(kvn, hd*4); ROW_VIEW(o, nh*hd*4);
    if (!s->no_engram && g_ds4_v41.n_engram) { ROW_VIEW(erows, erow); }
#undef ROW_VIEW
    if (!v->tok || !v->pos || !v->xn || !v->qrn || !v->q || !v->kvn || !v->o ||
        (!s->no_engram && g_ds4_v41.n_engram && !v->erows)) return false;
    s->tok = v->tok; s->pos = v->pos; s->xn = v->xn; s->erows = v->erows;
    s->qrn = v->qrn; s->q = v->q; s->kvn = v->kvn; s->o = v->o;
    s->n = n; s->pos0 = s->n_past;
    s->idx_owner = -1; s->cand_owner = -1; s->idx_topk = s->idx_ratio = 0;
    s->graph = 0; s->ced_skip = s->ced_done = 0; s->vrows_on = 0; s->mainh_wrote = 0;
    return true;
}
struct ds4_v41_batch *ds4_v41_batch_open(void *engine, int width) {
    ds4_engine *e = engine;
    if (!e || DS4_MODEL_VARIANT != DS4_VARIANT_V41 || !e->metal_ready || width < 1 || width > DS41_BATCH_MAX) return NULL;
    struct ds4_v41_batch *b = calloc(1, sizeof *b);
    if (!b) return NULL;
    b->engine = e; b->width = (uint32_t)width;
    b->capacity = (uint32_t)width*DS41_VERIFY_MAX;
    b->rows.cap_tok = b->capacity;
    bool ok = true;
    v41_alloc_dense(&b->rows, b->capacity, b->capacity, &ok);
    v41_alloc_attn_rows(&b->rows, b->capacity, &ok);
    if (g_ds4_v41.n_engram) {
        const ds4_v41_cfg *v = &g_ds4_v41;
        uint64_t in_dim = (uint64_t)(v->engram_max_ngram-1)*v->engram_heads*v->engram_head_dim;
        b->rows.erows = v41_alloc((uint64_t)b->capacity*in_dim*4, &ok);
        b->rows.ekv = v41_alloc((uint64_t)b->capacity*(DS4_N_HC+1)*DS4_N_EMBD*4, &ok);
    }
    b->onehot = calloc((size_t)b->capacity*DS4_N_HC, sizeof(float));
    if (!b->onehot) { ds4_v41_batch_close(b); return NULL; }
    for (uint32_t i = 0; i < b->capacity*DS4_N_HC; i++) b->onehot[i] = i%DS4_N_HC == 0 ? 1.f : 0.f;
    if (!ok) { ds4_v41_batch_close(b); return NULL; }
    return b;
}
void ds4_v41_batch_close(struct ds4_v41_batch *b) {
    if (!b) return;
    (void)ds4_gpu_synchronize();
    v41_state_free(&b->rows);
    free(b->onehot); free(b);
}
static int v41_batch_poison(struct ds4_v41_batch *b, struct ds4_v41_request **req, uint32_t n) {
    if (b) b->poisoned = true;
    for (uint32_t i = 0; i < n; i++) if (req[i]) req[i]->poisoned = true;
    return 1;
}
static bool v41_batch_body(struct ds4_v41_batch *b, struct ds4_v41_request **req,
                           const uint32_t *off, const uint32_t *nr, uint32_t n, uint32_t total) {
    ds4_engine *e = b->engine;
    ds4_v41_state *s = &b->rows;
    const ds4_model *m = &e->model;
    uint32_t emb = DS4_N_EMBD, hc = DS4_N_HC;
    s->n = total; s->pos0 = 0; s->graph = 0; s->ced_skip = s->ced_done = 0;
    s->no_engram = req[0]->state.no_engram;
    s->idx_owner = s->cand_owner = -1;
    bool ok = v41_embed(m, s->x, s->tok, e->weights.token_embd, DS4_N_VOCAB, total, emb) &&
              ds4_gpu_v41_expand_hc_tensor(s->hc, s->x, emb, hc, total);
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        const ds4_layer_weights *l = &e->weights.layer[il];
        if (!s->no_engram && g_ds4_v41.engram_index_of[il] >= 0) {
            for (uint32_t i = 0; ok && i < n; i++) ok = v41_engram_rows(e, &req[i]->state, il);
            if (ok) ok = v41_engram_apply(e, s, il);
        }
        /* The target is HC mean AFTER Engram and BEFORE attention. */
        if (ok && g_ds4_v41.mtp_target_slot[il] >= 0) {
            for (uint32_t i = 0; ok && i < n; i++) {
                ds4_v41_state *r = &req[i]->state;
                if (!r->mainh) continue;
                uint32_t k = nr[i] < r->mainh_cap ? nr[i] : r->mainh_cap;
                ok = ds4_gpu_v41_hc_mean_tensor(r->mainh, s->hc, emb, hc, k,
                    off[i]+nr[i]-k, (uint32_t)g_ds4_v41.mtp_target_slot[il],
                    g_ds4_v41.n_mtp_target, r->pos0+nr[i]-k, r->mainh_cap, NULL) != 0;
                r->mainh_wrote = k;
            }
        }
        if (ok) ok = v41_hc_half(e, s, l, true) && v41_attn_in(e, s, il);
        /* One owner stream for shared backend scratch; caches stay separate. */
        for (uint32_t i = 0; ok && i < n; i++) ok = v41_attn_cache(e, &req[i]->state, il);
        if (ok) ok = v41_attn_out(e, s, il) && ds4_gpu_v41_hc_post_tensor(s->hc2, s->attn_out, s->hc, s->post, s->comb, emb, hc, total);
        { ds4_gpu_tensor *t = s->hc; s->hc = s->hc2; s->hc2 = t; }
        if (ok) ok = v41_hc_half(e, s, l, false) && v41_moe(m, l, s, il) &&
                     ds4_gpu_v41_hc_post_tensor(s->hc2, s->y, s->hc, s->post, s->comb, emb, hc, total);
        { ds4_gpu_tensor *t = s->hc; s->hc = s->hc2; s->hc2 = t; }
    }
    if (ok) ok = ds4_gpu_v41_hc_pre_tensor(s->x, s->hc, s->pre_mix, emb, hc, total) &&
        ds4_gpu_v41_rms_norm_tensor(s->xn, s->x, m->map, m->size, e->weights.output_norm->abs_offset, emb, total, DS4_RMS_EPS);
    /* Keep the generation head on its unrounded GEMV path for 24+ rows. */
    for (uint32_t at = 0; ok && at < total; at += DS4_V41_GEMV_MAX_TOK) {
        uint32_t k = total-at < DS4_V41_GEMV_MAX_TOK ? total-at : DS4_V41_GEMV_MAX_TOK;
        ds4_gpu_tensor *x = ds4_gpu_tensor_view(s->xn, (uint64_t)at*emb*4, (uint64_t)k*emb*4);
        ds4_gpu_tensor *y = ds4_gpu_tensor_view(s->logits, (uint64_t)at*DS4_N_VOCAB*4, (uint64_t)k*DS4_N_VOCAB*4);
        ok = x && y && v41_tproj(m, y, e->weights.output, emb, DS4_N_VOCAB, x, k, 0);
        if (x) ds4_gpu_tensor_free(x);
        if (y) ds4_gpu_tensor_free(y);
    }
    return ok;
}

/* Every request must commit or be dropped before another verification. */
static int v41_batch_forward(struct ds4_v41_batch *b, struct ds4_v41_request **req,
        const int32_t *tokens, const uint32_t *nr, uint32_t n, uint32_t *off, uint32_t *total) {
    if (!b || b->poisoned || !req || !tokens || !nr ||
        !ds41_verify_layout(nr, n, b->width, b->capacity, off, total)) return 1;
    int32_t pos[DS41_BATCH_MAX*DS41_VERIFY_MAX];
    v41_batch_views views[DS41_BATCH_MAX] = {{0}};
    for (uint32_t i = 0; i < n; i++) {
        struct ds4_v41_request *r = req[i];
        if (!r || r->engine != b->engine || !r->ready || r->poisoned || r->round_rows ||
            r->state.n_past > r->state.ctx || nr[i] > r->state.ctx-r->state.n_past ||
            r->state.hc || nr[i] > r->state.cap_tok ||
            r->state.no_engram != req[0]->state.no_engram) return 1;
        for (uint32_t j = 0; j < i; j++) if (req[j] == r) return 1;
        for (uint32_t j = 0; j < nr[i]; j++) {
            if (tokens[off[i]+j] < 0 || (uint32_t)tokens[off[i]+j] >= DS4_N_VOCAB) return 1;
            pos[off[i]+j] = (int32_t)(r->state.n_past+j);
        }
    }
    bool ok = ds4_gpu_tensor_write(b->rows.tok, 0, tokens, (uint64_t)*total*4) &&
              ds4_gpu_tensor_write(b->rows.pos, 0, pos, (uint64_t)*total*4) &&
              ds4_gpu_tensor_write(b->rows.pre_mix, 0, b->onehot, (uint64_t)*total*DS4_N_HC*4);
    uint32_t attached = 0;
    for (uint32_t i = 0; ok && i < n; i++) {
        ds4_v41_state *s = &req[i]->state;
        if (nr[i] > 1) ok = v41_spec_snapshot(s, nr[i]);
        if (!ok) break;
        attached = i+1;
        ok = v41_batch_attach(s, &b->rows, off[i], nr[i], &views[i]);
        s->blk_ids = tokens+off[i];
        if (s->himg) memset(s->himg+s->n_past, 0, nr[i]);
        if (ok && s->efeed && s->efeed->prepare)
            ok = s->efeed->prepare(s->pos0, tokens+off[i], (int)nr[i], s->efeed->ud) == 0;
    }
    bool begun = false;
    if (ok) { begun = ds4_gpu_begin_commands() != 0; ok = begun; }
    if (ok) ok = v41_batch_body(b, req, off, nr, n, *total);
    if (begun) {
        bool ended = ds4_gpu_end_commands() != 0;
        bool synced = ds4_gpu_synchronize() != 0;
        ok = ok && ended && synced;
    }
    for (uint32_t i = 0; i < attached; i++) v41_batch_detach(&req[i]->state, &views[i]);
    if (!ok) return v41_batch_poison(b, req, n);
    for (uint32_t i = 0; i < n; i++) {
        ds4_v41_state *s = &req[i]->state;
        if (s->mainh && s->mainh_wrote) {
            int64_t first = (int64_t)s->pos0 + nr[i] - s->mainh_wrote;
            bool contiguous = s->mainh_n && s->mainh_end == first-1;
            uint32_t count = contiguous ? s->mainh_n+s->mainh_wrote : s->mainh_wrote;
            s->mainh_n = count < s->mainh_cap ? count : s->mainh_cap;
            s->mainh_end = (int64_t)s->pos0+nr[i]-1;
        }
        req[i]->round_start = s->n_past;
        req[i]->round_rows = nr[i];
        s->n_past += nr[i];
    }
    return 0;
}
int ds4_v41_request_commit(struct ds4_v41_request *r, uint32_t keep) {
    if (!r || r->poisoned || !r->round_rows || !keep || keep > r->round_rows ||
        r->state.n_past != r->round_start+r->round_rows) return 1;
    bool ok = r->round_rows == 1 || v41_spec_rollback(&r->state, keep);
    if (ok) ok = ds4_gpu_synchronize() != 0;
    if (!ok) { r->poisoned = true; return 1; }
    r->round_rows = 0;
    return 0;
}
int ds4_v41_batch_step(struct ds4_v41_batch *b, struct ds4_v41_request **req,
        const int32_t *tokens, int count, float *logits, size_t floats) {
    if (!b || !req || !tokens || !logits || count < 1 || count > DS41_BATCH_MAX ||
        floats != (size_t)count*DS4_N_VOCAB) return 1;
    uint32_t nr[DS41_BATCH_MAX], off[DS41_BATCH_MAX], total;
    for (int i=0; i<count; i++) nr[i] = 1;
    if (v41_batch_forward(b, req, tokens, nr, (uint32_t)count, off, &total)) return 1;
    bool ok = ds4_gpu_tensor_read(b->rows.logits, 0, logits, floats*sizeof(float)) != 0;
    for (int i=0; ok && i<count; i++) ok = ds4_v41_request_commit(req[i], 1) == 0;
    return ok ? 0 : v41_batch_poison(b, req, (uint32_t)count);
}
int ds4_v41_batch_verify(struct ds4_v41_batch *b, struct ds4_v41_request **req,
        const int32_t *tokens, const uint32_t *remaining, int count,
        int32_t *candidates, uint32_t *nr, float *logits, size_t floats) {
    if (!b || b->poisoned || !req || !tokens || !remaining || !candidates || !nr || !logits ||
        count < 1 || (uint32_t)count > b->width ||
        floats != (size_t)count*DS41_VERIFY_MAX*DS4_N_VOCAB) return 1;
    int32_t packed[DS41_BATCH_MAX*DS41_VERIFY_MAX];
    uint32_t off[DS41_BATCH_MAX], total=0;
    /* Validate the whole cohort before mutating any draft window. */
    for (int i=0; i<count; i++) {
        struct ds4_v41_request *r=req[i];
        if (!r || r->engine != b->engine || !r->ready || r->poisoned || r->round_rows ||
            !remaining[i] || r->state.n_past >= r->state.ctx ||
            tokens[i] < 0 || (uint32_t)tokens[i] >= DS4_N_VOCAB) return 1;
        for (int j=0; j<i; j++) if (req[j] == r) return 1;
    }
    for (int i=0; i<count; i++) {
        struct ds4_v41_request *r=req[i];
        uint32_t n=ds41_verify_width(r->mtp ? r->draft.block : 0, remaining[i],
                                    r->state.ctx-r->state.n_past, r->verify_k);
        int32_t *ids=candidates+(size_t)i*DS41_VERIFY_MAX;
        ids[0]=tokens[i];
        if (n > 1) {
            /* No request-count cutoff: both C4 and C8 execute real drafts. */
            if (!r->mtp || !r->draft.ready || !r->state.n_past ||
                !v41_draft_step(r->engine, &r->state, &r->draft, tokens[i], r->state.n_past-1))
                return v41_batch_poison(b, req, (uint32_t)count);
            for (uint32_t j=1; j<n; j++) ids[j]=r->draft.host_ids[j];
        }
        nr[i]=n;
        memcpy(packed+total, ids, (size_t)n*sizeof *ids);
        total += n;
    }
    if (v41_batch_forward(b, req, packed, nr, (uint32_t)count, off, &total))
        return v41_batch_poison(b, req, (uint32_t)count);
    for (int i=0; i<count; i++) {
        if (!ds4_gpu_tensor_read(b->rows.logits, (uint64_t)off[i]*DS4_N_VOCAB*4,
                logits+(size_t)i*DS41_VERIFY_MAX*DS4_N_VOCAB, (uint64_t)nr[i]*DS4_N_VOCAB*4))
            return v41_batch_poison(b, req, (uint32_t)count);
    }
    return 0;
}
#else
struct ds4_v41_request *ds4_v41_request_open(void *e, const int32_t *p, int n, int g, int no,
        struct ds4_v41_images *i, const ds41_engram_feed *f) {
    (void)e; (void)p; (void)n; (void)g; (void)no; (void)i; (void)f; return NULL;
}
void ds4_v41_request_close(struct ds4_v41_request *r) { (void)r; }
int ds4_v41_request_mtp(struct ds4_v41_request *r,int on,int k) { (void)r;(void)on;(void)k;return 1; }
uint64_t ds4_v41_request_mtp_bytes(void *e) { (void)e;return UINT64_MAX; }
int ds4_v41_request_commit(struct ds4_v41_request *r,uint32_t n) { (void)r;(void)n;return 1; }
int ds4_v41_request_prefill(struct ds4_v41_request *r, float *l, size_t n) { (void)r; (void)l; (void)n; return -1; }
uint32_t ds4_v41_request_position(const struct ds4_v41_request *r) { (void)r; return 0; }
uint64_t ds4_v41_request_peak(void *e, int n, int g, int no) { (void)e; (void)n; (void)g; (void)no; return UINT64_MAX; }
struct ds4_v41_batch *ds4_v41_batch_open(void *e, int n) { (void)e; (void)n; return NULL; }
void ds4_v41_batch_close(struct ds4_v41_batch *b) { (void)b; }
int ds4_v41_batch_step(struct ds4_v41_batch *b, struct ds4_v41_request **r, const int32_t *t, int n, float *l, size_t f) {
    (void)b; (void)r; (void)t; (void)n; (void)l; (void)f; return 1;
}
int ds4_v41_batch_verify(struct ds4_v41_batch *b, struct ds4_v41_request **r,
    const int32_t *t,const uint32_t *rem,int n,int32_t *ids,uint32_t *nr,float *l,size_t f) {
    (void)b;(void)r;(void)t;(void)rem;(void)n;(void)ids;(void)nr;(void)l;(void)f;return 1;
}
#endif
