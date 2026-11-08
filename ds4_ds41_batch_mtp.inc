/* SPDX-License-Identifier: MIT
 * Request-local V4.1 packed decode and MTP target verification.
 * Derived from YoungAi's core_v41_{multi,req,state}.c at
 * 7fb26cc332d740416792204a29c7d13d2986caf5. Rust owns the scheduler/sampler.
 * Requests share immutable weights, never draft windows, KV or Engram feeds.
 */
#include "ds41_batch.h"

#ifndef DS4_NO_GPU
struct ds4_v41_request {
    ds4_engine *engine;
    ds4_v41_state state;
    ds4_v41_draft draft;
    v41_sched schedule;
    ds41_round round;
    int32_t *prompt;
    uint32_t np, cursor, cap, verify_k;
    int mtp_mode;
    bool ready, poisoned;
};
struct ds4_v41_batch {
    ds4_engine *engine;
    ds4_v41_state rows;
    ds4_v41_state *tiles;
    uint32_t width, rowcap, ntiles;
    float *onehot;
    bool poisoned;
};
typedef struct {
    ds4_gpu_tensor *tok, *pos, *xn, *erows, *qrn, *q, *kvn, *o;
} v41_batch_views;

static uint32_t v41_batch_cap(uint32_t np, uint32_t ctx) {
    uint32_t cap = np < DS4_V41_CHUNK ? np : DS4_V41_CHUNK;
    uint32_t verify = g_ds4_v41.mtp_block + 1u;
    if (verify > DS41_VERIFY_MAX) return 0;
    if (cap < verify) cap = verify;
    return cap < ctx ? cap : ctx;
}

/* Deliberately conservative: price the main-state rows and KV, the full
 * request-local drafter/Markov cache, speculative snapshots and host logits.
 * The host also applies a 2x transient band and its explicit memory floor.
 * The shared batch workspace is already allocated before admission. */
uint64_t ds4_v41_request_peak(void *engine, int np, int predict, int no_engram) {
    ds4_engine *e = engine;
    if (!e || DS4_MODEL_VARIANT != DS4_VARIANT_V41 || np < 1 || predict < 1 ||
        (uint64_t)np + (uint64_t)predict > g_ds4_v41.ctx) return UINT64_MAX;
    const ds4_v41_cfg *v = &g_ds4_v41;
    const uint32_t ctx = (uint32_t)np + (uint32_t)predict;
    const uint32_t cap = v41_batch_cap((uint32_t)np, ctx);
    if (!cap) return UINT64_MAX;
    const uint32_t lr = cap < DS41_VERIFY_MAX ? cap : DS41_VERIFY_MAX;
    uint64_t b = v41_bytes_rows(cap, lr, 1) + v41_bytes_kv(ctx, cap);
    b += (uint64_t)ctx * (2u*sizeof(int32_t) + 1u);
    b += (uint64_t)(DS41_VERIFY_MAX + 1u) * DS4_N_VOCAB * sizeof(float);
    b += (uint64_t)cap * DS4_N_LAYER * sizeof(int32_t);
    if (!no_engram && v->n_engram) {
        uint64_t cols = (uint64_t)(v->engram_max_ngram - 1u) * v->engram_heads;
        uint64_t hd = v->engram_head_dim;
        b += (uint64_t)cap * (2u*v->n_engram*cols*(hd + hd/32u) +
             v->n_engram*cols*8u + cols*hd*4u + (DS4_N_HC+1u)*DS4_N_EMBD*4u);
    }
    uint64_t groups = 1;
    while (groups < ctx) groups <<= 1;
    uint64_t bs = v->candidate_block_size > 0 ? (uint64_t)v->candidate_block_size : 1u;
    b += (uint64_t)(cap/32u + DS41_VERIFY_MAX + 1u)*groups*4u +
         (uint64_t)cap*((groups+bs-1u)/bs);
    if (v->mtp_towers && v->mtp_block && v->n_mtp_target) {
        /* v41_draft_state_alloc: a SWA-sized main projection/window; tiny
         * row buffers; Markov cache; pinned slots; full target-ring linearizer.
         * Pricing all SWA rows with main-model dimensions is an upper bound
         * for the smaller tower's row allocations, not a measured reservation. */
        const uint64_t swa = DS4_N_SWA, hd = DS4_N_HEAD_DIM, emb = DS4_N_EMBD;
        b += v41_bytes_rows((uint32_t)swa, DS41_VERIFY_MAX, 1);
        b += 2u*swa*v->mtp_towers*hd*4u;
        b += 2u*swa*emb*4u + swa*v->n_mtp_target*emb*4u;
        b += (uint64_t)DS4_V41_MKCACHE_SLOTS*DS4_N_VOCAB*4u;
        b += (uint64_t)(2u*DS41_VERIFY_MAX+1u)*v->mtp_markov_rank*4u;
        b += (uint64_t)(DS41_VERIFY_MAX+1u)*DS4_N_VOCAB*4u;
        b += (uint64_t)DS4_N_LAYER*DS41_VERIFY_MAX*hd*4u*3u;
        b += 65536u; /* pinned indices, Markov metadata, handles/alignment */
    }
    return b;
}

/* Compact only row scratch. Keep the prompt's main-hidden ring for MTP.
 * The request-local attention buffers must hold the full verify width, not
 * just one row; accepted SWA/cpre rows survive the resize byte-for-byte. */
static bool v41_batch_shrink(ds4_v41_state *st, uint32_t cap) {
    if (!cap || cap > st->cap_tok || cap > DS41_VERIFY_MAX) return false;
    const uint64_t hd = DS4_N_HEAD_DIM, swa = DS4_N_SWA, rowb = hd*4u;
    v41_graph_free(st);
    V41_FREE(&st->erows, &st->ekv, &st->iscore, &st->cand);
    for (uint32_t k = 0; k < DS4_V41_MAX_ENGRAM; k++) V41_FREE(&st->eraw[k]);
    st->iscap = st->isrows = st->icrows = 0; st->iscap_gen++;
    v41_free_dense(st); v41_free_attn(st);
    bool ok = true;
    v41_alloc_attn_cache(st, cap, &ok);
    if (!st->no_engram && g_ds4_v41.n_engram) {
        const ds4_v41_cfg *v = &g_ds4_v41;
        uint64_t cols = (uint64_t)(v->engram_max_ngram-1u)*v->engram_heads;
        uint64_t stride = v->engram_head_dim + v->engram_head_dim/32u;
        for (uint32_t k = 0; k < v->n_engram; k++)
            st->eraw[k] = v41_alloc((uint64_t)cap*cols*stride, &ok);
    }
    for (uint32_t il = 0; ok && il < DS4_N_LAYER; il++) {
        ds4_gpu_tensor *w = v41_alloc((swa+cap)*rowb, &ok);
        if (ok) ok = ds4_gpu_tensor_fill_f32(w, 0.f, (swa+cap)*hd) &&
                     ds4_gpu_tensor_copy(w, 0, st->win[il], 0, swa*rowb);
        ds4_gpu_tensor_free(st->win[il]); st->win[il] = w;
        if (st->cpre_kv[il]) {
            uint32_t ratio = ds4_layer_compress_ratio(il);
            ds4_gpu_tensor *ck = v41_alloc((ratio+(uint64_t)cap)*rowb, &ok);
            ds4_gpu_tensor *cs = v41_alloc((ratio+(uint64_t)cap)*rowb, &ok);
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
        const int32_t *prompt, int np, int predict, int no_engram,
        struct ds4_v41_images *images, const ds41_engram_feed *feed) {
    ds4_engine *e = engine;
    if (!e || DS4_MODEL_VARIANT != DS4_VARIANT_V41 || !e->metal_ready || !prompt || np < 1 || predict < 1 ||
        (uint64_t)np + (uint64_t)predict > g_ds4_v41.ctx ||
        (!no_engram && g_ds4_v41.n_engram && !feed) ||
        (images && images->np != (uint32_t)np)) return NULL;
    for (int i = 0; i < np; i++) if (prompt[i] < 0 || (uint32_t)prompt[i] >= DS4_N_VOCAB) return NULL;
    struct ds4_v41_request *r = calloc(1, sizeof *r);
    if (!r) return NULL;
    r->engine = e; r->np = (uint32_t)np;
    r->cap = v41_batch_cap(r->np, r->np+(uint32_t)predict);
    if (!r->cap) { free(r); return NULL; }
    r->prompt = malloc((size_t)np*sizeof *prompt);
    if (!r->prompt) { free(r); return NULL; }
    memcpy(r->prompt, prompt, (size_t)np*sizeof *prompt);
    /* Short (<=8 token) prefills use the multi-row GEMV head, not the
     * last-only GEMM arm. They must have a logits slot for every short row. */
    if (!v41_state_alloc(&r->state, r->cap, r->np+(uint32_t)predict,
                         DS41_VERIFY_MAX)) { free(r->prompt); free(r); return NULL; }
    r->state.head_last_only = 1; r->state.no_engram = !!no_engram;
    r->state.efeed = feed; r->state.vimg = images;
    return r;
}
int ds4_v41_request_set_mtp(struct ds4_v41_request *r, int mode, uint32_t verify, int graph) {
    if (!r || r->ready || r->cursor || r->poisoned || r->draft.ready ||
        mode < 0 || mode > 2 || verify > g_ds4_v41.mtp_block) return 1;
    if (mode && (!r->state.mainh || !g_ds4_v41.mtp_block ||
        g_ds4_v41.mtp_block + 1u > DS41_VERIFY_MAX ||
        !v41_draft_alloc(r->engine, &r->draft))) {
        r->poisoned = true; return 1;
    }
    r->mtp_mode = mode; r->verify_k = verify;
    if (mode) r->draft.graph_off = !graph;
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
    uint32_t dc = r->mtp_mode ? r->draft.block + 1u : 1u;
    if (dc > r->cap) dc = r->cap;
    if (!ds4_gpu_tensor_read(r->state.logits, (uint64_t)r->state.last_logit_row*count*sizeof(float),
                            logits, count*sizeof(float)) || !v41_batch_shrink(&r->state, dc)) goto fail;
    r->ready = true; free(r->prompt); r->prompt = NULL;
    return 1;
fail:
    r->poisoned = true; return -1;
}

/* Cached non-owning row tiles keep ALL projections on the decode arithmetic
 * path. Passing sum(1+k)>8 directly would choose the rounded prefill GEMM
 * head instead of the target's GEMV head and can change sampled tokens.
 * These are row tiles of one layer across requests, NOT serialized requests. */
static bool v41_batch_tile_open(ds4_v41_state *t, ds4_v41_state *b, uint32_t row, uint32_t n) {
    const uint64_t e=DS4_N_EMBD, hc=DS4_N_HC, ff=DS4_N_FF_EXP, nh=DS4_N_HEAD, hd=DS4_N_HEAD_DIM;
    const uint64_t q=DS4_N_LORA_Q, low=(uint64_t)DS4_N_OUT_GROUP*DS4_N_LORA_O;
    const uint64_t mix=2u*hc+hc*hc, ne=DS4_N_EXPERT, k=DS4_N_EXPERT_USED;
    const uint64_t erow=g_ds4_v41.n_engram ?
        (uint64_t)(g_ds4_v41.engram_max_ngram-1u)*g_ds4_v41.engram_heads*g_ds4_v41.engram_head_dim : 0u;
    bool ok=true;
#define TV(f, words) do { uint64_t z=(words)*4u; t->f=ds4_gpu_tensor_view(b->f, (uint64_t)row*z, (uint64_t)n*z); if(!t->f)ok=false; } while(0)
    TV(tok,1); TV(pos,1); TV(hc,hc*e); TV(hc2,hc*e); TV(mix,mix);
    TV(pre,hc); TV(post,hc); TV(comb,hc*hc); TV(pre_mix,hc);
    TV(x,e); TV(xn,e); TV(attn_out,e); TV(glog,ne); TV(sel,k); TV(rw,k);
    TV(routed,e); TV(sg,ff); TV(su,ff); TV(sh,ff); TV(so,e); TV(y,e); TV(logits,DS4_N_VOCAB);
    TV(qr,q); TV(qrn,q); TV(q,nh*hd); TV(kv,hd); TV(kvn,hd); TV(o,nh*hd); TV(low,low);
    if(erow) { TV(erows,erow); TV(ekv,(hc+1u)*e); }
#undef TV
    t->cap_tok=n; t->logits_rows=n;
    return ok;
}
static void v41_batch_tile_close(ds4_v41_state *t) {
    v41_free_dense(t); v41_free_attn(t); V41_FREE(&t->erows,&t->ekv);
}
static void v41_batch_hc_swap(struct ds4_v41_batch *b) {
    ds4_gpu_tensor *p=b->rows.hc; b->rows.hc=b->rows.hc2; b->rows.hc2=p;
    for(uint32_t i=0;i<b->ntiles;i++) {
        p=b->tiles[i].hc; b->tiles[i].hc=b->tiles[i].hc2; b->tiles[i].hc2=p;
    }
}
enum v41_packed_op { V41_EMBED, V41_ENGRAM, V41_ATTN_IN, V41_ATTN_OUT, V41_FFN, V41_HEAD };
static bool v41_batch_project(struct ds4_v41_batch *b, uint32_t rows, uint32_t il, enum v41_packed_op op) {
    const ds4_model *m=&b->engine->model;
    const ds4_layer_weights *l=il<DS4_N_LAYER ? &b->engine->weights.layer[il] : NULL;
    const uint32_t e=DS4_N_EMBD,hc=DS4_N_HC;
    for(uint32_t first=0,tile=0;first<rows;first+=DS4_V41_GEMV_MAX_TOK,tile++) {
        ds4_v41_state *s=&b->tiles[tile];
        uint32_t n=rows-first; if(n>DS4_V41_GEMV_MAX_TOK)n=DS4_V41_GEMV_MAX_TOK;
        s->n=n; s->no_engram=b->rows.no_engram;
        bool ok=false;
        switch(op) {
        case V41_EMBED:
            ok=v41_embed(m,s->x,s->tok,b->engine->weights.token_embd,DS4_N_VOCAB,n,e) &&
               ds4_gpu_v41_expand_hc_tensor(s->hc,s->x,e,hc,n); break;
        case V41_ENGRAM: ok=v41_engram_apply(b->engine,s,il); break;
        case V41_ATTN_IN: ok=v41_hc_half(b->engine,s,l,true) && v41_attn_in(b->engine,s,il); break;
        case V41_ATTN_OUT:
            ok=v41_attn_out(b->engine,s,il) &&
               ds4_gpu_v41_hc_post_tensor(s->hc2,s->attn_out,s->hc,s->post,s->comb,e,hc,n); break;
        case V41_FFN:
            ok=v41_hc_half(b->engine,s,l,false) && v41_moe(m,l,s,il) &&
               ds4_gpu_v41_hc_post_tensor(s->hc2,s->y,s->hc,s->post,s->comb,e,hc,n); break;
        case V41_HEAD:
            ok=ds4_gpu_v41_hc_pre_tensor(s->x,s->hc,s->pre_mix,e,hc,n) &&
               ds4_gpu_v41_rms_norm_tensor(s->xn,s->x,m->map,m->size,b->engine->weights.output_norm->abs_offset,e,n,DS4_RMS_EPS) &&
               v41_tproj(m,s->logits,b->engine->weights.output,e,DS4_N_VOCAB,s->xn,n,0); break;
        }
        if(!ok)return false;
    }
    return true;
}

static void v41_batch_detach(ds4_v41_state *s, v41_batch_views *v) {
    V41_FREE(&v->tok,&v->pos,&v->xn,&v->erows,&v->qrn,&v->q,&v->kvn,&v->o);
    s->tok=s->pos=s->xn=s->erows=s->qrn=s->q=s->kvn=s->o=NULL; s->blk_ids=NULL;
}
static bool v41_batch_attach(ds4_v41_state *s, ds4_v41_state *b, uint32_t row, uint32_t n, v41_batch_views *v) {
    uint64_t e=DS4_N_EMBD,hd=DS4_N_HEAD_DIM,nh=DS4_N_HEAD,q=DS4_N_LORA_Q;
    uint64_t erow=g_ds4_v41.n_engram ?
        (uint64_t)(g_ds4_v41.engram_max_ngram-1u)*g_ds4_v41.engram_heads*g_ds4_v41.engram_head_dim*4u : 0u;
#define RV(f, bytes) v->f=ds4_gpu_tensor_view(b->f,(uint64_t)row*(bytes),(uint64_t)n*(bytes))
    RV(tok,4); RV(pos,4); RV(xn,e*4); RV(qrn,q*4); RV(q,nh*hd*4); RV(kvn,hd*4); RV(o,nh*hd*4);
    if(!s->no_engram && erow) { RV(erows,erow); }
#undef RV
    if(!v->tok||!v->pos||!v->xn||!v->qrn||!v->q||!v->kvn||!v->o||(!s->no_engram&&erow&&!v->erows)) return false;
    s->tok=v->tok;s->pos=v->pos;s->xn=v->xn;s->erows=v->erows;s->qrn=v->qrn;s->q=v->q;s->kvn=v->kvn;s->o=v->o;
    s->n=n;s->pos0=s->n_past;s->idx_owner=s->cand_owner=-1;s->idx_topk=s->idx_ratio=0;
    s->graph=0;s->ced_skip=s->ced_done=0;s->vrows_on=0;s->mainh_wrote=0;
    return true;
}
struct ds4_v41_batch *ds4_v41_batch_open(void *engine,int width) {
    ds4_engine *e=engine;
    if(!e||DS4_MODEL_VARIANT!=DS4_VARIANT_V41||!e->metal_ready||width<1||width>DS41_BATCH_MAX||
       g_ds4_v41.mtp_block+1u>DS41_VERIFY_MAX) return NULL;
    struct ds4_v41_batch *b=calloc(1,sizeof *b);if(!b)return NULL;
    b->engine=e;b->width=(uint32_t)width;b->rowcap=(uint32_t)width*(g_ds4_v41.mtp_block+1u);
    b->rows.cap_tok=b->rowcap;
    bool ok=true;
    v41_alloc_dense(&b->rows,b->rowcap,b->rowcap,&ok);
    v41_alloc_attn_rows(&b->rows,b->rowcap,&ok);
    if(g_ds4_v41.n_engram) {
        const ds4_v41_cfg *v=&g_ds4_v41;
        uint64_t in=(uint64_t)(v->engram_max_ngram-1u)*v->engram_heads*v->engram_head_dim;
        b->rows.erows=v41_alloc((uint64_t)b->rowcap*in*4,&ok);
        b->rows.ekv=v41_alloc((uint64_t)b->rowcap*(DS4_N_HC+1u)*DS4_N_EMBD*4,&ok);
    }
    b->onehot=calloc((size_t)b->rowcap*DS4_N_HC,sizeof(float));
    b->ntiles=(b->rowcap+DS4_V41_GEMV_MAX_TOK-1u)/DS4_V41_GEMV_MAX_TOK;
    b->tiles=calloc(b->ntiles,sizeof *b->tiles);
    if(!b->onehot||!b->tiles)ok=false;
    for(uint32_t i=0;ok&&i<b->rowcap*DS4_N_HC;i++) b->onehot[i]=i%DS4_N_HC==0?1.f:0.f;
    for(uint32_t i=0;ok&&i<b->ntiles;i++) {
        uint32_t off=i*DS4_V41_GEMV_MAX_TOK,n=b->rowcap-off;
        if(n>DS4_V41_GEMV_MAX_TOK)n=DS4_V41_GEMV_MAX_TOK;
        ok=v41_batch_tile_open(&b->tiles[i],&b->rows,off,n);
    }
    if(!ok){ds4_v41_batch_close(b);return NULL;}
    return b;
}
void ds4_v41_batch_close(struct ds4_v41_batch *b) {
    if(!b)return;
    (void)ds4_gpu_synchronize();
    if(b->tiles)for(uint32_t i=0;i<b->ntiles;i++)v41_batch_tile_close(&b->tiles[i]);
    free(b->tiles);v41_state_free(&b->rows);free(b->onehot);free(b);
}

static bool v41_batch_body(struct ds4_v41_batch *b,struct ds4_v41_request **req,
        const uint32_t *off,const uint32_t *nr,uint32_t count,uint32_t total) {
    ds4_engine *e=b->engine;
    b->rows.no_engram=req[0]->state.no_engram;
    bool ok=v41_batch_project(b,total,0,V41_EMBED);
    for(uint32_t il=0;ok&&il<DS4_N_LAYER;il++) {
        if(!b->rows.no_engram&&g_ds4_v41.engram_index_of[il]>=0) {
            for(uint32_t i=0;ok&&i<count;i++)ok=v41_engram_rows(e,&req[i]->state,il);
            if(ok)ok=v41_batch_project(b,total,il,V41_ENGRAM);
        }
        /* The drafter consumes the HC mean BEFORE target-layer attention,
         * not the layer output. The packed source offset and the request's
         * absolute destination position are intentionally independent. */
        if(g_ds4_v41.mtp_target_slot[il]>=0)for(uint32_t i=0;ok&&i<count;i++) {
            ds4_v41_state *s=&req[i]->state;
            if(s->mainh) {
                ok=ds4_gpu_v41_hc_mean_tensor(s->mainh,b->rows.hc,DS4_N_EMBD,DS4_N_HC,nr[i],off[i],
                     (uint32_t)g_ds4_v41.mtp_target_slot[il],g_ds4_v41.n_mtp_target,
                     s->pos0,s->mainh_cap,NULL)!=0;
                if(ok)s->mainh_wrote=nr[i];
            }
        }
        if(ok)ok=v41_batch_project(b,total,il,V41_ATTN_IN);
        /* Cache/attention is request-local; the CUDA workspace has one owner. */
        for(uint32_t i=0;ok&&i<count;i++)ok=v41_attn_cache(e,&req[i]->state,il);
        if(ok)ok=v41_batch_project(b,total,il,V41_ATTN_OUT);
        v41_batch_hc_swap(b);
        if(ok)ok=v41_batch_project(b,total,il,V41_FFN);
        v41_batch_hc_swap(b);
    }
    return ok&&v41_batch_project(b,total,DS4_N_LAYER,V41_HEAD);
}
static void v41_batch_advance(ds4_v41_state *s,uint32_t n) {
    if(s->mainh&&s->mainh_wrote) {
        int64_t first=(int64_t)s->pos0+(int64_t)(n-s->mainh_wrote);
        uint32_t nn=s->mainh_n>0&&s->mainh_end==first-1 ? s->mainh_n+s->mainh_wrote : s->mainh_wrote;
        s->mainh_n=nn>s->mainh_cap?s->mainh_cap:nn;
        s->mainh_end=(int64_t)s->pos0+n-1;
    }
    s->n_past+=n;
}
static int v41_batch_forward(struct ds4_v41_batch *b,struct ds4_v41_request **req,
        const int32_t *tokens,const uint32_t *nr,int count,float *logits,size_t floats,bool provisional) {
    if(!b||b->poisoned||!req||!tokens||!nr||!logits||count<1||(uint32_t)count>b->width)return 1;
    uint32_t off[DS41_BATCH_MAX],total=0;
    int32_t pos[DS41_BATCH_MAX*DS41_VERIFY_MAX];
    v41_batch_views views[DS41_BATCH_MAX]={{0}};
    for(int i=0;i<count;i++) {
        struct ds4_v41_request *r=req[i];
        if(!r||r->engine!=b->engine||!r->ready||r->poisoned||r->round.rows||
           !nr[i]||nr[i]>r->state.cap_tok||nr[i]>DS41_VERIFY_MAX||r->state.hc||
           (uint64_t)r->state.n_past+nr[i]>r->state.ctx||
           r->state.no_engram!=req[0]->state.no_engram||total+nr[i]>b->rowcap)return 1;
        for(int j=0;j<i;j++)if(req[j]==r)return 1;
        off[i]=total;
        for(uint32_t j=0;j<nr[i];j++) {
            if(tokens[total+j]<0||(uint32_t)tokens[total+j]>=DS4_N_VOCAB)return 1;
            pos[total+j]=(int32_t)(r->state.n_past+j);
        }
        total+=nr[i];
    }
    if(floats<(size_t)total*DS4_N_VOCAB)return 1;
    bool ok=ds4_gpu_tensor_write(b->rows.tok,0,tokens,(uint64_t)total*4)&&
            ds4_gpu_tensor_write(b->rows.pos,0,pos,(uint64_t)total*4)&&
            ds4_gpu_tensor_write(b->rows.pre_mix,0,b->onehot,(uint64_t)total*DS4_N_HC*4);
    int attached=0;
    for(int i=0;ok&&i<count;i++) {
        ds4_v41_state *s=&req[i]->state;
        if(provisional&&nr[i]>1u)ok=v41_spec_snapshot(s,nr[i]);
        if(!ok)break;
        attached=i+1;
        ok=v41_batch_attach(s,&b->rows,off[i],nr[i],&views[i]);
        s->blk_ids=tokens+off[i];
        if(s->himg)memset(s->himg+s->n_past,0,nr[i]);
        if(ok&&s->efeed&&s->efeed->prepare)
            ok=s->efeed->prepare(s->pos0,tokens+off[i],(int)nr[i],s->efeed->ud)==0;
    }
    bool begun=false;
    if(ok){begun=ds4_gpu_begin_commands()!=0;ok=begun;}
    if(ok)ok=v41_batch_body(b,req,off,nr,(uint32_t)count,total);
    if(begun) {
        bool ended=ds4_gpu_end_commands()!=0;
        bool synced=ds4_gpu_synchronize()!=0;
        ok=ok&&ended&&synced;
    }
    if(ok)ok=ds4_gpu_tensor_read(b->rows.logits,0,logits,(uint64_t)total*DS4_N_VOCAB*sizeof(float))!=0;
    for(int i=0;i<attached;i++)v41_batch_detach(&req[i]->state,&views[i]);
    for(int i=0;i<count;i++) {
        if(ok) {
            v41_batch_advance(&req[i]->state,nr[i]);
            if(provisional&&!ds41_round_begin(&req[i]->round,tokens+off[i],nr[i]))ok=false;
        }
    }
    if(!ok){b->poisoned=true;for(int i=0;i<count;i++)req[i]->poisoned=true;}
    return ok?0:1;
}
int ds4_v41_batch_step(struct ds4_v41_batch *b,struct ds4_v41_request **req,
        const int32_t *tokens,int count,float *logits,size_t floats) {
    if(count<1||count>DS41_BATCH_MAX||floats!=(size_t)count*DS4_N_VOCAB)return 1;
    uint32_t nr[DS41_BATCH_MAX];for(int i=0;i<count;i++)nr[i]=1;
    return v41_batch_forward(b,req,tokens,nr,count,logits,floats,false);
}
int ds4_v41_batch_verify(struct ds4_v41_batch *b,struct ds4_v41_request **req,
        const int32_t *tokens,int count,float *logits,size_t floats,
        int32_t *proposals,size_t proposal_slots,uint32_t *widths) {
    if(!b||b->poisoned||!req||!tokens||!logits||!proposals||!widths||count<1||
       (uint32_t)count>b->width||proposal_slots<(size_t)count*DS41_VERIFY_MAX||
       floats<(size_t)count*DS41_VERIFY_MAX*DS4_N_VOCAB)return 1;
    /* Validate every member BEFORE advancing any draft state. */
    for(int i=0;i<count;i++) {
        if(!req[i]||req[i]->engine!=b->engine||!req[i]->ready||req[i]->poisoned||req[i]->round.rows||
           req[i]->state.n_past>=req[i]->state.ctx||tokens[i]<0||(uint32_t)tokens[i]>=DS4_N_VOCAB)return 1;
        for(int j=0;j<i;j++)if(req[i]==req[j])return 1;
    }
    int32_t packed[DS41_BATCH_MAX*DS41_VERIFY_MAX];uint32_t total=0;
    for(int i=0;i<count;i++) {
        struct ds4_v41_request *r=req[i];uint32_t k=0;
        proposals[(size_t)i*DS41_VERIFY_MAX]=tokens[i];
        const uint32_t room=r->state.ctx-r->state.n_past;
        if(r->mtp_mode&&room>1u) {
            if(!r->draft.ready||!r->state.mainh_n||!r->state.n_past)goto fail;
            const double started=now_sec();
            if(!v41_draft_step(r->engine,&r->state,&r->draft,tokens[i],r->state.n_past-1u))goto fail;
            if(!r->draft.last_warm)v41_sched_draft_cost(&r->schedule,(now_sec()-started)*1e3);
            float gain=0;
            k=r->verify_k?r->verify_k:(r->mtp_mode==2?r->draft.block:
                v41_draft_pick_k(r->draft.host_conf,r->draft.block,&gain,&r->schedule));
            /* Keep adaptive calibration alive even after an early rejection. */
            if(!k)k=1u;
            if(k>r->draft.block)k=r->draft.block;
            if(k>=room)k=room-1u;
            if(k>=r->state.cap_tok)k=r->state.cap_tok-1u;
            for(uint32_t j=0;j<k;j++)proposals[(size_t)i*DS41_VERIFY_MAX+j+1u]=r->draft.host_ids[j+1u];
        }
        widths[i]=1u+k;
        memcpy(packed+total,proposals+(size_t)i*DS41_VERIFY_MAX,(size_t)widths[i]*sizeof(int32_t));
        total+=widths[i];
    }
    const double started=now_sec();
    if(v41_batch_forward(b,req,packed,widths,count,logits,floats,true))goto fail;
    const double elapsed=(now_sec()-started)*1e3;
    for(int i=0;i<count;i++)v41_sched_cost(&req[i]->schedule,widths[i],elapsed/count);
    return 0;
fail:
    b->poisoned=true;for(int i=0;i<count;i++)req[i]->poisoned=true;
    return 1;
}
int ds4_v41_request_commit(struct ds4_v41_request *r,uint32_t keep) {
    if(!r||r->poisoned||!r->round.rows||keep<1u||keep>r->round.rows)return 1;
    if(r->round.rows>1u) {
        if(!v41_spec_rollback(&r->state,keep)||!ds4_gpu_synchronize()) {r->poisoned=true;return 1;}
        if(r->mtp_mode==1)v41_sched_observe(&r->schedule,r->draft.host_conf,keep>1u);
    }
    ds41_round_clear(&r->round);return 0;
}
#else
struct ds4_v41_request *ds4_v41_request_open(void *e,const int32_t *p,int n,int g,int no,
        struct ds4_v41_images *i,const ds41_engram_feed *f) {
    (void)e;(void)p;(void)n;(void)g;(void)no;(void)i;(void)f;return NULL;
}
void ds4_v41_request_close(struct ds4_v41_request *r){(void)r;}
int ds4_v41_request_prefill(struct ds4_v41_request *r,float *l,size_t n){(void)r;(void)l;(void)n;return -1;}
uint32_t ds4_v41_request_position(const struct ds4_v41_request *r){(void)r;return 0;}
uint64_t ds4_v41_request_peak(void *e,int n,int g,int no){(void)e;(void)n;(void)g;(void)no;return UINT64_MAX;}
int ds4_v41_request_set_mtp(struct ds4_v41_request *r,int m,uint32_t v,int g){(void)r;(void)m;(void)v;(void)g;return 1;}
struct ds4_v41_batch *ds4_v41_batch_open(void *e,int n){(void)e;(void)n;return NULL;}
void ds4_v41_batch_close(struct ds4_v41_batch *b){(void)b;}
int ds4_v41_batch_step(struct ds4_v41_batch *b,struct ds4_v41_request **r,const int32_t *t,int n,float *l,size_t f) {
    (void)b;(void)r;(void)t;(void)n;(void)l;(void)f;return 1;
}
int ds4_v41_batch_verify(struct ds4_v41_batch *b,struct ds4_v41_request **r,const int32_t *t,int n,float *l,size_t f,
        int32_t *p,size_t z,uint32_t *w){(void)b;(void)r;(void)t;(void)n;(void)l;(void)f;(void)p;(void)z;(void)w;return 1;}
int ds4_v41_request_commit(struct ds4_v41_request *r,uint32_t k){(void)r;(void)k;return 1;}
#endif
