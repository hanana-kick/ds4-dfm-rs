/* SPDX-License-Identifier: MIT
 * Executes ds4_ds41_batch_mtp.inc itself against checked, in-memory tensor
 * doubles. Tests control flow, row isolation and provisional-state lifetime.
 * Does NOT validate CUDA kernels, model logits, image accuracy or throughput.
 */
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Keep the production implementation, replace only the engine/GPU boundary. */
#define DS41_BATCH_H
#include "../ds41_round.h"
#define DS41_BATCH_MAX 8
#define DS4_VARIANT_V41 16
#define DS4_MODEL_VARIANT DS4_VARIANT_V41
#define DS4_N_EMBD 4u
#define DS4_N_HC 2u
#define DS4_N_FF_EXP 8u
#define DS4_N_EXPERT 4u
#define DS4_N_EXPERT_USED 2u
#define DS4_N_HEAD 1u
#define DS4_N_HEAD_DIM 4u
#define DS4_N_LORA_Q 2u
#define DS4_N_LORA_O 2u
#define DS4_N_OUT_GROUP 1u
#define DS4_N_LAYER 4u
#define DS4_MAX_LAYER 4u
#define DS4_N_SWA 8u
#define DS4_N_VOCAB 128u
#define DS4_V41_CHUNK 16u
#define DS4_V41_GEMV_MAX_TOK 8u
#define DS4_V41_MAX_ENGRAM 2u
#define DS4_V41_MKCACHE_SLOTS 4u
#define DS4_RMS_EPS 1e-20f
#define DS4_MTP_MAX_BLOCK 8u

typedef struct ds4_gpu_tensor {
    unsigned char *data;
    uint64_t bytes;
    bool view;
} ds4_gpu_tensor;
static int alive_tensors, fail_alloc_after=-1, allocations, fail_kernel;
static int head_calls, head_rows, head_max, attention_rows, draft_calls, rollbacks;
static int mock_draft_mismatch, mock_draft_absent, scopes, fail_target, fail_feed;

static ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    if(fail_alloc_after>=0 && allocations++>=fail_alloc_after)return NULL;
    ds4_gpu_tensor *t=calloc(1,sizeof *t);assert(t);
    t->data=calloc(1,bytes?bytes:1);assert(t->data);t->bytes=bytes;alive_tensors++;return t;
}
static ds4_gpu_tensor *ds4_gpu_tensor_view(ds4_gpu_tensor *base,uint64_t off,uint64_t bytes) {
    if(!base||off>base->bytes||bytes>base->bytes-off)return NULL;
    if(fail_alloc_after>=0 && allocations++>=fail_alloc_after)return NULL;
    ds4_gpu_tensor *v=calloc(1,sizeof *v);assert(v);
    v->data=base->data+off;v->bytes=bytes;v->view=true;alive_tensors++;return v;
}
static void ds4_gpu_tensor_free(ds4_gpu_tensor *t) {
    if(!t)return;
    if(!t->view)free(t->data);free(t);assert(alive_tensors>0);alive_tensors--;
}
static int ds4_gpu_tensor_copy(ds4_gpu_tensor *to,uint64_t a,const ds4_gpu_tensor *from,uint64_t b,uint64_t n) {
    assert(to&&from&&a<=to->bytes&&n<=to->bytes-a&&b<=from->bytes&&n<=from->bytes-b);
    memmove(to->data+a,from->data+b,(size_t)n);return 1;
}
static int ds4_gpu_tensor_write(ds4_gpu_tensor *to,uint64_t a,const void *data,uint64_t n) {
    assert(to&&data&&a<=to->bytes&&n<=to->bytes-a);memcpy(to->data+a,data,(size_t)n);return 1;
}
static int ds4_gpu_tensor_read(const ds4_gpu_tensor *from,uint64_t a,void *data,uint64_t n) {
    assert(from&&data&&a<=from->bytes&&n<=from->bytes-a);memcpy(data,from->data+a,(size_t)n);return 1;
}
static int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *t,float value,uint64_t n) {
    assert(t&&n*4<=t->bytes);for(uint64_t i=0;i<n;i++)((float *)t->data)[i]=value;return 1;
}
static int ds4_gpu_synchronize(void) { return 1; }
static int ds4_gpu_begin_commands(void) { assert(!scopes);scopes=1;return 1; }
static int ds4_gpu_end_commands(void) { assert(scopes);scopes=0;return 1; }
static double now_sec(void) { return (double)clock()/CLOCKS_PER_SEC; }

typedef struct { uint64_t abs_offset; } ds4_tensor;
typedef struct { const void *map; uint64_t size; } ds4_model;
typedef struct { int il; } ds4_layer_weights;
typedef struct {
    bool metal_ready;ds4_model model;
    struct { ds4_layer_weights layer[DS4_N_LAYER];ds4_tensor *token_embd,*output_norm,*output; } weights;
} ds4_engine;
typedef struct {
    uint32_t ctx,mtp_block,mtp_towers,n_mtp_target,mtp_markov_rank;
    uint32_t n_engram,engram_max_ngram,engram_heads,engram_head_dim;
    int candidate_block_size;int engram_index_of[DS4_N_LAYER],mtp_target_slot[DS4_N_LAYER];
} ds4_v41_cfg;
static ds4_v41_cfg g_ds4_v41;
typedef struct {
    int (*prepare)(uint32_t,const int32_t *,int,void *);void *ud;
} ds41_engram_feed;
struct ds4_v41_images { uint32_t np; };
#define DENSE_FIELDS(X) \
 X(tok) X(pos) X(hc) X(hc2) X(mix) X(pre) X(post) X(comb) X(pre_mix) \
 X(x) X(xn) X(attn_out) X(glog) X(sel) X(rw) X(routed) X(sg) X(su) X(sh) X(so) X(y) X(logits)
#define ATT_FIELDS(X) X(qr) X(qrn) X(q) X(kv) X(kvn) X(o) X(low)
typedef struct {
#define FIELD(name) ds4_gpu_tensor *name;
    DENSE_FIELDS(FIELD) ATT_FIELDS(FIELD)
#undef FIELD
    ds4_gpu_tensor *erows,*ekv,*iscore,*cand,*mainh;
    ds4_gpu_tensor *eraw[DS4_V41_MAX_ENGRAM],*win[DS4_N_LAYER],*cpre_kv[DS4_N_LAYER],*cpre_sc[DS4_N_LAYER];
    ds4_gpu_tensor *snap_win[DS4_N_LAYER],*snap_cpre_kv[DS4_N_LAYER],*snap_cpre_sc[DS4_N_LAYER];
    uint32_t snap_past,snap_n,snap_on,snap_cpend[DS4_N_LAYER],ng_src[DS4_N_LAYER],cpend[DS4_N_LAYER];
    uint32_t n_past,ctx,cap_tok,logits_rows,last_logit_row,n,pos0;
    uint32_t mainh_cap,mainh_n,mainh_wrote;int64_t mainh_end;
    uint32_t iscap,isrows,icrows,iscap_gen,n_direct1,n_direct_n[10];
    int head_last_only,no_engram,idx_owner,cand_owner,idx_topk,idx_ratio,graph,ced_skip,ced_done,vrows_on;
    uint8_t *himg;const int32_t *blk_ids;const ds41_engram_feed *efeed;
    struct ds4_v41_images *vimg;
} ds4_v41_state;
typedef struct {
    int ready,graph_off,last_warm;uint32_t block;int32_t host_ids[10];float host_conf[9];ds4_gpu_tensor *owned;
} ds4_v41_draft;
typedef struct { uint32_t samples,hits; } v41_sched;
static ds4_gpu_tensor *v41_alloc(uint64_t n,bool *ok) {
    ds4_gpu_tensor *p=ds4_gpu_tensor_alloc(n);if(!p)*ok=false;return p;
}
static void v41_free_list(ds4_gpu_tensor ***p,size_t n) {
    for(size_t i=0;i<n;i++){ds4_gpu_tensor_free(*p[i]);*p[i]=NULL;}
}
#define V41_FREE(...) do {ds4_gpu_tensor **l[]={__VA_ARGS__};v41_free_list(l,sizeof(l)/sizeof(l[0]));}while(0)
static void v41_free_dense(ds4_v41_state *s) {
#define FREE(name) V41_FREE(&s->name);
    DENSE_FIELDS(FREE)
#undef FREE
}
static void v41_free_attn(ds4_v41_state *s) {
#define FREE(name) V41_FREE(&s->name);
    ATT_FIELDS(FREE)
#undef FREE
}
static void v41_graph_free(ds4_v41_state *s) {(void)s;}
static void v41_state_free(ds4_v41_state *s) {
    v41_free_dense(s);v41_free_attn(s);V41_FREE(&s->erows,&s->ekv,&s->iscore,&s->cand,&s->mainh);
    for(unsigned i=0;i<DS4_V41_MAX_ENGRAM;i++)V41_FREE(&s->eraw[i]);
    for(unsigned i=0;i<DS4_N_LAYER;i++)V41_FREE(&s->win[i],&s->cpre_kv[i],&s->cpre_sc[i],&s->snap_win[i],&s->snap_cpre_kv[i],&s->snap_cpre_sc[i]);
    free(s->himg);s->himg=NULL;
}
static void v41_alloc_dense(ds4_v41_state *s,uint32_t cap,uint32_t lr,bool *ok) {
#define ALLOC(name,words) s->name=v41_alloc((uint64_t)cap*(words)*4,ok)
    ALLOC(tok,1);ALLOC(pos,1);ALLOC(hc,DS4_N_HC*DS4_N_EMBD);ALLOC(hc2,DS4_N_HC*DS4_N_EMBD);
    ALLOC(mix,2*DS4_N_HC+DS4_N_HC*DS4_N_HC);ALLOC(pre,DS4_N_HC);ALLOC(post,DS4_N_HC);ALLOC(comb,DS4_N_HC*DS4_N_HC);ALLOC(pre_mix,DS4_N_HC);
    ALLOC(x,DS4_N_EMBD);ALLOC(xn,DS4_N_EMBD);ALLOC(attn_out,DS4_N_EMBD);ALLOC(glog,DS4_N_EXPERT);ALLOC(sel,DS4_N_EXPERT_USED);ALLOC(rw,DS4_N_EXPERT_USED);
    ALLOC(routed,DS4_N_EMBD);ALLOC(sg,DS4_N_FF_EXP);ALLOC(su,DS4_N_FF_EXP);ALLOC(sh,DS4_N_FF_EXP);ALLOC(so,DS4_N_EMBD);ALLOC(y,DS4_N_EMBD);
#undef ALLOC
    s->logits_rows=lr<cap?lr:cap;s->logits=v41_alloc((uint64_t)s->logits_rows*DS4_N_VOCAB*4,ok);
}
static void v41_alloc_attn_rows(ds4_v41_state *s,uint32_t cap,bool *ok) {
#define ALLOC(name,words) s->name=v41_alloc((uint64_t)cap*(words)*4,ok)
    ALLOC(qr,DS4_N_LORA_Q);ALLOC(qrn,DS4_N_LORA_Q);ALLOC(q,DS4_N_HEAD*DS4_N_HEAD_DIM);
    ALLOC(kv,DS4_N_HEAD_DIM);ALLOC(kvn,DS4_N_HEAD_DIM);ALLOC(o,DS4_N_HEAD*DS4_N_HEAD_DIM);ALLOC(low,DS4_N_OUT_GROUP*DS4_N_LORA_O);
#undef ALLOC
}
static void v41_alloc_attn_cache(ds4_v41_state *s,uint32_t n,bool *ok) {(void)s;(void)n;(void)ok;}
static uint64_t v41_bytes_rows(uint32_t n,uint32_t lr,int a) {return n*1024u+lr*DS4_N_VOCAB*4u+(uint64_t)a*128u;}
static uint64_t v41_bytes_kv(uint32_t ctx,uint32_t cap) {return (uint64_t)ctx*128u+cap*128u;}
static uint32_t ds4_layer_compress_ratio(uint32_t il) {return il==0?2:0;}
static bool v41_state_alloc(ds4_v41_state *s,uint32_t cap,uint32_t ctx,uint32_t lr) {
    memset(s,0,sizeof *s);s->cap_tok=cap;s->ctx=ctx;bool ok=true;
    v41_alloc_dense(s,cap,lr,&ok);v41_alloc_attn_rows(s,cap,&ok);
    s->himg=calloc(ctx,1);assert(s->himg);
    s->mainh_cap=DS4_N_SWA;s->mainh=v41_alloc((uint64_t)DS4_N_SWA*g_ds4_v41.n_mtp_target*DS4_N_EMBD*4,&ok);s->mainh_end=-1;
    for(unsigned i=0;i<DS4_N_LAYER;i++)s->win[i]=v41_alloc((DS4_N_SWA+(uint64_t)cap)*DS4_N_HEAD_DIM*4,&ok);
    s->cpre_kv[0]=v41_alloc((2+(uint64_t)cap)*DS4_N_HEAD_DIM*4,&ok);
    s->cpre_sc[0]=v41_alloc((2+(uint64_t)cap)*DS4_N_HEAD_DIM*4,&ok);
    if(!ok)v41_state_free(s);return ok;
}
static bool v41_prefill_chunk(ds4_engine *e,ds4_v41_state *s,const int32_t *p,uint32_t np,uint32_t *cursor,uint32_t cap) {
    (void)e;(void)cap;*cursor=np;s->n_past=np;s->last_logit_row=np<=8?np-1u:0;
    assert(s->last_logit_row<s->logits_rows);
    s->mainh_n=np<DS4_N_SWA?np:DS4_N_SWA;s->mainh_end=(int64_t)np-1;
    s->cpend[0]=np%2;s->ng_src[0]=np/2;
    for(unsigned il=0;il<DS4_N_LAYER;il++)for(unsigned j=0;j<DS4_N_SWA;j++)
        for(unsigned d=0;d<DS4_N_HEAD_DIM;d++)((float *)s->win[il]->data)[j*DS4_N_HEAD_DIM+d]=(float)(p[0]*100+il*10+j);
    ds4_gpu_tensor_fill_f32(s->cpre_kv[0],17.f,s->cpre_kv[0]->bytes/4);
    ds4_gpu_tensor_fill_f32(s->cpre_sc[0],29.f,s->cpre_sc[0]->bytes/4);
    for(uint32_t i=0;i<DS4_N_VOCAB;i++)((float *)s->logits->data)[s->last_logit_row*DS4_N_VOCAB+i]=i==(uint32_t)(p[np-1]+1)?1.f:0.f;
    return true;
}
static bool v41_draft_alloc(ds4_engine *e,ds4_v41_draft *d) {
    (void)e;memset(d,0,sizeof *d);if(mock_draft_absent)return false;
    d->owned=ds4_gpu_tensor_alloc(512);if(!d->owned)return false;
    d->ready=1;d->block=g_ds4_v41.mtp_block;return true;
}
static void v41_draft_free(ds4_v41_draft *d) {ds4_gpu_tensor_free(d->owned);memset(d,0,sizeof *d);}
static bool v41_draft_step(ds4_engine *e,ds4_v41_state *s,ds4_v41_draft *d,int32_t token,uint32_t pos) {
    (void)e;assert(s->mainh && s->mainh_n && s->mainh_end==(int64_t)pos);draft_calls++;
    d->host_ids[0]=token;
    for(uint32_t j=1;j<=d->block;j++)d->host_ids[j]=token+(int32_t)j;
    if(mock_draft_mismatch)d->host_ids[2]=63;
    return !fail_kernel;
}
static uint32_t v41_draft_pick_k(const float *c,uint32_t b,float *g,v41_sched *s) {(void)c;(void)g;(void)s;return b;}
static void v41_sched_draft_cost(v41_sched *s,double t) {(void)s;(void)t;}
static void v41_sched_cost(v41_sched *s,uint32_t n,double t) {(void)s;(void)n;(void)t;}
static void v41_sched_observe(v41_sched *s,const float *c,int hit) {(void)c;s->samples++;s->hits+=!!hit;}

static bool v41_spec_snapshot(ds4_v41_state *s,uint32_t n) {
    assert(n>1&&n<=9&&!s->snap_on);s->snap_on=1;s->snap_n=n;s->snap_past=s->n_past;
    for(unsigned il=0;il<DS4_N_LAYER;il++) {
        if(!s->snap_win[il])s->snap_win[il]=ds4_gpu_tensor_alloc(9*DS4_N_HEAD_DIM*4);
        for(unsigned j=0;j<n;j++)ds4_gpu_tensor_copy(s->snap_win[il],j*DS4_N_HEAD_DIM*4,s->win[il],((s->n_past+j)%DS4_N_SWA)*DS4_N_HEAD_DIM*4,DS4_N_HEAD_DIM*4);
    }
    s->snap_cpend[0]=s->cpend[0];return true;
}
static bool v41_spec_rollback(ds4_v41_state *s,uint32_t keep) {
    assert(s->snap_on&&keep>=1&&keep<=s->snap_n);rollbacks++;
    for(unsigned il=0;il<DS4_N_LAYER;il++)for(unsigned j=keep;j<s->snap_n;j++)
        ds4_gpu_tensor_copy(s->win[il],((s->snap_past+j)%DS4_N_SWA)*DS4_N_HEAD_DIM*4,s->snap_win[il],j*DS4_N_HEAD_DIM*4,DS4_N_HEAD_DIM*4);
    s->n_past=s->snap_past+keep;s->cpend[0]=s->n_past%2;s->ng_src[0]=s->n_past/2;
    s->snap_on=0;s->mainh_n-=s->snap_n-keep;s->mainh_end=(int64_t)s->n_past-1;return true;
}
static int ds4_gpu_v41_expand_hc_tensor(ds4_gpu_tensor *to,const ds4_gpu_tensor *from,uint32_t e,uint32_t hc,uint32_t n) {
    for(uint32_t i=0;i<n;i++)for(uint32_t h=0;h<hc;h++)ds4_gpu_tensor_copy(to,(i*hc+h)*(uint64_t)e*4,from,i*(uint64_t)e*4,e*4);return 1;
}
static bool v41_embed(const ds4_model *m,ds4_gpu_tensor *x,const ds4_gpu_tensor *tok,const ds4_tensor *w,uint32_t v,uint32_t n,uint32_t e) {
    (void)m;(void)w;(void)v;assert(n<=8);
    for(uint32_t i=0;i<n;i++)for(uint32_t k=0;k<e;k++)((float *)x->data)[i*e+k]=(float)((int32_t *)tok->data)[i];return !fail_target;
}
static bool v41_engram_rows(ds4_engine *e,ds4_v41_state *s,uint32_t il) {
    (void)e;(void)il;assert(s->efeed&&s->erows&&s->erows->bytes>=s->n*4);
    return true;
}
static bool v41_engram_apply(ds4_engine *e,ds4_v41_state *s,uint32_t il) {(void)e;(void)s;(void)il;return true;}
static bool v41_hc_half(ds4_engine *e,ds4_v41_state *s,const ds4_layer_weights *l,bool a) {
    (void)e;(void)l;(void)a;
    for(uint32_t i=0;i<s->n;i++)ds4_gpu_tensor_copy(s->xn,i*DS4_N_EMBD*4,s->hc,i*DS4_N_HC*DS4_N_EMBD*4,DS4_N_EMBD*4);
    /* Mirror the real helper's ping-pong ownership, even though the tensor
     * arithmetic here is a double. Two halves must restore each cached tile. */
    ds4_gpu_tensor *tmp=s->pre;s->pre=s->pre_mix;s->pre_mix=tmp;
    return true;
}
static bool v41_attn_in(ds4_engine *e,ds4_v41_state *s,uint32_t il) {(void)e;(void)s;(void)il;return true;}
static bool v41_attn_cache(ds4_engine *e,ds4_v41_state *s,uint32_t il) {
    (void)e;attention_rows+=(int)s->n;
    for(uint32_t i=0;i<s->n;i++) {
        int32_t t=((int32_t *)s->tok->data)[i];assert(((int32_t *)s->pos->data)[i]==(int32_t)(s->pos0+i));
        for(uint32_t k=0;k<DS4_N_HEAD_DIM;k++)((float *)s->win[il]->data)[((s->pos0+i)%DS4_N_SWA)*DS4_N_HEAD_DIM+k]=(float)t;
    }
    return true;
}
static bool v41_attn_out(ds4_engine *e,ds4_v41_state *s,uint32_t il) {(void)e;(void)il;return ds4_gpu_tensor_copy(s->attn_out,0,s->xn,0,s->n*DS4_N_EMBD*4);}
static bool v41_moe(const ds4_model *m,const ds4_layer_weights *l,ds4_v41_state *s,uint32_t il) {(void)m;(void)l;(void)il;return ds4_gpu_tensor_copy(s->y,0,s->xn,0,s->n*DS4_N_EMBD*4);}
static int ds4_gpu_v41_hc_post_tensor(ds4_gpu_tensor *dst,const ds4_gpu_tensor *x,const ds4_gpu_tensor *hc,const ds4_gpu_tensor *p,const ds4_gpu_tensor *c,uint32_t e,uint32_t h,uint32_t n) {
    (void)x;(void)p;(void)c;return ds4_gpu_tensor_copy(dst,0,hc,0,(uint64_t)n*h*e*4);
}
static int ds4_gpu_v41_hc_pre_tensor(ds4_gpu_tensor *dst,const ds4_gpu_tensor *hc,const ds4_gpu_tensor *mix,uint32_t e,uint32_t h,uint32_t n) {
    (void)mix;for(uint32_t i=0;i<n;i++)ds4_gpu_tensor_copy(dst,(uint64_t)i*e*4,hc,(uint64_t)i*h*e*4,e*4);return 1;
}
static int ds4_gpu_v41_rms_norm_tensor(ds4_gpu_tensor *dst,const ds4_gpu_tensor *src,const void *map,uint64_t size,uint64_t offset,uint32_t e,uint32_t n,float eps) {
    (void)map;(void)size;(void)offset;(void)eps;return ds4_gpu_tensor_copy(dst,0,src,0,(uint64_t)n*e*4);
}
static bool v41_tproj(const ds4_model *m,ds4_gpu_tensor *dst,const ds4_tensor *w,uint64_t e,uint64_t vocab,const ds4_gpu_tensor *src,uint32_t n,int round) {
    (void)m;(void)w;(void)round;assert(n<=DS4_V41_GEMV_MAX_TOK);head_calls++;head_rows+=n;if((int)n>head_max)head_max=n;
    for(uint32_t i=0;i<n;i++)for(uint64_t v=0;v<vocab;v++)((float *)dst->data)[i*vocab+v]=v==(uint64_t)(((float *)src->data)[i*e]+1.f)?1.f:0.f;
    return true;
}
static int ds4_gpu_v41_hc_mean_tensor(ds4_gpu_tensor *dst,const ds4_gpu_tensor *src,uint32_t e,uint32_t hc,uint32_t n,uint32_t first,uint32_t slot,uint32_t slots,uint32_t pos,uint32_t cap,const ds4_gpu_tensor *device_pos) {
    (void)device_pos;assert(dst&&slots&&slot<slots);
    for(uint32_t i=0;i<n;i++)ds4_gpu_tensor_copy(dst,(((uint64_t)pos+i)%cap*slots+slot)*e*4,src,((uint64_t)first+i)*hc*e*4,e*4);return 1;
}

struct ds4_v41_request;struct ds4_v41_batch;
void ds4_v41_request_close(struct ds4_v41_request *);
void ds4_v41_batch_close(struct ds4_v41_batch *);
/* This is the production code under test, not a second implementation. */
#include "../ds4_ds41_batch_mtp.inc"

static ds4_tensor dummy_weight;
static ds4_engine engine={.metal_ready=true,.weights={.token_embd=&dummy_weight,.output_norm=&dummy_weight,.output=&dummy_weight}};
static int feed_calls;
static int prepare_feed(uint32_t pos,const int32_t *t,int n,void *ud) {
    uint32_t *stamp=ud;*stamp=pos;assert(t&&n>=1&&n<=9);feed_calls++;return fail_feed;
}
static ds41_engram_feed feed[8];static uint32_t feed_pos[8];
static struct ds4_v41_request *request(int i,int predict,int mode) {
    int32_t prompt[10];for(int j=0;j<10;j++)prompt[j]=i*7;
    feed[i]=(ds41_engram_feed){.prepare=prepare_feed,.ud=&feed_pos[i]};
    struct ds4_v41_request *r=ds4_v41_request_open(&engine,prompt,10,predict,0,NULL,&feed[i]);assert(r);
    assert(!ds4_v41_request_set_mtp(r,mode,0,0));
    float logits[DS4_N_VOCAB];assert(ds4_v41_request_prefill(r,logits,DS4_N_VOCAB)==1);
    assert(r->state.mainh&&r->state.mainh_n);
    return r;
}
static void defaults(void) {
    memset(&g_ds4_v41,0,sizeof g_ds4_v41);g_ds4_v41.ctx=1024;g_ds4_v41.mtp_block=5;g_ds4_v41.mtp_towers=3;g_ds4_v41.n_mtp_target=1;g_ds4_v41.mtp_markov_rank=2;
    g_ds4_v41.n_engram=1;g_ds4_v41.engram_max_ngram=2;g_ds4_v41.engram_heads=1;g_ds4_v41.engram_head_dim=4;
    for(unsigned i=0;i<DS4_N_LAYER;i++){g_ds4_v41.engram_index_of[i]=i==0?0:-1;g_ds4_v41.mtp_target_slot[i]=i==3?0:-1;}
    mock_draft_absent=mock_draft_mismatch=fail_kernel=fail_target=fail_feed=0;fail_alloc_after=-1;
}
static int tests;
static void test_c4_varied_commit(void) {
    defaults();struct ds4_v41_batch *b=ds4_v41_batch_open(&engine,4);assert(b);
    struct ds4_v41_request *r[4];int32_t tokens[4];for(int i=0;i<4;i++){r[i]=request(i,64,2);tokens[i]=i*7+1;}
    float logits[4*DS41_VERIFY_MAX*DS4_N_VOCAB];int32_t p[4*DS41_VERIFY_MAX];uint32_t w[4];head_rows=head_max=head_calls=0;
    int before=draft_calls;
    assert(!ds4_v41_batch_verify(b,r,tokens,4,logits,sizeof logits/sizeof *logits,p,36,w));
    assert(draft_calls-before==4&&head_rows==24&&head_calls==3&&head_max==8);
    for(int i=0;i<4;i++) {
        assert(w[i]==6&&r[i]->state.n_past==16&&feed_pos[i]==10);
        for(unsigned j=0;j<w[i];j++)assert(logits[(i*6+j)*DS4_N_VOCAB+tokens[i]+j+1]==1.f);
        assert(!ds4_v41_request_commit(r[i],(uint32_t)i+1));
        assert(r[i]->state.n_past==(uint32_t)(11+i)&&!r[i]->round.rows);
        /* Rejected SWA cells are the old, request-specific contents. */
        for(unsigned j=(unsigned)i+1;j<6;j++)assert(((float *)r[i]->state.win[0]->data)[((10+j)%8)*4]==(float)(i*7*100+(10+j)%8));
    }
    for(int i=0;i<4;i++)ds4_v41_request_close(r[i]);ds4_v41_batch_close(b);assert(alive_tensors==0&&scopes==0);tests++;
}
static void test_rejection_depths(void) {
    for(unsigned keep=1;keep<=6;keep++) {
        defaults();struct ds4_v41_batch *b=ds4_v41_batch_open(&engine,1);assert(b);struct ds4_v41_request *r=request(0,64,2);
        float logits[9*DS4_N_VOCAB];int32_t p[9],t=1;uint32_t w;
        assert(!ds4_v41_batch_verify(b,&r,&t,1,logits,9*DS4_N_VOCAB,p,9,&w));
        assert(ds4_v41_request_commit(r,0)!=0&&r->round.rows==6);
        assert(ds4_v41_request_commit(r,7)!=0&&r->round.rows==6);
        assert(!ds4_v41_request_commit(r,keep));assert(r->state.n_past==10+keep);
        assert(ds4_v41_request_commit(r,1)!=0); /* double commit */
        t=(int32_t)(keep+1);assert(!ds4_v41_batch_verify(b,&r,&t,1,logits,9*DS4_N_VOCAB,p,9,&w));
        assert(!ds4_v41_request_commit(r,1));
        ds4_v41_request_close(r);ds4_v41_batch_close(b);assert(alive_tensors==0&&scopes==0);tests++;
    }
}
static void test_mixed_widths_tail_cancel(void) {
    defaults();struct ds4_v41_batch *b=ds4_v41_batch_open(&engine,4);assert(b);
    struct ds4_v41_request *r[4]={request(0,64,2),request(1,3,2),request(2,1,2),request(3,64,0)};
    float logits[36*DS4_N_VOCAB];int32_t p[36],t[4]={1,8,15,22};uint32_t w[4];
    assert(!ds4_v41_batch_verify(b,r,t,4,logits,36*DS4_N_VOCAB,p,36,w));
    assert(w[0]==6&&w[1]==3&&w[2]==1&&w[3]==1);
    for(int i=0;i<4;i++)assert(!ds4_v41_request_commit(r[i],1));
    ds4_v41_request_close(r[2]);r[2]=request(2,32,2); /* slot reuse after tail completion */
    assert(!ds4_v41_batch_verify(b,r,t,4,logits,36*DS4_N_VOCAB,p,36,w));
    ds4_v41_request_close(r[1]);r[1]=NULL; /* cancellation while provisional */
    for(int i=0;i<4;i++)if(r[i]){assert(!ds4_v41_request_commit(r[i],1));ds4_v41_request_close(r[i]);}
    ds4_v41_batch_close(b);assert(!alive_tensors&&!scopes);tests++;
}
static void test_failures_and_refusals(void) {
    defaults();struct ds4_v41_batch *b=ds4_v41_batch_open(&engine,2);assert(b);
    struct ds4_v41_request *r[2]={request(0,32,2),request(1,32,2)};
    struct ds4_v41_request *dup[2]={r[0],r[0]};float logits[18*DS4_N_VOCAB];int32_t t[]={1,8},p[18];uint32_t w[2];int before=draft_calls;
    assert(ds4_v41_batch_verify(b,dup,t,2,logits,18*DS4_N_VOCAB,p,18,w)!=0);assert(draft_calls==before&&!b->poisoned);
    assert(ds4_v41_batch_verify(b,r,t,2,logits,1,p,18,w)!=0);assert(draft_calls==before);
    fail_kernel=1;assert(ds4_v41_batch_verify(b,r,t,2,logits,18*DS4_N_VOCAB,p,18,w)!=0);assert(b->poisoned&&r[0]->poisoned&&r[1]->poisoned);fail_kernel=0;
    for(int i=0;i<2;i++)ds4_v41_request_close(r[i]);ds4_v41_batch_close(b);assert(!alive_tensors&&!scopes);tests++;
    defaults();int32_t ids[]={1};struct ds4_v41_request *q=ds4_v41_request_open(&engine,ids,1,32,0,NULL,&feed[0]);assert(q);mock_draft_absent=1;
    assert(ds4_v41_request_set_mtp(q,2,0,0)!=0);assert(q->poisoned);ds4_v41_request_close(q);assert(!alive_tensors);tests++;
}
static void test_short_prefill(void) {
    for(int np=1;np<=8;np++) {
        defaults();int32_t ids[8]={1};struct ds4_v41_request *r=ds4_v41_request_open(&engine,ids,np,32,0,NULL,&feed[0]);assert(r);
        assert(!ds4_v41_request_set_mtp(r,2,0,0));float logits[DS4_N_VOCAB];assert(ds4_v41_request_prefill(r,logits,DS4_N_VOCAB)==1);
        assert(r->state.mainh_n==(unsigned)np&&r->state.cap_tok==6);ds4_v41_request_close(r);assert(!alive_tensors);tests++;
    }
}
static void test_partial_allocations(void) {
    for(int fail=0;fail<220;fail++) {
        defaults();allocations=0;fail_alloc_after=fail;
        struct ds4_v41_batch *b=ds4_v41_batch_open(&engine,4);
        ds4_v41_batch_close(b);assert(!alive_tensors&&!scopes);tests++;
    }
    fail_alloc_after=-1;
}
static void test_round_state(void) {
    for(unsigned n=1;n<=DS41_VERIFY_MAX;n++)for(unsigned accepted=0;accepted<n;accepted++) {
        ds41_round r={0};int32_t p[9];for(unsigned i=0;i<n;i++)p[i]=(int32_t)i+1;
        assert(ds41_round_begin(&r,p,n));assert(!ds41_round_begin(&r,p,n));
        for(unsigned j=0;j<accepted;j++)assert(ds41_round_accept(&r,(int32_t)j+2));
        assert(!ds41_round_accept(&r,99));assert(ds41_round_keep(&r)==accepted+1);
        ds41_round_clear(&r);assert(!ds41_round_keep(&r));tests++;
    }
}

static int argmax(const float *p) {
    int best=0;for(unsigned i=1;i<DS4_N_VOCAB;i++)if(p[i]>p[best])best=(int)i;return best;
}
static void test_widths_and_multi_round_parity(void) {
    for(int width=2;width<=8;width++)for(int wrong=0;wrong<=1;wrong++) {
        defaults();mock_draft_mismatch=wrong;
        struct ds4_v41_batch *b=ds4_v41_batch_open(&engine,width);assert(b);
        struct ds4_v41_request *r[8];int32_t t[8],p[8*9];uint32_t w[8];float logits[8*9*DS4_N_VOCAB];
        unsigned outputs[8]={0};
        for(int i=0;i<width;i++){r[i]=request(i,40,2);t[i]=i*7+1;}
        for(unsigned round=0;round<5;round++) {
            assert(!ds4_v41_batch_verify(b,r,t,width,logits,8*9*DS4_N_VOCAB,p,8*9,w));
            unsigned offset=0;
            for(int i=0;i<width;i++) {
                unsigned accepted=0;
                while(true) {
                    int target=argmax(logits+(offset+accepted)*DS4_N_VOCAB);
                    assert(target==i*7+2+(int)outputs[i]);outputs[i]++;
                    if(accepted+1<w[i] && target==p[i*9+accepted+1] && outputs[i]<7) {
                        accepted++;continue;
                    }
                    t[i]=target;break;
                }
                assert(!ds4_v41_request_commit(r[i],accepted+1));offset+=w[i];
            }
        }
        for(int i=0;i<width;i++)ds4_v41_request_close(r[i]);ds4_v41_batch_close(b);
        assert(!alive_tensors&&!scopes);tests++;
    }
}
static void test_target_and_feed_failures(void) {
    for(int phase=0;phase<2;phase++) {
        defaults();struct ds4_v41_batch *b=ds4_v41_batch_open(&engine,4);assert(b);
        struct ds4_v41_request *r[4];for(int i=0;i<4;i++)r[i]=request(i,32,2);
        int32_t t[]={1,8,15,22},p[36];uint32_t w[4];float logits[36*DS4_N_VOCAB];
        if(phase)fail_feed=1;else fail_target=1;
        assert(ds4_v41_batch_verify(b,r,t,4,logits,36*DS4_N_VOCAB,p,36,w));
        assert(b->poisoned&&!scopes);
        for(int i=0;i<4;i++){assert(r[i]->poisoned);ds4_v41_request_close(r[i]);}
        ds4_v41_batch_close(b);assert(!alive_tensors);tests++;
    }
}
static void test_no_engram_and_short_tail(void) {
    defaults();g_ds4_v41.n_engram=0;
    struct ds4_v41_batch *b=ds4_v41_batch_open(&engine,1);assert(b);
    int32_t ids[]={1};struct ds4_v41_request *r=ds4_v41_request_open(&engine,ids,1,1,1,NULL,NULL);assert(r);
    assert(!ds4_v41_request_set_mtp(r,2,0,0));float row[DS4_N_VOCAB];assert(ds4_v41_request_prefill(r,row,DS4_N_VOCAB)==1);
    float logits[9*DS4_N_VOCAB];int32_t p[9],t=2;uint32_t w;
    assert(!ds4_v41_batch_verify(b,&r,&t,1,logits,9*DS4_N_VOCAB,p,9,&w));assert(w==1);
    assert(!ds4_v41_request_commit(r,1));assert(r->state.n_past==2);
    ds4_v41_request_close(r);ds4_v41_batch_close(b);assert(!alive_tensors);tests++;
}
static void test_view_allocation_failures(void) {
    for(int fail=0;fail<32;fail++) {
        defaults();struct ds4_v41_batch *b=ds4_v41_batch_open(&engine,4);assert(b);
        struct ds4_v41_request *r[4];for(int i=0;i<4;i++)r[i]=request(i,32,2);
        /* Warm snapshot buffers first; fault each of the 4 x 8 borrowed views. */
        for(int i=0;i<4;i++){assert(v41_spec_snapshot(&r[i]->state,6));r[i]->state.snap_on=0;}
        allocations=0;fail_alloc_after=fail;
        int32_t t[]={1,8,15,22},p[36];uint32_t w[4];float logits[36*DS4_N_VOCAB];
        assert(ds4_v41_batch_verify(b,r,t,4,logits,36*DS4_N_VOCAB,p,36,w));
        fail_alloc_after=-1;
        assert(b->poisoned&&!scopes);
        for(int i=0;i<4;i++)ds4_v41_request_close(r[i]);ds4_v41_batch_close(b);
        assert(!alive_tensors);tests++;
    }
}


/* The production bridge implementation is checked against the real native
 * control functions above, not replacement verify/commit implementations. */
typedef struct { ds4_engine *engine; } ds4_bridge_model;
typedef struct { ds4_bridge_model *model; struct ds4_v41_request *native; } ds4_bridge_v41_request;
typedef struct { ds4_bridge_model *model; struct ds4_v41_batch *native; } ds4_bridge_v41_batch;
static void set_err(char *out,size_t n,const char *message) { if(out&&n)snprintf(out,n,"%s",message); }
#include "../native/bridge/ds4_bridge_v41_mtp.inc"
static void test_bridge(void) {
    defaults();ds4_bridge_model m={&engine},other={&engine};char err[256]={0};
    ds4_bridge_v41_batch b={&m,ds4_v41_batch_open(&engine,4)};assert(b.native);
    ds4_bridge_v41_request owned[4],*r[4];
    int32_t t[4]={1,8,15,22},p[36];float logits[36*DS4_N_VOCAB];uint32_t w[4];
    for(int i=0;i<4;i++){owned[i]=(ds4_bridge_v41_request){&m,request(i,32,2)};r[i]=&owned[i];}
    r[2]->model=&other;int prior=draft_calls;
    assert(ds4_bridge_v41_batch_verify(&b,r,t,4,logits,36*DS4_N_VOCAB,p,36,w,err,sizeof err));
    assert(draft_calls==prior&&strstr(err,"mix models"));r[2]->model=&m;
    assert(!ds4_bridge_v41_batch_verify(&b,r,t,4,logits,36*DS4_N_VOCAB,p,36,w,err,sizeof err));
    for(int i=0;i<4;i++)assert(!ds4_bridge_v41_request_commit(r[i],i+1,err,sizeof err));
    assert(ds4_bridge_v41_request_set_mtp(NULL,2,0,1,err,sizeof err));
    for(int i=0;i<4;i++)ds4_v41_request_close(r[i]->native);ds4_v41_batch_close(b.native);
    assert(!alive_tensors&&!scopes);tests++;
}

int main(void) {
    test_c4_varied_commit();test_rejection_depths();test_mixed_widths_tail_cancel();test_failures_and_refusals();test_short_prefill();test_partial_allocations();test_round_state();
    test_widths_and_multi_round_parity();test_target_and_feed_failures();test_no_engram_and_short_tail();test_view_allocation_failures();
    assert(!alive_tensors&&!scopes);
    test_bridge();
    printf("PASS: %d cases; production packed-MTP control path with checked tensor doubles (NOT CUDA qualification)\n",tests);
    return 0;
}
