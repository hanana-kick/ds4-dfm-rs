/* Model-free ABI/lifetime gate for the production batch bridge. Native GPU
 * entry points below are doubles: this proves ownership/dispatch, NOT logits.
 * In particular every feed belongs to one request and is drained on all exits.
 */
#include <assert.h>
#include "../native/bridge/ds4_bridge.c"

typedef struct { int pending, drains, closed, identity; } provider;
struct ds4_v41_feed { provider *p; };
struct ds4_v41_request { provider *p; uint32_t pos; int image; };
struct ds4_v41_batch { int width; };
static int fail_open, image_frees, live_requests, live_feeds, steps, widest;

int ds4_v41_engram_count(void *engine) { (void)engine; return 1; }
ds4_v41_feed *ds4_v41_feed_open(void *engine, int np, ds4_v41_rows_fn rows, void *ud,
        ds4_v41_wait_fn wait, void *wu, ds4_v41_next_fn next, void *nu) {
    (void)engine; (void)np; (void)rows; (void)wait; (void)wu; (void)next; (void)nu;
    ds4_v41_feed *f = calloc(1,sizeof *f); assert(f); f->p=ud; live_feeds++; return f;
}
const ds41_engram_feed *ds4_v41_feed_handle(const ds4_v41_feed *f) { return (const ds41_engram_feed *)f; }
void ds4_v41_feed_close(ds4_v41_feed *f) {
    if (!f) return; assert(!f->p->pending); f->p->closed++; live_feeds--; free(f);
}
struct ds4_v41_images *ds4_v41_images_build(const ds4_v41_image_src *in, int n,
        const int32_t *row, uint32_t np, uint32_t nr, const uint64_t *sig) {
    assert(in && n==1 && row && np==1 && nr==1 && sig);
    return (struct ds4_v41_images *)calloc(1,1);
}
void ds4_v41_images_free(struct ds4_v41_images *i) { if(i) { image_frees++; free(i); } }
uint64_t ds4_v41_images_bytes(const struct ds4_v41_images *i) { return i ? 32 : 0; }
uint64_t ds4_v41_request_peak(void *e,int np,int n,int no) { (void)e;(void)no;return (uint64_t)(np+n)*16; }
struct ds4_v41_request *ds4_v41_request_open(void *e,const int32_t *ids,int np,int n,int no,
        struct ds4_v41_images *i,const ds41_engram_feed *feed) {
    (void)e;(void)ids;(void)n;(void)no; provider *p=((const ds4_v41_feed *)feed)->p;
    p->pending=1; if(fail_open) return NULL;
    struct ds4_v41_request *r=calloc(1,sizeof *r); assert(r); r->p=p;r->pos=(uint32_t)np;r->image=i!=NULL;live_requests++;return r;
}
void ds4_v41_request_close(struct ds4_v41_request *r) {
    if(!r) return; assert(!r->p->pending);live_requests--;free(r);
}
int ds4_v41_request_prefill(struct ds4_v41_request *r,float *out,size_t n) {
    for(size_t k=0;k<n;k++)out[k]=(float)r->p->identity;
    return 1;
}
uint32_t ds4_v41_request_position(const struct ds4_v41_request *r) { return r?r->pos:0; }
struct ds4_v41_batch *ds4_v41_batch_open(void *e,int n) { (void)e;struct ds4_v41_batch *b=malloc(sizeof *b);assert(b);b->width=n;return b; }
void ds4_v41_batch_close(struct ds4_v41_batch *b) { free(b); }
int ds4_v41_batch_step(struct ds4_v41_batch *b,struct ds4_v41_request **r,const int32_t *ids,int n,float *out,size_t count) {
    assert(n<=b->width && count==(size_t)n*3); steps++; if(n>widest)widest=n;
    for(int j=0;j<n;j++) { r[j]->pos++;for(int k=0;k<3;k++)out[j*3+k]=(float)(r[j]->p->identity*100+ids[j]+k); }
    return 0;
}
static int rows(uint32_t pos,const int32_t *ids,int n,uint8_t *const *dst,void *ud) {
    (void)pos;(void)ids;(void)n;(void)dst;(void)ud;return 0;
}
static int drain(void *ud) { provider *p=ud;p->pending=0;p->drains++;return 0; }
int main(void) {
    ds4_bridge_model model={.engine=(ds4_engine *)1}, other={.engine=(ds4_engine *)2};
    char err[256];const int32_t prompt[]={7}, tok[]={11,22,33,44};float out[12];
    provider p[4]={{.identity=1},{.identity=2},{.identity=3},{.identity=4}};
    ds4_bridge_v41_request *r[4];
    ds4_bridge_v41_batch *b=ds4_bridge_v41_batch_open(&model,4,err,sizeof err); assert(b);
    for(int j=0;j<4;j++) {
        r[j]=ds4_bridge_v41_request_open(&model,prompt,1,8,0,NULL,rows,NULL,NULL,drain,&p[j],err,sizeof err);
        assert(r[j]);assert(ds4_bridge_v41_request_prefill(r[j],out,3,err,sizeof err)==1);
    }
    assert(live_requests==4 && live_feeds==4);
    assert(!ds4_bridge_v41_batch_step(b,r,tok,4,out,12,err,sizeof err));
    assert(steps==1 && widest==4);
    for(int j=0;j<4;j++){assert(out[3*j]==(float)(p[j].identity*100+tok[j]));assert(ds4_bridge_v41_request_position(r[j])==2);}
    ds4_bridge_v41_request_close(r[1]);r[1]=NULL;
    assert(p[1].drains==1 && p[1].closed==1 && p[0].drains==0 && p[2].pending);
    ds4_bridge_v41_request *keep[]={r[0],r[2],r[3]};
    assert(!ds4_bridge_v41_batch_step(b,keep,tok,3,out,9,err,sizeof err));
    assert(live_requests==3 && steps==2);
    keep[1]->model=&other;
    assert(ds4_bridge_v41_batch_step(b,keep,tok,3,out,9,err,sizeof err)!=0 && steps==2);
    keep[1]->model=&model;
    for(int j=0;j<4;j++)if(r[j])ds4_bridge_v41_request_close(r[j]);
    assert(live_requests==0 && live_feeds==0);
    fail_open=1;
    assert(!ds4_bridge_v41_request_open(&model,prompt,1,8,0,NULL,rows,NULL,NULL,drain,&p[0],err,sizeof err));
    assert(!p[0].pending && live_feeds==0);
    fail_open=0;
    assert(!ds4_bridge_v41_request_open(&model,prompt,1,8,0,NULL,rows,NULL,NULL,NULL,&p[0],err,sizeof err));
    float patch[3]={0};int32_t row[]={0};uint64_t sig[]={1};
    ds4_bridge_v41_image img={.patches=patch,.patches_len=3,.n_vit_h=1,.n_vit_w=1,.n_llm_h=1,.n_llm_w=1};
    ds4_bridge_v41_images images={.img=&img,.n=1,.row_of=row,.sig_of=sig,.np=1,.n_rows=1};
    ds4_bridge_v41_request *image=ds4_bridge_v41_request_open(&model,prompt,1,8,0,&images,rows,NULL,NULL,drain,&p[0],err,sizeof err);
    assert(image && image->native->image);assert(image_frees==0);
    assert(ds4_bridge_v41_request_prefill(image,out,3,err,sizeof err)==1 && image_frees==1);
    ds4_bridge_v41_request_close(image);assert(image_frees==1 && live_feeds==0 && live_requests==0);
    ds4_bridge_v41_batch_close(b);
    assert(!ds4_bridge_v41_batch_open(&model,9,err,sizeof err));
    puts("V4.1 packed bridge: isolation, cohort dispatch, cancellation, drain, failed open and image lifetime PASS (mock GPU)");
    return 0;
}
