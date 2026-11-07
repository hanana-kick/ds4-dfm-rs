/* Model-free ABI transaction gate. Reuses the production bridge plus fake
 * GPU entry points from the lifetime test; does NOT claim numerical parity. */
#define main plain_bridge_main
#include "test_ds41_batch_lifetime.c"
#undef main

static unsigned mtp_on[16], pending_rows[16], start_pos[16];
static int verify_calls, mtp_calls;
uint64_t ds4_v41_request_mtp_bytes(void *e) { return e ? 1024 : UINT64_MAX; }
int ds4_v41_request_mtp(struct ds4_v41_request *r,int on,int k) {
    if(!r || k<0 || k>=9)return 1;
    mtp_on[r->p->identity]=!!on;mtp_calls++;return 0;
}
int ds4_v41_batch_verify(struct ds4_v41_batch *b,struct ds4_v41_request **r,
    const int32_t *tokens,const uint32_t *rem,int n,int32_t *ids,uint32_t *nr,float *out,size_t f) {
    assert(n<=b->width && f==(size_t)n*9*3);verify_calls++;
    for(int i=0;i<n;i++) {
        int id=r[i]->p->identity;
        assert(mtp_on[id] && !pending_rows[id] && rem[i]);
        nr[i]=rem[i]<6?rem[i]:6;
        start_pos[id]=r[i]->pos;pending_rows[id]=nr[i];r[i]->pos+=nr[i];
        r[i]->p->pending=1;
        for(unsigned j=0;j<nr[i];j++) {
            ids[i*9+j]=tokens[i]+(int)j;
            for(int k=0;k<3;k++)out[(i*9+j)*3+k]=(float)(id*1000+j*10+k);
        }
    }
    return 0;
}
int ds4_v41_request_commit(struct ds4_v41_request *r,uint32_t keep) {
    if(!r)return 1;
    int id=r->p->identity;
    assert(!r->p->pending); /* Actual bridge must drain BEFORE rollback. */
    if(!keep || keep>pending_rows[id])return 1;
    r->pos=start_pos[id]+keep;pending_rows[id]=0;return 0;
}
int main(void) {
    assert(!plain_bridge_main());
    for(int width=4;width<=8;width+=4) {
        memset(mtp_on,0,sizeof mtp_on);memset(pending_rows,0,sizeof pending_rows);
        ds4_bridge_model m={.engine=(ds4_engine *)1},other={.engine=(ds4_engine *)2};
        provider p[8]={{0}};ds4_bridge_v41_request *req[8];
        int32_t prompt[]={7},tokens[8],ids[8*9];uint32_t rem[8],nr[8];
        float out[8*9*3];char err[256];
        ds4_bridge_v41_batch *b=ds4_bridge_v41_batch_open(&m,width,err,sizeof err);assert(b);
        for(int i=0;i<width;i++) {
            p[i].identity=i+1;tokens[i]=20+i*10;rem[i]=32;
            req[i]=ds4_bridge_v41_request_open(&m,prompt,1,128,0,NULL,rows,NULL,NULL,drain,&p[i],err,sizeof err);
            assert(req[i]);assert(!ds4_bridge_v41_request_mtp(req[i],1,0,err,sizeof err));
            assert(ds4_bridge_v41_request_prefill(req[i],out,3,err,sizeof err)==1);
        }
        int before=verify_calls;
        assert(!ds4_bridge_v41_batch_verify(b,req,tokens,rem,width,ids,nr,out,width*9*3,err,sizeof err));
        assert(verify_calls==before+1);
        for(int i=0;i<width;i++) {
            assert(nr[i]==6 && ids[i*9+5]==tokens[i]+5);
            assert(out[(i*9+5)*3]==(float)((i+1)*1000+50));
            uint32_t keep=(unsigned)i%6+1;
            assert(!ds4_bridge_v41_request_commit(req[i],keep,err,sizeof err));
            assert(ds4_bridge_v41_request_position(req[i])==1+keep);
            assert(p[i].drains==1 && !p[i].pending);
        }
        ds4_bridge_v41_request *saved=req[1];req[1]=req[0];before=verify_calls;
        assert(ds4_bridge_v41_batch_verify(b,req,tokens,rem,width,ids,nr,out,width*9*3,err,sizeof err));
        assert(verify_calls==before);req[1]=saved;
        req[1]->model=&other;
        assert(ds4_bridge_v41_batch_verify(b,req,tokens,rem,width,ids,nr,out,width*9*3,err,sizeof err));
        assert(verify_calls==before);req[1]->model=&m;
        for(int i=0;i<width;i++)rem[i]=(unsigned)i%6+1;
        assert(!ds4_bridge_v41_batch_verify(b,req,tokens,rem,width,ids,nr,out,width*9*3,err,sizeof err));
        ds4_bridge_v41_request_close(req[0]);req[0]=NULL;
        assert(p[0].closed==1 && p[1].pending && !p[1].closed);
        for(int i=1;i<width;i++) {
            assert(nr[i]==rem[i]);
            assert(!ds4_bridge_v41_request_commit(req[i],nr[i],err,sizeof err));
            ds4_bridge_v41_request_close(req[i]);
        }
        assert(!live_requests && !live_feeds);
        ds4_bridge_v41_batch_close(b);
    }
    assert(mtp_calls==12);
    puts("DS41 C4/C8 MTP bridge: packed dispatch, ragged rows, independent commits, cancellation and draining PASS (mock GPU)");
    return 0;
}
