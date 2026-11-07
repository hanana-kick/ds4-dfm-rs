#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include "../ds41_spec_plan.h"
int main(void) {
    uint32_t rows[8], off[8], total = 99;
    for (int n=1;n<=8;n++) {
        for (int i=0;i<n;i++) rows[i]=6;
        assert(ds41_verify_layout(rows,n,8,72,off,&total));
        assert(total==(uint32_t)n*6);
        for(int i=0;i<n;i++) assert(off[i]==(uint32_t)i*6);
    }
    for(uint32_t k=0;k<=8;k++) for(uint32_t rem=0;rem<=10;rem++)
      for(uint32_t room=0;room<=10;room++) for(uint32_t limit=0;limit<=8;limit++) {
        uint32_t got=ds41_verify_width(k,rem,room,limit);
        uint32_t want=k;if(limit&&want>limit)want=limit;want++;
        if(want>rem)want=rem;
        if(want>room)want=room;
        assert(got==want);
      }
    uint32_t ragged[]={6,1,4,2};
    assert(ds41_verify_layout(ragged,4,8,72,off,&total));
    assert(total==13&&off[0]==0&&off[1]==6&&off[2]==7&&off[3]==11);
    total=99;off[0]=99;
    assert(!ds41_verify_layout(ragged,4,3,72,off,&total));
    assert(!ds41_verify_layout(ragged,4,8,12,off,&total));
    assert(total==99&&off[0]==99);
    rows[0]=UINT32_MAX;
    assert(!ds41_verify_layout(rows,1,8,UINT32_MAX,off,&total));
    rows[0]=0;assert(!ds41_verify_layout(rows,1,8,72,off,&total));
    puts("DS41 verification layout: C1..C8, ragged widths, output/context bounds PASS");
}
