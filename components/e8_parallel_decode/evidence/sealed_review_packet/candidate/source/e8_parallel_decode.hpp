#pragma once
#include <cstdint>
#ifdef E8_PARALLEL_CPU
#define E8P_FN inline
#else
#define E8P_FN __simt_callee__ inline __attribute__((always_inline))
#endif

namespace e8_parallel {
// Eight independent coordinate registers. There is no indexed device array.
struct I8 { int a,b,c,d,e,f,g,h; };
E8P_FN unsigned Xor8(unsigned a,unsigned b,unsigned c,unsigned d,
                    unsigned e,unsigned f,unsigned g,unsigned h) {
    const unsigned ab=a^b,cd=c^d,ef=e^f,gh=g^h;
    const unsigned abcd=ab^cd,efgh=ef^gh;
    return abcd^efgh;
}
E8P_FN int Sum8(int a,int b,int c,int d,int e,int f,int g,int h) {
    const int ab=a+b,cd=c+d,ef=e+f,gh=g+h;
    const int abcd=ab+cd,efgh=ef+gh;
    return abcd+efgh;
}
E8P_FN unsigned Max2(unsigned a,unsigned b) { return a>b?a:b; }
E8P_FN unsigned Max8(unsigned a,unsigned b,unsigned c,unsigned d,
                    unsigned e,unsigned f,unsigned g,unsigned h) {
    const unsigned ab=Max2(a,b),cd=Max2(c,d),ef=Max2(e,f),gh=Max2(g,h);
    const unsigned abcd=Max2(ab,cd),efgh=Max2(ef,gh);
    return Max2(abcd,efgh);
}
E8P_FN unsigned RepairKey(int value,unsigned index) {
    const unsigned positive=unsigned(value>=0);
    const unsigned absolute=unsigned(value<0?-value:value);
    // Frozen scan tie: first positive max; otherwise last negative max.
    return (absolute<<8)|(positive<<7)|(positive?7u-index:index);
}
E8P_FN I8 Repair(I8 r,unsigned parity) {
    const unsigned key=Max8(RepairKey(r.a,0),RepairKey(r.b,1),
        RepairKey(r.c,2),RepairKey(r.d,3),RepairKey(r.e,4),RepairKey(r.f,5),
        RepairKey(r.g,6),RepairKey(r.h,7));
    const unsigned positive=(key>>7)&1u;
    const unsigned index=positive?7u-(key&7u):key&7u;
    const int delta=positive?-8:8;
    return {r.a+int(parity&unsigned(index==0))*delta,
            r.b+int(parity&unsigned(index==1))*delta,
            r.c+int(parity&unsigned(index==2))*delta,
            r.d+int(parity&unsigned(index==3))*delta,
            r.e+int(parity&unsigned(index==4))*delta,
            r.f+int(parity&unsigned(index==5))*delta,
            r.g+int(parity&unsigned(index==6))*delta,
            r.h+int(parity&unsigned(index==7))*delta};
}
E8P_FN I8 Project(unsigned label) {
    const int a=2*int((label>>14)&1u)+int((label>>15)&1u)-2;
    const int b=int(label&3u)-2,c=int((label>>2)&3u)-2;
    const int d=int((label>>4)&3u)-2,e=int((label>>6)&3u)-2;
    const int f=int((label>>8)&3u)-2,g=int((label>>10)&3u)-2;
    const int h=int((label>>12)&3u)-2;
    return {a,a+2*b,a-2*b+2*c,a-2*c+2*d,a-2*d+2*e,
            a-2*e+2*f,a-2*f+2*g+2*h,a-2*g+2*h};
}
E8P_FN I8 ShapeResidualBits(I8 p) {
    const unsigned a=unsigned(p.a)+4u,b=unsigned(p.b)+4u;
    const unsigned c=unsigned(p.c)+4u,d=unsigned(p.d)+4u;
    const unsigned e=unsigned(p.e)+4u,f=unsigned(p.f)+4u;
    const unsigned g=unsigned(p.g)+4u,h=unsigned(p.h)+4u;
    const I8 r{int(a&7u)-4,int(b&7u)-4,int(c&7u)-4,int(d&7u)-4,
               int(e&7u)-4,int(f&7u)-4,int(g&7u)-4,int(h&7u)-4};
    const unsigned parity=Xor8((a>>3)&1u,(b>>3)&1u,(c>>3)&1u,(d>>3)&1u,
                               (e>>3)&1u,(f>>3)&1u,(g>>3)&1u,(h>>3)&1u);
    const unsigned negative=Xor8(unsigned(r.a<0),unsigned(r.b<0),
        unsigned(r.c<0),unsigned(r.d<0),unsigned(r.e<0),unsigned(r.f<0),
        unsigned(r.g<0),unsigned(r.h<0));
    const I8 q{r.a<0?r.a+4:r.a-4,r.b<0?r.b+4:r.b-4,
        r.c<0?r.c+4:r.c-4,r.d<0?r.d+4:r.d-4,r.e<0?r.e+4:r.e-4,
        r.f<0?r.f+4:r.f-4,r.g<0?r.g+4:r.g-4,r.h<0?r.h+4:r.h-4};
    const I8 integer=Repair(r,parity),half=Repair(q,parity^negative);
    const int id=Sum8(integer.a*integer.a,integer.b*integer.b,
        integer.c*integer.c,integer.d*integer.d,integer.e*integer.e,
        integer.f*integer.f,integer.g*integer.g,integer.h*integer.h);
    const int hd=Sum8(half.a*half.a,half.b*half.b,half.c*half.c,half.d*half.d,
                     half.e*half.e,half.f*half.f,half.g*half.g,half.h*half.h);
    const bool choose=(hd<id)|((hd==id)&(half.a<integer.a));
    return {choose?half.a:integer.a,choose?half.b:integer.b,
        choose?half.c:integer.c,choose?half.d:integer.d,choose?half.e:integer.e,
        choose?half.f:integer.f,choose?half.g:integer.g,choose?half.h:integer.h};
}
E8P_FN I8 Decode(unsigned label) { return ShapeResidualBits(Project(label)); }
} // namespace e8_parallel
#undef E8P_FN
