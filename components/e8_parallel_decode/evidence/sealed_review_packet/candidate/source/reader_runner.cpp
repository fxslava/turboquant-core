#include "e8_decode.hpp"
#include "e8_packet_layout.hpp"
#define E8_CPU_TEST
#include <routine_e8_bf16_dequant_simt.hpp>
#ifndef E8_READER_HOST_ONLY
#include "camodel_common.hpp"
#include "compiled_resources.hpp"
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr unsigned kTiles=6,kLast=32,kContext=352,kH=2,kHead=1,kPages=6;
constexpr unsigned kCacheBytes=kPages*2*kH*4352;
constexpr std::uint32_t kPoison=0x7fc0beef;
void Require(bool x,const char* why) { if(!x) throw std::runtime_error(why); }
std::vector<std::uint8_t> Read(const std::string& p) {
    std::ifstream f(p,std::ios::binary);
    if(!f) throw std::runtime_error("cannot read frozen fixture: "+p);
    return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
}
std::uint16_t U16(const std::uint8_t* p) { return p[0]|(std::uint16_t(p[1])<<8); }
float F32(const std::uint8_t* p) {
    const std::uint32_t b=p[0]|(std::uint32_t(p[1])<<8)|(std::uint32_t(p[2])<<16)|(std::uint32_t(p[3])<<24);
    float x; std::memcpy(&x,&b,4); return x;
}
std::uint32_t Bits(float x) { std::uint32_t b;std::memcpy(&b,&x,4);return b; }
std::uint16_t BF16(float x) {
    const auto b=Bits(x);return std::uint16_t((b+0x7fffu+((b>>16)&1u))>>16);
}
float Widen(std::uint16_t x) {
    const std::uint32_t b=std::uint32_t(x)<<16;
    float y;std::memcpy(&y,&b,4);return y;
}
void Put16(std::uint8_t* p,std::uint16_t x) { p[0]=x;p[1]=x>>8; }
void Put32(std::uint8_t* p,float x) {
    auto b=Bits(x);for(unsigned i=0;i<4;++i) p[i]=std::uint8_t(b>>(8*i));
}

struct Fixture {
    bool nonuniform;
    std::vector<std::uint8_t> labels_file,scales_file,golden_file;
    std::vector<std::uint8_t> key,value,key_after,value_after;
    std::array<std::uint64_t,8> offsets{};
    std::array<float,16*256> query{};
    std::array<std::uint16_t,32> update_k{},update_v{};
    float update_k_scale=0,update_v_scale=0;
    std::vector<std::uint16_t> key_ref,value_ref,output_ref;
    std::vector<float> score_ref;

    Fixture(const std::string& root,bool varied):nonuniform(varied),
        labels_file(Read(root+"/bf16_indices.u16le")),
        scales_file(Read(root+"/bf16_scales.f32le")),
        golden_file(Read(root+"/bf16_dequant.bf16le")),
        key(kCacheBytes),value(kCacheBytes),
        key_ref(6*64*256,0),value_ref(6*64*256,0),output_ref(6*256),
        score_ref(6*6*64,0) {
        Require(labels_file.size()==2048&&scales_file.size()==128&&golden_file.size()==16384,
            "frozen E8 fixture sizes changed");
        for(unsigned i=0;i<kCacheBytes;++i) key[i]=value[i]=std::uint8_t(kPoison>>(8*(i%4)));
        for(unsigned tile=0;tile<6;++tile) {
            const unsigned physical=2*(tile/2)+1;
            offsets[tile]=((physical*2+tile%2)*kH+kHead)*4352u;
            for(unsigned row=0;row<(tile==5?kLast:64);++row) {
                const unsigned ks=(tile*5+row*11+3)%32;
                const unsigned vs=varied?7:(tile*7+row*13+7)%32;
                WriteRow(key,offsets[tile],row,ks,0,key_ref,tile);
                WriteRow(value,offsets[tile],row,vs,0,value_ref,tile);
            }
        }
        key_after=key;value_after=value;
        // Two actual device writer calls update tile4,row31, then the reader
        // consumes the same cache allocations. This is encoded-label input.
        WriteRow(key_after,offsets[4],31,31,0,key_ref,4);
        WriteRow(value_after,offsets[4],31,varied?7:11,varied?0:6,value_ref,4);
        for(unsigned i=0;i<32;++i) {
            update_k[i]=U16(labels_file.data()+(31*32+i)*2);
            update_v[i]=U16(labels_file.data()+((varied?7:11)*32+i)*2);
        }
        update_k_scale=F32(scales_file.data()+31*4);
        update_v_scale=std::ldexp(F32(scales_file.data()+(varied?7:11)*4),varied?0:6);
        for(unsigned h=0;h<6;++h)
            if(varied) query[h*256+(h*37)%256]=std::ldexp(1.0f,-int(h%3));
        BuildAttentionReference();
    }
    void WriteRow(std::vector<std::uint8_t>& cache,std::uint64_t offset,unsigned row,
        unsigned source,int exponent,std::vector<std::uint16_t>& expected,unsigned tile) {
        for(unsigned b=0;b<32;++b)
            Put16(cache.data()+offset+row*64+b*2,U16(labels_file.data()+(source*32+b)*2));
        Put32(cache.data()+offset+4096+row*4,
            std::ldexp(F32(scales_file.data()+source*4),exponent));
        for(unsigned d=0;d<256;++d) {
            const float frozen=Widen(U16(golden_file.data()+(source*256+d)*2));
            expected[(tile*64+row)*256+d]=BF16(std::ldexp(frozen,exponent));
        }
    }
    void BuildAttentionReference() {
        std::array<double,kContext> logits{},weights{};
        for(unsigned h=0;h<6;++h) {
            double maximum=-1e300;
            for(unsigned row=0;row<kContext;++row) {
                double score=0;
                for(unsigned d=0;d<256;++d)
                    score+=double(query[h*256+d])*Widen(key_ref[row*256+d]);
                score_ref[((row/64)*6+h)*64+row%64]=static_cast<float>(score);
                logits[row]=score*0.0625;
                maximum=std::max(maximum,logits[row]);
            }
            double denominator=0;
            for(unsigned row=0;row<kContext;++row) {
                weights[row]=std::exp(logits[row]-maximum);denominator+=weights[row];
            }
            for(unsigned d=0;d<256;++d) {
                double numerator=0;
                for(unsigned row=0;row<kContext;++row)
                    numerator+=weights[row]*Widen(value_ref[row*256+d]);
                output_ref[h*256+d]=BF16(static_cast<float>(numerator/denominator));
            }
        }
    }
    void HostCheck() const {
        // Frozen BF16 outputs are the independent boundary oracle. Helpers
        // are used only to audit the packet fixture, never to replace device input.
        for(unsigned plane=0;plane<2;++plane) {
            const auto& cache=plane?value_after:key_after;
            const auto& expected=plane?value_ref:key_ref;
            for(unsigned tile=0;tile<6;++tile)
                for(unsigned row=0;row<(tile==5?kLast:64);++row) {
                    const auto* p=cache.data()+offsets[tile];
                    const float scale=F32(p+4096+row*4);
                    for(unsigned block=0;block<32;++block) {
                        int residual[8];
                        ascend_micro::e8::Project(U16(p+row*64+block*2),residual);
                        ascend_micro::e8::ShapeResidualBits(residual);
                        for(unsigned c=0;c<8;++c)
                            Require(BF16((float(residual[c])*0.5f)*scale)==
                                expected[(tile*64+row)*256+block*8+c],
                                "packet reconstruction disagrees with archived BF16 golden");
                    }
                }
        }
        for(unsigned i=6*256;i<query.size();++i) Require(query[i]==0,"query padding not zero");
        std::printf("PASS E8 encoded-input host fixture: six tiles,352rows,one logical KV task, "
            "outerH2/head1/odd physical pages, fixture=%s,staticUB=%u,L1=%u\n",
            nonuniform?"nonuniform-QK/constant-V":"uniform-score/varying-V",
            e8_decode::kUbStorageBytes,e8_decode::kL1Bytes);
    }
};

#ifndef E8_READER_HOST_ONLY
extern "C" void launch_e8_packet_v1_scatter(void*,std::uint8_t*,std::uint8_t*,std::uint8_t*,std::uint32_t);
using Buffer=camodel::GuardedDeviceBuffer;
template<class T> void Upload(Buffer& b,const T* p,std::size_t n) {
    Require(n*sizeof(T)<=b.payload_bytes,"upload bounds");
    std::fill(b.host.begin()+16,b.host.end()-16,kPoison);
    std::memcpy(b.host_payload(),p,n*sizeof(T));b.upload();
}
void Fill(Buffer& b) {
    std::fill(b.host.begin()+16,b.host.end()-16,kPoison);b.upload();
}
void CheckBits(const Buffer& b,const std::vector<std::uint16_t>& expected,const char* name) {
    const auto* p=static_cast<const std::uint16_t*>(const_cast<Buffer&>(b).host_payload());
    for(std::size_t i=0;i<expected.size();++i)
        if(p[i]!=expected[i]) {
            std::fprintf(stderr,"%s mismatch at%zu: got%04x expected%04x\n",name,i,p[i],expected[i]);
            throw std::runtime_error(name);
        }
}
int Run(const Fixture& f) {
    camodel::Session session;
    Require(parallel_resources::kAudited,"compiled resources missing");
    std::size_t privateLimit=0,divergenceLimit=0;
    camodel::check(aclrtDeviceGetLimit(ACL_RT_DEV_LIMIT_SIMT_STACK_SIZE,&privateLimit),"reader private stack query");
    camodel::check(aclrtDeviceGetLimit(ACL_RT_DEV_LIMIT_SIMT_DVG_WARP_STACK_SIZE,&divergenceLimit),"reader divergence query");
    Require(privateLimit%32==0&&privateLimit/32>=parallel_resources::kReaderFrame,
            "reader private stack capacity");
    Require(divergenceLimit>=parallel_resources::kReaderDivergence,"reader divergence capacity");
    std::printf("SIMT reader limits privateWarp=%zu divergenceWarp=%zu\n",privateLimit,divergenceLimit);

    Buffer keys(kCacheBytes),values(kCacheBytes),offsets(64),query(16*256*4),
        klabels(64),vlabels(64),kscale(32),vscale(32),
        output(8*256*2),kw(6*64*256*2),vw(6*64*256*2),scores(6*6*64*4);
    Upload(keys,f.key.data(),f.key.size());Upload(values,f.value.data(),f.value.size());
    Upload(offsets,f.offsets.data(),f.offsets.size());Upload(query,f.query.data(),f.query.size());
    Upload(klabels,f.update_k.data(),32);Upload(vlabels,f.update_v.data(),32);
    Upload(kscale,&f.update_k_scale,1);Upload(vscale,&f.update_v_scale,1);
    const std::array<Buffer*,6> inputs{&offsets,&query,&klabels,&vlabels,&kscale,&vscale};
    std::vector<std::vector<std::uint32_t>> snapshots;
    for(auto* b:inputs) snapshots.push_back(b->host);
    Fill(output);Fill(kw);Fill(vw);Fill(scores);
    launch_e8_packet_v1_scatter(session.stream,static_cast<std::uint8_t*>(klabels.payload()),
        static_cast<std::uint8_t*>(kscale.payload()),static_cast<std::uint8_t*>(keys.payload())+f.offsets[4],31);
    launch_e8_packet_v1_scatter(session.stream,static_cast<std::uint8_t*>(vlabels.payload()),
        static_cast<std::uint8_t*>(vscale.payload()),static_cast<std::uint8_t*>(values.payload())+f.offsets[4],31);
    camodel::check(aclrtSynchronizeStream(session.stream),"encoded writer synchronize");
    keys.download();values.download();keys.assert_guards();values.assert_guards();
    Require(std::memcmp(keys.host_payload(),f.key_after.data(),kCacheBytes)==0,"K packet transport");
    Require(std::memcmp(values.host_payload(),f.value_after.data(),kCacheBytes)==0,"V packet transport");
    // No upload or host-decoded cache substitution after these actual writes.
    const e8_decode::Launch launch{6,32,0.0625f};
    const e8_decode::Buffers buffers{keys.payload(),values.payload(),offsets.payload(),query.payload(),
        output.payload(),kw.payload(),vw.payload(),scores.payload()};
    std::printf("launch encoded writer->actual E8 fused reader:6tiles352rows6heads,fixture=%s\n",
        f.nonuniform?"nonuniform":"uniform");
    launch_e8_fused_decode_fp32(session.stream,&launch,&buffers);
    camodel::check(aclrtSynchronizeStream(session.stream),"E8 fused reader synchronize");
    output.download();kw.download();vw.download();scores.download();
    output.assert_guards();kw.assert_guards();vw.assert_guards();scores.assert_guards();
    CheckBits(output,f.output_ref,"attention BF16");
    CheckBits(kw,f.key_ref,"K reconstruction BF16");
    CheckBits(vw,f.value_ref,"V reconstruction BF16");
    const auto* actual_scores=static_cast<const float*>(scores.host_payload());
    for(unsigned i=0;i<f.score_ref.size();++i)
        if(Bits(actual_scores[i])!=Bits(f.score_ref[i])) {
            std::fprintf(stderr,"QK score mismatch at%u got%g expected%g\n",i,
                double(actual_scores[i]),double(f.score_ref[i]));
            throw std::runtime_error("actual FP32 Cube QK witness");
        }
    const auto* raw=static_cast<const std::uint32_t*>(output.host_payload());
    for(unsigned i=6*256/2;i<output.payload_bytes/4;++i) Require(raw[i]==kPoison,"output inactive tail");
    for(unsigned i=0;i<inputs.size();++i) {
        inputs[i]->download();inputs[i]->assert_guards();
        Require(inputs[i]->host==snapshots[i],"read-only input changed");
    }
    keys.download();values.download();keys.assert_guards();values.assert_guards();
    Require(std::memcmp(keys.host_payload(),f.key_after.data(),kCacheBytes)==0,"reader changed K packets");
    Require(std::memcmp(values.host_payload(),f.value_after.data(),kCacheBytes)==0,"reader changed V packets");
    std::printf("PASS actual encoded scatter->E8 fused attention:0BF16ULP, "
        "K/V frozen BF16 witnesses exact,FP32 Cube QK witness exact,6tiles,352rows, "
        "all guards/input bytes/inactive output intact. Raw BF16 encoder NOT tested.\n");
    return 0;
}
#endif
} // namespace
int main(int argc,char** argv) {
    try {
        bool host=false,varied=false;
        std::string root="/mnt/c/ascend-micro-harness/routines/e8_wave1_goldens";
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--host-check") host=true;
            else if(arg=="--nonuniform") varied=true;
            else if(arg=="--golden-root"&&i+1<argc) root=argv[++i];
            else throw std::runtime_error("usage: e8_reader [--host-check] [--nonuniform] [--golden-root PATH]");
        }
        Fixture f(root,varied);f.HostCheck();
#ifdef E8_READER_HOST_ONLY
        (void)host;return 0;
#else
        return host?0:Run(f);
#endif
    } catch(const std::exception& error) {
        std::fprintf(stderr,"FAIL E8 fused reader:%s\n",error.what());return 1;
    }
}

