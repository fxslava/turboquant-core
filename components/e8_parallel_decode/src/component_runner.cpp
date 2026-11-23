#include <acl/acl.h>
#include "camodel_common.hpp"
#define E8_CPU_TEST
#include <routine_e8_bf16_dequant_simt.hpp>
#include <routine_e8_int8_dequant_simt.hpp>
#include <array>
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include "compiled_resources.hpp"
extern "C" void launch_e8_parallel_component(void*,std::uint16_t*,float*,std::uint16_t*,std::int8_t*);
static std::uint16_t bf16_rne(float x) {
    std::uint32_t bits=0;std::memcpy(&bits,&x,4);
    return std::uint16_t((bits+0x7fffU+((bits>>16)&1U))>>16);
}
int main()try {
    std::fesetround(FE_TONEAREST);
    camodel::Session session;
    if(!parallel_resources::kAudited)throw std::runtime_error("compiled resource audit missing");
    std::size_t privateLimit=0,divergenceLimit=0;
    camodel::check(aclrtDeviceGetLimit(ACL_RT_DEV_LIMIT_SIMT_STACK_SIZE,&privateLimit),"private stack query");
    camodel::check(aclrtDeviceGetLimit(ACL_RT_DEV_LIMIT_SIMT_DVG_WARP_STACK_SIZE,&divergenceLimit),"divergence query");
    // Final required thresholds are checked against compiled ELF before model
    // admission; these printed values are retained with the actual device run.
    std::printf("SIMT limits privateWarp=%zu divergenceWarp=%zu\n",privateLimit,divergenceLimit);
    if(privateLimit%32||privateLimit/32<parallel_resources::kComponentFrame||
       divergenceLimit<parallel_resources::kComponentDivergence)
        throw std::runtime_error("runtime SIMT capacity below compiled component requirement");
    camodel::GuardedDeviceBuffer labels(1024),scales(64),bf16(8192),int8(4096);
    auto* l=static_cast<std::uint16_t*>(labels.host_payload());
    auto* s=static_cast<float*>(scales.host_payload());
    const std::array<float,16> values{1.0f,-1.0f,0.5f,-0.5f,0.1f,127.0f,
        0x0.fffffep-126f,-0x0.fffffep-126f,0x1p-126f,-0x1p-126f,
        0x1p-132f,-0x1p-132f,0.0f,-0.0f,16.0f,0.03125f};
    for(unsigned i=0;i<512;++i)l[i]=std::uint16_t((i*127u+27306u)&65535u);
    l[0]=0;l[1]=65535;l[2]=27306;l[3]=52115;
    std::memcpy(s,values.data(),64);
    std::array<std::uint16_t,4096> goldenBf16{};
    std::array<std::int8_t,4096> goldenInt8{};
    for(unsigned i=0;i<512;++i) {
        int p[8];ascend_micro::e8::Project(l[i],p);ascend_micro::e8::ShapeResidualBits(p);
        for(unsigned c=0;c<8;++c) {
            const float x=(float(p[c])*0.5f)*s[i/32];
            goldenBf16[i*8+c]=bf16_rne(x);
            goldenInt8[i*8+c]=ascend_micro::e8_int8_simt::detail::PhysicalInt8(x);
        }
    }
    std::array<std::uint16_t,512> inputLabels{};std::memcpy(inputLabels.data(),l,1024);
    labels.upload();scales.upload();
    launch_e8_parallel_component(session.stream,static_cast<std::uint16_t*>(labels.payload()),
        static_cast<float*>(scales.payload()),static_cast<std::uint16_t*>(bf16.payload()),
        static_cast<std::int8_t*>(int8.payload()));
    camodel::check(aclrtSynchronizeStream(session.stream),"component synchronize");
    labels.download();scales.download();bf16.download();int8.download();
    labels.assert_guards();scales.assert_guards();bf16.assert_guards();int8.assert_guards();
    if(std::memcmp(labels.host_payload(),inputLabels.data(),1024)||std::memcmp(scales.host_payload(),values.data(),64))
        throw std::runtime_error("read-only input mismatch");
    const auto* actualBf16=static_cast<const std::uint16_t*>(bf16.host_payload());
    const auto* actualInt8=static_cast<const std::int8_t*>(int8.host_payload());
    unsigned bf16Mismatch=0,int8Mismatch=0;
    for(unsigned i=0;i<4096;++i) {
        if(actualBf16[i]!=goldenBf16[i]) {
            if(bf16Mismatch<8)std::fprintf(stderr,"BF16 mismatch %u actual%04x expected%04x scaleRow%u\n",i,actualBf16[i],goldenBf16[i],i/256);
            ++bf16Mismatch;
        }
        if(actualInt8[i]!=goldenInt8[i])++int8Mismatch;
    }
    if(bf16Mismatch||int8Mismatch)throw std::runtime_error("Frozen reference mismatch: BF16="+
        std::to_string(bf16Mismatch)+" INT8="+std::to_string(int8Mismatch));
    std::puts("PASS 4096/4096 BF16words0ULP;4096/4096 INT8bytesexact;inputs+512guardbytesintact;FP32/BF16subnormalboundaryscalesincluded");
    return 0;
}catch(const std::exception&e){std::fprintf(stderr,"FAIL %s\n",e.what());return 1;}
