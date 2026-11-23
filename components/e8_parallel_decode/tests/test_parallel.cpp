#define E8_CPU_TEST
#define E8_PARALLEL_CPU
#include "e8_parallel_decode.hpp"
#include "routine_e8_bf16_dequant_simt.hpp"
#include "routine_e8_int8_dequant_simt.hpp"
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>

static std::uint16_t bf16(float value) {
    std::uint32_t bits=0;std::memcpy(&bits,&value,4);
    if((bits&0x7fffffffU)>0x7f800000U)return std::uint16_t((bits>>16)|0x40U);
    return std::uint16_t((bits+0x7fffU+((bits>>16)&1U))>>16);
}
static std::int8_t physical(float value) {
    const float low=value< -128.0f?-128.0f:value;
    const float clipped=low>127.0f?127.0f:low;
    return static_cast<std::int8_t>(std::nearbyint(clipped));
}
int main() {
    std::fesetround(FE_TONEAREST);
    const std::array<float,24> scales{0.0f,-0.0f,0.000001f,0.03125f,0.1f,
        0.5f,1.0f,1.0009765625f,1.00390625f,1.5f,2.0f,16.0f,127.0f,
        -0.25f,-1.0f,0x1p-126f,0x1p-149f,0x1.fffffep127f,
        0x0.fffffep-126f,-0x0.fffffep-126f,-0x1p-126f,
        0x1p-132f,-0x1p-132f,-0x1p-149f};
    std::uint64_t coordinates=0,bf16words=0,int8words=0;
    for(unsigned label=0;label<65536;++label) {
        int reference[8],legacy[8],int8ref[8];
        ascend_micro::e8::Project(label,reference);
        ascend_micro::e8::Project(label,legacy);
        ascend_micro::e8_int8_simt::detail::Project(label,int8ref);
        ascend_micro::e8::ShapeResidualBits(reference);
        ascend_micro::e8::Shape(legacy);
        ascend_micro::e8_int8_simt::detail::ShapeResidualBits(int8ref);
        const auto q=e8_parallel::Decode(label);
        const int values[8]{q.a,q.b,q.c,q.d,q.e,q.f,q.g,q.h};
        for(unsigned c=0;c<8;++c) {
            if(values[c]!=reference[c]||values[c]!=legacy[c]||values[c]!=int8ref[c]) {
                std::fprintf(stderr,"FAIL label%u coordinate%u new%d residual%d legacy%d int8%d\n",
                    label,c,values[c],reference[c],legacy[c],int8ref[c]);return 1;
            }
            ++coordinates;
            for(float scale:scales) {
                const float expected=(float(reference[c])*0.5f)*scale;
                const float actual=(float(values[c])*0.5f)*scale;
                if(bf16(actual)!=bf16(expected))return 2;
                if(physical(actual)!=ascend_micro::e8_int8_simt::detail::PhysicalInt8(expected))return 3;
                ++bf16words;++int8words;
            }
        }
    }
    std::printf("PASS labels=65536 coordinates=%llu bf16_words=%llu int8_words=%llu mismatches=0\n",
        static_cast<unsigned long long>(coordinates),static_cast<unsigned long long>(bf16words),
        static_cast<unsigned long long>(int8words));
}
