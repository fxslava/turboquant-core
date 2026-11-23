#define E8_CPU_TEST
#include <routine_e8_bf16_dequant_simt.hpp>
#include <routine_e8_int8_dequant_simt.hpp>
#define ascend_micro parallel_api
#include "routines/routine_e8_bf16_dequant_simt.hpp"
#include "routines/routine_e8_int8_dequant_simt.hpp"
#undef ascend_micro
#include <array>
#include <cstdio>
#include <vector>
template<parallel_api::e8::TableLayout L>
static bool table_test() {
    using namespace parallel_api::e8;
    std::vector<std::int8_t> table(PartialTableFootprint<L>::TotalBytes,0);
    int zero[8];ascend_micro::e8::Project(0,zero);
    for(unsigned b=0;b<256;++b) {
        int lo[8],hi[8];ascend_micro::e8::Project(b,lo);ascend_micro::e8::Project(b<<8,hi);
        for(unsigned c=0;c<8;++c) {
            table[PartialOffset<L>(b,c)]=std::int8_t(lo[c]);
            table[PartialTableFootprint<L>::PartialBytes+PartialOffset<L>(b,c)]=std::int8_t(hi[c]-zero[c]);
        }
    }
    for(unsigned label=0;label<65536;++label) {
        int p[8],gold[8];CompactProject<L>(label,table.data(),p);
        ascend_micro::e8::Project(label,gold);
        for(unsigned c=0;c<8;++c)if(p[c]!=gold[c])return false;
        Shape(p);ascend_micro::e8::Shape(gold);
        for(unsigned c=0;c<8;++c)if(p[c]!=gold[c])return false;
    }
    return true;
}
int main() {
    for(unsigned label=0;label<65536;++label) {
        int a[8],b[8],c[8];
        ascend_micro::e8::Project(label,a);ascend_micro::e8::Shape(a);
        parallel_api::e8::Project(label,b);parallel_api::e8::Shape(b);
        parallel_api::e8_int8_simt::detail::Project(label,c);
        parallel_api::e8_int8_simt::detail::ShapeResidualBits(c);
        for(unsigned coordinate=0;coordinate<8;++coordinate)
            if(a[coordinate]!=b[coordinate]||a[coordinate]!=c[coordinate])return 1;
    }
    using parallel_api::e8::TableLayout;
    if(!table_test<TableLayout::AoS>()||!table_test<TableLayout::SoA>()||
       !table_test<TableLayout::PaddedSoA>()||!table_test<TableLayout::Blocked>())return 2;
    std::puts("PASS publicBF16+INT8 API labels65536; Arithmetic and4CompactLayouts; LegacyNearest=ResidualBits; zero mismatches");
}
