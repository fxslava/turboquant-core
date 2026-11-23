#include "e8_packet_v1.hpp"
#define E8_CPU_TEST
#include "routine_e8_bf16_dequant_simt.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
std::vector<std::uint8_t> load(const std::string& p) {
  std::ifstream f(p,std::ios::binary);
  if (!f) throw std::runtime_error("Cannot read "+p);
  return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
}
int main(int argc,char** argv) {
  try {
    if (argc!=3) throw std::runtime_error("Pass run and fixture directories");
    const auto golden=load(std::string(argv[2])+"/bf16_dequant.bf16le");
    unsigned compared=0;
    for (unsigned p=0;p<2;++p) {
      const auto packet=load(std::string(argv[1])+(p?"/value_packet.bin":"/key_packet.bin"));
      if (packet.size()!=4352) throw std::runtime_error("Invalid device packet length");
      for (unsigned pos=0;pos<3;++pos) {
        constexpr unsigned rows[]={0,31,63};
        const unsigned source=(pos*11+p*7)%32;
        const auto labels=e8_packet_v1::ReadLabels(packet.data(),rows[pos]);
        const float scale=e8_packet_v1::ReadScale(packet.data(),rows[pos]);
        for (unsigned b=0;b<32;++b) {
          int q[8]; ascend_micro::e8::Project(labels[b],q);
          ascend_micro::e8::ShapeResidualBits(q);
          for (unsigned c=0;c<8;++c) {
            const float value=(float(q[c])*0.5f)*scale;
            std::uint32_t bits=0; std::memcpy(&bits,&value,4);
            const auto got=std::uint16_t((bits+0x7fffu+((bits>>16)&1u))>>16);
            const unsigned i=(source*256+b*8+c)*2;
            const auto want=std::uint16_t(golden.at(i))|(std::uint16_t(golden.at(i+1))<<8);
            if (got!=want) throw std::runtime_error("Device packet -> frozen leaf BF16 mismatch");
            ++compared;
          }
        }
      }
    }
    std::cout<<"PASS device packet -> unchanged frozen E8 leaf CPU: BF16_values="<<compared
             <<" mismatches=0 max_BF16_ULP=0 (CPU reconstruction, not attention)\n";
  } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
