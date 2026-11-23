#include "batch_fixture.hpp"
#include <array>
#include <iostream>
int main(int argc,char**argv) {
 try {
  if(argc!=2) throw std::runtime_error("Pass fixture directory");
  for(unsigned plane=0;plane<2;++plane) {
   const auto input=batch_fixture::Build(argv[1],plane);
   auto actual=input.expected;
   constexpr std::uint8_t canary[]={0xef,0xbe,0xad,0xde};
   for(std::size_t i=0;i<actual.size();++i) actual[i]=canary[i%4];
   std::array<int,4> free{1,1,1,1},ready{0,0,0,0};
   for(unsigned strip=0;strip<6;++strip) {
    const unsigned slot=strip%4;
    if(free[slot]!=1||ready[slot]!=0) throw std::runtime_error("Invalid slot ownership");
    --free[slot]; ++ready[slot]; --ready[slot];
    const unsigned packet=strip/4,row=(strip%4)*16;
    const unsigned labels=packet*4352+row*64;
    const unsigned scales=packet*4352+4096+row*4;
    const auto before=actual;
    std::memcpy(actual.data()+labels,input.labels.data()+strip*1024,1024);
    std::memcpy(actual.data()+scales,input.scales.data()+strip*64,64);
    for(std::size_t i=0;i<actual.size();++i)
     if(!((i>=labels&&i<labels+1024)||(i>=scales&&i<scales+64))&&actual[i]!=before[i])
      throw std::runtime_error("Strip changed unowned packet bytes");
    ++free[slot];
   }
   for(unsigned slot=0;slot<4;++slot) {
    --free[slot];
    if(free[slot]!=0||ready[slot]!=0) throw std::runtime_error("Epilogue leaked token");
   }
   if(actual!=input.expected) throw std::runtime_error("Strip DMA model differs from independent row packet ABI");
  }
  std::cout<<"PASS sixstrip ring4 token/ownership model: K/V separate2x4352B packets, wrap afterstrip3, last32rows preserved, zero leaked tokens; CPU only\n";
 } catch(const std::exception&e) {std::cerr<<e.what()<<'\n';return 1;}
}
