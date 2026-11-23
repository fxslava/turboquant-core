#pragma once
#include "e8_packet_v1.hpp"
#include "e8_packet_layout.hpp"
#include <fstream>
#include <string>
#include <vector>
namespace batch_fixture {
constexpr unsigned kStrips=6, kRows=96, kPackets=2;
struct Plane {
 std::vector<std::uint8_t> labels,scales,expected;
};
inline std::vector<std::uint8_t> Load(const std::string& path) {
 std::ifstream file(path,std::ios::binary);
 if(!file) throw std::runtime_error("Cannot read "+path);
 return {std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};
}
inline Plane Build(const std::string& directory,unsigned plane) {
 static_assert(e8_packet_v1::kPacketBytes==e8_packet_abi::kPacketBytes);
 const auto labels=Load(directory+"/bf16_indices.u16le");
 const auto scales=Load(directory+"/bf16_scales.f32le");
 if(labels.size()!=2048||scales.size()!=128) throw std::runtime_error("Frozen fixture sizes changed");
 Plane result;
 result.labels.resize(kRows*64); result.scales.resize(kRows*4);
 result.expected.resize(kPackets*e8_packet_abi::kPacketBytes);
 constexpr std::uint8_t canary[]={0xef,0xbe,0xad,0xde};
 for(std::size_t i=0;i<result.expected.size();++i) result.expected[i]=canary[i%4];
 for(unsigned row=0;row<kRows;++row) {
  const unsigned source=(row*17+plane*7)%32;
  std::memcpy(result.labels.data()+row*64,labels.data()+source*64,64);
  std::memcpy(result.scales.data()+row*4,scales.data()+source*4,4);
  e8_packet_v1::Labels token{};
  for(unsigned i=0;i<32;++i) token[i]=std::uint16_t(labels[source*64+2*i])|
    (std::uint16_t(labels[source*64+2*i+1])<<8);
  float scale=0; std::memcpy(&scale,scales.data()+source*4,4);
  e8_packet_v1::Scatter(result.expected.data()+(row/64)*e8_packet_abi::kPacketBytes,
    e8_packet_abi::kPacketBytes,row%64,token,scale);
 }
 return result;
}
}
