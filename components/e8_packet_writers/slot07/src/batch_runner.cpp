#include "camodel_common.hpp"
#include "batch_fixture.hpp"
#include <array>
#include <iostream>
extern "C" void launch_e8_packet_v1_batch_encoded(void*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint8_t*,uint32_t,uint32_t);
void save(const char* path,const void* bytes,std::size_t size) {
 std::ofstream f(path,std::ios::binary); f.write(static_cast<const char*>(bytes),static_cast<std::streamsize>(size));
 if(!f) throw std::runtime_error("Could not save packet evidence");
}
int main(int argc,char**argv) {
 try {
  if(argc!=2) throw std::runtime_error("Pass fixture directory");
  const auto key=batch_fixture::Build(argv[1],0),value=batch_fixture::Build(argv[1],1);
  camodel::Session session;
  camodel::GuardedDeviceBuffer kl(key.labels.size()),ks(key.scales.size()),vl(value.labels.size()),vs(value.scales.size());
  camodel::GuardedDeviceBuffer kp(key.expected.size()),vp(value.expected.size());
  std::memcpy(kl.host_payload(),key.labels.data(),key.labels.size());
  std::memcpy(ks.host_payload(),key.scales.data(),key.scales.size());
  std::memcpy(vl.host_payload(),value.labels.data(),value.labels.size());
  std::memcpy(vs.host_payload(),value.scales.data(),value.scales.size());
  kl.upload();ks.upload();vl.upload();vs.upload();
  const std::array<std::vector<std::uint32_t>,4> inputBefore{kl.host,ks.host,vl.host,vs.host};
  launch_e8_packet_v1_batch_encoded(session.stream,static_cast<uint8_t*>(kl.payload()),static_cast<uint8_t*>(ks.payload()),
    static_cast<uint8_t*>(vl.payload()),static_cast<uint8_t*>(vs.payload()),static_cast<uint8_t*>(kp.payload()),
    static_cast<uint8_t*>(vp.payload()),6,2);
  camodel::check(aclrtSynchronizeStream(session.stream),"sixstrip E8 encoded batch sync");
  std::array<camodel::GuardedDeviceBuffer*,6> buffers{&kl,&ks,&vl,&vs,&kp,&vp};
  for(auto* b:buffers) {b->download();b->assert_guards();}
  for(unsigned i=0;i<4;++i) if(buffers[i]->host!=inputBefore[i]) throw std::runtime_error("Input changed");
  if(std::memcmp(kp.host_payload(),key.expected.data(),key.expected.size()) ||
     std::memcmp(vp.host_payload(),value.expected.data(),value.expected.size()))
    throw std::runtime_error("Full E8 batch packet bytes mismatch");
  save("key_packets.bin",kp.host_payload(),key.expected.size());save("value_packets.bin",vp.host_payload(),value.expected.size());
  save("key_expected.bin",key.expected.data(),key.expected.size());save("value_expected.bin",value.expected.data(),value.expected.size());
  std::ofstream report("packet_result.json");
  report<<"{\"scope\":\"encoded E8 batch packet transport\",\"strips\":6,\"ring_depth\":4,\"rows\":96,\"planes\":2,\"packets_per_plane\":2,\"packet_bytes\":4352,\"compared_bytes\":17408,\"byte_mismatches\":0,\"guards\":\"PASS\",\"raw_encoder\":false}\n";
  std::cout<<"PASS E8 encoded batch: sixstrips ring4 K/V17408fullpacketbytes,0mismatches,inputs/guards/last32rows preserved; rawencoder=false\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
