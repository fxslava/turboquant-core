#include "camodel_common.hpp"
#include "e8_packet_v1.hpp"
#include "e8_packet_layout.hpp"
#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
extern "C" void launch_e8_packet_v1_scatter(void*, std::uint8_t*, std::uint8_t*,
                                           std::uint8_t*, std::uint32_t);
namespace packet = e8_packet_v1;
static_assert(packet::kRows == e8_packet_abi::kRows);
static_assert(packet::kPacketBytes == e8_packet_abi::kPacketBytes);
static_assert(packet::kScaleOffset == e8_packet_abi::kScaleOffset);
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
std::vector<std::uint8_t> load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot read " + path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
void save(const std::string& path, const void* data, std::size_t bytes) {
    std::ofstream file(path, std::ios::binary);
    file.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    require(bool(file), "Cannot save packet evidence");
}
int main(int argc, char** argv) {
  try {
    require(argc == 2, "Pass frozen E8 fixture directory");
    const std::string fixture = argv[1];
    const auto indices = load(fixture + "/bf16_indices.u16le");
    const auto scaleBits = load(fixture + "/bf16_scales.f32le");
    require(indices.size() == 2048 && scaleBits.size() == 128, "Frozen fixture sizes changed");
    camodel::Session session;
    camodel::GuardedDeviceBuffer labels(64), scale(32);
    camodel::GuardedDeviceBuffer key(packet::kPacketBytes), value(packet::kPacketBytes);
    std::array<camodel::GuardedDeviceBuffer*, 2> output{&key, &value};
    std::array<std::vector<std::uint8_t>, 2> expected;
    for (unsigned plane = 0; plane < 2; ++plane) {
      auto* initial = static_cast<std::uint8_t*>(output[plane]->host_payload());
      expected[plane] = {initial, initial+packet::kPacketBytes};
    }
    constexpr std::array<std::uint32_t,3> rows{0,31,63};
    std::uint64_t compared = 0;
    unsigned launches = 0;
    for (unsigned position = 0; position < rows.size(); ++position) {
      for (unsigned plane = 0; plane < 2; ++plane) {
        const unsigned fixtureRow = (position*11 + plane*7) % 32;
        std::memcpy(labels.host_payload(),indices.data()+fixtureRow*64,64);
        std::memcpy(scale.host_payload(),scaleBits.data()+fixtureRow*4,4);
        const auto sourceLabels = labels.host;
        const auto sourceScale = scale.host;
        packet::Labels input{};
        for (unsigned i = 0; i < 32; ++i)
          input[i] = std::uint16_t(indices[fixtureRow*64+2*i]) |
                     (std::uint16_t(indices[fixtureRow*64+2*i+1]) << 8);
        float inputScale = 0;
        std::memcpy(&inputScale, scaleBits.data()+fixtureRow*4,4);
        packet::Scatter(expected[plane].data(),packet::kPacketBytes,rows[position],input,inputScale);
        labels.upload(); scale.upload();
        launch_e8_packet_v1_scatter(session.stream,
          static_cast<std::uint8_t*>(labels.payload()),
          static_cast<std::uint8_t*>(scale.payload()),
          static_cast<std::uint8_t*>(output[plane]->payload()), rows[position]);
        camodel::check(aclrtSynchronizeStream(session.stream), "packet scatter sync");
        ++launches;
        for (unsigned p = 0; p < 2; ++p) {
          output[p]->download(); output[p]->assert_guards();
          const auto* got = static_cast<const std::uint8_t*>(output[p]->host_payload());
          for (std::size_t byte = 0; byte < packet::kPacketBytes; ++byte) {
            if (got[byte] != expected[p][byte]) {
              std::cerr << "Mismatch launch=" << launches << " plane=" << p
                        << " byte=" << byte << " got=" << unsigned(got[byte])
                        << " expected=" << unsigned(expected[p][byte]) << '\n';
              throw std::runtime_error("Full 4352B packet mismatch");
            }
            ++compared;
          }
        }
        labels.download(); scale.download(); labels.assert_guards(); scale.assert_guards();
        require(labels.host == sourceLabels && scale.host == sourceScale,
                "Kernel modified input labels/scale");
        std::cout << "PASS plane=" << plane << " row=" << rows[position]
                  << " both_packets=8704B all_unowned_bytes_and_guards_unchanged\n";
      }
    }
    save("key_packet.bin",key.host_payload(),packet::kPacketBytes);
    save("value_packet.bin",value.host_payload(),packet::kPacketBytes);
    save("key_expected.bin",expected[0].data(),packet::kPacketBytes);
    save("value_expected.bin",expected[1].data(),packet::kPacketBytes);
    std::ofstream report("packet_result.json");
    report << "{\"scope\":\"encoded D256 label+FP32 scale packet transport\","
           << "\"camodel_launches\":" << launches << ",\"packet_bytes\":4352,"
           << "\"planes\":2,\"rows\":[0,31,63],\"compared_packet_bytes\":" << compared
           << ",\"byte_mismatches\":0,\"guards\":\"PASS\","
           << "\"raw_encoder_tested\":false,\"fused_attention_tested\":false}\n";
    require(bool(report), "Cannot save result report");
    std::cout << "PASS E8 packet CAModel: launches=" << launches
              << " bytes=" << compared << " mismatches=0 guards=PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n'; return 1;
  }
}
