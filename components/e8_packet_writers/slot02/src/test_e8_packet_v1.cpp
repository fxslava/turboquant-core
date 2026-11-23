#include "e8_packet_v1.hpp"
#define E8_CPU_TEST
#include "routine_e8_bf16_dequant_simt.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace packet = e8_packet_v1;
void Require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
std::vector<std::uint8_t> Load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot read fixture " + path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
std::uint16_t U16(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    return std::uint16_t(bytes.at(offset)) | (std::uint16_t(bytes.at(offset+1)) << 8);
}
float F32(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    std::uint32_t bits = 0;
    for (unsigned i = 0; i < 4; ++i) bits |= std::uint32_t(bytes.at(offset+i)) << (8*i);
    float result = 0;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
std::uint16_t Bf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
std::array<std::uint16_t, packet::kChannels> Decode(
    const std::uint8_t* bytes, std::size_t row) {
    const auto labels = packet::ReadLabels(bytes, row);
    const float scale = packet::ReadScale(bytes, row);
    std::array<std::uint16_t, packet::kChannels> output{};
    for (std::size_t block = 0; block < labels.size(); ++block) {
        int doubled[8];
        ascend_micro::e8::Project(labels[block], doubled);
        ascend_micro::e8::ShapeResidualBits(doubled);
        for (unsigned coordinate = 0; coordinate < 8; ++coordinate)
            output[block*8 + coordinate] = Bf16((float(doubled[coordinate])*0.5f)*scale);
    }
    return output;
}
int main(int argc, char** argv) {
    try {
        Require(argc == 2, "Pass frozen e8_wave1_goldens directory");
        const std::string root = argv[1];
        const auto indices = Load(root + "/bf16_indices.u16le");
        const auto scales = Load(root + "/bf16_scales.f32le");
        const auto golden = Load(root + "/bf16_dequant.bf16le");
        Require(indices.size() == 2048 && scales.size() == 128 && golden.size() == 16384,
                "Frozen fixture sizes changed");
        // Two adjacent plane packets, distinct prefix/suffix guards. Every byte outside
        // a row's owned label/scale addresses must survive each individual write.
        constexpr std::size_t guard = 32;
        alignas(32) std::array<std::uint8_t, 2*packet::kPacketBytes+2*guard> arena{};
        arena.fill(0x5a);
        auto* key = arena.data()+guard;
        auto* value = key+packet::kPacketBytes;
        std::uint64_t compared = 0;
        for (std::size_t turn = 0; turn < packet::kRows; ++turn) {
            const std::size_t row = (turn*17) % packet::kRows;
            packet::Labels labels{};
            for (std::size_t i = 0; i < labels.size(); ++i)
                labels[i] = U16(indices, ((row%32)*32+i)*2);
            const auto before = arena;
            packet::Scatter(key, packet::kPacketBytes, row, labels, F32(scales, (row%32)*4));
            const std::size_t lo = guard + row*64;
            const std::size_t scaleLo = guard+packet::kScaleOffset+row*4;
            for (std::size_t i = 0; i < arena.size(); ++i)
                if (!((i>=lo && i<lo+64) || (i>=scaleLo && i<scaleLo+4)))
                    Require(arena[i] == before[i], "Scatter clobbered unowned byte");
            Require(packet::ReadLabels(key,row) == labels, "Label bit transport failed");
            const auto decoded = Decode(key,row);
            for (std::size_t channel = 0; channel < decoded.size(); ++channel) {
                Require(decoded[channel] == U16(golden, ((row%32)*256+channel)*2),
                        "Packet reader disagrees with frozen BF16 golden");
                ++compared;
            }
        }
        // Exercise every possible uint16 label and both byte positions, repeated writes,
        // all row positions, and independent value-plane ownership.
        for (std::uint32_t first = 0; first < 65536; first += 32) {
            packet::Labels labels{};
            for (unsigned i = 0; i < 32; ++i) labels[i] = static_cast<std::uint16_t>(first+i);
            const std::size_t row = (first/32)%64;
            const auto before = arena;
            packet::Scatter(value, packet::kPacketBytes, row, labels, 1.0f);
            Require(packet::ReadLabels(value,row) == labels, "Exhaustive LE label transport failed");
            Require(std::equal(key, key+packet::kPacketBytes, before.data()+guard),
                    "Value write changed key packet");
            Require(value[row*64] == std::uint8_t(first) &&
                    value[row*64+1] == std::uint8_t(first>>8), "Explicit little endian failed");
        }
        const auto beforeInvalid = arena;
        bool rejected = false;
        try { packet::Scatter(key, packet::kPacketBytes, 64, packet::Labels{}, 1.0f); }
        catch (const std::out_of_range&) { rejected = true; }
        Require(rejected && arena == beforeInvalid, "Invalid row changed packet");
        rejected = false;
        try { packet::Scatter(key, packet::kPacketBytes-1, 0, packet::Labels{}, 1.0f); }
        catch (const std::invalid_argument&) { rejected = true; }
        Require(rejected && arena == beforeInvalid, "Truncated packet changed bytes");
        // Transport preserves FP32 payload bits; it does not silently normalize NaNs,
        // signed zeros, subnormals or scale policy. This test does not decode them.
        for (std::uint32_t bits : {0u, 0x80000000u, 1u, 0x7f800000u, 0x7fc01234u}) {
            float scale = 0; std::memcpy(&scale,&bits,4);
            packet::Scatter(value,packet::kPacketBytes,63,packet::Labels{},scale);
            const float restored = packet::ReadScale(value,63);
            std::uint32_t back = 0; std::memcpy(&back,&restored,4);
            Require(back == bits, "Scale payload changed");
        }
        Require(std::all_of(arena.begin(),arena.begin()+guard,[](auto b){return b==0x5a;}),
                "Prefix guard changed");
        Require(std::all_of(arena.end()-guard,arena.end(),[](auto b){return b==0x5a;}),
                "Suffix guard changed");
        std::cout << "PASS packet scatter: 64 row placements, all 65536 labels, K/V isolation, "
                  << "scale bits, guards; frozen BF16 comparisons=" << compared
                  << ", mismatches=0 (CPU only, not fused attention)\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n'; return 1;
    }
}
