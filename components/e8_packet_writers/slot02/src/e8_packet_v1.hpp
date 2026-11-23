#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

// E8PacketV1: one D256 plane. K and V use separate packets.
// Labels: LE uint16 [64 token rows][32 E8 blocks], bytes [0,4096).
// Scales: LE IEEE754 FP32 [64 token rows], bytes [4096,4352).
namespace e8_packet_v1 {
constexpr std::size_t kRows = 64;
constexpr std::size_t kChannels = 256;
constexpr std::size_t kLabelsPerRow = kChannels / 8;
constexpr std::size_t kLabelRowBytes = kLabelsPerRow * 2;
constexpr std::size_t kScaleOffset = kRows * kLabelRowBytes;
constexpr std::size_t kPacketBytes = kScaleOffset + kRows * 4;
static_assert(kScaleOffset == 4096 && kPacketBytes == 4352);
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
using Labels = std::array<std::uint16_t, kLabelsPerRow>;

inline std::size_t LabelOffset(std::size_t row) {
    if (row >= kRows) throw std::out_of_range("E8 packet row must be below 64");
    return row * kLabelRowBytes;
}
inline void Scatter(std::uint8_t* packet, std::size_t bytes, std::size_t row,
                    const Labels& labels, float scale) {
    const std::size_t offset = LabelOffset(row);
    if (bytes != kPacketBytes || packet == nullptr)
        throw std::invalid_argument("E8 scatter requires one complete packet");
    for (std::size_t i = 0; i < labels.size(); ++i) {
        packet[offset + 2*i] = static_cast<std::uint8_t>(labels[i]);
        packet[offset + 2*i + 1] = static_cast<std::uint8_t>(labels[i] >> 8);
    }
    std::uint32_t bits = 0;
    std::memcpy(&bits, &scale, sizeof(bits));
    for (unsigned i = 0; i < 4; ++i)
        packet[kScaleOffset + 4*row + i] = static_cast<std::uint8_t>(bits >> (8*i));
}
inline Labels ReadLabels(const std::uint8_t* packet, std::size_t row) {
    const std::size_t offset = LabelOffset(row);
    Labels labels{};
    for (std::size_t i = 0; i < labels.size(); ++i)
        labels[i] = std::uint16_t(packet[offset+2*i]) |
                    (std::uint16_t(packet[offset+2*i+1]) << 8);
    return labels;
}
inline float ReadScale(const std::uint8_t* packet, std::size_t row) {
    (void)LabelOffset(row);
    std::uint32_t bits = 0;
    for (unsigned i = 0; i < 4; ++i)
        bits |= std::uint32_t(packet[kScaleOffset+4*row+i]) << (8*i);
    float scale = 0;
    std::memcpy(&scale, &bits, sizeof(scale));
    return scale;
}
}  // namespace e8_packet_v1
