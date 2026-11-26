#pragma once
#include <cstddef>
#include <cstdint>
#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#endif

#include "e8_packet_layout.hpp"

namespace e8_decode {
constexpr std::uint32_t kRows = 64, kHalfRows = 32, kChannels = 256;
constexpr std::uint32_t kHeads = 6, kLocalHeads = 3, kPaddedHeads = 16;
constexpr std::uint32_t kRing = 2, kMaxTiles = 256, kPacketBytes = e8_packet_abi::kPacketBytes;
constexpr std::uint32_t kMaxSplits = 8, kPartialTail = 16, kPartialStride = kChannels + kPartialTail;
constexpr std::uint32_t kMaxGrid = 65535;
constexpr std::uint32_t kPad = 256;
constexpr std::uint32_t Round32(std::uint32_t n) { return (n + 31u) & ~31u; }
constexpr std::uint32_t kLabelAllocation = 2048 + kPad;
constexpr std::uint32_t kScaleAllocation = 128 + kPad;
constexpr std::uint32_t kPackedAllocation = 32768 + kPad;
constexpr std::uint32_t kKeyGroupStrideBytes = 1056;
constexpr std::uint32_t kKeyPackedAllocation = 32 * kKeyGroupStrideBytes + kPad;
constexpr std::uint32_t kKeyPlaneAllocation = kLabelAllocation + kScaleAllocation + kKeyPackedAllocation;
constexpr std::uint32_t kPlaneAllocation = kLabelAllocation + kScaleAllocation + kPackedAllocation;
constexpr std::uint32_t kSlotAllocation = kKeyPlaneAllocation + kPlaneAllocation + 32;
constexpr std::uint32_t kUbScores = 2 * kSlotAllocation;
constexpr std::uint32_t kUbContext = kUbScores + 768 + kPad;
constexpr std::uint32_t kUbAccumulator = kUbContext + 3072 + kPad;
constexpr std::uint32_t kUbProbabilities = kUbAccumulator + 3072 + kPad;
constexpr std::uint32_t kUbMaximum = kUbProbabilities + 768 + kPad;
constexpr std::uint32_t kUbSum = kUbMaximum + 96 + kPad;
constexpr std::uint32_t kUbOutput = kUbSum + 96 + kPad;
constexpr std::uint32_t kUbOffsets = kUbOutput + 1536 + kPad;
constexpr std::uint32_t kUbStorageBytes = kUbOffsets + kMaxTiles * 8 + kPad;
static_assert(kUbStorageBytes == 158464);
static_assert(kUbStorageBytes <= 192u * 1024u);
static_assert(256u * 1024u - kUbStorageBytes - 8u * 1024u >= 32u * 1024u);
constexpr std::uint32_t kL1Plane = 65536 + 32;
constexpr std::uint32_t kL1Slot = 2 * kL1Plane;
constexpr std::uint32_t kL1Query = 2 * kL1Slot;
constexpr std::uint32_t kL1Probabilities = kL1Query + 16384 + 32;
constexpr std::uint32_t kL1Bytes = kL1Probabilities + 4096 + 32;
static_assert(kL1Bytes == 282816 && kL1Bytes <= 512u * 1024u);
constexpr std::uint32_t PlaneBase(std::uint32_t slot, std::uint32_t plane) {
    return slot * kSlotAllocation + plane * kKeyPlaneAllocation;
}
constexpr std::uint32_t Labels(std::uint32_t s, std::uint32_t p) { return PlaneBase(s, p); }
constexpr std::uint32_t Scales(std::uint32_t s, std::uint32_t p) { return Labels(s, p) + kLabelAllocation; }
constexpr std::uint32_t Packed(std::uint32_t s, std::uint32_t p) { return Scales(s, p) + kScaleAllocation; }
constexpr std::uint32_t KeyPackedIndex(std::uint32_t row, std::uint32_t channel) {
    return (channel / 8) * (kKeyGroupStrideBytes / 4) + row * 8 + channel % 8;
}
constexpr std::uint32_t ValuePackedIndex(std::uint32_t row, std::uint32_t channel) {
    return (row / 8) * 256 * 8 + channel * 8 + row % 8;
}

struct Launch {
    std::uint32_t tiles = 6;
    std::uint32_t last_rows = 32;
    float attention_scale = 0.0625f;
    std::uint32_t lanes = 1, groups = 1, splits = 1, context_tiles = 0;
};
struct SplitRange {
    std::uint32_t first_tile, tiles, last_rows;
};
constexpr SplitRange split_range(std::uint32_t length, std::uint32_t block,
                                 std::uint32_t splits, std::uint32_t split) {
    if ((block != 64 && block != 128) || !splits || splits > kMaxSplits || split >= splits)
        return {0, 0, 0};
    const auto pages = (std::uint64_t{length} + block - 1) / block;
    const auto first = pages * split / splits * block;
    const auto limit = pages * (split + 1) / splits * block;
    const auto begin = first < length ? first : length;
    const auto end = limit < length ? limit : length;
    const auto tiles = static_cast<std::uint32_t>((end - begin + kRows - 1) / kRows);
    return {static_cast<std::uint32_t>(first / kRows), tiles,
            tiles ? static_cast<std::uint32_t>(end - begin - (tiles - 1) * kRows) : 0};
}
struct alignas(32) Task {
    std::uint64_t offsets_index, witness_tile, score_tile;
    std::uint32_t tiles, last_rows, lane, reserved = 0;
};
static_assert(sizeof(Task) == 64);
constexpr std::uint64_t workspace_floats(std::uint32_t lanes, std::uint32_t groups, std::uint32_t splits) {
    return splits == 1 ? 0 : std::uint64_t{lanes} * groups * kHeads * splits * kPartialStride;
}
#if defined(__CCE_AICORE__)
__aicore__
#endif
constexpr std::uint64_t partial_offset(std::uint64_t head, std::uint32_t split, std::uint32_t splits) {
    return (head * splits + split) * kPartialStride;
}
struct Buffers {
    void *key_packets, *value_packets, *packet_offsets, *query_padded;
    void *output, *key_witness, *value_witness, *score_witness;
    void *tasks = nullptr, *workspace = nullptr;
};
constexpr bool valid_launch(const Launch& c, const Buffers& b) {
    return c.lanes && c.groups && c.splits && c.splits <= kMaxSplits &&
        std::uint64_t{c.lanes} * c.splits <= kMaxGrid &&
        std::uint64_t{c.lanes} * c.groups <= kMaxGrid / kHeads &&
        c.tiles <= kMaxTiles && c.last_rows && c.last_rows <= kRows &&
        c.attention_scale > 0 && c.attention_scale <= 3.4028235e38f &&
        b.key_packets && b.value_packets && b.packet_offsets && b.query_padded && b.output &&
        (c.splits == 1 || b.workspace) &&
        (b.tasks ? c.context_tiles >= c.tiles : (c.lanes == 1 && c.groups == 1 && c.splits == 1 && c.tiles));
}
}
extern "C" void launch_e8_fused_decode_fp32(void* stream, const e8_decode::Launch* launch,
                                            const e8_decode::Buffers* buffers);
