/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// THE INGRESS REPACK REFERENCE: the host-side shim's contract, as the oracle a write-time
// NZ ingress is compared against.
//
// `repack_strip_to_nz_tiled` is the conversion the fused decode's cache needs when something
// else produced an RFC 3.1 strip: it READS the strip contract (tq_test::strip_nibble, and the
// plane-major scale index) and WRITES the NZ contract (NzTiledChannelByte,
// NzTiledChannelNibble, NzTiledScaleIndex). It is deliberately the slowest possible form --
// one logical element at a time, through the contract's own address functions -- because its
// only job is to be obviously right.
//
// WHY THAT MAKES IT AN ORACLE FOR THE KERNEL. The device writer does the same permutation as
// four DeInterleave passes, eight high-dimensional-split copies, one Interleave and one framed
// MTE3 descriptor, and derives its GM addresses from an open-coded base plus three strides.
// Nothing in those two derivations is shared: they agree only if both match the contract. So a
// byte-for-byte comparison against this reference catches a wrong group stride, a swapped
// nibble half, a transposed value plane and a mis-strided scale record -- each of which still
// produces a plausible-looking cache.
//
// `count_*` return a write census beside the image, so the reference itself is checkable: a
// repack whose destination addresses collide, or that leaves a byte of a mapped tile unwritten,
// is a broken oracle and must fail before it is used to judge a kernel.

#ifndef TURBOQUANT_NZ_INGRESS_HPP
#define TURBOQUANT_NZ_INGRESS_HPP

#include "turboquant_test_common.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tq_nz_ingress {

namespace tq = vllm_ascend::turboquant;
using tq_test::kPlanes;
using tq_test::kStripHeight;

// The NZ cache geometry of a launch. `block_size` must be a whole number of 64-row tiles;
// everything else is the same shape the strip writer has.
struct NzGeometry {
    int num_kv_heads = 4;
    int head_size = 256;
    int block_size = 64;
    int num_blocks = 4;

    int packed_bytes() const { return head_size / tq_test::kNibblesPerByte; }
    int scale_slot() const
    {
        return static_cast<int>(tq::NzScaleSlotFloats(static_cast<std::uint32_t>(num_kv_heads)));
    }
    // A physical block is num_kv_heads NZ planes a tile, each kCubeTileRows rows of packed
    // bytes -- the same total as the strip layout's page, laid out differently.
    std::size_t cache_bytes() const
    {
        return static_cast<std::size_t>(num_blocks) * block_size * num_kv_heads * packed_bytes();
    }
    std::size_t scale_floats() const
    {
        return static_cast<std::size_t>(num_blocks) * block_size * scale_slot();
    }
    bool valid() const
    {
        return num_kv_heads > 0 && num_blocks > 0 && head_size > 0 && block_size > 0 &&
               block_size % static_cast<int>(tq::kCubeTileRows) == 0 &&
               head_size % static_cast<int>(tq::kOperandC0) == 0 &&
               head_size % tq_test::kNibblesPerByte == 0;
    }
};

// Where (slot, kv head, channel) lives in the NZ packed cache, and which nibble of it.
struct NzNibble {
    std::size_t byte;
    int half;  // 0 = low nibble, 1 = high
};

inline NzNibble nz_nibble(const NzGeometry& g, int slot, int kv_head, int channel)
{
    return NzNibble{
        static_cast<std::size_t>(tq::NzTiledChannelByte(
            static_cast<std::uint64_t>(slot), static_cast<std::uint64_t>(kv_head),
            static_cast<std::uint64_t>(channel), static_cast<std::uint64_t>(g.num_kv_heads),
            static_cast<std::uint64_t>(g.head_size))),
        static_cast<int>(tq::NzTiledChannelNibble(static_cast<std::uint64_t>(channel),
                                                  static_cast<std::uint64_t>(g.head_size)))};
}

inline std::size_t nz_scale_index(const NzGeometry& g, int slot, int kv_head, int kv)
{
    return static_cast<std::size_t>(tq::NzTiledScaleIndex(
        static_cast<std::uint64_t>(slot), static_cast<std::uint64_t>(kv_head),
        static_cast<std::uint64_t>(kv), static_cast<std::uint64_t>(g.num_kv_heads)));
}

// One (strip, plane, kv head) panel, in the form the golden prefill reference produces it:
// `packed` is the RFC 3.1 strip page of ONE 16-token strip (kStripHeight * head_size / 2 bytes,
// this plane's layout) and `scale` its kStripHeight FP32 scales.
struct StripPanel {
    int slot0 = 0;    // GLOBAL cache row of the strip's first token
    int plane = 0;    // 0 = key, 1 = value
    int kv_head = 0;
    const std::uint8_t* packed = nullptr;
    const float* scale = nullptr;
};

// The destination images and the write census.
struct NzImage {
    std::vector<std::uint8_t> key, value;
    std::vector<float> scale;
    std::vector<std::uint32_t> count_packed;  // per cache byte, per plane summed
    std::vector<std::uint32_t> count_scale;   // per scale float

    explicit NzImage(const NzGeometry& g)
        : key(g.cache_bytes(), 0), value(g.cache_bytes(), 0), scale(g.scale_floats(), 0.0f),
          count_packed(g.cache_bytes(), 0), count_scale(g.scale_floats(), 0)
    {
    }
};

// The repack itself. The strip's height is kStripHeight, so a panel covers slots
// [slot0, slot0 + kStripHeight); the strip index inside its page never enters, because
// NzTiledChannelByte is defined on the global row.
inline void repack_strip_to_nz_tiled(const NzGeometry& g, const StripPanel& panel, NzImage& out)
{
    tq_test::require(g.valid(), "the NZ repack needs a whole number of 64-row tiles");
    tq_test::require(panel.packed != nullptr && panel.scale != nullptr,
                     "the NZ repack needs a strip panel");
    tq_test::require(panel.slot0 % kStripHeight == 0, "a strip starts on a multiple of R");

    tq_test::Shape strip;
    strip.num_kv_heads = g.num_kv_heads;
    strip.head_size = g.head_size;
    strip.block_size = kStripHeight;  // the panel IS one strip, so its page is one strip high
    strip.num_blocks = 1;

    std::vector<std::uint8_t>& dst = panel.plane == 0 ? out.key : out.value;
    for (int token = 0; token < kStripHeight; ++token) {
        const int slot = panel.slot0 + token;
        for (int channel = 0; channel < g.head_size; ++channel) {
            // Read: the RFC 3.1 nibble of (token, channel) inside this one-strip page.
            const std::size_t from =
                tq_test::strip_nibble(strip, panel.plane, 0, 0, token, channel);
            const std::uint8_t byte = panel.packed[from / tq_test::kNibblesPerByte];
            const std::uint8_t raw = (from % tq_test::kNibblesPerByte) == 0
                                         ? static_cast<std::uint8_t>(byte & 0x0Fu)
                                         : static_cast<std::uint8_t>((byte >> 4) & 0x0Fu);
            // Write: the kv4fp8 nibble of (slot, kv head, channel).
            const NzNibble to = nz_nibble(g, slot, panel.kv_head, channel);
            tq_test::require(to.byte < dst.size(), "the NZ repack ran off the cache");
            dst[to.byte] |= to.half == 0 ? raw : static_cast<std::uint8_t>(raw << 4);
            ++out.count_packed[to.byte];
        }
        const std::size_t lane = nz_scale_index(g, slot, panel.kv_head, panel.plane);
        tq_test::require(lane < out.scale.size(), "the NZ repack ran off the scale plane");
        out.scale[lane] = panel.scale[token];
        ++out.count_scale[lane];
    }
}

// The census gate. Over the slots a repack was asked to fill, every packed byte must have been
// written by exactly kNibblesPerByte logical elements (its two nibbles) and every scale lane by
// exactly one. `slots` is the set of global rows the caller repacked, in any order.
struct Census {
    std::size_t packed_bytes = 0;
    std::size_t scale_lanes = 0;
    bool packed_exact = true;
    bool scale_exact = true;
    bool outside_clean = true;  // nothing written outside the repacked slots
};

inline Census audit_repack(const NzGeometry& g, const NzImage& image, const std::vector<int>& slots)
{
    Census census;
    std::vector<char> owned(image.count_packed.size(), 0);
    std::vector<char> owned_scale(image.count_scale.size(), 0);
    for (const int slot : slots)
        for (int head = 0; head < g.num_kv_heads; ++head) {
            for (int channel = 0; channel < g.head_size; ++channel)
                owned[nz_nibble(g, slot, head, channel).byte] = 1;
            for (int kv = 0; kv < kPlanes; ++kv) owned_scale[nz_scale_index(g, slot, head, kv)] = 1;
        }
    for (std::size_t i = 0; i < owned.size(); ++i) {
        if (owned[i] == 0) {
            if (image.count_packed[i] != 0) census.outside_clean = false;
            continue;
        }
        // Both planes repack into their own image, so a byte owned by one (slot, head) is
        // written by its two nibbles in EACH plane: twice per plane, 2 * kPlanes in the sum.
        if (image.count_packed[i] !=
            static_cast<std::uint32_t>(tq_test::kNibblesPerByte * kPlanes))
            census.packed_exact = false;
        ++census.packed_bytes;
    }
    for (std::size_t i = 0; i < owned_scale.size(); ++i) {
        if (owned_scale[i] == 0) {
            if (image.count_scale[i] != 0) census.outside_clean = false;
            continue;
        }
        if (image.count_scale[i] != 1u) census.scale_exact = false;
        ++census.scale_lanes;
    }
    return census;
}

}  // namespace tq_nz_ingress

#endif
