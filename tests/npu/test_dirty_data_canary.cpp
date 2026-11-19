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

// THE DIRTY-DATA CANARY: one or two 64-row tiles of the fused decode, with every byte the
// kernel is not allowed to consume poisoned.
//
// WHY THIS IS SMALL. A tile is the unit the decode's dirty-data safety is defined on: the tail
// mask is per tile, the page cursor is per tile, and the block-table validation is per worker.
// Two tiles (context 72 or 80) exercise the live tile plus the short tile; one tile (context 64)
// exercises an aligned page. Anything larger only repeats those states, so the runner stays at
// 1-2 tiles and the simulator run stays bounded.
//
// WHAT IS POISONED
//
//   * every physical page the block table does not map -- 0xDEADBEEF bytes in both packed
//     planes and 0x7FC0BEEF (a NaN) in every scale lane;
//   * on the mapped page, every row at or past the live context -- the rows the Cube's 64-row
//     tile still reads but the mask must discard. Key/value bytes are 0xDEADBEEF garbage and
//     every scale lane is a NaN, so a softmax that merely Min-ed the dead logits could still
//     let 0 * NaN through the value-scale multiply;
//   * the per-row scale lanes past 2 * num_kv_heads, which no gather may name.
//
// WHAT IS CHECKED
//
//   1. the output of every live channel against the fixture's double-precision oracle, at the
//      same 2-ULP bf16 budget the e2e ladder uses;
//   2. 0xDEADBEEF guard bands on every allocation, so a read/write past a plane is named;
//   3. the input allocations are unchanged (the operator is read-only over them).
//
// The third case is a synthetic tiling descriptor: blocks_in_split = 65 (past the kernel's
// 64-entry block-table cache) while only the first tile is live, with entry 64 negative on one
// token and equal to num_blocks on the other. Both are outside [0, num_blocks), so the hardened
// PrefetchBlockTable must skip both workers and write zeros. The old sign-OR over the first 64
// entries would have accepted the table, read page 0, and produced a nonzero (wrong) answer.

#include "../common/camodel_session.hpp"
#include "../common/fused_decode_fixture.hpp"
#include "../../src/host/turboquant_tiling.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fd = ascend_micro::fused_decode_opt;
namespace fdt = ascend_micro::fused_decode;
using fused_decode_fixture::Element;
using fused_decode_fixture::Fixture;

constexpr std::uint32_t kGuardWord = 0xDEADBEEFu;
constexpr std::uint32_t kNaNBits = 0x7FC0BEEFu;
constexpr int kUlpBudget = 2;
constexpr std::size_t kScaleSlot = 8;  // NzScaleSlotFloats at H <= 4

void require(bool ok, const char* what)
{
    if (!ok) {
        std::fprintf(stderr, "DIRTY-DATA CANARY FAILED: %s\n", what);
        throw std::runtime_error(what);
    }
}

float bits_float(std::uint32_t bits)
{
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::uint8_t poison_byte(std::size_t i)
{
    return static_cast<std::uint8_t>(kGuardWord >> ((i % 4u) * 8u));
}

// One page of a packed plane, entirely 0xDEADBEEF words.
void poison_page(std::vector<std::uint8_t>& plane, std::size_t page, std::size_t page_bytes)
{
    for (std::size_t i = 0; i < page_bytes; ++i) plane[page * page_bytes + i] = poison_byte(i);
}

void poison_scale_row(std::vector<float>& scales, int slot, float poison)
{
    const std::size_t base = std::size_t(slot) * kScaleSlot;
    for (std::size_t lane = 0; lane < kScaleSlot; ++lane) scales[base + lane] = poison;
}

std::size_t packed_byte(int physical, int page_tokens, int row, int head, int col, int num_kv_heads)
{
    const std::uint64_t page_base =
        std::uint64_t(physical) * std::uint64_t(page_tokens) * std::uint64_t(num_kv_heads) * 128u;
    return std::size_t(page_base) +
           std::size_t(vllm_ascend::turboquant::NzTiledPackedByte(
               std::uint64_t(row), std::uint64_t(head), std::uint64_t(col),
               std::uint64_t(num_kv_heads), 128u));
}

struct LaunchBuffers {
    camodel::GuardedDeviceBuffer query;
    camodel::GuardedDeviceBuffer keys;
    camodel::GuardedDeviceBuffer values;
    camodel::GuardedDeviceBuffer scales;
    camodel::GuardedDeviceBuffer table;
    camodel::GuardedDeviceBuffer lengths;
    camodel::GuardedDeviceBuffer modes;
    camodel::GuardedDeviceBuffer workspace;
    camodel::GuardedDeviceBuffer out;
    camodel::GuardedDeviceBuffer tiling;
    std::size_t out_bytes;

    LaunchBuffers(std::size_t query_bytes, std::size_t cache_bytes, std::size_t scale_bytes,
                  std::size_t table_bytes, std::size_t length_bytes, std::size_t workspace_bytes,
                  std::size_t out_payload, std::size_t tiling_bytes)
        : query(query_bytes),
          keys(cache_bytes),
          values(cache_bytes),
          scales(scale_bytes),
          table(table_bytes),
          lengths(length_bytes),
          modes(32),
          workspace(workspace_bytes),
          out(out_payload),
          tiling(tiling_bytes),
          out_bytes(out_payload)
    {
    }

    fd::Buffers buffers()
    {
        return fd::Buffers{query.payload(), keys.payload(), values.payload(), scales.payload(),
                           table.payload(), lengths.payload(), modes.payload(), workspace.payload(),
                           out.payload(), tiling.payload()};
    }

    void download_all()
    {
        query.download();
        keys.download();
        values.download();
        scales.download();
        table.download();
        lengths.download();
        modes.download();
        workspace.download();
        out.download();
        tiling.download();
    }

    void assert_guards()
    {
        query.assert_guards();
        keys.assert_guards();
        values.assert_guards();
        scales.assert_guards();
        table.assert_guards();
        lengths.assert_guards();
        modes.assert_guards();
        workspace.assert_guards();
        out.assert_guards();
        tiling.assert_guards();
    }
};

std::size_t aligned32(std::size_t bytes) { return (bytes + 31u) / 32u * 32u; }

// One fixture launch with the cache poisoned the way the header describes.
void run_tail_case(int context, int page_tokens, camodel::Session& session)
{
    const int requests = 1, kv_heads = 4, group_heads = 4;
    Fixture f(context, requests, kv_heads, group_heads, page_tokens, Element::kBf16);
    const int physical_pages = f.physical_pages;
    const std::size_t page_bytes = std::size_t(page_tokens) * kv_heads * 128u;

    std::vector<std::uint8_t> key = f.key, value = f.value;
    std::vector<float> scales = f.scales;
    const float poison = bits_float(kNaNBits);

    // Every physical page the table does not map is entirely poison.
    std::vector<bool> mapped(std::size_t(physical_pages), false);
    int mapped_pages = 0;
    for (int request = 0; request < requests; ++request)
        for (int page = 0; page < f.pages; ++page) {
            mapped[std::size_t(f.table[std::size_t(request) * f.pages + page])] = true;
            ++mapped_pages;
        }
    for (int physical = 0; physical < physical_pages; ++physical) {
        if (mapped[std::size_t(physical)]) continue;
        poison_page(key, std::size_t(physical), page_bytes);
        poison_page(value, std::size_t(physical), page_bytes);
        for (int row = 0; row < page_tokens; ++row)
            poison_scale_row(scales, physical * page_tokens + row, poison);
    }

    // On the mapped page, rows at/past the live context are the dead rows of the last tile. The
    // scale lanes are set to NaN on purpose: 0 * NaN is NaN, which is the leak the value-scale
    // SELECT in quantize_probabilities exists to stop.
    int dead_rows = 0;
    for (int request = 0; request < requests; ++request)
        for (int page = 0; page < f.pages; ++page) {
            const int physical = f.table[std::size_t(request) * f.pages + page];
            const int live = f.page_rows(page);
            for (int row = live; row < page_tokens; ++row) {
                for (int head = 0; head < kv_heads; ++head)
                    for (int col = 0; col < 128; ++col) {
                        key[packed_byte(physical, page_tokens, row, head, col, kv_heads)] =
                            poison_byte(std::size_t(col));
                        value[packed_byte(physical, page_tokens, row, head, col, kv_heads)] =
                            poison_byte(std::size_t(col + 2));
                    }
                poison_scale_row(scales, physical * page_tokens + row, poison);
                ++dead_rows;
            }
        }

    const fd::Launch config = f.launch(1);
    require(config.num_blocks == static_cast<std::uint32_t>(physical_pages),
            "the fixture did not publish its physical page count");

    const std::size_t table_bytes = aligned32(f.table.size() * sizeof(std::int32_t));
    const std::size_t length_bytes = aligned32(f.lengths.size() * sizeof(std::int32_t));
    const std::size_t workspace_bytes =
        std::size_t(requests) * config.heads * (256 + 16) * sizeof(float);
    const std::size_t tiling_bytes =
        aligned32(fdt::decode_tiling_words(config.tokens, config.splits, config.grid) *
                  sizeof(std::uint32_t));
    LaunchBuffers guard(f.query.size() * sizeof(float), f.key.size(), f.scales.size() * sizeof(float),
                        table_bytes, length_bytes, workspace_bytes,
                        f.query.size() * sizeof(std::uint16_t), tiling_bytes);

    std::memcpy(guard.query.host_payload(), f.query.data(), f.query.size() * sizeof(float));
    std::memcpy(guard.keys.host_payload(), key.data(), key.size());
    std::memcpy(guard.values.host_payload(), value.data(), value.size());
    std::memcpy(guard.scales.host_payload(), scales.data(), scales.size() * sizeof(float));
    {
        std::vector<std::int32_t> table(table_bytes / sizeof(std::int32_t), -1);
        std::memcpy(table.data(), f.table.data(), f.table.size() * sizeof(std::int32_t));
        std::memcpy(guard.table.host_payload(), table.data(), table_bytes);
    }
    {
        std::vector<std::int32_t> lengths(length_bytes / sizeof(std::int32_t), 0);
        std::memcpy(lengths.data(), f.lengths.data(), f.lengths.size() * sizeof(std::int32_t));
        std::memcpy(guard.lengths.host_payload(), lengths.data(), length_bytes);
    }
    {
        std::vector<std::uint32_t> modes(8, 0);
        std::memcpy(guard.modes.host_payload(), modes.data(), 32);
    }
    {
        std::vector<std::uint32_t> image(fdt::decode_tiling_words(config.tokens, config.splits,
                                                                  config.grid),
                                         0);
        const auto status = fdt::build_decode_tiling(fdt::decode_tiling_request(config),
                                                     f.lengths.data(), f.table.data(), image.data());
        require(status == fdt::DecodeTilingStatus::kOk, fdt::decode_tiling_status_text(status));
        std::memcpy(guard.tiling.host_payload(), image.data(), image.size() * sizeof(std::uint32_t));
    }

    guard.query.upload();
    guard.keys.upload();
    guard.values.upload();
    guard.scales.upload();
    guard.table.upload();
    guard.lengths.upload();
    guard.modes.upload();
    guard.tiling.upload();

    const fd::Buffers buffers = guard.buffers();
    camodel::check(aclrtSynchronizeStream(session.stream), "pre-launch synchronize");
    launch_fused_decode_opt(session.stream, &config, &buffers);
    camodel::check(aclrtSynchronizeStream(session.stream), "post-launch synchronize");

    guard.download_all();
    guard.assert_guards();

    const std::size_t live = f.expected.size();
    require(live == std::size_t(requests) * config.heads * fd::kHeadSize,
            "the fixture's expected image is not [requests, heads, 256]");
    const auto* got = static_cast<const std::uint16_t*>(guard.out.host_payload());
    int worst_ulp = 0;
    for (std::size_t i = 0; i < live; ++i) {
        const std::uint16_t want = fused_decode_fixture::reference_bfloat16(f.expected[i]);
        const int ulp = std::abs(int(got[i]) - int(want));
        if (ulp > worst_ulp) worst_ulp = ulp;
        if (ulp > kUlpBudget) {
            std::fprintf(stderr,
                         "context %d live channel %zu: got %g (%04x) expected %g (%04x), %d ULP\n",
                         context, i, double(tq_test::bf16_to_float(got[i])), got[i],
                         double(f.expected[i]), want, ulp);
            require(false, "a poisoned tail/unmapped page changed a live output channel");
        }
    }
    std::printf("  ok  context %3d  page %3d  tiles %d  tail %2d  worst %d ULP  %d unmapped "
                "pages (%d) and %d dead rows poisoned\n",
                context, page_tokens, f.tiles(), context % 64, worst_ulp, physical_pages - mapped_pages,
                physical_pages, dead_rows);
}

// The synthetic >64-page span: one live tile per token, but blocks_in_split = 65 so the worker
// declares a table span past the kernel's 64-entry cache. Entry 64 is the dirty one.
void run_blocktable_case(camodel::Session& session)
{
    constexpr int kTokens = 2, kHeads = 16, kKvHeads = 1, kBlock = 64;
    constexpr int kNumBlocks = 2, kSpan = 65;

    fd::Launch config;
    config.element_type = static_cast<std::uint32_t>(Element::kBf16);
    config.tokens = kTokens;
    config.heads = kHeads;
    config.kv_heads = kKvHeads;
    config.head_size = 256;
    config.block_size = kBlock;
    config.max_blocks = kSpan;
    config.num_blocks = kNumBlocks;
    config.splits = 1;
    config.heads_per_task = kHeads;
    config.tasks_per_block = 2;
    config.reduce_tasks_per_block = kTokens * kHeads;
    config.grid = 1;
    config.fused_context_limit = 4096;
    config.scale = 0.0625f;
    config.inv_sqrt_len = 0.0625f;
    require(fd::valid_launch(config.grid, config.tokens, config.heads, config.kv_heads,
                             config.head_size, config.block_size, config.max_blocks, config.splits,
                             config.heads_per_task, config.tasks_per_block,
                             config.reduce_tasks_per_block),
            "the synthetic block-table geometry is not a valid launch");

    const std::size_t query_bytes = std::size_t(kTokens) * kHeads * 256 * sizeof(float);
    const std::size_t out_bytes = std::size_t(kTokens) * kHeads * 256 * sizeof(std::uint16_t);
    const std::size_t cache_bytes = std::size_t(kNumBlocks) * kBlock * kKvHeads * 128;
    const std::size_t scale_bytes = std::size_t(kNumBlocks) * kBlock * kScaleSlot * sizeof(float);
    const std::size_t table_bytes = aligned32(std::size_t(kTokens) * kSpan * sizeof(std::int32_t));
    const std::size_t length_bytes = aligned32(std::size_t(kTokens) * sizeof(std::int32_t));
    const std::size_t workspace_bytes = std::size_t(kTokens) * kHeads * (256 + 16) * sizeof(float);
    const std::size_t tiling_bytes = aligned32(
        fdt::decode_tiling_words(config.tokens, config.splits, config.grid) * sizeof(std::uint32_t));
    LaunchBuffers guard(query_bytes, cache_bytes, scale_bytes, table_bytes, length_bytes,
                        workspace_bytes, out_bytes, tiling_bytes);

    // A live page 0 with finite, nonzero content: the old code would have produced a nonzero
    // answer from it, so "all zeros" below is a real assertion about the validation, not an
    // accident of zeroed inputs.
    {
        std::vector<float> query(std::size_t(kTokens) * kHeads * 256, 0.25f);
        std::memcpy(guard.query.host_payload(), query.data(), query_bytes);

        std::vector<std::uint8_t> plane(cache_bytes, 0x11u);
        std::memcpy(guard.keys.host_payload(), plane.data(), cache_bytes);
        std::memcpy(guard.values.host_payload(), plane.data(), cache_bytes);

        std::vector<float> scales(scale_bytes / sizeof(float), 2.0f);
        for (std::size_t i = std::size_t(kBlock) * kScaleSlot; i < scales.size(); ++i)
            scales[i] = bits_float(kNaNBits);
        std::memcpy(guard.scales.host_payload(), scales.data(), scale_bytes);

        std::vector<std::int32_t> table(table_bytes / sizeof(std::int32_t), -1);
        for (int token = 0; token < kTokens; ++token)
            for (int page = 0; page < 64; ++page)
                table[std::size_t(token) * kSpan + std::size_t(page)] = 0;
        table[0 * kSpan + 64] = -1;          // negative, first-token row
        table[1 * kSpan + 64] = kNumBlocks;  // == num_blocks, one past the cache
        std::memcpy(guard.table.host_payload(), table.data(), table_bytes);

        std::vector<std::int32_t> lengths(length_bytes / sizeof(std::int32_t), 0);
        lengths[0] = lengths[1] = kBlock;
        std::memcpy(guard.lengths.host_payload(), lengths.data(), length_bytes);

        std::vector<std::uint32_t> modes(8, 0);
        std::memcpy(guard.modes.host_payload(), modes.data(), 32);
    }

    {
        std::vector<std::uint32_t> image(
            fdt::decode_tiling_words(config.tokens, config.splits, config.grid), 0);
        static_assert(fdt::kDecodeTilingWords == 16, "the tiling descriptor width moved");
        static_assert(sizeof(fdt::TurboQuantDecodeTilingData) == 64,
                      "the tiling descriptor is not one 64-byte line");
        auto* d = reinterpret_cast<fdt::TurboQuantDecodeTilingData*>(image.data());
        for (int token = 0; token < kTokens; ++token) {
            auto& e = d[token];
            e.num_tiles = 1;
            e.regular_tiles = 1;
            e.tail_rows = 0;
            e.block_start_idx = 0;
            e.blocks_in_split = kSpan;  // > the 64-entry prefetch cache
            e.head_dim = 256;
            e.group_heads = kHeads;
            e.split_k = 1;
            e.tile_origin = 0;
            e.tiles_per_page_shift = 0;
            e.flags = fdt::kDecodeSplitLive;
            e.chunks_per_group = 1;
        }
        auto* dispatch = reinterpret_cast<fdt::TurboQuantDecodeDispatch*>(
            image.data() + fdt::decode_dispatch_word_base(config.tokens, config.splits));
        dispatch[0].token = 0;
        dispatch[0].kv_head = 0;
        dispatch[0].split = 0;
        dispatch[0].chunk = 0;
        dispatch[0].reduce_token = 0;
        dispatch[0].reduce_head = 0;
        dispatch[0].reduce_token_hi = 1;
        dispatch[0].reduce_head_hi = 0;
        std::memcpy(guard.tiling.host_payload(), image.data(), image.size() * sizeof(std::uint32_t));
    }

    guard.query.upload();
    guard.keys.upload();
    guard.values.upload();
    guard.scales.upload();
    guard.table.upload();
    guard.lengths.upload();
    guard.modes.upload();
    guard.tiling.upload();

    const fd::Buffers buffers = guard.buffers();
    camodel::check(aclrtSynchronizeStream(session.stream), "pre-launch synchronize");
    launch_fused_decode_opt(session.stream, &config, &buffers);
    camodel::check(aclrtSynchronizeStream(session.stream), "post-launch synchronize");

    guard.download_all();
    guard.assert_guards();
    const auto* out = static_cast<const std::uint16_t*>(guard.out.host_payload());
    for (std::size_t i = 0; i < out_bytes / sizeof(std::uint16_t); ++i)
        require(out[i] == 0, "an out-of-range block-table entry past the 64-page cache was not "
                             "rejected: the worker wrote a nonzero output");
    std::printf("  ok  synthetic span %d (> 64-page cache): negative and past-num_blocks entries "
                "at index 64 skipped both workers, all %zu output words zero\n",
                kSpan, out_bytes / sizeof(std::uint16_t));
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode != "--tail" && mode != "--blocktable") {
            std::fprintf(stderr, "usage: %s --tail | --blocktable\n", argv[0]);
            return 2;
        }
        const auto started = std::chrono::steady_clock::now();
        camodel::Session session;
        if (mode == "--tail") {
            run_tail_case(64, 64, session);
            run_tail_case(72, 128, session);
            run_tail_case(80, 128, session);
        } else {
            run_blocktable_case(session);
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::printf("PASS dirty-data canary (%s) in %.1f s\n", mode.c_str(), seconds);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
