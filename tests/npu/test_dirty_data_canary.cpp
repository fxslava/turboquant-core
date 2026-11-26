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
// Tail cases stay at 1-2 live tiles: contexts 17, 80 and 81 cover unaligned/aligned align16
// bounds and a whole dead tile inside a mapped page. Split-K exercises an empty worker too.
//
// WHAT IS POISONED
//
//   * every unmapped physical page and every mapped row at/past context_len, all the way to
//     the page end (including past align16(context_len)): signed +1 nibbles, K_scale=10 and
//     V_scale=100. Queries are all +1; live K/V nibbles and scales are zero. All inputs are
//     finite. A leaked spike has logit 160 versus live logits 0 and drives output near 100.
//     NaNs are unsuitable here: bitwise Select and maxNum can silently neutralize them.
//
// WHAT IS CHECKED
//
//   1. every output channel is exactly zero (no tolerance); a positive control admits one
//      spike by incrementing context_len and must instead produce approximately 100;
//   2. 0xDEADBEEF guard bands on every allocation, so a write past a plane is named;
//   3. the input allocations are unchanged (the operator is read-only over them).
//
// The --blocktable mode uses a synthetic tiling descriptor: blocks_in_split = 65 (past the kernel's
// 64-entry block-table cache) while only the first tile is live, with entry 64 negative on one
// token and equal to num_blocks on the other. Both are outside [0, num_blocks), so the hardened
// PrefetchBlockTable must skip both workers and write zeros. The old sign-OR over the first 64
// entries would have accepted the table, read page 0, and produced a nonzero (wrong) answer.

#include "../common/camodel_session.hpp"
#include "../common/fused_decode_fixture.hpp"
#include "../../src/turboquant/common/turboquant_tiling.h"

#include <algorithm>
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

constexpr std::uint32_t kNaNBits = 0x7FC0BEEFu;
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
void run_tail_case(int context, int page_tokens, int splits, camodel::Session& session,
                   bool admit_spike = false)
{
    const int requests = 1, kv_heads = 1, group_heads = 4;
    Fixture f(context, requests, kv_heads, group_heads, page_tokens, Element::kBf16);
    const int physical_pages = f.physical_pages;
    std::fill(f.query.begin(), f.query.end(), 1.0f);
    // Start with finite spikes everywhere, then clear ONLY live rows using the NZ layout.
    // Leaving the entire page tail intact catches bounds rounded past align16(context).
    std::vector<std::uint8_t> key(f.key.size(), 0x11u), value(f.value.size(), 0x11u);
    std::vector<float> scales(f.scales.size(), 0.0f);
    for (int physical = 0; physical < physical_pages; ++physical) {
        for (int row = 0; row < page_tokens; ++row)
            for (int head = 0; head < kv_heads; ++head) {
                const auto base = std::size_t(physical * page_tokens + row) * kScaleSlot;
                scales[base + head] = 10.0f;
                scales[base + kv_heads + head] = 100.0f;
            }
    }

    int dead_rows = 0;
    for (int request = 0; request < requests; ++request)
        for (int page = 0; page < f.pages; ++page) {
            const int physical = f.table[std::size_t(request) * f.pages + page];
            const int live = f.page_rows(page);
            dead_rows += page_tokens - live;
            for (int row = 0; row < live; ++row) {
                for (int head = 0; head < kv_heads; ++head) {
                    const auto base = std::size_t(physical * page_tokens + row) * kScaleSlot;
                    scales[base + head] = scales[base + kv_heads + head] = 0.0f;
                    for (int col = 0; col < 128; ++col) {
                        key[packed_byte(physical, page_tokens, row, head, col, kv_heads)] =
                            0x00u;
                        value[packed_byte(physical, page_tokens, row, head, col, kv_heads)] =
                            0x00u;
                    }
                }
            }
        }
    require(dead_rows > 0, "tail test has no poisoned mapped rows");
    if (admit_spike) ++f.lengths[0];  // Same cache/query: only the mask/tiling bound changes.

    const fd::Launch config = f.launch(splits);
    require(config.num_blocks == static_cast<std::uint32_t>(physical_pages),
            "the fixture did not publish its physical page count");

    const std::size_t table_bytes = aligned32(f.table.size() * sizeof(std::int32_t));
    const std::size_t length_bytes = aligned32(f.lengths.size() * sizeof(std::int32_t));
    const std::size_t workspace_bytes =
        std::size_t(requests) * config.heads * splits * (256 + 16) * sizeof(float);
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

    const std::vector<camodel::GuardedDeviceBuffer*> inputs{
        &guard.query, &guard.keys, &guard.values, &guard.scales,
        &guard.table, &guard.lengths, &guard.modes, &guard.tiling};
    std::vector<std::vector<std::uint32_t>> input_images;
    for (const auto* input : inputs) input_images.push_back(input->host);

    const fd::Buffers buffers = guard.buffers();
    camodel::check(aclrtSynchronizeStream(session.stream), "pre-launch synchronize");
    launch_fused_decode_opt(session.stream, &config, &buffers);
    camodel::check(aclrtSynchronizeStream(session.stream), "post-launch synchronize");

    guard.download_all();
    guard.assert_guards();
    for (std::size_t i = 0; i < inputs.size(); ++i)
        require(inputs[i]->host == input_images[i], "decode changed an input allocation");

    const std::size_t live = f.query.size();
    require(live == std::size_t(requests) * config.heads * fd::kHeadSize,
            "the output shape is not [requests, heads, 256]");
    const auto* got = static_cast<const std::uint16_t*>(guard.out.host_payload());
    float max_abs = 0.0f;
    for (std::size_t i = 0; i < live; ++i) {
        const float output = tq_test::bf16_to_float(got[i]);
        // Test equality per lane BEFORE reducing: NaN must fail, even if maxNum ignores it.
        const bool ok = admit_spike ? std::fabs(output - 100.0f) <= 1.0f : output == 0.0f;
        if (!ok) {
            std::fprintf(stderr,
                         "context %d splits %d control %d channel %zu: got %g (%04x), expected %g\n",
                         context, splits, int(admit_spike), i, double(output), got[i],
                         admit_spike ? 100.0 : 0.0);
            require(false, "zero-live / finite-spike output mismatch");
        }
        max_abs = std::max(max_abs, std::fabs(output));
    }
    std::printf("  ok  context %3d  page %3d  splits %d  max_abs %g  %d finite-spike tail rows%s\n",
                context, page_tokens, splits, double(max_abs), dead_rows,
                admit_spike ? " (positive control: first spike admitted)" : "");
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
            for (int context : {17, 80})
                run_tail_case(context, 128, 1, session);
            run_tail_case(81, 128, 3, session);  // Includes an empty split.
            run_tail_case(17, 128, 1, session, true);
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
