#pragma once

#include "../../src/turboquant/common/turboquant_fused_decode_opt.hpp"
#include "../../src/turboquant/common/turboquant_gemm_reduce.hpp"
#include "turboquant_golden_oracle.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen35_bench {
namespace fd = ascend_micro::fused_decode_opt;
namespace tiling = ascend_micro::fused_decode;
namespace tq = vllm_ascend::turboquant;
namespace reduce = ascend_micro::gemm_reduce;

constexpr std::array<std::uint32_t, 2> kBlocks{128, 1024};
constexpr std::array<std::uint32_t, 2> kBatches{1, 2};
constexpr std::array<std::uint32_t, 4> kContexts{2048, 32768, 262144, 1048576};
constexpr std::array<std::uint32_t, 4> kSplits{1, 2, 4, 8};
constexpr std::uint64_t kPackedRowBytes = 256 / 2;
constexpr std::uint64_t kScaleRowBytes = tq::NzScaleSlotFloats(1) * sizeof(float);
constexpr std::uint64_t kCacheRowBytes = 2 * kPackedRowBytes + kScaleRowBytes;
constexpr std::uint64_t kIssuedKvRowBytes = 2 * (kPackedRowBytes + kScaleRowBytes);
constexpr std::uint64_t kUbBound = fd::kUbEnd + fd::kBasisUbBytes + fd::kReducerUbBytes + 256;
constexpr std::uint64_t kL1Bound = reduce::kAttentionL1Bytes + reduce::kReduceL1Bytes;
static_assert(kCacheRowBytes == 288 && kIssuedKvRowBytes == 320);
static_assert(kUbBound <= 192 * 1024 && kL1Bound <= 512 * 1024);

struct Geometry {
    std::uint32_t block, batch, context, splits;
    constexpr std::uint32_t pages() const { return block ? context / block : 0; }
    constexpr std::uint64_t slots() const { return std::uint64_t{batch} * context; }
    constexpr std::uint64_t baseline_bytes() const { return slots() * 2 * 256 * 2; }
    constexpr std::uint64_t cache_bytes() const { return slots() * kCacheRowBytes; }
    constexpr std::uint64_t workspace_bytes() const
    {
        return std::uint64_t{batch} * 6 * splits * (256 + tq::kPartialTail) * sizeof(float);
    }
    constexpr std::uint64_t kv_read_bytes() const { return slots() * kIssuedKvRowBytes; }
    constexpr fd::Launch launch() const
    {
        fd::Launch c;
        c.tokens = batch;
        c.heads = 6;
        c.kv_heads = 1;
        c.block_size = block;
        c.max_blocks = pages();
        c.num_blocks = batch * pages();
        c.splits = splits;
        c.heads_per_task = 6;
        c.grid = batch * splits;
        c.tasks_per_block = 1;
        c.reduce_tasks_per_block = c.grid ? (batch * 6 + c.grid - 1) / c.grid : 0;
        c.prologue_vectors_per_block = c.reduce_tasks_per_block;
        c.element_type = 1;
        c.fused_context_limit = 0;
        c.output_stage = 0;
        return c;
    }
};

constexpr bool valid_static_matrix()
{
    for (auto block : kBlocks) for (auto batch : kBatches)
        for (auto context : kContexts) for (auto splits : kSplits) {
            const Geometry g{block, batch, context, splits};
            const auto c = g.launch();
            if (!fd::valid_launch(c.grid, c.tokens, c.heads, c.kv_heads, c.head_size,
                    c.block_size, c.max_blocks, c.splits, c.heads_per_task,
                    c.tasks_per_block, c.reduce_tasks_per_block)) return false;
            if (g.slots() - 1 > INT32_MAX || g.slots() * kPackedRowBytes - 1 > UINT32_MAX ||
                g.slots() * kScaleRowBytes - 1 > UINT32_MAX || g.workspace_bytes() > UINT32_MAX)
                return false;
        }
    return true;
}
static_assert(valid_static_matrix(), "Qwen3.5 sweep exceeds the production launch contract");

struct Plan {
    Geometry geometry;
    fd::Launch config;
    std::vector<std::int32_t> table, lengths;
    std::vector<std::uint32_t> descriptors;
    std::uint32_t live_splits = 0;
    std::string reason;
    explicit Plan(Geometry g, bool canary = false) : geometry(g), config(g.launch())
    {
        const bool tiny = canary && (g.context == 64 || g.context == 128) && g.block == 64 &&
            g.batch == 1 && (g.splits == 1 || g.splits == 2);
        if (!tiny && (std::find(kBlocks.begin(), kBlocks.end(), g.block) == kBlocks.end() ||
            std::find(kBatches.begin(), kBatches.end(), g.batch) == kBatches.end() ||
            std::find(kContexts.begin(), kContexts.end(), g.context) == kContexts.end() ||
            std::find(kSplits.begin(), kSplits.end(), g.splits) == kSplits.end())) {
            reason = "outside the requested 64-configuration sweep";
            return;
        }
        if (!fd::valid_launch(config.grid, config.tokens, config.heads, config.kv_heads,
                config.head_size, config.block_size, config.max_blocks, config.splits,
                config.heads_per_task, config.tasks_per_block, config.reduce_tasks_per_block)) {
            reason = "valid_launch rejected geometry or task coverage";
            return;
        }
        if (g.slots() - 1 > INT32_MAX || g.slots() * kPackedRowBytes - 1 > UINT32_MAX ||
            g.slots() * kScaleRowBytes - 1 > UINT32_MAX || g.workspace_bytes() > UINT32_MAX) {
            reason = "slot, plane byte offset or workspace exceeds 32-bit range";
            return;
        }
        table.resize(std::size_t(g.batch) * g.pages());
        lengths.assign(g.batch, static_cast<std::int32_t>(g.context));
        for (std::size_t i = 0; i < table.size(); ++i)
            table[i] = static_cast<std::int32_t>(table.size() - 1 - i);
        descriptors.resize(tiling::decode_tiling_words(g.batch, g.splits, config.grid));
        const auto status = tiling::build_decode_tiling(tiling::decode_tiling_request(config),
            lengths.data(), table.data(), descriptors.data());
        if (status != tiling::DecodeTilingStatus::kOk) {
            reason = tiling::decode_tiling_status_text(status);
            return;
        }
        for (std::uint32_t token = 0; token < g.batch; ++token) {
            std::uint32_t tiles = 0, pages = 0;
            for (std::uint32_t split = 0; split < g.splits; ++split) {
                tiling::TurboQuantDecodeTilingData d{};
                const std::size_t offset = (std::size_t(token) * g.splits + split) * tiling::kDecodeTilingWords;
                std::memcpy(&d, descriptors.data() + offset, sizeof(d));
                if (!(d.flags & tiling::kDecodeSplitLive)) continue;
                if (d.block_start_idx != pages || d.tail_rows != 0)
                    throw std::runtime_error("non-contiguous benchmark tiling");
                pages += d.blocks_in_split;
                tiles += d.num_tiles;
                if (token == 0) ++live_splits;
            }
            if (pages != g.pages() || tiles != g.context / fd::kTileRows)
                throw std::runtime_error("benchmark tiling does not cover the full context");
        }
    }
    bool valid() const { return reason.empty(); }
};

struct Fixture {
    Geometry geometry;
    std::vector<std::uint8_t> key, value;
    std::vector<float> scales, query, signs;
    std::vector<std::uint16_t> raw;
    explicit Fixture(Geometry g) : geometry(g),
        key(g.slots() * kPackedRowBytes), value(key.size()),
        scales(g.slots() * tq::NzScaleSlotFloats(1), 0.0f),
        query(std::size_t(g.batch) * 6 * 256, 0.0f),
        signs(tq_golden::kernel_pi_signs(256)), raw(query.size(), 0)
    {
        for (std::uint64_t slot = 0; slot < g.slots(); ++slot) {
            scales[tq::NzTiledScaleIndex(slot, 0, 0, 1)] =
                0.125f * static_cast<float>(1u << (slot % 3));
            scales[tq::NzTiledScaleIndex(slot, 0, 1, 1)] =
                0.125f * static_cast<float>(1u << ((slot % g.context) / (g.context / 4)));
            for (std::uint32_t col = 0; col < 128; ++col) {
                const auto hash = tq_golden::mix(static_cast<std::uint32_t>(slot) * 0x9e3779b9u ^ col);
                const auto index = tq::NzTiledPackedByte(slot, 0, col, 1, 128);
                key[index] = static_cast<std::uint8_t>(hash);
                value[index] = static_cast<std::uint8_t>((1 + (hash >> 8) % 7) |
                    ((1 + (hash >> 16) % 7) << 4));
            }
        }
        for (std::uint32_t b = 0; b < g.batch; ++b)
            for (std::uint32_t h = 0; h < 6; ++h) {
                const std::size_t i = (std::size_t(b) * 6 + h) * 256 + (h * 37 + b * 19) % 256;
                query[i] = 0.5f + static_cast<float>(h + b) * 0.25f;
                raw[i] = tq_test::float_to_bf16_rne(query[i]);
            }
    }
    std::vector<double> expected(const Plan& plan) const
    {
        tq_golden::DecodeGeometry g;
        g.requests = static_cast<int>(geometry.batch);
        g.group_heads = 6;
        g.page_tokens = static_cast<int>(geometry.block);
        g.max_pages = static_cast<int>(geometry.pages());
        return tq_golden::decode_reference(g, tq_golden::DensePi(signs), query.data(),
            key.data(), value.data(), scales.data(), plan.table.data(), plan.lengths.data()).stage0;
    }
};
}
