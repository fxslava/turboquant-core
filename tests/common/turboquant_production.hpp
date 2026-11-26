#pragma once

#include "src/turboquant/common/e8_decode.hpp"
#include "src/turboquant/common/turboquant_fused_decode_opt.hpp"
#define E8_CPU_TEST
#define ascend_micro frozen_micro
#include <routine_e8_bf16_dequant_simt.hpp>
#undef ascend_micro
#undef E8_CPU_TEST
#include "tests/common/turboquant_test_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace tq_production {

namespace fd = ascend_micro::fused_decode_opt;
namespace tiling = ascend_micro::fused_decode;
namespace layout = vllm_ascend::turboquant;
namespace packet = e8_packet_abi;

constexpr std::uint32_t kDim = 256;
constexpr std::uint32_t kTileRows = 64;
constexpr std::uint32_t kPatternPages = 5;
constexpr std::uint32_t kNpuGrid = 32;
constexpr float kAttentionScale = 1.0f / 16;
constexpr auto& kLm4Centroids = layout::TurboQuantLloydMaxTraits<layout::TurboQuantMode::KV4_FP8>::kCentroids;

enum class Mode { lm4, lm3, e8 };
constexpr std::array kModes{Mode::lm4, Mode::lm3, Mode::e8};
inline const char* name(Mode mode)
{
    switch (mode) {
        case Mode::lm4: return "lm4";
        case Mode::lm3: return "lm3";
        case Mode::e8: return "e8";
    }
    throw std::invalid_argument("invalid codec");
}

struct Profile {
    const char* name;
    std::uint32_t sequences, context, kv_heads, group, splits;
    bool ragged;
};
constexpr Profile kProfiles[] = {
    {"h1", 1, 2048, 1, 1, 1, true},
    {"qwen35-gqa4", 1, 2048, 4, 4, 1, false},
    {"qwen35-gqa6", 4, 4096, 4, 6, 1, true},
    {"glm4-gqa8", 4, 2048, 8, 8, 1, true},
    {"glm4-gqa16", 1, 2048, 8, 16, 1, false},
    {"split4", 4, 4096, 4, 8, 4, true},
    {"split8", 16, 8192, 1, 6, 8, true},
    {"split16", 32, 16384, 4, 4, 16, true},
    {"deepseek-v4-flash-16k", 16, 16384, 4, 4, 1, true},
    {"deepseek-v4-flash-32k", 32, 32768, 1, 6, 4, true},
    {"deepseek-v4-flash-64k", 4, 65536, 1, 8, 8, true},
    {"deepseek-v4-flash-128k", 1, 131072, 4, 16, 8, true},
};
constexpr std::array<std::uint32_t, 2> kBlocks{64, 128};

constexpr std::size_t max_tiling_words()
{
    std::size_t words = 0;
    for (const auto& profile : kProfiles)
        words = std::max(words, tiling::decode_tiling_words(profile.sequences, profile.splits, kNpuGrid));
    return words;
}

using TilingWords = std::array<std::uint32_t, max_tiling_words()>;

constexpr std::uint32_t ceil_div(std::uint32_t a, std::uint32_t b)
{
    return a / b + (a % b != 0);
}

constexpr std::uint64_t packet_offset(unsigned page, unsigned tile, unsigned kv,
    unsigned tiles_per_page, unsigned kv_heads)
{
    return ((std::uint64_t{page} * tiles_per_page + tile) * kv_heads + kv) * packet::kPacketBytes;
}

static_assert(packet_offset(1u << 20, 1, 7, 2, 8) > UINT32_MAX);
static_assert(layout::NzTiledPackedByte(1u << 28, 0, 0, 8, 128) > UINT32_MAX);

inline void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

struct Plan {
    Profile p;
    std::uint32_t block, max_pages, pages = 0;
    std::vector<std::int32_t> lengths, table;
    std::uint64_t live_rows = 0, read_tiles = 0;

    explicit Plan(Profile profile, std::uint32_t block_size)
        : p(profile), block(block_size), max_pages(ceil_div(p.context, block)),
          lengths(p.sequences), table(std::uint64_t{p.sequences} * max_pages, -1)
    {
        for (std::uint32_t seq = 0; seq < p.sequences; ++seq) {
            const auto trim = p.ragged ? 1 + (seq * 13 + 7) % 63 : 0;
            lengths[seq] = static_cast<std::int32_t>(p.context - trim);
            pages += ceil_div(lengths[seq], block);
            live_rows += lengths[seq];
            read_tiles += ceil_div(lengths[seq], kTileRows);
        }
        auto next_page = pages;
        for (std::uint32_t seq = 0; seq < p.sequences; ++seq)
            for (std::uint32_t i = 0; i < ceil_div(lengths[seq], block); ++i)
                table[std::uint64_t{seq} * max_pages + i] = static_cast<std::int32_t>(--next_page);
        require(next_page == 0, "incomplete physical page map");
    }

    std::uint32_t heads() const { return p.kv_heads * p.group; }
    std::uint64_t slots() const { return std::uint64_t{pages} * block; }

    std::uint32_t physical(std::uint32_t seq, std::uint32_t logical_page) const
    {
        return static_cast<std::uint32_t>(table[std::uint64_t{seq} * max_pages + logical_page]);
    }

    std::uint32_t pattern_row(std::uint32_t seq, std::uint32_t row) const
    {
        return physical(seq, row / block) % kPatternPages * block + row % block;
    }

    fd::Launch native_launch() const
    {
        fd::Launch c;
        c.tokens = p.sequences;
        c.heads = heads();
        c.kv_heads = p.kv_heads;
        c.block_size = block;
        c.max_blocks = max_pages;
        c.num_blocks = pages;
        c.splits = p.splits;
        c.heads_per_task = p.group;
        c.fused_context_limit = 0;
        c.element_type = 1;
        c.decode_unpack = static_cast<std::uint32_t>(layout::DecodeUnpack::kLloydMaxLut);
        const auto dist = tiling::decode_launch_split(tiling::decode_tiling_request(c), kNpuGrid);
        c.grid = dist.grid;
        c.tasks_per_block = dist.tasks_per_block;
        c.reduce_tasks_per_block = dist.reduce_tasks_per_block;
        return c;
    }

    const char* device_limit(Mode mode, std::string_view stage) const
    {
        if (mode == Mode::lm3) return "no native Lloyd-Max 3-bit kernel";
        if (stage == "ingress") return mode == Mode::e8 ? "E8 device encoder gated" : nullptr;
        if (stage == "dequant") return mode == Mode::lm4 ? "no standalone Lloyd-Max 4-bit dequantizer" : nullptr;
        if (mode == Mode::lm4) {
            const auto c = native_launch();
            if (!fd::valid_launch(c.grid, c.tokens, c.heads, c.kv_heads, c.head_size,
                    c.block_size, c.max_blocks, c.splits, c.heads_per_task,
                    c.tasks_per_block, c.reduce_tasks_per_block))
                return "native fused launch requires at most 4 KV heads and 8 splits";
            return nullptr;
        }
        if (p.splits != 1) return "sealed E8 reader has no Split-K merge";
        for (const auto length : lengths) {
            const auto tiles = ceil_div(length, kTileRows);
            if (tiles < 4 || tiles > e8_decode::kMaxTiles || tiles % 2)
                return "sealed E8 reader requires an even 4..256 tiles per sequence";
        }
        return nullptr;
    }

    std::uint64_t kv_bytes(Mode mode, bool allocated) const
    {
        const auto tiles = allocated ? slots() / kTileRows : read_tiles;
        if (mode == Mode::e8) return tiles * p.kv_heads * packet::kPacketBytes * 2;
        const auto packed = mode == Mode::lm4 ? kDim / 2 : 3 * kDim / 8;
        return tiles * kTileRows * (std::uint64_t{2} * p.kv_heads * packed +
            std::uint64_t{layout::NzScaleSlotFloats(p.kv_heads)} * sizeof(float));
    }
};

inline std::uint32_t hash(std::uint32_t x)
{
    x = (x ^ (x >> 16)) * 0x7feb352du;
    x = (x ^ (x >> 15)) * 0x846ca68bu;
    return x ^ (x >> 16);
}

inline std::uint32_t sample(std::uint32_t row, std::uint32_t kv, unsigned plane, unsigned channel)
{
    return hash(1 + row * 0x9e3779b9u + kv * 0x85ebca6bu + plane * 0xc2b2ae35u + channel);
}

inline std::uint16_t e8_label(std::uint32_t row, std::uint32_t kv, unsigned plane, unsigned group)
{
    return static_cast<std::uint16_t>(sample(row, kv, plane, group));
}

inline float row_scale(std::uint32_t row, std::uint32_t kv, unsigned plane)
{
    return float(2 + (row + kv + plane) % 13) * 0.03125f;
}

inline unsigned lm_code(Mode mode, std::uint32_t row, std::uint32_t kv, unsigned plane, unsigned channel)
{
    return sample(row, kv, plane, channel) % (mode == Mode::lm4 ? 16 : 8);
}

using Packet = std::array<std::uint8_t, packet::kPacketBytes>;
static_assert(sizeof(Packet) == 4352);

inline Packet packet_fixture(std::uint32_t first_row, std::uint32_t kv, unsigned plane)
{
    Packet out{};
    for (std::uint32_t r = 0; r < kTileRows; ++r) {
        for (unsigned g = 0; g < packet::kLabelsPerRow; ++g) {
            const auto label = e8_label(first_row + r, kv, plane, g);
            std::memcpy(out.data() + std::uint64_t{r} * packet::kLabelRowBytes + std::uint64_t{g} * 2,
                        &label, sizeof(label));
        }
        const auto scale = row_scale(first_row + r, kv, plane);
        std::memcpy(out.data() + packet::kScaleOffset + std::uint64_t{r} * sizeof(scale), &scale, sizeof(scale));
    }
    return out;
}

inline void verify_packet(const Packet& bytes, unsigned first_row, unsigned kv, unsigned plane)
{
    for (std::uint32_t r = 0; r < 64; ++r) {
        for (unsigned g = 0; g < 32; ++g) {
            const auto offset = std::uint64_t{r} * 64 + std::uint64_t{g} * 2;
            const unsigned label = bytes[offset] | static_cast<unsigned>(bytes[offset + 1]) << 8;
            require(label == e8_label(first_row + r, kv, plane, g), "E8 packet label mismatch");
        }
        std::uint32_t raw = 0;
        for (unsigned byte = 0; byte < 4; ++byte)
            raw |= std::uint32_t{bytes[4096 + std::uint64_t{r} * 4 + byte]} << (8 * byte);
        require(raw == tq_test::float_bits(row_scale(first_row + r, kv, plane)), "E8 packet scale mismatch");
    }
}

inline const std::array<double, 8>& lm3_centroids()
{
    static const auto codebook = [] {
        std::array<double, 8> centers{-2.2, -1.3, -0.7, -0.2, 0.2, 0.7, 1.3, 2.2};
        constexpr double inv_sqrt_2pi = 0.39894228040143267794;
        constexpr double inv_sqrt_2 = 0.70710678118654752440;
        for (unsigned iteration = 0; iteration < 1000; ++iteration) {
            auto next = centers;
            double previous_density = 0, previous_cdf = 0;
            for (unsigned i = 0; i < centers.size(); ++i) {
                const bool last = i + 1 == centers.size();
                const double boundary = last ? 0 : (centers[i] + centers[i + 1]) * 0.5;
                const double density = last ? 0 : inv_sqrt_2pi * std::exp(-0.5 * boundary * boundary);
                const double cdf = last ? 1 : 0.5 * std::erfc(-boundary * inv_sqrt_2);
                next[i] = (previous_density - density) / (cdf - previous_cdf);
                previous_density = density;
                previous_cdf = cdf;
            }
            centers = next;
        }
        return centers;
    }();
    return codebook;
}

inline void e8_golden(std::uint16_t label, int (&coordinates)[8])
{
    frozen_micro::e8::Project(label, coordinates);
    frozen_micro::e8::ShapeResidualBits(coordinates);
}

inline void reconstruct(Mode mode, unsigned row, unsigned kv, unsigned plane, float* output)
{
    const float scale = row_scale(row, kv, plane);
    if (mode == Mode::e8) {
        for (unsigned group = 0; group < 32; ++group) {
            int coordinates[8];
            e8_golden(e8_label(row, kv, plane, group), coordinates);
            for (unsigned j = 0; j < 8; ++j)
                output[std::uint64_t{group} * 8 + j] = tq_test::bf16_to_float(
                    tq_test::float_to_bf16_rne(float(coordinates[j]) * 0.5f * scale));
        }
        return;
    }
    for (unsigned c = 0; c < kDim; ++c) {
        const auto index = lm_code(mode, row, kv, plane, c);
        const double centroid = mode == Mode::lm4 ? kLm4Centroids[index] : lm3_centroids()[index];
        output[c] = static_cast<float>(centroid * scale);
    }
}

inline std::uint32_t query_channel(unsigned seq, unsigned head)
{
    return (seq * 31 + head * 37) % kDim;
}

inline float query_value(unsigned seq, unsigned head)
{
    return float(8 + (seq + head) % 7) * 0.5f;
}

inline std::vector<std::uint32_t> pattern_counts(const Plan& plan, unsigned seq,
    unsigned first_page, unsigned pages)
{
    std::vector<std::uint32_t> counts(std::uint64_t{kPatternPages} * plan.block);
    const auto begin = std::uint64_t{first_page} * plan.block;
    const auto end = std::min<std::uint64_t>(plan.lengths[seq], (std::uint64_t{first_page} + pages) * plan.block);
    for (auto row = begin; row < end; ++row)
        ++counts[plan.pattern_row(seq, static_cast<std::uint32_t>(row))];
    return counts;
}

struct Attention {
    std::array<double, kDim> context{};
    double weight = 0;
};

inline std::vector<double> attention_reference(const Plan& plan, Mode mode, bool split)
{
    std::vector<double> result(std::uint64_t{plan.p.sequences} * plan.heads() * kDim);
    const auto pattern_rows = std::uint64_t{kPatternPages} * plan.block;
    std::vector<float> keys(pattern_rows * kDim), values(keys.size());
    const auto splits = split ? plan.p.splits : 1;
    for (unsigned kv = 0; kv < plan.p.kv_heads; ++kv) {
        for (unsigned row = 0; row < pattern_rows; ++row) {
            reconstruct(mode, row, kv, 0, keys.data() + std::uint64_t{row} * kDim);
            reconstruct(mode, row, kv, 1, values.data() + std::uint64_t{row} * kDim);
        }
        for (unsigned seq = 0; seq < plan.p.sequences; ++seq) {
            const auto pages = ceil_div(plan.lengths[seq], plan.block);
            std::vector<std::vector<std::uint32_t>> counts;
            for (unsigned s = 0; s < splits; ++s) {
                const auto first = std::min(pages, s * ceil_div(pages, splits));
                const auto end = std::min(pages, (s + 1) * ceil_div(pages, splits));
                counts.push_back(pattern_counts(plan, seq, first, end - first));
            }
            for (unsigned group = 0; group < plan.p.group; ++group) {
                const auto head = kv * plan.p.group + group;
                const auto channel = query_channel(seq, head);
                const double scale = query_value(seq, head) * kAttentionScale;
                Attention merged;
                for (const auto& count : counts) {
                    Attention partial;
                    for (unsigned row = 0; row < pattern_rows; ++row) {
                        if (!count[row]) continue;
                        const auto offset = std::uint64_t{row} * kDim;
                        const double weight = count[row] * std::exp(keys[offset + channel] * scale);
                        partial.weight += weight;
                        for (unsigned c = 0; c < kDim; ++c)
                            partial.context[c] += weight * values[offset + c];
                    }
                    merged.weight += partial.weight;
                    for (unsigned c = 0; c < kDim; ++c) merged.context[c] += partial.context[c];
                }
                require(merged.weight > 0, "empty CPU softmax");
                auto* out = result.data() + (std::uint64_t{seq} * plan.heads() + head) * kDim;
                for (unsigned c = 0; c < kDim; ++c) out[c] = merged.context[c] / merged.weight;
            }
        }
    }
    return result;
}

struct Metrics {
    double rmse, cosine, relative_l2;
    unsigned max_ulp;
};

inline Metrics metrics(const std::vector<std::uint16_t>& actual, const std::vector<double>& expected)
{
    require(!actual.empty() && actual.size() == expected.size(), "BF16 metric shape mismatch");
    double sum_error = 0, sum_a = 0, sum_b = 0, dot = 0;
    unsigned max_ulp = 0;
    const auto ordered = [](std::uint16_t bits) {
        return bits & 0x8000 ? 0x8000 - int(bits & 0x7fff) : 0x8000 + int(bits);
    };
    for (std::uint64_t i = 0; i < actual.size(); ++i) {
        const double a = tq_test::bf16_to_float(actual[i]), b = expected[i];
        require(std::isfinite(a) && std::isfinite(b), "nonfinite BF16 output");
        const auto want = tq_test::float_to_bf16_rne(static_cast<float>(b));
        max_ulp = std::max(max_ulp, static_cast<unsigned>(std::abs(ordered(actual[i]) - ordered(want))));
        sum_error += (a - b) * (a - b);
        sum_a += a * a;
        sum_b += b * b;
        dot += a * b;
    }
    const double cosine = sum_a == 0 && sum_b == 0 ? 1 : dot / std::max(1e-40, std::sqrt(sum_a * sum_b));
    return {std::sqrt(sum_error / actual.size()), cosine, std::sqrt(sum_error / std::max(1e-40, sum_b)), max_ulp};
}

struct Filters {
    std::string_view mode = "all", stage = "all", profile = "all";
    unsigned block = 0, device = 0, warmup = 10, iterations = 50;
    bool list = false, help = false, verify = false;

    Filters(int argc, char** argv)
    {
        for (int i = 1; i < argc; ++i) {
            const std::string_view flag(argv[i]);
            if (flag == "--help" || flag == "-h") { help = true; continue; }
            if (flag == "--list") { list = true; continue; }
            if (flag == "--verify") { verify = true; continue; }
            require(i + 1 < argc, "missing filter value (see --help)");
            const std::string_view value(argv[++i]);
            if (flag == "--mode") mode = value;
            else if (flag == "--stage") stage = value;
            else if (flag == "--profile") profile = value;
            else {
                require(!value.empty() && value.size() <= 5 &&
                        value.find_first_not_of("0123456789") == std::string_view::npos,
                    "invalid integer filter");
                const auto number = static_cast<unsigned>(std::stoul(std::string(value)));
                if (flag == "--block-size") block = number;
                else if (flag == "--device") device = number;
                else if (flag == "--warmup") warmup = number;
                else if (flag == "--iterations") iterations = number;
                else throw std::invalid_argument("unknown option (see --help)");
            }
        }
        require(mode == "all" || mode == "lm4" || mode == "lm3" || mode == "e8", "invalid --mode");
        require(stage == "all" || stage == "ingress" || stage == "dequant" || stage == "decode", "invalid --stage");
        require(profile == "all" || std::any_of(std::begin(kProfiles), std::end(kProfiles),
            [&](const auto& p) { return p.name == profile; }), "unknown --profile (see --list)");
        require(block == 0 || block == 64 || block == 128, "block-size must be 64 or 128");
        require(iterations > 0 && iterations <= 512, "iterations must be in [1, 512]");
    }

    void usage(const char* program) const
    {
        std::printf("Usage: %s [--list] [--mode all|lm4|lm3|e8]"
                    " [--stage all|ingress|dequant|decode] [--profile NAME]"
                    " [--block-size 64|128] [--verify] [--device ID]"
                    " [--warmup N] [--iterations N]\n", program);
    }

    template<class Visitor>
    void visit(Visitor visitor) const
    {
        for (const auto& candidate : kProfiles) {
            if (profile != "all" && profile != candidate.name) continue;
            for (const auto size : kBlocks) {
                if (block && block != size) continue;
                const Plan plan(candidate, size);
                for (const auto codec : kModes) {
                    if (mode != "all" && mode != name(codec)) continue;
                    for (const std::string_view s : {"ingress", "dequant", "decode"})
                        if (stage == "all" || stage == s) visitor(plan, codec, s);
                }
            }
        }
    }
};

inline void print_case(const Plan& plan, Mode mode, std::string_view stage)
{
    std::printf("mode=%s stage=%.*s profile=%s dim=%u block=%u seqs=%u context=%u"
                " kv_heads=%u group=%u split_k=%u ragged=%u",
        name(mode), static_cast<int>(stage.size()), stage.data(), plan.p.name, kDim,
        plan.block, plan.p.sequences, plan.p.context, plan.p.kv_heads, plan.p.group,
        plan.p.splits, static_cast<unsigned>(plan.p.ragged));
}

inline void print_accuracy(const Metrics& m)
{
    std::printf(" rmse=%.8g cosine=%.9f relative_l2=%.8g max_bf16_ulp=%u", m.rmse, m.cosine, m.relative_l2, m.max_ulp);
}

struct Timing {
    double p50, p95, mean;
};

inline Timing summarize(std::vector<double> samples)
{
    require(!samples.empty(), "no latency samples");
    std::sort(samples.begin(), samples.end());
    double sum = 0;
    for (const double value : samples) {
        require(std::isfinite(value) && value > 0, "invalid latency sample");
        sum += value;
    }
    const auto n = samples.size();
    return {(samples[(n - 1) / 2] + samples[n / 2]) * 0.5,
        samples[(95 * n + 99) / 100 - 1], sum / n};
}

inline void print_timing(const Timing& time, std::uint64_t bytes)
{
    std::printf(" p50_us=%.3f p95_us=%.3f mean_us=%.3f effective_bytes=%llu effective_GB_s=%.3f",
        time.p50, time.p95, time.mean, static_cast<unsigned long long>(bytes), bytes / time.mean / 1000);
}

}
