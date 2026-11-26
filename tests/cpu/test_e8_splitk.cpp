#include "tests/common/turboquant_production.hpp"
#include "src/turboquant/decode/native/vector/routine_vector_online_softmax.hpp"

#include <limits>

using namespace tq_production;
namespace e8 = e8_decode;

namespace {
struct Merge {
    float maximum = -std::numeric_limits<float>::max(), sum = 0, next = 0, alpha = 0, beta = 0;
    std::array<float, kDim> accumulator{};
    const float* partial = nullptr;
    void update_max_and_weights() {
        next = std::max(maximum, partial[kDim]);
        alpha = std::exp(maximum - next);
        beta = std::exp(partial[kDim] - next);
    }
    void update_denominator() { sum = sum * alpha + partial[kDim + 8] * beta; }
    void update_accumulator() {
        for (unsigned c = 0; c < kDim; ++c)
            accumulator[c] = accumulator[c] * alpha + partial[c] * beta;
    }
    void commit_max() { maximum = next; }
};

void coverage() {
    for (const unsigned length : {0u, 1u, 63u, 64u, 65u, 127u, 128u, 129u, 193u, 4097u, 65535u, 65536u, 131071u})
        for (const unsigned block : {64u, 128u})
            for (unsigned splits = 1; splits <= 8; ++splits) {
                std::vector<unsigned> visits(ceil_div(length, 64), 0);
                unsigned rows = 0;
                for (unsigned split = 0; split < splits; ++split) {
                    const auto r = e8::split_range(length, block, splits, split);
                    require(r.first_tile % (block / 64) == 0, "split is not page aligned");
                    require((r.tiles != 0) == (r.last_rows != 0), "empty split tail is not neutral");
                    if (!r.tiles) continue;
                    require(r.last_rows <= 64, "invalid tail");
                    rows += (r.tiles - 1) * 64 + r.last_rows;
                    for (unsigned t = 0; t < r.tiles; ++t)
                        require(++visits.at(r.first_tile + t) == 1, "overlapping split");
                }
                require(rows == length && std::all_of(visits.begin(), visits.end(), [](auto x) { return x == 1; }), "split misses rows");
            }
    require(e8::workspace_floats(16, 1, 1) == 0, "unsplit must not allocate workspace");
    require(e8::partial_offset(1u << 24, 7, 8) > UINT32_MAX, "partial offsets truncate");
    static_assert(e8::kPartialTail == layout::kPartialTail);
    static_assert(e8::kPartialStride == kDim + layout::kPartialTail);
}

void numerical(unsigned length, unsigned block, unsigned splits) {
    constexpr unsigned heads = e8::kHeads;
    constexpr float sentinel = 123456.0f;
    std::vector<float> work(2 + std::uint64_t{heads} * splits * e8::kPartialStride, sentinel);
    std::vector<std::array<float, kDim>> values(length);
    std::vector<float> scores(length);
    for (unsigned row = 0; row < length; ++row) {
        reconstruct(Mode::e8, row, 0, 1, values[row].data());
        scores[row] = (int(row % 31) - 15) * 4.0f;
    }
    const auto reference_max = length ? *std::max_element(scores.begin(), scores.end()) : 0;
    std::array<long double, kDim> reference{};
    long double denominator = 0;
    for (unsigned row = 0; row < length; ++row) {
        const long double weight = std::exp(static_cast<long double>(scores[row] - reference_max));
        denominator += weight;
        for (unsigned c = 0; c < kDim; ++c) reference[c] += weight * values[row][c];
    }
    for (unsigned head = 0; head < heads; ++head) {
        for (unsigned split = 0; split < splits; ++split) {
            const auto r = e8::split_range(length, block, splits, split);
            auto* partial = work.data() + 1 + e8::partial_offset(head, split, splits);
            std::fill_n(partial, e8::kPartialStride, 0.0f);
            partial[kDim] = -std::numeric_limits<float>::max();
            const auto begin = r.first_tile * 64;
            const auto end = r.tiles ? begin + (r.tiles - 1) * 64 + r.last_rows : begin;
            for (unsigned row = begin; row < end; ++row) partial[kDim] = std::max(partial[kDim], scores[row]);
            for (unsigned row = begin; row < end; ++row) {
                const float weight = std::exp(scores[row] - partial[kDim]);
                partial[kDim + 8] += weight;
                for (unsigned c = 0; c < kDim; ++c) partial[c] += weight * values[row][c];
            }
        }
        Merge merge;
        for (unsigned split = 0; split < splits; ++split) {
            merge.partial = work.data() + 1 + e8::partial_offset(head, split, splits);
            ascend_micro::vector_merge_partials(merge);
        }
        for (unsigned c = 0; c < kDim; ++c) {
            const auto expected = denominator ? reference[c] / denominator : 0;
            const auto actual = merge.accumulator[c] / std::max(1.0f, merge.sum);
            require(std::isfinite(actual) && std::abs(actual - expected) < 2e-5L, "FP32 merge differs from unsplit long-double oracle");
        }
    }
    require(work.front() == sentinel && work.back() == sentinel, "workspace guard modified");
}

void profiles() {
    for (const auto& p : kProfiles) {
        if (std::string_view(p.name) != "split8" && std::string_view(p.name) != "deepseek-v4-flash-64k" && std::string_view(p.name) != "qwen35-gqa6") continue;
        auto selected = p;
        selected.splits = 8;
        for (const auto block : kBlocks)
            require(!Plan(selected, block).device_limit(Mode::e8, "decode"), "requested Split-K profile rejected");
    }
    auto* pointer = reinterpret_cast<void*>(std::uintptr_t{256});
    e8::Buffers b{pointer, pointer, pointer, pointer, pointer, nullptr, nullptr, nullptr};
    e8::Launch c;
    require(e8::valid_launch(c, b), "legacy launch rejected");
    c.lanes = e8::kMaxGrid + 1;
    require(!e8::valid_launch(c, b), "oversized launch grid accepted");
    c.lanes = 1;
    c.splits = 8;
    require(!e8::valid_launch(c, b), "missing split workspace accepted");
    b.tasks = b.workspace = pointer;
    c.context_tiles = 64;
    require(e8::valid_launch(c, b), "batched launch rejected");
    c.splits = 9;
    require(!e8::valid_launch(c, b), "split nine accepted");
    c.splits = 1;
    c.attention_scale = std::numeric_limits<float>::quiet_NaN();
    require(!e8::valid_launch(c, b), "NaN scale accepted");
}
}

int main() try {
    coverage();
    profiles();
    for (const unsigned length : {0u, 1u, 65u, 193u, 1025u})
        for (const auto block : kBlocks)
            for (unsigned split = 1; split <= 8; ++split) numerical(length, block, split);
    std::puts("E8 Split-K page coverage, FP32 merge, empty/odd tails, guards and launch validation PASS");
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
