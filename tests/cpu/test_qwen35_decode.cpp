#include "../common/qwen35_decode_benchmark.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace bench = qwen35_bench;

void check_predicate_lifetime()
{
    using Layout = bench::tq::TelemetryPredicateLayout;
    for (std::uint32_t improved = 0; improved < 8; ++improved) {
        std::array<std::uint8_t, Layout::kArenaFloats * sizeof(float)> arena{};
        arena[Layout::kPosition * sizeof(float)] = 1u << bench::tq::kTelemetryPosLane;
        arena[Layout::kPage * sizeof(float)] = 1u << bench::tq::kTelemetryPageLane;
        std::uint32_t dynamic = 0;
        for (std::uint32_t head = 0; head < 3; ++head)
            if ((improved >> head) & 1u) dynamic |= 0xffu << (head * 8);
        std::memcpy(arena.data() + Layout::kArgmax * sizeof(float), &dynamic, sizeof(dynamic));
        std::uint16_t position_mask = 0, page_mask = 0;
        std::memcpy(&position_mask, arena.data() + Layout::kPosition * sizeof(float), sizeof(position_mask));
        std::memcpy(&page_mask, arena.data() + Layout::kPage * sizeof(float), sizeof(page_mask));
        std::array<float, 16> tail{};
        tail[bench::tq::kPartialMaxLane] = 1.25f;
        tail[bench::tq::kPartialSumLane] = 64.0f;
        for (std::size_t lane = 0; lane < tail.size(); ++lane) {
            if ((position_mask >> lane) & 1u) tail[lane] = 37.0f;
            if ((page_mask >> lane) & 1u) tail[lane] = 2.0f;
        }
        if (tail[bench::tq::kPartialMaxLane] != 1.25f || tail[bench::tq::kPartialSumLane] != 64.0f ||
            tail[bench::tq::kTelemetryPosLane] != 37.0f || tail[bench::tq::kTelemetryPageLane] != 2.0f)
            throw std::runtime_error("argmax scratch corrupted the partial-tail publication masks");
    }
    std::puts("PASS telemetry: all three-head improvement masks preserve softmax max/sum");
}

int main()
{
    try {
        check_predicate_lifetime();
        for (auto block : bench::kBlocks) for (auto batch : bench::kBatches)
            for (auto context : bench::kContexts) for (auto splits : bench::kSplits) {
                const bench::Geometry g{block, batch, context, splits};
                const bench::Plan p(g);
                if (!p.valid()) throw std::runtime_error(p.reason);
                std::vector<bool> seen(g.slots() / 64, false);
                for (std::uint32_t token = 0; token < batch; ++token)
                    for (std::uint32_t split = 0; split < splits; ++split) {
                        bench::tiling::TurboQuantDecodeTilingData d{};
                        std::memcpy(&d, p.descriptors.data() +
                            (std::size_t(token) * splits + split) * bench::tiling::kDecodeTilingWords, sizeof(d));
                        for (std::uint32_t tile = 0; tile < d.num_tiles; ++tile) {
                            const auto page = d.block_start_idx + tile / (block / 64);
                            const auto physical = p.table[std::size_t(token) * g.pages() + page];
                            const auto physical_tile = std::size_t(physical) * (block / 64) + tile % (block / 64);
                            if (physical_tile >= seen.size() || seen[physical_tile])
                                throw std::runtime_error("duplicate or out-of-range physical tile");
                            seen[physical_tile] = true;
                        }
                    }
                if (std::find(seen.begin(), seen.end(), false) != seen.end())
                    throw std::runtime_error("tiling skipped a physical tile");
            }
        for (auto batch : bench::kBatches) {
            const bench::Geometry a{128, batch, 2048, 1}, b{1024, batch, 2048, 1};
            const bench::Plan pa(a), pb(b);
            const bench::Fixture fa(a), fb(b);
            if (fa.key != fb.key || fa.value != fb.value || fa.scales != fb.scales)
                throw std::runtime_error("physical NZ image depends on page size");
            const auto ea = fa.expected(pa), eb = fb.expected(pb);
            double error2 = 0, reference2 = 0;
            for (std::size_t i = 0; i < ea.size(); ++i) {
                error2 += (ea[i] - eb[i]) * (ea[i] - eb[i]);
                reference2 += ea[i] * ea[i];
            }
            const double relative = std::sqrt(error2 / reference2);
            std::printf("host: batch=%u context=2048 pages=128/1024 relative_l2=%.9g\n", batch, relative);
            if (relative >= 1e-12) throw std::runtime_error("host golden depends on page size");
        }
        std::puts("PASS Qwen3.5: 64 physical tile partitions and both 2K page layouts");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL Qwen3.5 host regression: %s\n", error.what());
        return 1;
    }
}
