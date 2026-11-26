#include "tests/common/turboquant_production.hpp"

#define E8_PARALLEL_CPU
#include "src/turboquant/decode/e8/e8_parallel_decode.hpp"
#undef E8_PARALLEL_CPU

#include <chrono>

namespace {
using namespace tq_production;

std::uint64_t verify_tiling(const Plan& plan)
{
    auto request = tiling::decode_tiling_request(plan.native_launch());
    alignas(tiling::TurboQuantDecodeTilingData) TilingWords words;
    require(tiling::build_decode_tiling(request, plan.lengths.data(), plan.table.data(), words.data()) ==
        tiling::DecodeTilingStatus::kOk, "host tiling builder rejected the profile");
    std::uint64_t live_tiles = 0;
    for (unsigned seq = 0; seq < request.tokens; ++seq) {
        unsigned covered_pages = 0, covered_rows = 0;
        for (unsigned split = 0; split < request.splits; ++split) {
            const auto* row = words.data() + (std::uint64_t{seq} * request.splits + split) * tiling::kDecodeTilingWords;
            const auto tiles = row[tiling::kTilingWordNumTiles];
            const auto tail = row[tiling::kTilingWordTailRows];
            if (!(row[tiling::kTilingWordFlags] & tiling::kDecodeSplitLive)) {
                require(tiles == 0 && tail == 0, "dead split contains live rows");
                continue;
            }
            require(row[tiling::kTilingWordBlockStart] == covered_pages, "host split overlaps or skips a page");
            covered_pages += row[tiling::kTilingWordBlocksInSplit];
            covered_rows += (tiles - (tail != 0)) * kTileRows + tail;
            live_tiles += tiles;
        }
        require(covered_pages == ceil_div(plan.lengths[seq], plan.block) &&
                covered_rows == static_cast<unsigned>(plan.lengths[seq]), "host split misses a live token");
    }
    require(live_tiles == plan.read_tiles, "host tiling misses a live tile");
    return live_tiles;
}

std::uint64_t check_lm_frame(const Plan& plan, Mode mode)
{
    const auto bits = mode == Mode::lm4 ? 4u : 3u;
    const auto bytes = std::uint64_t{kDim} * bits / 8;
    std::vector<std::uint8_t> payload(bytes);
    for (unsigned kv = 0; kv < plan.p.kv_heads; ++kv)
        for (unsigned plane = 0; plane < 2; ++plane)
            for (unsigned row = 0; row < plan.block; ++row) {
                std::fill(payload.begin(), payload.end(), 0);
                for (unsigned c = 0; c < kDim; ++c) {
                    const auto col = mode == Mode::lm4 ? (c % (kDim / 2)) * 2 + c / (kDim / 2) : c;
                    const auto index = std::uint64_t{col} * bits;
                    const auto code = (lm_code(mode, row, kv, plane, c) + (1u << (bits - 1))) & ((1u << bits) - 1);
                    for (unsigned bit = 0; bit < bits; ++bit)
                        payload[(index + bit) / 8] |= ((code >> bit) & 1) << ((index + bit) % 8);
                }
                for (unsigned c = 0; c < kDim; ++c) {
                    const auto col = mode == Mode::lm4 ? (c % (kDim / 2)) * 2 + c / (kDim / 2) : c;
                    const auto index = std::uint64_t{col} * bits;
                    const auto byte = index / 8;
                    const auto packed = payload[byte] |
                        (byte + 1 < bytes ? std::uint32_t{payload[byte + 1]} << 8 : 0u);
                    const auto slot = ((packed >> (index % 8)) + (1u << (bits - 1))) & ((1u << bits) - 1);
                    require(slot == lm_code(mode, row, kv, plane, c), "Lloyd-Max packed code mismatch");
                }
            }
    require(std::is_sorted(lm3_centroids().begin(), lm3_centroids().end()) &&
        std::abs(lm3_centroids().back() - 2.15195) < 0.001, "unexpected 3-bit Gaussian Lloyd-Max codebook");
    return std::uint64_t{plan.block} * plan.p.kv_heads * 2 * bytes;
}

std::uint64_t check_e8_frames(const Plan& plan, bool dequant)
{
    constexpr std::size_t guard = 32;
    constexpr std::uint8_t sentinel = 0xec;
    for (unsigned kv = 0; kv < plan.p.kv_heads; ++kv)
        for (unsigned plane = 0; plane < 2; ++plane)
            for (unsigned first = 0; first < plan.block; first += kTileRows) {
                auto frame = packet_fixture(first, kv, plane);
                std::array<std::uint8_t, sizeof(Packet) + 2 * guard> storage;
                storage.fill(sentinel);
                std::memcpy(storage.data() + guard, frame.data(), sizeof(frame));
                std::memcpy(frame.data(), storage.data() + guard, sizeof(frame));
                verify_packet(frame, first, kv, plane);
                require(std::all_of(storage.begin(), storage.begin() + guard,
                    [](auto byte) { return byte == sentinel; }) &&
                    std::all_of(storage.end() - guard, storage.end(), [](auto byte) { return byte == sentinel; }),
                    "E8 host packet canary modified");
                if (!dequant) continue;
                for (unsigned row = 0; row < kTileRows; ++row)
                    for (unsigned group = 0; group < packet::kLabelsPerRow; ++group) {
                        const auto label = e8_label(first + row, kv, plane, group);
                        auto d = e8_parallel::Decode(label);
                        const int actual[8]{d.a, d.b, d.c, d.d, d.e, d.f, d.g, d.h};
                        int expected[8];
                        e8_golden(label, expected);
                        require(std::equal(std::begin(actual), std::end(actual), expected),
                            "E8 production decoder disagrees with frozen CPU golden");
                    }
            }
    return std::uint64_t{plan.block / kTileRows} * plan.p.kv_heads * 2 * sizeof(Packet);
}

void run_host(const Plan& plan, Mode mode, std::string_view stage)
{
    if (stage != "decode") {
        const auto bytes = mode == Mode::e8 ? check_e8_frames(plan, stage == "dequant") : check_lm_frame(plan, mode);
        std::printf(" status=PASS checked_sample_bytes=%llu",
            static_cast<unsigned long long>(bytes));
        return;
    }
    const auto tiles = verify_tiling(plan);
    const auto expected = attention_reference(plan, mode, false);
    const auto split = attention_reference(plan, mode, true);
    std::vector<std::uint16_t> actual(split.size());
    for (std::size_t i = 0; i < split.size(); ++i) {
        require(std::abs(split[i] - expected[i]) < 1e-12, "CPU split merge disagrees with unsplit softmax");
        actual[i] = tq_test::float_to_bf16_rne(static_cast<float>(split[i]));
    }
    const auto m = metrics(actual, expected);
    require(m.max_ulp <= 1 && m.relative_l2 < 0.003 && m.cosine > 0.99999, "CPU BF16 decode metric failed");
    std::printf(" status=PASS live_tiles=%llu", static_cast<unsigned long long>(tiles));
    print_accuracy(m);
}

}

int main(int argc, char** argv) try
{
    const Filters filters(argc, argv);
    if (filters.help) { filters.usage(argv[0]); return 0; }
    filters.visit([&](const Plan& plan, Mode mode, std::string_view stage) {
        print_case(plan, mode, stage);
        if (filters.list) {
            const auto limitation = plan.device_limit(mode, stage);
            std::printf(" cache_bytes=%llu device=%s%s%s",
                static_cast<unsigned long long>(plan.kv_bytes(mode, true)),
                limitation ? "UNSUPPORTED" : "READY", limitation ? " reason=" : "",
                limitation ? limitation : "");
        } else {
            const auto start = std::chrono::steady_clock::now();
            run_host(plan, mode, stage);
            const std::chrono::duration<double, std::micro> elapsed = std::chrono::steady_clock::now() - start;
            std::printf(" host_validation_us=%.3f", elapsed.count());
        }
        std::putchar('\n');
    });
    return 0;
}
catch (const std::exception& error) {
    std::fprintf(stderr, "TurboQuant host validation failed: %s\n", error.what());
    return 1;
}
