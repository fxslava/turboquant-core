// Bounded production-entry gate adapted from ascend-micro-harness d026b81's
// tests/sim/kernel/test_camodel_fused_decode_opt.cpp. No diagnostic kernel wrapper.
// Uniform key rows make softmax weights exactly one; a FP64 weighted mean is the
// independent oracle. Core's wire contract is signed nibble * stored scale.
#include "../../include/turboquant/turboquant_fused_decode_opt.hpp"
#include "../../src/host/turboquant_telemetry.h"
#include "../common/fused_decode_fixture.hpp"
#ifndef TQ_WAVE_V11_HOST_ONLY
#include "../common/camodel_session.hpp"
#endif

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fd = ascend_micro::fused_decode_opt;
namespace ft = ascend_micro::fused_decode;
namespace tel = ascend_micro::fused_decode_telemetry;
namespace tq = vllm_ascend::turboquant;
constexpr unsigned kTiles = 4, kTail = 16, kHeads = 6, kDim = 256;
constexpr unsigned kContext = (kTiles - 1) * 64 + kTail;
constexpr unsigned kPages = 6, kPageRows = 128, kScaleLanes = 8;
constexpr std::uint32_t kPoison = 0x7fc0beefu;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

struct Fixture {
    fd::Launch config;
    std::vector<float> query, scales;
    std::vector<std::uint8_t> key, value;
    std::array<std::int32_t, 8> table{};
    std::array<std::int32_t, 8> lengths{};
    std::array<std::uint32_t, 8> modes{};
    std::vector<std::uint32_t> tiling;
    std::array<std::uint16_t, kDim> expected{};

    Fixture() : query(kHeads * kDim, 1.0f / kDim),
                scales(kPages * kPageRows * kScaleLanes, tq_test::bits_float(kPoison)),
                key(kPages * kPageRows * kDim / 2), value(key.size())
    {
        config.heads = kHeads;
        config.reduce_tasks_per_block = kHeads;
        config.element_type = 1;
        config.num_blocks = kPages;
        config.scale = 1.0f;
        for (std::size_t i = 0; i < key.size(); ++i)
            key[i] = value[i] = static_cast<std::uint8_t>(kPoison >> (8 * (i % 4)));
        table.fill(-1);
        table[0] = 1;
        table[1] = 3;
        lengths[0] = kContext;
        for (unsigned tile = 0; tile < kTiles; ++tile) {
            const unsigned physical = static_cast<unsigned>(table[tile / 2]);
            for (unsigned row = (tile % 2) * 64; row < (tile % 2 + 1) * 64; ++row) {
                const auto si = (physical * kPageRows + row) * kScaleLanes;
                scales[si] = scales[si + 1] = 2.0f;
                for (unsigned col = 0; col < kDim / 2; ++col) {
                    const auto offset = physical * kPageRows * kDim / 2 +
                        tq::NzTiledPackedByte(row, 0, col, 1, kDim / 2);
                    key[offset] = 0x33;  // every dequantized key is exactly 6
                    const auto lo = (tile + col) % 7 + 1;
                    const auto hi = (tile + (col + 3) % 128) % 7 + 1;
                    value[offset] = static_cast<std::uint8_t>(lo | (hi << 4));
                }
            }
        }
        // An independent formula over tiles, not the device's online recurrence.
        for (unsigned d = 0; d < kDim; ++d) {
            double numerator = 0;
            for (unsigned tile = 0; tile < kTiles; ++tile) {
                const auto col = d % 128;
                const auto lane = d < 128 ? col : (col + 3) % 128;
                const auto level = (tile + lane) % 7 + 1;
                numerator += (tile + 1 == kTiles ? kTail : 64) * (2.0 * level);
            }
            expected[d] = fused_decode_fixture::reference_bfloat16(
                static_cast<float>(numerator / kContext));
        }
        tiling.assign(ft::decode_tiling_words(config.tokens, config.splits, config.grid), 0);
        const auto status = ft::build_decode_tiling(ft::decode_tiling_request(config),
            lengths.data(), table.data(), tiling.data());
        require(status == ft::DecodeTilingStatus::kOk, ft::decode_tiling_status_text(status));
    }

    void host_check() const
    {
        require(fd::valid_launch(config.grid, config.tokens, config.heads, config.kv_heads,
            config.head_size, config.block_size, config.max_blocks, config.splits,
            config.heads_per_task, config.tasks_per_block, config.reduce_tasks_per_block),
            "invalid bounded launch");
        // Decode the actual packed cache by rows and compare it with the independent
        // tile formula. This catches lane-order, page-map and tail errors in this fixture.
        for (unsigned d = 0; d < kDim; ++d) {
            double numerator = 0;
            for (unsigned row = 0; row < kContext; ++row) {
                const auto physical = static_cast<unsigned>(table[row / kPageRows]);
                const auto within = row % kPageRows;
                const auto offset = physical * kPageRows * kDim / 2 +
                    tq::NzTiledPackedByte(within, 0, d % 128, 1, kDim / 2);
                const unsigned nibble = d < 128 ? value[offset] & 15 : value[offset] >> 4;
                numerator += 2.0 * nibble;
                require(key[offset] == 0x33, "nonuniform live key in exact fixture");
            }
            require(tq_test::float_to_bf16_rne(static_cast<float>(numerator / kContext)) ==
                        expected[d], "packed-cache oracle disagrees with independent mean");
        }
        std::printf("PASS Wave-v11 host fixture: %u tiles, %u rows, %u heads, "
                    "signed-nibble wire, independent packed-cache mean\n",
                    kTiles, kContext, kHeads);
    }
};

#ifndef TQ_WAVE_V11_HOST_ONLY
using Buffer = camodel::GuardedDeviceBuffer;

template <class T>
void upload(Buffer& buffer, const T* source, std::size_t count)
{
    require(count * sizeof(T) <= buffer.payload_bytes, "fixture upload overrun");
    std::fill(buffer.host.begin() + 16, buffer.host.end() - 16, kPoison);
    std::memcpy(buffer.host_payload(), source, count * sizeof(T));
    buffer.upload();
}

int run(const Fixture& fixture)
{
    camodel::Session session;
    Buffer query(fixture.query.size() * 4), key(fixture.key.size()), value(fixture.value.size()),
        scales(fixture.scales.size() * 4), table(32), lengths(32), modes(32),
        tiling(fused_decode_fixture::aligned(fixture.tiling.size() * 4)),
        workspace(kHeads * (kDim + 16) * 4 + 4096), output((kHeads + 2) * kDim * 2);
    upload(query, fixture.query.data(), fixture.query.size());
    upload(key, fixture.key.data(), fixture.key.size());
    upload(value, fixture.value.data(), fixture.value.size());
    upload(scales, fixture.scales.data(), fixture.scales.size());
    upload(table, fixture.table.data(), fixture.table.size());
    upload(lengths, fixture.lengths.data(), fixture.lengths.size());
    upload(modes, fixture.modes.data(), fixture.modes.size());
    upload(tiling, fixture.tiling.data(), fixture.tiling.size());
    for (Buffer* buffer : {&workspace, &output}) {
        std::fill(buffer->host.begin() + 16, buffer->host.end() - 16, kPoison);
        buffer->upload();
    }
    const std::array<Buffer*, 8> inputs{&query, &key, &value, &scales, &table, &lengths,
                                      &modes, &tiling};
    std::vector<std::vector<std::uint32_t>> snapshots;
    for (auto* input : inputs) snapshots.push_back(input->host);
    const fd::Buffers buffers{query.payload(), key.payload(), value.payload(), scales.payload(),
        table.payload(), lengths.payload(), modes.payload(), workspace.payload(), output.payload(),
        tiling.payload()};
    std::printf("launch Wave-v11 production entry: grid 1 tiles 4 tail 16 heads 6/1 N 208\n");
    launch_fused_decode_opt(session.stream, &fixture.config, &buffers);
    camodel::check(aclrtSynchronizeStream(session.stream), "Wave-v11 synchronize");
    output.download();
    workspace.download();
    output.assert_guards();
    workspace.assert_guards();
    const auto* actual = static_cast<const std::uint16_t*>(output.host_payload());
    for (unsigned h = 0; h < kHeads; ++h)
        for (unsigned d = 0; d < kDim; ++d)
            if (actual[h * kDim + d] != fixture.expected[d]) {
                std::fprintf(stderr, "BF16 mismatch h=%u d=%u got=%04x expected=%04x\n",
                    h, d, actual[h * kDim + d], fixture.expected[d]);
                throw std::runtime_error("Wave-v11 output is not 0 BF16 ULP");
            }
    const auto* words = static_cast<const std::uint32_t*>(output.host_payload());
    for (std::size_t i = kHeads * kDim / 2; i < output.payload_bytes / 4; ++i)
        require(words[i] == kPoison, "inactive output changed");
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        inputs[i]->download();
        inputs[i]->assert_guards();
        require(inputs[i]->host == snapshots[i], "read-only input changed");
    }
    const auto* partials = static_cast<const std::uint32_t*>(workspace.host_payload());
#if ASCEND_TQ_DECODE_TELEMETRY
    const tel::Geometry geometry{1, kHeads, 1, kDim, kPageRows, 8, kPages, 4096};
    const auto verified = tel::verify(partials, workspace.payload_bytes / 4, geometry,
        fixture.lengths.data(), fixture.table.data(), kPoison);
    if (!verified.ok) {
        std::fprintf(stderr, "workspace word %zu: %s\n", verified.word, verified.what);
        throw std::runtime_error("invalid Wave-v11 workspace telemetry");
    }
#else
    for (std::size_t i = 0; i < workspace.payload_bytes / 4; ++i)
        require(partials[i] == kPoison, "inactive workspace changed");
#endif
    std::printf("PASS Wave-v11 production entry: 0 BF16 ULP, 4 tiles, "
                "inputs unchanged, output/workspace guards and telemetry intact\n");
    return 0;
}
#endif
}  // namespace

int main(int argc, char** argv)
{
    try {
        require(argc == 1 || (argc == 2 && std::string(argv[1]) == "--host-check"),
                "usage: test_wave_v11_exact [--host-check]");
        Fixture fixture;
        fixture.host_check();
#ifdef TQ_WAVE_V11_HOST_ONLY
        return 0;
#else
        return argc == 2 ? 0 : run(fixture);
#endif
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL Wave-v11 exact gate: %s\n", error.what());
        return 1;
    }
}

