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

// THE SINGLE-TILE NZ INGRESS CANARY: turboquant_reshape_and_cache_bf16 built with
// -DTURBOQUANT_INGRESS_NZ=1, over EXACTLY one 64-row NZ tile, against the repack oracle.
//
// WHY ONE TILE IS THE WHOLE GATE. The write-time NZ ingress changes nothing about how this
// operator is scheduled: no multi-block reduction, no cross-core SyncAll, no partial
// accumulator, and -- because a strip owns its own 16 tile rows of every column group -- no
// shared destination byte between strips or between grid blocks. What the change CAN get wrong
// is a permutation and four strides. All of that is observable inside one tile, so the gate is
// one tile exercised to the byte rather than a large launch sampled.
//
// The tile is four SEQUENTIAL 16-token strips in ONE grid block (tokensPerCore = 64), which is
// what makes the four tile-row offsets 0, 16, 32 and 48 all live, back to back, through the
// same depth-2 pack ring and the same two vector subcores. A bug in the tile-row term of the
// base address, or a pack slot read a chunk late, lands on a different quarter of the tile and
// is a named byte mismatch here.
//
// WHAT IS COMPARED, AND AGAINST WHAT
//
// 1. BIT PARITY, BOTH PLANES, EVERY BYTE. The 32768 packed bytes of the written page and all
//    512 of its FP32 scale lanes, against tq_nz_ingress::repack_strip_to_nz_tiled -- the
//    element-at-a-time conversion that READS the RFC 3.1 strip contract and WRITES the NZ one
//    through turboquant_layout.h's own address functions. The device derives the same addresses
//    from an open-coded base and three strides, and the permutation from four DeInterleave
//    passes, eight split copies and one Interleave; nothing is shared between the two
//    derivations, so they agree only if both match the contract. The quantised values
//    themselves come from the integer golden oracle, so there is no tolerance anywhere.
//
// 2. THE ORACLE IS AUDITED FIRST. The repack's own write census has to show every owned byte
//    written by exactly its two nibbles in each plane and every owned scale lane written once,
//    and nothing written outside the tile. An oracle whose addresses collide would otherwise
//    agree with a kernel that collided the same way.
//
// 3. GUARD BANDS AND UNWRITTEN TILES. Every allocation carries camodel::kGuardBytes of
//    0xDEADBEEF on both sides. The cache payloads are additionally pre-filled with 0x5A before
//    the launch, and the three physical blocks no strip maps to must still read back as 0x5A
//    afterwards -- so an over-long burst, a wrong tile term or a skipped block-table slot is a
//    named failure rather than a wrong number elsewhere.
//
// WHAT THIS RUNNER CANNOT SEE: whether a cross-core announcement overtook the pipeline work it
// reports. A bit-exact cache says nothing about that; audit the trace.
//
//   --perf [N]    after every check has passed, re-issue the same launch N times and report the
//                 DEVICE-side per-launch median. The writer is a pure function of its inputs,
//                 so the re-launches leave the cache byte for byte as the checks found it.

#include "../common/camodel_session.hpp"
#include "../common/device_timer.hpp"
#include "../common/turboquant_golden_oracle.hpp"
#include "../common/turboquant_nz_ingress.hpp"
#include "../../include/turboquant/turboquant_wave_protocol.hpp"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string_view>
#include <vector>

extern "C" void launch_turboquant_reshape_and_cache_bf16(
    void* stream, const std::uint16_t* key, const std::uint16_t* value,
    std::int8_t* key_cache, std::int8_t* value_cache, float* scale_cache,
    const std::int32_t* slot_mapping, const float* pi_signs,
    const std::int32_t* tables, std::uint32_t num_tokens,
    std::uint32_t num_kv_heads, std::uint32_t head_size, std::uint32_t block_size,
    std::uint32_t num_blocks, std::uint32_t tokens_per_core, float inv_sqrt_len);

namespace nz_canary {

namespace tq = vllm_ascend::turboquant;

constexpr int kHeads = 4;
constexpr int kHeadSize = 256;
constexpr int kTileRows = static_cast<int>(tq::kCubeTileRows);          // 64
constexpr int kStrip = static_cast<int>(tq::kStripTokens);              // 16
constexpr int kStrips = kTileRows / kStrip;                             // 4 strips a tile
constexpr int kBlocks = 4;                                              // physical blocks
constexpr int kTilePage = 1;                                            // the one page written
constexpr std::uint8_t kFill = 0x5Au;  // the pre-launch cache fill, per the canary contract

static_assert(kTileRows % kStrip == 0, "a tile is a whole number of strips");

using tq_golden::ChunkRef;

void require(bool ok, const char* what)
{
    if (!ok) {
        std::fprintf(stderr, "FIXTURE/CHECK FAILED: %s\n", what);
        throw std::runtime_error(what);
    }
}

// The kernel's chunk numbering is value heads first, then key heads; this runner only needs the
// (plane, head) pairs, in any order.
void usage(std::FILE* output, const char* argv0)
{
    std::fprintf(output,
                 "Usage: %s [--perf [iterations]]\n"
                 "  One 64-row NZ tile: four sequential 16-token strips in ONE grid block, on\n"
                 "  physical block %d of %d. Compares every packed byte and every FP32 scale\n"
                 "  lane of that block against the strip->NZ repack oracle, bit for bit, and\n"
                 "  requires the other %d blocks to still read back as 0x%02X.\n"
                 "  --perf times the launch on the DEVICE after every check has passed:\n"
                 "  `iterations` timed launches (default %d, max %d) bracketed by\n"
                 "  ACL_EVENT_TIME_LINE markers. A median wall interval is a timing, not an\n"
                 "  overlap or liveness claim -- audit the trace for those.\n",
                 argv0, kTilePage, kBlocks, kBlocks - 1, kFill, camodel::kDefaultPerfIters,
                 camodel::kMaxPerfIters);
}

int run(int perf_iters)
{
    // ---- the two geometries -----------------------------------------------
    // The DEVICE geometry is the NZ one: a physical block is one whole 64-row tile.
    tq_nz_ingress::NzGeometry nz;
    nz.num_kv_heads = kHeads;
    nz.head_size = kHeadSize;
    nz.block_size = kTileRows;
    nz.num_blocks = kBlocks;
    require(nz.valid(), "the NZ canary geometry is not a whole number of tiles");

    // The ORACLE geometry is the one the golden prefill reference is defined for: a page exactly
    // one 16-token strip high. Only its per-(strip, plane, head) panels are used -- the rotation,
    // the power-of-two scale and the RFC 3.1 packed strip -- and the repack then places those
    // panels at the NZ geometry's global rows. Its own page numbering never leaves this function.
    tq_test::Shape strip_shape;
    strip_shape.num_tokens = kTileRows;
    strip_shape.num_kv_heads = kHeads;
    strip_shape.head_size = kHeadSize;
    strip_shape.block_size = kStrip;
    strip_shape.num_blocks = kStrips;
    require(strip_shape.valid(), "unsupported oracle shape");

    // ---- fixture and oracle ------------------------------------------------
    std::vector<std::int8_t> key_in, value_in;
    tq_golden::build_activations(strip_shape, key_in, value_in);
    const std::vector<std::int8_t> table_blob = tq_golden::build_tables(strip_shape);

    std::vector<std::int32_t> oracle_slots(static_cast<std::size_t>(kTileRows));
    for (int t = 0; t < kTileRows; ++t) oracle_slots[static_cast<std::size_t>(t)] = t;
    const tq_golden::Reference ref = tq_golden::compute_oracle_reference(
        strip_shape, key_in.data(), value_in.data(), oracle_slots.data());
    require(ref.bad_slot_strip < 0, "the oracle's slot mapping breaks its one-strip-one-page form");

    const tq_golden::Audit audit = tq_golden::audit_reference(ref);
    require(audit.nibbles_in_range, "the oracle produced a value outside INT4");
    require(audit.nibbles_covered == 16, "the fixture does not exercise all sixteen nibbles");
    require(audit.clamped_chunks > 0, "the INT4 upper clamp is never taken");
    require(audit.mixed_exponent_chunks > 0, "no chunk carries two different scale exponents");
    require(audit.distinct_exponents >= 3, "the scale exponent barely varies");
    require(audit.pages_distinct, "two chunks of one strip pack to identical pages");
    require(audit.tokens_distinct, "two tokens of one chunk quantise identically");
    require(audit.strips_distinct, "two strips of the tile pack to identical pages");

    const long long composed = tq_golden::audit_tables(strip_shape, table_blob.data());
    require(composed == static_cast<long long>(kHeadSize) * kHeadSize,
            "the resident rotation blocks do not compose to the oracle's matrix");

    // ---- the repack oracle, and its own census ----------------------------
    tq_nz_ingress::NzImage want(nz);
    std::vector<int> slots;
    slots.reserve(static_cast<std::size_t>(kTileRows));
    for (int s = 0; s < kStrips; ++s) {
        const int slot0 = kTilePage * kTileRows + s * kStrip;
        for (int t = 0; t < kStrip; ++t) slots.push_back(slot0 + t);
        for (int plane = 0; plane < tq_test::kPlanes; ++plane)
            for (int head = 0; head < kHeads; ++head) {
                const ChunkRef& panel = ref.chunk(s, plane, head);
                require(!panel.packed.empty(), "the oracle skipped a panel of the tile");
                tq_nz_ingress::StripPanel in;
                in.slot0 = slot0;
                in.plane = plane;
                in.kv_head = head;
                in.packed = panel.packed.data();
                in.scale = panel.scale.data();
                tq_nz_ingress::repack_strip_to_nz_tiled(nz, in, want);
            }
    }
    const tq_nz_ingress::Census census = tq_nz_ingress::audit_repack(nz, want, slots);
    require(census.packed_exact,
            "the repack oracle does not write every byte of the tile exactly twice a plane");
    require(census.scale_exact, "the repack oracle does not write every scale lane exactly once");
    require(census.outside_clean, "the repack oracle wrote outside the tile it was given");
    require(census.packed_bytes ==
                static_cast<std::size_t>(kTileRows) * kHeads * nz.packed_bytes(),
            "the repack oracle covered the wrong number of bytes");
    require(census.scale_lanes ==
                static_cast<std::size_t>(kTileRows) * tq_test::kPlanes * kHeads,
            "the repack oracle covered the wrong number of scale lanes");

    // ---- device buffers ---------------------------------------------------
    camodel::Session session;
    camodel::GuardedDeviceBuffer key(key_in.size() * sizeof(std::uint16_t));
    camodel::GuardedDeviceBuffer value(value_in.size() * sizeof(std::uint16_t));
    camodel::GuardedDeviceBuffer key_cache(nz.cache_bytes());
    camodel::GuardedDeviceBuffer value_cache(nz.cache_bytes());
    camodel::GuardedDeviceBuffer scale_cache(nz.scale_floats() * sizeof(float));
    camodel::GuardedDeviceBuffer slot_mapping(static_cast<std::size_t>(kTileRows) *
                                              sizeof(std::int32_t));
    camodel::GuardedDeviceBuffer pi_signs(kHeadSize * sizeof(float));
    camodel::GuardedDeviceBuffer tables(table_blob.size() * sizeof(std::uint16_t));

    auto* kb = static_cast<std::uint16_t*>(key.host_payload());
    auto* vb = static_cast<std::uint16_t*>(value.host_payload());
    for (std::size_t i = 0; i < key_in.size(); ++i) {
        kb[i] = tq_test::exact_bf16(key_in[i]);
        vb[i] = tq_test::exact_bf16(value_in[i]);
    }
    auto* tb = static_cast<std::uint16_t*>(tables.host_payload());
    for (std::size_t i = 0; i < table_blob.size(); ++i)
        tb[i] = tq_test::exact_bf16(table_blob[i]);
    auto* ps = static_cast<float*>(pi_signs.host_payload());
    for (int ch = 0; ch < kHeadSize; ++ch) ps[ch] = static_cast<float>(tq_golden::pi_sign(ch));

    // The DEVICE slot mapping: the whole tile on physical block kTilePage, so the four strips
    // land at tile rows 0, 16, 32 and 48 of one fractal image.
    auto* sm = static_cast<std::int32_t*>(slot_mapping.host_payload());
    for (int t = 0; t < kTileRows; ++t) sm[t] = kTilePage * kTileRows + t;

    // 0x5A over both cache payloads and the scale plane, so every byte the writer must NOT
    // touch is a distinct, checkable pattern. The guard bands either side stay 0xDEADBEEF.
    std::memset(key_cache.host_payload(), kFill, nz.cache_bytes());
    std::memset(value_cache.host_payload(), kFill, nz.cache_bytes());
    std::memset(scale_cache.host_payload(), kFill, nz.scale_floats() * sizeof(float));
    const std::uint32_t fill_bits = 0x5A5A5A5Au;

    key.upload();
    value.upload();
    slot_mapping.upload();
    pi_signs.upload();
    tables.upload();
    key_cache.upload();
    value_cache.upload();
    scale_cache.upload();

    const auto launch = [&] {
        launch_turboquant_reshape_and_cache_bf16(
            session.stream, static_cast<const std::uint16_t*>(key.payload()),
            static_cast<const std::uint16_t*>(value.payload()),
            static_cast<std::int8_t*>(key_cache.payload()),
            static_cast<std::int8_t*>(value_cache.payload()),
            static_cast<float*>(scale_cache.payload()),
            static_cast<const std::int32_t*>(slot_mapping.payload()),
            static_cast<const float*>(pi_signs.payload()),
            static_cast<const std::int32_t*>(tables.payload()),
            static_cast<std::uint32_t>(kTileRows), static_cast<std::uint32_t>(kHeads),
            static_cast<std::uint32_t>(kHeadSize), static_cast<std::uint32_t>(nz.block_size),
            static_cast<std::uint32_t>(nz.num_blocks), static_cast<std::uint32_t>(kTileRows),
            strip_shape.inv_sqrt_len());
    };

    // ONE grid block: tokensPerCore = 64 is four strips, so the tile is a single chunk stream.
    require(turboquant_wave::TotalStrips(kTileRows, kTileRows) == 1u,
            "the canary must be one grid block");
    require(turboquant_wave::StripsPerBlock(kTileRows) == static_cast<std::uint32_t>(kStrips),
            "the canary must be four strips in that block");
    launch();
    camodel::check(aclrtSynchronizeStream(session.stream), "aclrtSynchronizeStream");

    key_cache.download();
    value_cache.download();
    scale_cache.download();
    key.download();
    value.download();
    slot_mapping.download();
    pi_signs.download();
    tables.download();

    key.assert_guards();
    value.assert_guards();
    key_cache.assert_guards();
    value_cache.assert_guards();
    scale_cache.assert_guards();
    slot_mapping.assert_guards();
    pi_signs.assert_guards();
    tables.assert_guards();

    const auto* kc = static_cast<const std::uint8_t*>(key_cache.host_payload());
    const auto* vc = static_cast<const std::uint8_t*>(value_cache.host_payload());
    const auto* sc = static_cast<const float*>(scale_cache.host_payload());

    // ---- check 1: nothing outside the one mapped tile moved ---------------
    const std::size_t block_bytes =
        static_cast<std::size_t>(kTileRows) * kHeads * nz.packed_bytes();
    const std::size_t block_floats = static_cast<std::size_t>(kTileRows) * nz.scale_slot();
    std::size_t untouched = 0;
    for (int block = 0; block < kBlocks; ++block) {
        if (block == kTilePage) continue;
        for (std::size_t i = 0; i < block_bytes; ++i) {
            const std::size_t at = static_cast<std::size_t>(block) * block_bytes + i;
            require(kc[at] == kFill, "keyCache byte on an unmapped block was overwritten");
            require(vc[at] == kFill, "valueCache byte on an unmapped block was overwritten");
            ++untouched;
        }
        for (std::size_t i = 0; i < block_floats; ++i) {
            const std::size_t at = static_cast<std::size_t>(block) * block_floats + i;
            require(tq_test::float_bits(sc[at]) == fill_bits,
                    "scaleCache float on an unmapped block moved");
        }
    }

    // ---- check 2: bit parity over the tile, both planes -------------------
    const std::size_t page_base = static_cast<std::size_t>(kTilePage) * block_bytes;
    std::size_t key_bytes = 0, value_bytes = 0;
    int failing = 0;
    for (std::size_t i = 0; i < block_bytes && failing < 4; ++i) {
        const std::size_t at = page_base + i;
        if (kc[at] == want.key[at]) {
            ++key_bytes;
        } else {
            ++failing;
            std::fprintf(stderr,
                         "  keyCache byte %zu of the tile (cache offset %zu): device 0x%02X "
                         "oracle 0x%02X\n",
                         i, at, kc[at], want.key[at]);
        }
        if (vc[at] == want.value[at]) {
            ++value_bytes;
        } else {
            ++failing;
            std::fprintf(stderr,
                         "  valueCache byte %zu of the tile (cache offset %zu): device 0x%02X "
                         "oracle 0x%02X\n",
                         i, at, vc[at], want.value[at]);
        }
    }

    // ---- check 3: bit parity over the tile's slot-major scale records -----
    std::size_t scale_lanes = 0;
    for (int row = 0; row < kTileRows && failing < 8; ++row)
        for (int kv = 0; kv < tq_test::kPlanes; ++kv)
            for (int head = 0; head < kHeads; ++head) {
                const std::size_t at =
                    tq_nz_ingress::nz_scale_index(nz, kTilePage * kTileRows + row, head, kv);
                if (tq_test::float_bits(sc[at]) == tq_test::float_bits(want.scale[at])) {
                    ++scale_lanes;
                    continue;
                }
                ++failing;
                std::fprintf(stderr,
                             "  scale lane (row %d, %s head %d) at float %zu: device %.9g "
                             "(0x%08X) oracle %.9g (0x%08X)\n",
                             row, kv == 0 ? "key" : "value", head, at,
                             static_cast<double>(sc[at]), tq_test::float_bits(sc[at]),
                             static_cast<double>(want.scale[at]),
                             tq_test::float_bits(want.scale[at]));
            }
    if (failing != 0) throw std::runtime_error("NZ-tiled cache mismatch");

    // ---- perf, after every check ------------------------------------------
    if (perf_iters > 0) {
        const camodel::PerfStats stats = camodel::measure_launch(
            session.stream, camodel::kDefaultPerfWarmup, perf_iters, launch);
        camodel::print_perf_preamble(camodel::kDefaultPerfWarmup, perf_iters,
                                     !stats.device_timed);
        // Traffic: both BF16 activation planes in, both NZ planes and the slot records out.
        const double rows = double(kTileRows) * double(kHeads);
        const double bytes = rows * double(kHeadSize) * 2.0 * 2.0 +
                             rows * double(nz.packed_bytes()) * 2.0 +
                             double(kTileRows) * double(nz.scale_slot()) * 4.0;
        camodel::print_perf_line("nz-ingress 1 tile/blk x1", stats, double(kTileRows), "tok",
                                 bytes);
    }

    std::printf("PASS turboquant_reshape_and_cache_bf16 (TURBOQUANT_INGRESS_NZ=1): one %d-row "
                "NZ tile, %d sequential %d-token strips in 1 grid block, %d chunks\n",
                kTileRows, kStrips, kStrip, kStrips * 2 * kHeads);
    std::printf("  keyCache   %zu/%zu bytes bit-exact, NZ-tiled\n", key_bytes, block_bytes);
    std::printf("  valueCache %zu/%zu bytes bit-exact, NZ-tiled (token-major, same address "
                "function as the key plane)\n",
                value_bytes, block_bytes);
    std::printf("  scaleCache %zu/%d FP32 bit-exact, slot-interleaved at a %d-float pitch\n",
                scale_lanes, kTileRows * tq_test::kPlanes * kHeads, nz.scale_slot());
    std::printf("  tile rows 0, %d, %d and %d all live: %d column groups of %d x %d bytes a "
                "(strip, head, plane), %d bursts of %d bytes %d apart\n",
                kStrip, 2 * kStrip, 3 * kStrip, nz.packed_bytes() / int(tq::kOperandC0), kStrip,
                int(tq::kOperandC0), nz.packed_bytes() / int(tq::kOperandC0),
                kStrip * int(tq::kOperandC0), kTileRows * int(tq::kOperandC0));
    std::printf("  repack oracle census: %zu bytes x %d writes and %zu scale lanes x 1, none "
                "outside the tile\n",
                census.packed_bytes, tq_test::kNibblesPerByte * tq_test::kPlanes,
                census.scale_lanes);
    std::printf("  %zu cache bytes on the %d unmapped blocks still 0x%02X, %d/%d guard bytes "
                "intact\n",
                untouched, kBlocks - 1, kFill, 8 * 2 * static_cast<int>(camodel::kGuardBytes),
                8 * 2 * static_cast<int>(camodel::kGuardBytes));
    std::printf("  every nibble and every scale reproduced by the integer oracle through "
                "turboquant_layout.h's own address functions: no tolerance anywhere\n");
    std::printf("  K0 = %d (BF16 fractal 16x16), Hadamard group 32 = 2 fractals, %d resident "
                "blocks of 32x32 BF16, %lld composed entries\n",
                32 / 2, 2 * tq_golden::group_count(strip_shape), composed);
    return 0;
}

int run_main(int argc, char** argv)
{
    int perf_iters = 0;
    bool saw_perf = false;
    const auto parse = [](std::string_view text, int& out, int lo, int hi) {
        int value = 0;
        const auto got = std::from_chars(text.data(), text.data() + text.size(), value);
        if (got.ec != std::errc() || got.ptr != text.data() + text.size() || value < lo ||
            value > hi)
            return false;
        out = value;
        return true;
    };
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--help" || arg == "-h") {
            usage(stdout, argv[0]);
            return 0;
        }
        if (arg == "--perf") {
            if (saw_perf) {
                usage(stderr, argv[0]);
                return 2;
            }
            saw_perf = true;
            perf_iters = camodel::kDefaultPerfIters;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                if (!parse(argv[i + 1], perf_iters, 1, camodel::kMaxPerfIters)) {
                    usage(stderr, argv[0]);
                    return 2;
                }
                ++i;
            }
            continue;
        }
        usage(stderr, argv[0]);
        return 2;
    }
    return run(perf_iters);
}

}  // namespace nz_canary

int main(int argc, char** argv)
{
    try {
        return nz_canary::run_main(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
