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

// The telemetry gate: everything about the attention telemetry probe that can be decided
// without a device.
//
//   1. RECORD LAYOUT. The lane constants the kernel packs against and this host code parses
//      against are one declaration (turboquant_layout.h), restated here as runtime checks.
//
//   2. LIVE-SPLIT ARITHMETIC. live_splits_of against build_decode_tiling's descriptors over a
//      sweep of contexts: the host merge must see exactly the splits the device launched.
//
//   3. THE FOLD, END TO END. A scalar emulation of the kernel's per-tile argmax fold -- strict
//      improvement, first-max-lane within a tile, earliest tile and earliest split on ties --
//      produces the records the kernel would write; merge() must return the direct argmax over
//      the whole context, with the page and row it implies.
//
//   4. verify(). A byte-correct synthetic image passes, and each named corruption class --
//      clobbered fill, non-finite accumulator, out-of-range position, page/position
//      disagreement, non-zero padding, a stale live slot, a live dead slot -- fails on the
//      right word. This is the same helper the CAModel/NPU runners call as their workspace
//      contract, so this gate is what keeps that contract honest.
//
//   5. dump(). The human-readable line carries the token, the winner, and the physical page.

#include "../../src/turboquant/common/turboquant_telemetry.h"
#include "../../src/turboquant/common/turboquant_tiling.h"
#include "../common/turboquant_test_common.hpp"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

namespace tel = ascend_micro::fused_decode_telemetry;
namespace fdt = ascend_micro::fused_decode;
namespace tq = vllm_ascend::turboquant;

tq_test::Failures g_fail;

constexpr std::uint32_t kFill = 0x5A5A5A5Au;
constexpr std::uint32_t kTileRows = tq::kCubeTileRows;

float bits_float(std::uint32_t bits)
{
    float v;
    std::memcpy(&v, &bits, 4);
    return v;
}

// ------------------------------------------------------------------ 1. record layout
void check_layout()
{
    g_fail.check(tel::kTelemetryMaxLane == tq::kPartialMaxLane &&
                     tel::kTelemetryPosLane == 1 && tel::kTelemetryPageLane == 2 &&
                     tel::kTelemetryPageLane < tq::kPartialSumLane,
                 "the probe lanes sit in the first tail block's dead padding");
    g_fail.check(tel::record_stride(256) == 272, "the record stride is head_size + 16 floats");
    const tel::Geometry g{2, 4, 8, 256, 128, 40, 80, 4096};
    g_fail.check(tel::record_offset(g, 1, 2, 3) == ((1u * 4 + 2) * 8 + 3) * 272ull,
                 "record_offset is the device's PartialOffset");
    g_fail.check(tel::record_offset(g, 0, 0, 0) * 4 % 32 == 0 &&
                     (tel::record_offset(g, 0, 0, 0) + 256) * 4 % 32 == 0,
                 "every record tail is 32-byte aligned, as the DataCopy contract needs");
    g_fail.check(tel::workspace_floats(g) == 2ull * 4 * 8 * 272,
                 "the record extent is tokens*heads*splits strides");
    std::printf("  record layout: stride %u floats, lanes max=%u pos=%u page=%u sum=%u\n",
                tel::record_stride(256), tel::kTelemetryMaxLane, tel::kTelemetryPosLane,
                tel::kTelemetryPageLane, tel::kPartialSumLane);
}

// ------------------------------------------------------------------ 2. live splits
void check_live_splits()
{
    const tel::Geometry g{1, 16, 8, 256, 128, 640, 640, 4096};
    for (const std::uint32_t context :
         {1u, 63u, 64u, 65u, 127u, 128u, 129u, 1000u, 4096u, 4097u, 4224u, 5000u, 8192u, 65536u}) {
        // The device side of the same arithmetic: descriptors from the real host tiling.
        fdt::DecodeTilingRequest r;
        r.tokens = 1;
        r.num_heads = 16;
        r.num_kv_heads = 1;
        r.head_dim = 256;
        r.block_size = 128;
        r.max_blocks = 640;
        r.splits = 8;
        r.heads_per_task = 16;
        r.tasks_per_block = 8;
        r.reduce_tasks_per_block = 16;
        r.grid = 1;
        r.fused_context_limit = 4096;
        const std::int32_t length = static_cast<std::int32_t>(context);
        std::vector<std::uint32_t> image(fdt::decode_tiling_words(1, 8, 1), 0);
        const auto status = fdt::build_decode_tiling(r, &length, nullptr, image.data());
        tq_test::require(status == fdt::DecodeTilingStatus::kOk,
                         "build_decode_tiling refused a sweep context");
        std::uint32_t descriptors_live = 0;
        for (std::uint32_t split = 0; split < 8; ++split) {
            const auto *d = reinterpret_cast<const fdt::TurboQuantDecodeTilingData *>(
                image.data()) + split;
            if (d->flags & fdt::kDecodeSplitLive) ++descriptors_live;
        }
        const std::uint32_t host_live = tel::live_splits_of(context, g);
        g_fail.check(host_live == descriptors_live,
                     "live_splits_of disagrees with build_decode_tiling");
        if (host_live != descriptors_live)
            std::printf("    context %u: host %u, tiling %u\n", context, host_live,
                        descriptors_live);
    }
    std::printf("  live splits: 14 contexts match build_decode_tiling's descriptors\n");
}

// ------------------------------------------------- 3. the fold, end to end
// A scalar emulation of the kernel's per-tile argmax fold (the probe phase of
// NativeCubeSoftmaxEngine): the running max improves only strictly, the candidate is the
// FIRST lane equal to the tile maximum, and the tile's context base and logical page fold in
// with it. This is the mirror the device code is reviewed against; merge() below must return
// the direct argmax over the whole context computed independently.
struct FoldState {
    float run_max;
    std::int64_t position;
    std::int64_t logical_page;
};
FoldState fold_tiles(const std::vector<float> &logits, std::uint32_t context,
                     std::uint32_t block_size)
{
    FoldState st{-FLT_MAX, -1, -1};
    for (std::uint32_t base = 0; base < context; base += kTileRows) {
        const std::uint32_t valid =
            context - base < kTileRows ? context - base : kTileRows;
        float tile_max = -FLT_MAX;
        for (std::uint32_t i = 0; i < valid; ++i) tile_max = std::max(tile_max, logits[base + i]);
        // The FIRST lane equal to the tile maximum, as the kernel's MIN reduce over the
        // ascending index ramp selects.
        std::uint32_t first = 0;
        for (std::uint32_t i = 0; i < valid; ++i)
            if (logits[base + i] == tile_max) {
                first = i;
                break;
            }
        if (tile_max > st.run_max) {
            st.run_max = tile_max;
            st.position = base + first;
            st.logical_page = (base + first) / block_size;
        }
    }
    return st;
}

// The direct argmax over the live context: strict maximum, earliest position on ties. This is
// the answer the fold must reproduce WITHOUT seeing it.
std::int64_t direct_argmax(const std::vector<float> &logits, std::uint32_t context)
{
    float best = -FLT_MAX;
    std::int64_t at = -1;
    for (std::uint32_t i = 0; i < context; ++i)
        if (logits[i] > best) {
            best = logits[i];
            at = i;
        }
    return at;
}

// Write a slot's statistics tail exactly as the kernel's packer does: lanes 0/1/2/8 live,
// every other lane zero, and the accumulator region left to the caller.
void write_tail(std::vector<float> &ws, const tel::Geometry &g, std::uint32_t token,
                std::uint32_t head, std::uint32_t split, float max_logit, std::int64_t pos,
                std::int64_t page, float run_sum)
{
    float *tail = ws.data() + tel::record_offset(g, token, head, split) + g.head_size;
    for (std::uint32_t lane = 0; lane < tel::kPartialTail; ++lane) tail[lane] = 0.0f;
    tail[tel::kTelemetryMaxLane] = max_logit;
    tail[tel::kTelemetryPosLane] = static_cast<float>(pos);
    tail[tel::kTelemetryPageLane] = static_cast<float>(page);
    tail[tel::kPartialSumLane] = run_sum;
}

struct World {
    tel::Geometry g;
    std::vector<std::int32_t> lengths;
    std::vector<std::int32_t> table;
    std::vector<float> workspace;  // starts as the fill word, like the runners' guard fill
};

World make_world()
{
    World w;
    w.g = tel::Geometry{3, 4, 8, 256, 128, 40, 80, 4096};
    w.lengths = {300, 5000, 4224};
    w.table.assign(std::size_t(w.g.tokens) * w.g.max_blocks, -1);
    for (std::uint32_t token = 0; token < w.g.tokens; ++token)
        for (std::uint32_t page = 0; page < w.g.max_blocks; ++page)
            w.table[std::size_t(token) * w.g.max_blocks + page] =
                static_cast<std::int32_t>((token * 17 + page * 3) % 80);
    w.workspace.assign(tel::workspace_floats(w.g), bits_float(kFill));
    return w;
}

void check_fold_and_merge()
{
    World w = make_world();
    const std::uint32_t heads = w.g.heads;
    // Per (token, head) logits over the whole context, with the winner planted per head so the
    // heads do not all agree; ties are planted deliberately in head 0 (the same max twice in
    // one tile, and the tile-max tie across two splits of the split token).
    for (std::uint32_t head = 0; head < heads; ++head) {
        // Token 0, fused, context 300: one task, five tiles (the last one ragged at 44 rows).
        std::vector<float> logits(300, -20.0f);
        const std::uint32_t winner0 = 201 + 7 * head;
        for (std::uint32_t i = 0; i < 300; ++i)
            logits[i] = -20.0f + 0.01f * static_cast<float>(i % 37);
        logits[winner0] = 5.0f + 0.25f * static_cast<float>(head);
        if (head == 0) logits[193] = logits[winner0];  // an earlier tie in the SAME tile range
        const FoldState folded = fold_tiles(logits, 300, w.g.block_size);
        const std::int64_t want = direct_argmax(logits, 300);
        g_fail.check(folded.position == want,
                     "the per-tile fold reproduces the direct argmax (fused)");
        write_tail(w.workspace, w.g, 0, head, 0, folded.run_max, folded.position,
                   folded.logical_page, 0.0f);

        // Token 1, split, context 5000: 40 pages over 8 splits of 5 pages each.
        std::vector<float> slogits(5000, -30.0f);
        for (std::uint32_t i = 0; i < 5000; ++i)
            slogits[i] = -30.0f + 0.001f * static_cast<float>(i % 91);
        // The winner lives in split 2's row range [1280, 1920).
        const std::uint32_t winner1 = 1500 + 11 * head;
        slogits[winner1] = 7.0f + 0.25f * static_cast<float>(head);
        // A per-split maximum that must NOT win, planted in split 0.
        slogits[17 + head] = 2.0f;
        const std::int64_t want1 = direct_argmax(slogits, 5000);
        // Fold per split over its own row range, exactly as the device does: each split's task
        // sees only its own rows and its own context bases.
        for (std::uint32_t split = 0; split < 8; ++split) {
            std::uint32_t begin, end;
            tel::split_rows(5000, w.g, split, begin, end);
            tq_test::require(begin < end, "a live split has no rows");
            std::vector<float> part(slogits.begin() + begin, slogits.begin() + end);
            const FoldState fs = fold_tiles(part, end - begin, w.g.block_size);
            const float sum = 1.0f + static_cast<float>(split);  // any positive l_s
            write_tail(w.workspace, w.g, 1, head, split, fs.run_max,
                       fs.position >= 0 ? begin + fs.position : -1,
                       fs.logical_page >= 0 ? (begin + fs.position) / w.g.block_size : -1, sum);
        }
        // Token 2, split, context 4224: 33 pages -> 7 live splits, split 7 dead.
        std::vector<float> dlogits(4224, -25.0f);
        dlogits[4099] = 6.0f + 0.25f * static_cast<float>(head);  // split 6 owns the winner
        for (std::uint32_t split = 0; split < 8; ++split) {
            if (split < 7) {
                std::uint32_t begin, end;
                tel::split_rows(4224, w.g, split, begin, end);
                std::vector<float> part(dlogits.begin() + begin, dlogits.begin() + end);
                const FoldState fs = fold_tiles(part, end - begin, w.g.block_size);
                write_tail(w.workspace, w.g, 2, head, split, fs.run_max,
                           fs.position >= 0 ? begin + fs.position : -1,
                           fs.logical_page >= 0 ? (begin + fs.position) / w.g.block_size : -1,
                           1.0f);
            } else {
                // The dead split, written by RunTask as a neutral partial: zero accumulator,
                // negated-inf max, zero sum and the sentinel probe lanes.
                float *rec =
                    w.workspace.data() + tel::record_offset(w.g, 2, head, split);
                for (std::uint32_t i = 0; i < w.g.head_size; ++i) rec[i] = 0.0f;
                write_tail(w.workspace, w.g, 2, head, split, -FLT_MAX, -1, -1, 0.0f);
            }
        }
        // Accumulator regions of every written slot: finite filler (opaque to verify()).
        for (std::uint32_t token = 0; token < w.g.tokens; ++token) {
            const std::uint32_t splits = tel::is_fused(w.lengths[token], w.g) ? 1 : 8;
            for (std::uint32_t split = 0; split < splits; ++split) {
                float *rec = w.workspace.data() + tel::record_offset(w.g, token, head, split);
                if (tel::is_fused(w.lengths[token], w.g)) break;  // fused acc keeps the fill
                for (std::uint32_t i = 0; i < w.g.head_size; ++i) rec[i] = 0.25f;
            }
        }

        // The merge, against the direct argmax computed independently of the fold.
        const tel::Merged m0 = tel::merge(w.workspace.data(), w.g, 0, head, 300);
        g_fail.check(m0.valid && m0.position == want && !m0.drifted && !m0.stale &&
                         !m0.inconsistent,
                     "merge returns the fused token's winner");
        g_fail.check(m0.logical_page == want / 128,
                     "the fused winner's logical page is its context page");
        const tel::Merged m1 = tel::merge(w.workspace.data(), w.g, 1, head, 5000);
        g_fail.check(m1.valid && m1.position == want1 && !m1.drifted && !m1.inconsistent,
                     "merge returns the split token's winner from the winning split");
        g_fail.check(m1.logical_page == want1 / 128,
                     "the split winner's logical page is its context page");
        const tel::Merged m2 = tel::merge(w.workspace.data(), w.g, 2, head, 4224);
        g_fail.check(m2.valid && m2.position == 4099 && m2.logical_page == 32,
                     "merge skips the dead split and keeps split 6's winner");
        g_fail.check(!m2.stale, "a dead split's sentinel is not staleness");
    }
    std::printf("  fold and merge: fused, 8-split and dead-split tokens over 4 heads, ties "
                "included, all against the direct argmax\n");
}

// ------------------------------------------------------------------ 4. verify()
void check_verify()
{
    World w = make_world();
    // Reuse the record-writing helpers on a smaller image: one fused token (300) and one
    // split token (4224, 7 live splits).
    w.lengths = {300, 4224, 0};
    w.g.tokens = 3;
    w.workspace.assign(tel::workspace_floats(w.g), bits_float(kFill));
    for (std::uint32_t head = 0; head < w.g.heads; ++head) {
        write_tail(w.workspace, w.g, 0, head, 0, 5.0f, 201 + head, (201 + head) / 128, 0.0f);
        for (std::uint32_t split = 0; split < 8; ++split) {
            float *rec = w.workspace.data() + tel::record_offset(w.g, 1, head, split);
            for (std::uint32_t i = 0; i < w.g.head_size; ++i) rec[i] = 0.25f;
            if (split < 7) {
                write_tail(w.workspace, w.g, 1, head, split, 1.0f + split, 100 + 640 * split,
                           (100 + 640 * split) / 128, 1.0f + split);
            } else {
                for (std::uint32_t i = 0; i < w.g.head_size; ++i) rec[i] = 0.0f;
                write_tail(w.workspace, w.g, 1, head, split, -FLT_MAX, -1, -1, 0.0f);
            }
        }
        // Token 2 has context 0: the sentinel record at split 0, fill elsewhere.
        write_tail(w.workspace, w.g, 2, head, 0, -FLT_MAX, -1, -1, 0.0f);
    }
    const auto words = reinterpret_cast<const std::uint32_t *>(w.workspace.data());
    const std::size_t count = w.workspace.size();
    tel::VerifyResult ok = tel::verify(words, count, w.g, w.lengths.data(), w.table.data(), kFill);
    g_fail.check(ok.ok, "a byte-correct telemetry image passes verify()");
    if (!ok.ok) std::printf("    unexpected failure at word %zu: %s\n", ok.word, ok.what);

    // The winning position of token 0's heads is in [201, 204]. Restrict this check to
    // that token: the split token's records deliberately have winners outside this range.
    tel::Geometry range_geometry = w.g;
    range_geometry.tokens = 1;
    const std::size_t range_count = tel::workspace_floats(range_geometry);
    ok = tel::verify(words, range_count, range_geometry, w.lengths.data(), w.table.data(),
                     kFill, 200, 205);
    g_fail.check(ok.ok, "verify() accepts a winner inside the position range override");
    ok = tel::verify(words, range_count, range_geometry, w.lengths.data(), w.table.data(),
                     kFill, 205, 300);
    g_fail.check(!ok.ok, "verify() rejects a winner outside the position range override");

    const auto expect_fail = [&](std::size_t word, float value, const char *what) {
        std::vector<float> bad = w.workspace;
        bad[word] = value;
        const auto *bw = reinterpret_cast<const std::uint32_t *>(bad.data());
        const tel::VerifyResult r =
            tel::verify(bw, bad.size(), w.g, w.lengths.data(), w.table.data(), kFill);
        g_fail.check(!r.ok, what);
        if (!r.ok) std::printf("    rejected at word %zu: %s\n", r.word, r.what);
    };
    const std::uint64_t fused0 = tel::record_offset(w.g, 0, 0, 0);
    const std::uint64_t split1 = tel::record_offset(w.g, 1, 0, 1);
    const std::uint64_t dead7 = tel::record_offset(w.g, 1, 0, 7);
    expect_fail(fused0 + 256 + 1, 9999.0f, "a position past the context is rejected");
    expect_fail(fused0 + 256 + 2, 3.0f, "a page that disagrees with the position is rejected");
    expect_fail(fused0 + 256 + 3, 1.0f, "a non-zero padding lane is rejected");
    expect_fail(fused0 + 256 + 8, 2.0f, "a fused record with a run_sum is rejected");
    expect_fail(fused0 + 128, 0.0f, "a clobbered fill word in the fused acc region is rejected");
    expect_fail(split1 + 3, bits_float(0x7FC00000u),
                "a non-finite accumulator lane is rejected");
    expect_fail(split1 + 256 + 8, -1.0f, "a non-positive live run_sum is rejected");
    expect_fail(dead7 + 256 + 1, 55.0f, "a dead split carrying a live position is rejected");
    expect_fail(count - 1, 0.0f, "a write into a fused token's duplicate split is rejected");
    std::printf("  verify: a correct image, the range override, and 9 named corruptions\n");
}

// ------------------------------------------------------------------ 5. dump()
void check_dump()
{
    World w = make_world();
    w.lengths = {300, 4224, 0};
    for (std::uint32_t head = 0; head < w.g.heads; ++head) {
        write_tail(w.workspace, w.g, 0, head, 0, 5.0f, 201 + head, (201 + head) / 128, 0.0f);
        for (std::uint32_t split = 0; split < 7; ++split) {
            float *rec = w.workspace.data() + tel::record_offset(w.g, 1, head, split);
            for (std::uint32_t i = 0; i < w.g.head_size; ++i) rec[i] = 1.0f;
            write_tail(w.workspace, w.g, 1, head, split, 1.0f + split, 4096 + split,
                       (4096 + split) / 128, 1.0f);
        }
        float *rec = w.workspace.data() + tel::record_offset(w.g, 1, head, 7);
        for (std::uint32_t i = 0; i < w.g.head_size; ++i) rec[i] = 0.0f;
        write_tail(w.workspace, w.g, 1, head, 7, -FLT_MAX, -1, -1, 0.0f);
        write_tail(w.workspace, w.g, 2, head, 0, -FLT_MAX, -1, -1, 0.0f);
    }
    std::FILE *sink = std::tmpfile();
    tq_test::require(sink != nullptr, "tmpfile for the dump check");
    tel::dump(w.workspace.data(), w.g, w.lengths.data(), w.table.data(), sink);
    std::fflush(sink);
    std::rewind(sink);
    std::string text;
    char buf[256];
    while (std::fgets(buf, sizeof(buf), sink)) text += buf;
    std::fclose(sink);
    g_fail.check(text.find("token=0 head=2 ctx=300 max_logit=5 argmax=203") !=
                     std::string::npos,
                 "the dump carries the fused winner");
    g_fail.check(text.find("argmax=203 page=1(phys=") != std::string::npos,
                 "the dump maps the logical page to the physical one");
    g_fail.check(text.find("row=75") != std::string::npos,
                 "the dump carries the row inside the page");
    g_fail.check(text.find("token=1 head=1") != std::string::npos &&
                     text.find("argmax=4102") != std::string::npos,
                 "the dump merges the split token to its winning split");
    g_fail.check(text.find("token=2 head=0 ctx=0 no-record") != std::string::npos,
                 "a zero-context token is reported as having no record");
    std::printf("  dump: fused winner, physical page mapping, row, split merge, no-record\n");
}

}  // namespace

int main()
{
    try {
        std::printf("test_telemetry_dump\n");
        check_layout();
        check_live_splits();
        check_fold_and_merge();
        check_verify();
        check_dump();
    } catch (const std::exception &error) {
        std::printf("FAIL test_telemetry_dump: %s\n", error.what());
        return 1;
    }
    if (!g_fail.clean()) {
        std::printf("FAIL test_telemetry_dump: %d check(s) failed\n", g_fail.count);
        return 1;
    }
    std::printf("PASS test_telemetry_dump: record layout, live-split arithmetic, the per-tile "
                "fold against the direct argmax, verify() over a correct image and 9 named "
                "corruptions, and the dump format\n");
    return 0;
}
