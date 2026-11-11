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

// END TO END: the modulo-scheduled fused decode on a cache an ingress writer could have
// produced, against the double-precision attention reference.
//
// WHAT IT CHECKS, PER RUNG
//
//   A. THE OUTPUT, against `expected` from tests/common/fused_decode_fixture.hpp, which is the
//      closed form of softmax(q k^T * scale) v over the live tiles evaluated in double. The
//      budget is exact (2 ULP for BF16) because the fixture is built so that the fp32 recurrence
//      has no rounding left to do.
//
//   B. GUARD BANDS on every allocation, 0x5A on both sides, plus the slack between a payload's
//      end and its 32-byte aligned length. An over-length MTE burst fails here by name.
//
//   C. INTERIOR CANARIES. The block table deliberately leaves physical pages unmapped, and the
//      scale plane of those pages is 0x5A5A5A5A. A decode that walked a page its descriptor
//      does not name reads a canary scale and cannot produce the right answer.
//
//   D. EVERY INPUT ALLOCATION UNCHANGED. The operator takes the cache, the query and the tiling
//      read-only; a write to any of them is a failure even if the output is right.
//
// WHAT IT DOES NOT CHECK BY DEFAULT: timing, overlap, or a second execution wave. `--perf [N]`
// adds a device-side measurement -- N timed launches per rung and split, bracketed by
// ACL_EVENT_TIME_LINE markers on the stream (tests/common/device_timer.hpp) -- and it runs AFTER
// every correctness and canary check of that configuration, so nothing a validation pass
// observed can have come from a perf re-launch. It is still not an overlap claim: a median wall
// interval says the launch got faster or slower and nothing about which pipe was busy.
//
// `--reproduce` replaces the ladder with ONE geometry: the serving launch that faults on an A5
// with AIV error 341, run through both entry points so the result says whether the raw-query
// prologue or the group_heads 6 work split is what the ladder was missing. It takes no other
// flag. See the block above `reproduce()` for the device log it was built from, the two axes no
// rung covers, and how to read each of the three outcomes.
//
// It also does not canary the TAIL ROWS of a mapped page: a masked
// lane's scale still flows through the Gather and the Muls in front of the mask, so a
// 0x5A5A5A5A scale there (~3.8e15) would be testing the masking phase's numerical headroom
// rather than its logic, and a failure could not be told apart from one. Those rows carry
// ordinary fixture data, which the reference already rejects if they are counted.

#include "../common/camodel_session.hpp"
#include "../common/device_timer.hpp"
#include "../common/fused_decode_fixture.hpp"
#include "../../include/turboquant/turboquant_splitk_contract.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <set>
#include <string>
#include <vector>

namespace {

namespace fd = ascend_micro::fused_decode_opt;
using fused_decode_fixture::Fixture;
using camodel::check;
using camodel::require;

// The canary this file guards with, as a byte and as the word it forms.
constexpr std::uint8_t kGuardByte = 0x5A;
constexpr std::uint32_t kGuardWord = 0x5A5A5A5Au;
// Guard band on each side of every allocation. 64 bytes keeps every payload 32-byte aligned,
// which camodel::Buffer requires, and is wider than one UB block so a single over-length burst
// cannot step past it unseen.
constexpr std::size_t kGuard = 64;

// Real operator geometry. The page size and the ULP budget are the ladder's; the head counts
// vary per rung and live in the rung table.
constexpr int kPageTokens = 128;
constexpr int kUlpBudget = 2;
// Scale lanes per cache row, of which the codec reads two per KV head.
constexpr std::size_t kScaleLanes = 8;

// A device allocation with 0x5A guard bands and a host snapshot of what went into it.
struct Guarded {
    camodel::Buffer allocation;
    std::size_t bytes;                   // the payload, past the leading guard
    std::vector<std::uint8_t> snapshot;  // the whole allocation, as uploaded
    explicit Guarded(std::size_t payload)
        : allocation(fused_decode_fixture::aligned(payload) + 2 * kGuard),
          bytes(fused_decode_fixture::aligned(payload))
    {
    }
    void* ptr() const { return static_cast<std::uint8_t*>(allocation.ptr) + kGuard; }

    // Upload `payload` into the middle and 0x5A everywhere else -- the guards AND the slack
    // between the payload's end and the aligned length.
    template <class T>
    void Upload(const std::vector<T>& payload)
    {
        const std::size_t n = payload.size() * sizeof(T);
        require(n <= bytes, "fused-opt NPU validation: payload exceeds its allocation");
        snapshot.assign(allocation.bytes, kGuardByte);
        std::memcpy(snapshot.data() + kGuard, payload.data(), n);
        allocation.upload(snapshot);
    }
    // No payload: the whole allocation is canary. Outputs and workspaces start here.
    void FillGuard()
    {
        snapshot.assign(allocation.bytes, kGuardByte);
        allocation.upload(snapshot);
    }
    std::vector<std::uint8_t> Download() const
    {
        std::vector<std::uint8_t> image(allocation.bytes);
        allocation.download(image);
        return image;
    }
    // The guard bands, on both sides, after a launch.
    void CheckGuards(const std::vector<std::uint8_t>& image, const char* what) const
    {
        for (std::size_t i = 0; i < kGuard; ++i) {
            if (image[i] != kGuardByte || image[kGuard + bytes + i] != kGuardByte) {
                std::fprintf(stderr, "guard band of %s clobbered at %zu: %02x / %02x\n", what, i,
                             image[i], image[kGuard + bytes + i]);
                require(false, "fused-opt NPU validation: 0x5A guard band clobbered");
            }
        }
    }
    void CheckUnchanged(const char* what) const
    {
        const auto image = Download();
        CheckGuards(image, what);
        if (image != snapshot) {
            std::fprintf(stderr, "input allocation %s changed under the kernel\n", what);
            require(false, "fused-opt NPU validation: an input allocation changed");
        }
    }
};

// The interior canary plan: which physical KV pages the block table does not map, and which
// scale lanes of a mapped page nothing may touch.
struct CanaryPlan {
    std::vector<std::uint8_t> unmapped_page;  // per physical page
    int unmapped_pages = 0;
    std::size_t scale_pad_floats = 0;
};

CanaryPlan plan_canaries(const Fixture& f)
{
    CanaryPlan plan;
    plan.unmapped_page.assign(std::size_t(f.physical_pages), 1);
    for (int request = 0; request < f.requests; ++request)
        for (int page = 0; page < f.pages; ++page) {
            const std::int32_t physical = f.table[std::size_t(request) * f.pages + page];
            require(physical >= 0 && physical < f.physical_pages,
                    "fused-opt NPU validation: fixture block table out of range");
            plan.unmapped_page[std::size_t(physical)] = 0;
        }
    for (std::uint8_t unmapped : plan.unmapped_page) plan.unmapped_pages += unmapped ? 1 : 0;
    require(plan.unmapped_pages > 0, "fused-opt NPU validation: fixture maps every physical page, "
                                     "so there is no unmapped-slot canary to check");
    return plan;
}

// 0x5A over every unmapped physical page of a byte-addressed cache plane.
void paint_unmapped(std::vector<std::uint8_t>& plane, const CanaryPlan& plan, std::size_t page_bytes)
{
    require(plane.size() == plan.unmapped_page.size() * page_bytes,
            "fused-opt NPU validation: cache plane is not a whole number of pages");
    for (std::size_t page = 0; page < plan.unmapped_page.size(); ++page)
        if (plan.unmapped_page[page])
            std::fill(plane.begin() + page * page_bytes, plane.begin() + (page + 1) * page_bytes,
                      kGuardByte);
}

// 0x5A over the unmapped pages of the scale plane AND over the inter-head padding lanes of
// every mapped page. The codec reads lanes [head] and [kv_heads + head] of each row's eight, so
// lanes [2 * kv_heads, 8) are padding at every row of every page -- six of eight at one KV
// head, and NONE at four, where all eight lanes are live. At four KV heads this paints the
// unmapped pages only and leaves plan.scale_pad_floats at zero for the mapped ones, which the
// run reports: a canary with nothing to guard is worth saying out loud rather than asserting.
void paint_scales(std::vector<float>& scales, const Fixture& f, CanaryPlan& plan)
{
    float canary;
    std::memcpy(&canary, &kGuardWord, 4);
    const std::size_t rows_per_page = std::size_t(f.page_tokens);
    const std::size_t used_lanes = std::size_t(2 * f.kv_heads);
    require(scales.size() == std::size_t(f.physical_pages) * rows_per_page * kScaleLanes,
            "fused-opt NPU validation: scale plane geometry drifted from the fixture");
    require(used_lanes <= kScaleLanes,
            "fused-opt NPU validation: the scale slot cannot hold this KV head count");
    for (std::size_t page = 0; page < plan.unmapped_page.size(); ++page)
        for (std::size_t row = 0; row < rows_per_page; ++row) {
            const std::size_t base = (page * rows_per_page + row) * kScaleLanes;
            const std::size_t from = plan.unmapped_page[page] ? 0 : used_lanes;
            for (std::size_t lane = from; lane < kScaleLanes; ++lane) scales[base + lane] = canary;
            if (!plan.unmapped_page[page]) plan.scale_pad_floats += kScaleLanes - from;
        }
}

struct OracleResult {
    int worst_ulp = 0;
    double worst_abs = 0;
};

// Every live output channel against the fixture's FP64 oracle, plus the output tail canary.
// `expected` rather than `f.expected` because the raw-query entry's output stage is not the
// fixture's stage-0 context: --reproduce passes Pi applied once more. `ulp_budget` likewise --
// see kReproUlpBudget for why that leg's is looser and what a failure there would mean.
OracleResult check_output(const std::vector<std::uint8_t>& image, const std::vector<float>& expected,
                          std::size_t payload_bytes, int ulp_budget)
{
    OracleResult result;
    const std::size_t live = expected.size();
    for (std::size_t i = 0; i < live; ++i) {
        std::uint16_t bits;
        std::memcpy(&bits, image.data() + kGuard + 2 * i, 2);
        const std::uint16_t reference = fused_decode_fixture::reference_bfloat16(expected[i]);
        // The fixture's results are positive finite normals, so the bf16 bit distance IS the
        // ULP distance -- no sign or subnormal case to fold in.
        const int ulp = std::abs(int(bits) - int(reference));
        const double got = tq_test::bf16_to_float(bits);
        const double absolute = std::fabs(got - double(expected[i]));
        if (ulp > result.worst_ulp) result.worst_ulp = ulp;
        if (absolute > result.worst_abs) result.worst_abs = absolute;
        if (ulp > ulp_budget) {
            std::fprintf(stderr,
                         "oracle mismatch at channel %zu: got %g (%04x) expected %g (%04x), "
                         "%d bf16 ULP over a budget of %d\n",
                         i, got, bits, double(expected[i]), reference, ulp, ulp_budget);
            require(false, "fused-opt NPU validation: output disagrees with the FP64 oracle");
        }
    }
    for (std::size_t i = 2 * live; i < payload_bytes; ++i)
        if (image[kGuard + i] != kGuardByte) {
            std::fprintf(stderr, "output tail byte %zu past the last live channel is %02x\n", i,
                         image[kGuard + i]);
            require(false, "fused-opt NPU validation: output tail canary clobbered");
        }
    return result;
}

struct Rung {
    int context;
    int requests;
    int kv_heads;
    int group_heads;  // query heads per KV head
    std::vector<int> splits;
    const char* note;
};

// The cache traffic one launch reads, for the GB/s column. Every live row of every KV head is
// read once per (request) in both planes -- 128 packed bytes a row at D = 256 -- plus that row's
// scale slot, which is kScaleLanes floats shared by the heads of the row rather than one per
// head. The query, the output, the block table and the tiling are a rounding error beside it and
// are left out, so the number is the KV read the operator is actually bandwidth-bound on.
double decode_cache_bytes(const Fixture& f)
{
    const double rows = double(f.requests) * double(f.context);
    const double packed = rows * double(f.kv_heads) * 2.0 * double(fd::kHeadSize / 2);
    const double scales = rows * double(kScaleLanes) * 4.0;
    return packed + scales;
}

// A label short enough for the perf column and specific enough to tell two rows apart.
std::string perf_label(const Fixture& f, int splits)
{
    char text[64];
    std::snprintf(text, sizeof(text), "N%d tok%d kvh%d S%d", f.context, f.requests, f.kv_heads,
                  splits);
    return std::string(text);
}

int run(bool camodel_smoke, int perf_iters)
{
    camodel::Session session;

    const std::vector<Rung> full_ladder = {
        // aligned, one KV head, 16 query heads
        {kPageTokens, 1, 1, 16, {1, 2, 4, 8}, "one page: S>1 leaves unmapped (empty) splits"},
        {8 * kPageTokens, 1, 1, 16, {1, 2, 4, 8}, "eight pages: bounded short path at S=8"},
        {16 * kPageTokens, 1, 1, 16, {1, 2, 3, 4, 5, 7, 8},
         "sixteen pages: deep path at every S, and the non-power-of-2 split sweep"},
        {16 * kPageTokens, 2, 1, 16, {1, 2, 4, 8},
         "sixteen pages, two tokens: the reduce phase's token dimension"},
        // ragged: context divides neither the page nor the 64-row tile
        {79, 1, 1, 16, {1, 2, 3}, "one page, 79 rows: one full tile and a 15-row tail"},
        {257, 1, 1, 16, {1, 2, 3, 5},
         "three pages, the last holding ONE token: a single-row tile across a page boundary"},
        {1055, 1, 1, 16, {1, 3, 5, 7},
         "nine pages, a 31-row tail: the tail lands on a different worker at each S"},
        // multi-KV-head: 4 x 16 = 64 query heads
        {8 * kPageTokens, 1, 4, 16, {1, 2, 4, 8},
         "4 KV heads, 64 query heads: grid reaches exactly the 32-block bound at S=8"},
        {1055, 1, 4, 16, {3, 7},
         "4 KV heads, ragged tail and a non-power-of-2 split count at once"},
    };
    const std::vector<Rung> smoke_ladder = {
        {kPageTokens, 1, 1, 16, {1, 2}, "aligned one-page CAModel smoke"},
        {79, 1, 1, 16, {1, 2}, "ragged 15-row CAModel smoke"},
        {257, 1, 1, 16, {1, 3}, "single-row tail CAModel smoke"},
    };
    const auto& ladder = camodel_smoke ? smoke_ladder : full_ladder;

    int launches = 0, worst_ulp = 0, ragged_rungs = 0, narrowest_tail = 64;
    double worst_abs = 0;
    long long canary_pages = 0, canary_pad_floats = 0;
    std::set<int> splits_seen, kv_heads_seen;
    // Perf bookkeeping. `perf_host_clocked` is carried rather than inferred so the summary can
    // say which clock produced the numbers it is quoting.
    int perf_configs = 0;
    bool perf_host_clocked = false;
    double perf_best_us = 0.0, perf_worst_us = 0.0;
    std::string perf_best_label, perf_worst_label;
    if (perf_iters > 0)
        camodel::print_perf_preamble(camodel::kDefaultPerfWarmup, perf_iters, false);

    for (const Rung& rung : ladder) {
        // One fixture per rung: the oracle is the expensive part and the split count changes
        // only how the work is partitioned, never the answer.
        const Fixture f(rung.context, rung.requests, rung.kv_heads, rung.group_heads, kPageTokens,
                        fused_decode_fixture::Element::kBf16);
        CanaryPlan plan = plan_canaries(f);
        const int tail = f.context % 64;
        if (tail != 0) {
            ++ragged_rungs;
            if (tail < narrowest_tail) narrowest_tail = tail;
            // The rung has to be ragged in the way its note claims, or it is testing the
            // aligned path under a ragged name.
            require(f.valid_rows(f.tiles() - 1) == tail,
                    "fused-opt NPU validation: a ragged rung's last tile is not short");
        }
        kv_heads_seen.insert(rung.kv_heads);

        std::vector<std::uint8_t> key = f.key, value = f.value;
        std::vector<float> scales = f.scales;
        const std::size_t page_bytes = std::size_t(f.page_tokens) * f.kv_heads * 128;
        paint_unmapped(key, plan, page_bytes);
        paint_unmapped(value, plan, page_bytes);
        paint_scales(scales, f, plan);

        for (const int k : rung.splits) {
            splits_seen.insert(k);
            const fd::Launch config = f.launch(k);
            require(config.grid <= turboquant_splitk::kMaxCoResidentBlocks,
                    "fused-opt NPU validation: grid exceeds the one-wave bound the reduction's "
                    "SyncAll needs");

            Guarded query(f.query.size() * 4), keys(key.size()), values(value.size()),
                scale(scales.size() * 4), table(f.table.size() * 4), lengths(f.lengths.size() * 4),
                modes(32),
                workspace(std::size_t(f.requests) * config.heads * std::size_t(k) * (256 + 16) * 4 +
                          4096),
                out(f.query.size() * 2 + 32),
                tiling(ascend_micro::fused_decode::decode_tiling_bytes_for(config));
            query.Upload(f.query);
            keys.Upload(key);
            values.Upload(value);
            scale.Upload(scales);
            table.Upload(f.table);
            lengths.Upload(f.lengths);
            modes.Upload(std::vector<std::uint32_t>(8, 0));
            // The host tiling, from the same context lengths and block table the device gets.
            // Every tile count, live-tail length and page span this launch acts on is decided
            // here; the device derives none of it.
            {
                namespace fd_tiling = ascend_micro::fused_decode;
                std::vector<std::uint32_t> image(
                    fd_tiling::decode_tiling_words(config.tokens, config.splits, config.grid), 0);
                const auto status = fd_tiling::build_decode_tiling(
                    fd_tiling::decode_tiling_request(config), f.lengths.data(), f.table.data(),
                    image.data());
                require(status == fd_tiling::DecodeTilingStatus::kOk,
                        fd_tiling::decode_tiling_status_text(status));
                tiling.Upload(image);
            }
            workspace.FillGuard();
            out.FillGuard();

            const fd::Buffers buffers{query.ptr(), keys.ptr(), values.ptr(), scale.ptr(),
                                      table.ptr(), lengths.ptr(), modes.ptr(), workspace.ptr(),
                                      out.ptr(), tiling.ptr()};
            check(aclrtSynchronizeStream(session.stream), "pre-launch synchronize");
            launch_fused_decode_opt(session.stream, &config, &buffers);
            check(aclrtSynchronizeStream(session.stream), "post-launch synchronize");
            ++launches;

            // 1 and 2d: the oracle and the output's tail canary.
            const auto image = out.Download();
            out.CheckGuards(image, "output");
            const OracleResult oracle = check_output(image, f.expected, out.bytes, kUlpBudget);
            worst_ulp = std::max(worst_ulp, oracle.worst_ulp);
            worst_abs = std::max(worst_abs, oracle.worst_abs);

            // 4, which also covers 2a for the inputs and 2b/2c: the unmapped pages and the
            // scale padding are part of the uploaded image, so a byte comparison of the whole
            // allocation is what checks them. They are counted separately below so the run
            // reports how much canary it actually held rather than only that nothing changed.
            query.CheckUnchanged("query");
            keys.CheckUnchanged("key cache");
            values.CheckUnchanged("value cache");
            scale.CheckUnchanged("scale cache");
            table.CheckUnchanged("block table");
            lengths.CheckUnchanged("context lengths");
            modes.CheckUnchanged("mode tables");
            canary_pages += 2 * plan.unmapped_pages;  // key and value planes
            canary_pad_floats += static_cast<long long>(plan.scale_pad_floats);

            // 3: the workspace.
            const auto partials = workspace.Download();
            workspace.CheckGuards(partials, "workspace");
            if (k > 1) {
                const std::size_t floats =
                    std::size_t(f.requests) * config.heads * std::size_t(k) * (256 + 16);
                for (std::size_t i = 0; i < floats; ++i) {
                    float v;
                    std::memcpy(&v, partials.data() + kGuard + i * 4, 4);
                    if (!std::isfinite(v)) {
                        std::fprintf(stderr, "partial float %zu is %g: left unwritten or spoiled\n",
                                     i, double(v));
                        require(false, "fused-opt NPU validation: a Split-K partial came back "
                                       "non-finite");
                    }
                }
                for (std::size_t i = floats * 4; i < workspace.bytes; ++i)
                    if (partials[kGuard + i] != kGuardByte)
                        require(false, "fused-opt NPU validation: the workspace past the live "
                                       "partial extent was written");
            } else if (partials != workspace.snapshot) {
                require(false, "fused-opt NPU validation: the workspace was written at S = 1, "
                               "where the operator takes its fused path and NeedsReduction() is "
                               "false");
            }

            // PERF, last: every check above has already passed on the output and on all eight
            // allocations of THIS configuration, so the re-launches below cannot be what any of
            // them saw. The operator rewrites its output and its Split-K workspace from inputs
            // it only reads, so N launches leave exactly what one left.
            if (perf_iters > 0) {
                const camodel::PerfStats stats = camodel::measure_launch(
                    session.stream, camodel::kDefaultPerfWarmup, perf_iters,
                    [&] { launch_fused_decode_opt(session.stream, &config, &buffers); });
                const std::string label = perf_label(f, k);
                camodel::print_perf_line(label.c_str(), stats,
                                         double(f.requests) * double(config.heads), "vec",
                                         decode_cache_bytes(f));
                ++perf_configs;
                perf_host_clocked = perf_host_clocked || !stats.device_timed;
                if (perf_best_label.empty() || stats.median_us() < perf_best_us) {
                    perf_best_us = stats.median_us();
                    perf_best_label = label;
                }
                if (perf_worst_label.empty() || stats.median_us() > perf_worst_us) {
                    perf_worst_us = stats.median_us();
                    perf_worst_label = label;
                }
            }

            std::printf("  ok  N %5d  tok %d  kvh %d  qh %2d  S %d  grid %2d  tiles %3d "
                        "tail %2d  worst %d ULP  worst abs %.3e  %d unmapped pages, %zu scale "
                        "pad floats intact\n",
                        rung.context, rung.requests, rung.kv_heads,
                        rung.kv_heads * rung.group_heads, k, config.grid, f.tiles(), tail,
                        oracle.worst_ulp, oracle.worst_abs, plan.unmapped_pages,
                        plan.scale_pad_floats);
        }
        std::printf("  rung N %5d tok %d kvh %d: %s\n", rung.context, rung.requests,
                    rung.kv_heads, rung.note);
    }

    require(ragged_rungs >= 3, "fused-opt NPU validation: the ragged rungs went missing");
    require(narrowest_tail == 1, "fused-opt NPU validation: no single-row tail was exercised");
    if (!camodel_smoke) {
        require(kv_heads_seen.count(4) == 1,
                "fused-opt NPU validation: no multi-KV-head rung ran");
        for (const int k : {3, 5, 7})
            require(splits_seen.count(k) == 1,
                    "fused-opt NPU validation: a non-power-of-2 split count went missing");
    }
    require(canary_pad_floats > 0,
            "fused-opt NPU validation: no rung had an inter-head scale padding lane to guard");

    std::string split_list;
    for (const int k : splits_seen) split_list += (split_list.empty() ? "" : ",") + std::to_string(k);
    std::printf("PASS fused decode opt NPU validation: %d launches over %zu rungs, S in {%s}, "
                "1 and 4 KV heads, 16 query heads per KV head, D=%d, %d-token pages. %d rungs "
                "are RAGGED (context divides neither the page nor the 64-row tile), the "
                "narrowest live tail being %d row. Worst bf16 ULP %d against a budget of %d; "
                "worst absolute residual %.3e (the bf16 quantisation of the oracle). 0x%08X "
                "canaries held: %lld unmapped physical KV pages, %lld inter-head scale padding "
                "floats, both guard bands of every allocation, the output tail and the "
                "workspace outside the live partial extent. Every input allocation "
                "byte-identical afterwards.\n",
                launches, ladder.size(), split_list.c_str(), int(fd::kHeadSize), kPageTokens,
                ragged_rungs, narrowest_tail, worst_ulp, kUlpBudget, worst_abs, kGuardWord,
                canary_pages, canary_pad_floats);
    if (perf_configs > 0)
        std::printf("PERF %d configurations timed, %d launches each, %s-clocked. Fastest median "
                    "%.1f us (%s), slowest %.1f us (%s). These are wall intervals per launch: "
                    "they do NOT establish MTE2/Cube overlap, which needs a trace.\n",
                    perf_configs, perf_iters, perf_host_clocked ? "partly host" : "device",
                    perf_best_us, perf_best_label.c_str(), perf_worst_us,
                    perf_worst_label.c_str());
    return 0;
}


// ------------------------------------------------------------------- --reproduce
//
// THE SERVING GEOMETRY THAT FAULTS, as the vllm-ascend TurboQuant backend launched it.
//
// Reported from a live vLLM EngineCore on an A5, out of the backend's own decode trace:
//
//     query        [1, 24, 256] bf16     output [1, 24, 256] bf16
//     key_cache    [6363, 128, 4, 128]   block_size 128   scale_cache [2, 6363, 4, 128]
//     block_tables [1, 518]              context_lens [272], then [316] on the next step
//     num_heads 24, num_kv_heads 4, group_heads 6, scale 0.0625, inv_sqrt_len 0.0625
//
// and the device answered, on that launch:
//
//     aivec error, error code = 341, "The address for VEC to access UB is out of bounds"
//         AIV core ids 0..3 and 38..41, every one at the same pc
//     aicore error, "timeout or trap error" subErrType 0x4, mte error: 0
//         AIC core ids 0, 1, 19, 20
//
// Eight AIVs and four AICs is a KERNEL_TYPE_MIX_AIC_1_2 launch of grid 4, which is what this
// geometry derives below. The 341 is the FAULT and the AIC line is its COLLATERAL -- the Cube
// halves left in CrossCoreWaitFlag until the watchdog reported them, which is why the device
// log prints the Cube first. The AIV is where to look.
//
// WHY THE LADDER ABOVE DOES NOT COVER IT -- two axes, and this flag runs both legs of the
// second so a run says which one matters:
//
//   group_heads 6. Every rung in both ladders is group_heads 16, which is kCubeTileM exactly:
//   one full M fractal, an even 8/8 dual-destination Fixpipe split, and heads.mine 8 against
//   the kHalfRows 8 the UB arena is sized for. 6 gives evenM 6, a 3/3 split and heads.mine 3.
//
//   the RAW-QUERY entry. Nothing under tests/ calls Fixture::launch_raw or RawQueryBasis, so
//   launch_fused_decode_opt_raw_query -- its AIV rotation prologue, its own SyncAll<false>()
//   and the kUnrotated output stage -- has never run on a device. Production uses exactly that
//   entry, at output_stage 1, with a NULL gate pointer.
//
// So each case runs twice, pre-rotated first and raw-query second, on one fixture:
//
//   pre-rotated passes, raw-query faults  -> the prologue or the output stage
//   both fault                            -> group_heads 6, common to the two
//   neither faults                        -> the geometry is not the whole story and what the
//                                            serving cache HELD is, which this fixture replaces
//
// FOR THE NPU RUNNER, not the CAModel one. Both contexts are over 256 rows, which is past the
// working limit the simulator is held to here (the --camodel-smoke ladder tops out at 257 for
// the same reason), and the fault this reproduces is a device report with no simulator
// equivalent. --reproduce refuses to combine with --camodel-smoke, but nothing stops
// test_fused_decode_e2e_sim from being handed it, so: run it from the NPU target.
//
// TWO DELIBERATE DEPARTURES from the serving launch, neither of them reachable by the device:
// `max_blocks` is this fixture's own page count rather than 518 (with one request the block
// table row base is 0 either way), and the host tiling is built with the fixture's real block
// table where the binding passes nullptr (the live span is dense in both, so the descriptors
// come out identical -- and the table is what keeps the unmapped-page canaries meaningful).
struct ReproCase {
    int context;
    int tail;  // context % 64, stated so a case note cannot drift from its own geometry
    const char* note;
};

// Serving-side geometry, from the trace above.
constexpr int kReproRequests = 1;
constexpr int kReproKvHeads = 4;
constexpr int kReproGroupHeads = 6;
constexpr int kReproSplits = 1;
constexpr unsigned kReproOutputStage = 1;  // TurboQuantOutputStage::kUnrotated
constexpr unsigned kReproCubeChunk = 0;    // 6 vectors a block is below kQueryBasisMinCubeVectors
constexpr std::uint32_t kReproFusedContextLimit = 4096;
// What the vllm-ascend binding derives for this geometry, restated so a drift in either
// derivation fails here instead of quietly making this a different launch.
constexpr std::uint32_t kReproGrid = 4;
constexpr std::uint32_t kReproTasksPerBlock = 1;
constexpr std::uint32_t kReproReduceTasksPerBlock = 6;
constexpr std::uint32_t kReproPrologueVectors = 6;
// Every live output channel finite. A probe has no oracle, so this is the arithmetic half of
// its contract: one tile of one page cannot produce a NaN or an infinity from finite inputs,
// and a mask or a scale read off the wrong rows is the way it would.
void check_output_finite(const std::vector<std::uint8_t>& image, std::size_t live,
                         std::size_t payload_bytes, const char* what)
{
    for (std::size_t i = 0; i < live; ++i) {
        std::uint16_t bits;
        std::memcpy(&bits, image.data() + kGuard + 2 * i, 2);
        const double got = tq_test::bf16_to_float(bits);
        if (!std::isfinite(got)) {
            std::fprintf(stderr, "%s: output channel %zu is %g (%04x)\n", what, i, got, bits);
            require(false, "fused-opt --probe: a probed tile produced a non-finite output");
        }
    }
    for (std::size_t i = 2 * live; i < payload_bytes; ++i)
        if (image[kGuard + i] != kGuardByte)
            require(false, "fused-opt --probe: output tail canary clobbered");
}

// The oracle rows for one launch: the fixture's stage-0 context, or Pi applied once more at the
// stages only the raw-query entry can emit.
std::vector<float> oracle_image(const Fixture& f, const fused_decode_fixture::RawQueryBasis& basis,
                                unsigned output_stage)
{
    if (output_stage == 0) return f.expected;
    std::vector<float> out(f.expected.size(), 0.0f);
    const std::size_t rows = f.expected.size() / fd::kHeadSize;
    for (std::size_t row = 0; row < rows; ++row) {
        const std::vector<float> image = f.expected_row(basis, row, output_stage);
        std::copy(image.begin(), image.end(), out.begin() + row * fd::kHeadSize);
    }
    return out;
}

int reproduce()
{
    camodel::Session session;
    const fused_decode_fixture::RawQueryBasis basis;

    const std::vector<ReproCase> cases = {
        {272, 16, "the first faulting step: 3 pages, 5 tiles, a 16-row live tail"},
        {316, 60, "the next step: the same 5 tiles with a 60-row tail"},
    };

    int launches = 0, worst_ulp_pre = 0, worst_ulp_raw = 0;
    for (const ReproCase& c : cases) {
        const Fixture f(c.context, kReproRequests, kReproKvHeads, kReproGroupHeads, kPageTokens,
                        fused_decode_fixture::Element::kBf16);
        require(f.context % 64 == c.tail,
                "fused-opt --reproduce: a case's stated tail is not its geometry's");
        require(f.tiles() == 5, "fused-opt --reproduce: the serving steps were 5 tiles each");
        require(f.valid_rows(f.tiles() - 1) == c.tail,
                "fused-opt --reproduce: the last tile is not the short one");

        CanaryPlan plan = plan_canaries(f);
        std::vector<std::uint8_t> key = f.key, value = f.value;
        std::vector<float> scales = f.scales;
        const std::size_t page_bytes = std::size_t(f.page_tokens) * f.kv_heads * 128;
        paint_unmapped(key, plan, page_bytes);
        paint_unmapped(value, plan, page_bytes);
        paint_scales(scales, f, plan);

        const std::vector<std::uint16_t> raw = f.raw_query(basis);
        const std::vector<std::int32_t> pool = basis.tables();
        const std::vector<std::uint16_t> hadamard = basis.h16();

        // Leg 0 is the pre-rotated entry, leg 1 the raw-query one. Same fixture, same cache,
        // same split count: the only thing that moves between them is the entry point.
        struct ReproLeg {
            bool raw_entry;
            unsigned stage;
            bool oracle;
            const char* name;
        };
        // THREE LEGS, and the middle one is the point.
        //
        //   pre-rotated, stage 0   the fixture's own entry, exact against its fp64 oracle
        //   RAW-QUERY,  stage 0    the SAME oracle, because the prologue's whole contract is
        //                          that it reproduces the pre-rotated `query` BIT FOR BIT from
        //                          `raw_query` -- fused_decode_fixture.hpp says so and builds
        //                          raw_query exact in both element formats. So this leg is a
        //                          bit-exact test OF ROTATE_Q: a rotation wrong by one ULP
        //                          anywhere misses the attention oracle on top of it, at the
        //                          same budget of 2.
        //   RAW-QUERY,  stage 1    what production launches. NO oracle: reference_bfloat16
        //                          refuses anything outside [1,15], which is the range the
        //                          fixture engineers its stage-0 context into, and Pi applied
        //                          once more is signed and unbounded. Finiteness and guards
        //                          instead, which is what a fault hunt needs.
        //
        // Asking the stage-0 oracle for a stage-1 output is what the first version of this did,
        // and on the device the fixture threw "oracle result outside fixture range" without
        // reporting anything about the launch. The stage was why, not the launch.
        static constexpr ReproLeg kLegs[] = {
            {false, 0u, true, "pre-rotated"},
            {true, 0u, true, "raw-query/st0"},
            {true, kReproOutputStage, false, "raw-query/st1"},
        };
        for (const ReproLeg& leg : kLegs) {
            const bool raw_entry = leg.raw_entry;
            fd::Launch config = raw_entry
                                    ? f.launch_raw(kReproSplits, leg.stage, kReproCubeChunk)
                                    : f.launch(kReproSplits);
            config.fused_context_limit = kReproFusedContextLimit;

            require(config.grid == kReproGrid && config.tasks_per_block == kReproTasksPerBlock &&
                        config.reduce_tasks_per_block == kReproReduceTasksPerBlock,
                    "fused-opt --reproduce: the derived work split is not the serving launch's");
            require(config.heads == unsigned(kReproKvHeads * kReproGroupHeads) &&
                        config.heads_per_task == unsigned(kReproGroupHeads) &&
                        config.block_size == unsigned(kPageTokens),
                    "fused-opt --reproduce: the derived geometry is not the serving launch's");
            if (raw_entry)
                require(config.prologue_vectors_per_block == kReproPrologueVectors &&
                            config.prologue_cube_chunk_vectors == kReproCubeChunk &&
                            config.output_stage == leg.stage,
                        "fused-opt --reproduce: the raw-query prologue share drifted");

            Guarded keys(key.size()), values(value.size()), scale(scales.size() * 4),
                table(f.table.size() * 4), lengths(f.lengths.size() * 4), modes(32),
                workspace(std::size_t(f.requests) * config.heads * kReproSplits * (256 + 16) * 4 +
                          4096),
                out(f.query.size() * 2 + 32),
                tiling(ascend_micro::fused_decode::decode_tiling_bytes_for(config));
            // The fp32 rotated query. The pre-rotated entry READS it; the raw-query entry's
            // prologue WRITES it, so that leg uploads canary and checks the guards afterwards
            // rather than checking the allocation unchanged.
            Guarded query(f.query.size() * 4);
            if (raw_entry) {
                query.FillGuard();
            } else {
                query.Upload(f.query);
            }
            Guarded raw_query(raw.size() * 2), signs(basis.signs.size() * 4),
                tables(pool.size() * 4), h16(hadamard.size() * 2);
            if (raw_entry) {
                raw_query.Upload(raw);
                signs.Upload(basis.signs);
                tables.Upload(pool);
                h16.Upload(hadamard);
            }
            keys.Upload(key);
            values.Upload(value);
            scale.Upload(scales);
            table.Upload(f.table);
            lengths.Upload(f.lengths);
            modes.Upload(std::vector<std::uint32_t>(8, 0));
            {
                namespace fd_tiling = ascend_micro::fused_decode;
                std::vector<std::uint32_t> image(
                    fd_tiling::decode_tiling_words(config.tokens, config.splits, config.grid), 0);
                const auto status = fd_tiling::build_decode_tiling(
                    fd_tiling::decode_tiling_request(config), f.lengths.data(), f.table.data(),
                    image.data());
                require(status == fd_tiling::DecodeTilingStatus::kOk,
                        fd_tiling::decode_tiling_status_text(status));
                tiling.Upload(image);
            }
            workspace.FillGuard();
            out.FillGuard();

            const fd::Buffers buffers{query.ptr(), keys.ptr(), values.ptr(), scale.ptr(),
                                      table.ptr(), lengths.ptr(), modes.ptr(), workspace.ptr(),
                                      out.ptr(), tiling.ptr()};
            // Printed BEFORE the launch and flushed: a device fault takes the process with it,
            // so this line is what says which leg of which case was on the stream.
            std::printf("  launch  N %4d  %-14s stage %u  %-6s  grid %u  tiles %d  tail %2d  "
                        "heads %u/%u  %s\n",
                        f.context, leg.name, leg.stage, leg.oracle ? "oracle" : "finite",
                        config.grid, f.tiles(), c.tail, config.heads, config.kv_heads, c.note);
            std::fflush(stdout);
            check(aclrtSynchronizeStream(session.stream), "pre-launch synchronize");
            if (raw_entry) {
                // gate is NULL exactly as the binding passes it at this stage. The launch
                // header says stages 1 and 2 read the gate, while the kernel returns before
                // touching gateGm_ unless the stage is kGated; if this leg faults and the
                // pre-rotated one does not, that disagreement is the first thing to test.
                const fd::RawQueryBuffers raw_buffers{raw_query.ptr(), signs.ptr(), tables.ptr(),
                                                      h16.ptr(), nullptr, buffers};
                launch_fused_decode_opt_raw_query(session.stream, &config, &raw_buffers);
            } else {
                launch_fused_decode_opt(session.stream, &config, &buffers);
            }
            check(aclrtSynchronizeStream(session.stream), "post-launch synchronize");
            ++launches;

            const auto image = out.Download();
            out.CheckGuards(image, "output");
            if (leg.oracle) {
                const OracleResult oracle =
                    check_output(image, oracle_image(f, basis, leg.stage), out.bytes, kUlpBudget);
                int& worst = raw_entry ? worst_ulp_raw : worst_ulp_pre;
                worst = std::max(worst, oracle.worst_ulp);
            } else {
                check_output_finite(image, f.expected.size(), out.bytes, leg.name);
            }

            keys.CheckUnchanged("key cache");
            values.CheckUnchanged("value cache");
            scale.CheckUnchanged("scale cache");
            table.CheckUnchanged("block table");
            lengths.CheckUnchanged("context lengths");
            modes.CheckUnchanged("mode tables");
            tiling.CheckUnchanged("tiling");
            if (raw_entry) {
                raw_query.CheckUnchanged("raw query");
                signs.CheckUnchanged("pi signs");
                tables.CheckUnchanged("rotation tables");
                h16.CheckUnchanged("h16");
                // The prologue's own output: its payload moved, so only the guards bound it.
                query.CheckGuards(query.Download(), "rotated query");
            } else {
                query.CheckUnchanged("query");
            }
            // S = 1 is the fused path, so NeedsReduction() is false and nothing may touch the
            // workspace at all -- the same invariant the ladder checks at S = 1.
            const auto partials = workspace.Download();
            workspace.CheckGuards(partials, "workspace");
            require(partials == workspace.snapshot,
                    "fused-opt --reproduce: the workspace was written on the fused path");
        }
    }

    std::printf("PASS fused decode opt --reproduce: %d launches over %zu serving geometries, "
                "group_heads %d on %d KV heads (%d query heads), D=%d, %d-token pages, S=%d. "
                "Three legs each: the pre-rotated entry; the raw-query entry at stage 0, which "
                "is a BIT-EXACT test of the rotation prologue, because its contract is to "
                "reproduce the pre-rotated query exactly; and the raw-query entry at the "
                "output_stage %u production launches, with a null gate and NO oracle, because "
                "the fixture's own range guard admits only [1,15] and Pi applied once more is "
                "signed and unbounded. Worst bf16 ULP %d pre-rotated and %d raw-query, both "
                "against a budget of %d. Every guard band, the unmapped-page and scale-padding "
                "canaries, every input allocation and the untouched fused-path workspace all "
                "held. So the serving GEOMETRY does not fault on its own AND the prologue "
                "rotates correctly over it: what the serving cache HELD is the remaining "
                "difference, and this fixture replaces it.",
                launches, cases.size(), kReproGroupHeads, kReproKvHeads,
                kReproKvHeads * kReproGroupHeads, int(fd::kHeadSize), kPageTokens, kReproSplits,
                kReproOutputStage, worst_ulp_pre, worst_ulp_raw, kUlpBudget);
    return 0;
}


// ---------------------------------------------------------------------- --probe
//
// ONE TILE PER LAUNCH, at the edges of the grid the serving decode faulted on.
//
// This is the cheap half of --reproduce. The serving fault is a geometry/addressing report
// (AIV 341, "The address for VEC to access UB is out of bounds"), not a handshake one, so it
// does not need the real 5-tile ring to show up: it needs the right TILE, with the right live
// row count, at the right page. `Launch::probe_page` / `probe_tile` ask the host tiling for
// exactly one 64-row tile, which makes every launch here S = 64 -- inside the simulator's
// working range, so this mode runs on the CAModel target where --reproduce cannot.
//
// WHAT THE SWEEP SEPARATES. Three axes, and each one is a line of the printed table:
//
//   group_heads 6 against 16. 16 is kCubeTileM and the only value any rung has ever used:
//   a full M fractal, an 8/8 dual-destination Fixpipe split, heads.mine 8 against the
//   kHalfRows 8 the arena is sized for, and -- the reason this matters for the reduction --
//   `ReduceRepeat(row_reduce, scores, 64, heads, ...)` at the full repeat count, followed by
//   a Brcb that reads 8 source elements whatever `heads` was. 6 gives evenM 6, a 3/3 split
//   and heads.mine 3.
//
//   the ENTRY POINT, and with it rotate_q. The pre-rotated entry reads the fp32 rotated query
//   straight from GM; the raw-query entry rotates it first, in a prologue nothing under tests/
//   has ever launched (grep for launch_raw). Both are run over the identical single tile, so
//   pre-rotated passing and raw-query faulting puts the fault in the prologue or the output
//   stage and NOWHERE in the tile pipeline.
//
//   the prologue's own two paths. cube_chunk 0 rotates every vector of the block's share with
//   ApplyPi on the vector cores; 16 rotates whole 16-vector chunks on the Cube. At group_heads
//   6 the share is 6 vectors, so `share / cubeChunk_` is 0 and 16 degenerates to the same
//   per-vector path -- which is why the Cube leg is swept at group_heads 16, where the share
//   is 16 and the chunk path actually runs.
//
//   the TILE. (0,0) is the first full tile, (1,1) a full tile at the far side of a page
//   boundary, and (2,0) the sequence's own short tile -- 16 live rows of 64 at context 272,
//   which is where the tail mask and `lastValid` come into it. The host tiling now gives a
//   probe its tile's real live count, so that last one is probed AS a short tile.
//
// WHAT IT CHECKS, and what it deliberately does not. A probe computes attention over ONE tile
// of one page, so the fixture's `expected` -- the closed form over the WHOLE live context --
// does not apply and no oracle is checked. What is checked is everything that still holds: no
// device fault, every 0x5A guard band, every input allocation byte-identical, and every live
// output float finite. That is the contract a fault hunt needs; --reproduce is where the
// numbers are checked, against the real grid.
struct ProbeTile {
    std::uint32_t page;
    std::uint32_t tile;
    const char* note;
};

struct ProbeLeg {
    bool raw_entry;
    unsigned cube_chunk;   // raw entry only
    int min_group_heads;   // the Cube chunk path needs a share of at least kQueryBasisCubeVectors
    const char* name;
};

int probe()
{
    camodel::Session session;
    const fused_decode_fixture::RawQueryBasis basis;

    // The serving context, so the short tile is the serving one: 272 rows is 4 full tiles and
    // 16 live rows in page 2 tile 0.
    constexpr int kProbeContext = 272;
    const std::vector<int> group_head_sweep = {kReproGroupHeads, 16};
    const std::vector<ProbeLeg> legs = {
        {false, 0, 1, "pre-rotated"},
        {true, 0, 1, "raw-query/vec"},
        {true, 16, 16, "raw-query/cube"},
    };
    const std::vector<ProbeTile> probes = {
        {0, 0, "first full tile"},
        {1, 1, "full tile past a page boundary"},
        {2, 0, "the sequence's own short tile, 16 live rows of 64"},
    };

    int launches = 0;
    for (const int group_heads : group_head_sweep) {
        const Fixture f(kProbeContext, kReproRequests, kReproKvHeads, group_heads, kPageTokens,
                        fused_decode_fixture::Element::kBf16);
        CanaryPlan plan = plan_canaries(f);
        std::vector<std::uint8_t> key = f.key, value = f.value;
        std::vector<float> scales = f.scales;
        const std::size_t page_bytes = std::size_t(f.page_tokens) * f.kv_heads * 128;
        paint_unmapped(key, plan, page_bytes);
        paint_unmapped(value, plan, page_bytes);
        paint_scales(scales, f, plan);

        const std::vector<std::uint16_t> raw = f.raw_query(basis);
        const std::vector<std::int32_t> pool = basis.tables();
        const std::vector<std::uint16_t> hadamard = basis.h16();

        for (const ProbeLeg& leg : legs) {
            if (group_heads < leg.min_group_heads) continue;
            for (const ProbeTile& probe_tile : probes) {
                fd::Launch config =
                    leg.raw_entry
                        ? f.launch_raw(kReproSplits, kReproOutputStage, leg.cube_chunk)
                        : f.launch(kReproSplits);
                config.fused_context_limit = kReproFusedContextLimit;
                config.probe_page = probe_tile.page;
                config.probe_tile = probe_tile.tile;

                Guarded keys(key.size()), values(value.size()), scale(scales.size() * 4),
                    table(f.table.size() * 4), lengths(f.lengths.size() * 4), modes(32),
                    workspace(std::size_t(f.requests) * config.heads * kReproSplits * (256 + 16) *
                                  4 +
                              4096),
                    out(f.query.size() * 2 + 32),
                    tiling(ascend_micro::fused_decode::decode_tiling_bytes_for(config));
                Guarded query(f.query.size() * 4);
                if (leg.raw_entry) {
                    query.FillGuard();
                } else {
                    query.Upload(f.query);
                }
                Guarded raw_query(raw.size() * 2), signs(basis.signs.size() * 4),
                    tables(pool.size() * 4), h16(hadamard.size() * 2);
                if (leg.raw_entry) {
                    raw_query.Upload(raw);
                    signs.Upload(basis.signs);
                    tables.Upload(pool);
                    h16.Upload(hadamard);
                }
                keys.Upload(key);
                values.Upload(value);
                scale.Upload(scales);
                table.Upload(f.table);
                lengths.Upload(f.lengths);
                modes.Upload(std::vector<std::uint32_t>(8, 0));
                {
                    namespace fd_tiling = ascend_micro::fused_decode;
                    std::vector<std::uint32_t> image(
                        fd_tiling::decode_tiling_words(config.tokens, config.splits, config.grid),
                        0);
                    const auto status = fd_tiling::build_decode_tiling(
                        fd_tiling::decode_tiling_request(config), f.lengths.data(),
                        f.table.data(), image.data());
                    require(status == fd_tiling::DecodeTilingStatus::kOk,
                            fd_tiling::decode_tiling_status_text(status));
                    // The descriptor the device is about to act on, so the printed line is the
                    // launch and not an intention: one tile, this page, this live row count.
                    const auto* d =
                        reinterpret_cast<const ascend_micro::fused_decode::TurboQuantDecodeTilingData*>(
                            image.data());
                    require(d->num_tiles == 1 && d->block_start_idx == probe_tile.page &&
                                d->tile_origin == probe_tile.tile,
                            "fused-opt --probe: the isolated descriptor is not the probe asked "
                            "for");
                    const unsigned valid = d->tail_rows ? d->tail_rows : 64u;
                    std::printf("  probe  gh %2d  %-14s  page %u tile %u  valid %2u  grid %u  %s\n",
                                group_heads, leg.name, probe_tile.page, probe_tile.tile, valid,
                                config.grid, probe_tile.note);
                    std::fflush(stdout);
                    tiling.Upload(image);
                }
                workspace.FillGuard();
                out.FillGuard();

                const fd::Buffers buffers{query.ptr(),     keys.ptr(),  values.ptr(),
                                          scale.ptr(),     table.ptr(), lengths.ptr(),
                                          modes.ptr(),     workspace.ptr(), out.ptr(),
                                          tiling.ptr()};
                check(aclrtSynchronizeStream(session.stream), "pre-launch synchronize");
                if (leg.raw_entry) {
                    const fd::RawQueryBuffers raw_buffers{raw_query.ptr(), signs.ptr(),
                                                          tables.ptr(),    h16.ptr(),
                                                          nullptr,         buffers};
                    launch_fused_decode_opt_raw_query(session.stream, &config, &raw_buffers);
                } else {
                    launch_fused_decode_opt(session.stream, &config, &buffers);
                }
                check(aclrtSynchronizeStream(session.stream), "post-launch synchronize");
                ++launches;

                const auto image = out.Download();
                out.CheckGuards(image, "output");
                check_output_finite(image, f.expected.size(), out.bytes, leg.name);
                keys.CheckUnchanged("key cache");
                values.CheckUnchanged("value cache");
                scale.CheckUnchanged("scale cache");
                table.CheckUnchanged("block table");
                lengths.CheckUnchanged("context lengths");
                modes.CheckUnchanged("mode tables");
                tiling.CheckUnchanged("tiling");
                if (leg.raw_entry) {
                    raw_query.CheckUnchanged("raw query");
                    signs.CheckUnchanged("pi signs");
                    tables.CheckUnchanged("rotation tables");
                    h16.CheckUnchanged("h16");
                    query.CheckGuards(query.Download(), "rotated query");
                } else {
                    query.CheckUnchanged("query");
                }
                const auto partials = workspace.Download();
                workspace.CheckGuards(partials, "workspace");
                require(partials == workspace.snapshot,
                        "fused-opt --probe: the workspace was written on the fused path");
            }
        }
    }

    std::printf("PASS fused decode opt --probe: %d single-tile launches, context %d, "
                "group_heads {%d, 16} x {pre-rotated, raw-query on the vector prologue, "
                "raw-query on the Cube chunk prologue} x {first full tile, a full tile past a "
                "page boundary, the short last tile}. Every launch is ONE 64-row tile, so no "
                "oracle: what held is the absence of a device fault, every 0x5A guard band, "
                "every input allocation byte-identical, a finite output and an untouched "
                "fused-path workspace. A fault here would have named its own line above.\n",
                launches, kProbeContext, kReproGroupHeads);
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        bool camodel_smoke = false;
        bool reproduce_only = false;
        bool probe_only = false;
        int perf_iters = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--camodel-smoke") {
                camodel_smoke = true;
                continue;
            }
            // Its own mode rather than another rung: the ladder's summary asserts a ragged
            // single-row tail, a multi-KV-head rung and the non-power-of-2 split sweep, none
            // of which this geometry has, and it launches an entry the ladder never does.
            if (arg == "--reproduce") {
                reproduce_only = true;
                continue;
            }
            // One tile per launch, so S is 64 and this one DOES run on the CAModel target.
            if (arg == "--probe") {
                probe_only = true;
                continue;
            }
            if (arg == "--perf") {
                // The count is optional and must not swallow the next flag, so a following
                // argument is only consumed when it parses whole as a positive integer.
                perf_iters = camodel::kDefaultPerfIters;
                if (i + 1 < argc) {
                    const std::string next = argv[i + 1];
                    int parsed = 0;
                    const auto got = std::from_chars(next.data(), next.data() + next.size(),
                                                     parsed);
                    if (got.ec == std::errc() && got.ptr == next.data() + next.size() &&
                        parsed >= 1 && parsed <= camodel::kMaxPerfIters) {
                        perf_iters = parsed;
                        ++i;
                    }
                }
                continue;
            }
            throw std::runtime_error("usage: test_fused_decode_e2e [--camodel-smoke] "
                                     "[--reproduce] [--probe] [--perf [iterations]]");
        }
        if (probe_only) {
            require(!reproduce_only && !camodel_smoke && perf_iters == 0,
                    "fused-opt --probe: takes no other flag");
            return probe();
        }
        if (reproduce_only) {
            require(!camodel_smoke && perf_iters == 0,
                    "fused-opt --reproduce: takes no other flag -- it is one geometry, two "
                    "launches, and a device fault is the result it is looking for");
            return reproduce();
        }
        return run(camodel_smoke, perf_iters);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL fused decode opt NPU validation: %s\n", error.what());
        return 1;
    }
}