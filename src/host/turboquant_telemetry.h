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

// Host side of the attention telemetry probe (ASCEND_TQ_DECODE_TELEMETRY): the reader for the
// argmax records the fused decode leaves in the Split-K workspace, the merge that turns one
// record per live split into one winner per (token, head), the post-launch workspace verifier
// the device runners use as their workspace contract, and the human-readable dump.
//
// The record layout is the kernel's, pinned in include/turboquant/turboquant_layout.h: the
// 16-float statistics tail of a (token, head, split) slot carries the maximum logit at lane
// kTelemetryMaxLane, the argmax context position at lane kTelemetryPosLane and the LOGICAL
// page index of that position at lane kTelemetryPageLane, with every other probe lane zero.
// The page is logical because the device records what the closed-form cursor computed -- the
// host owns the block table and maps logical to physical here, which also lets the dump say
// when the two derivations disagree.
//
// Toolkit-free on purpose, like turboquant_tiling.h: the CPU gates, the CAModel runners and
// any adapter include this without a kernel toolchain, so the merge and the verifier are
// compiled and exercised everywhere the workspace image can be inspected.

#ifndef TURBOQUANT_HOST_TELEMETRY_H
#define TURBOQUANT_HOST_TELEMETRY_H

#include "../../include/turboquant/turboquant_layout.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace ascend_micro::fused_decode_telemetry {

using vllm_ascend::turboquant::kPartialSumLane;
using vllm_ascend::turboquant::kPartialTail;
using vllm_ascend::turboquant::kTelemetryInvalid;
using vllm_ascend::turboquant::kTelemetryMaxLane;
using vllm_ascend::turboquant::kTelemetryPageLane;
using vllm_ascend::turboquant::kTelemetryPosLane;

// The launch geometry the records were written against: the subset of the launch descriptor
// that decides the workspace layout and the fused/split split, named separately so this layer
// stays free of the launch ABI (the same decoupling DecodeTilingRequest has).
struct Geometry {
    std::uint32_t tokens = 0;
    std::uint32_t heads = 0;
    std::uint32_t splits = 1;
    std::uint32_t head_size = 256;
    std::uint32_t block_size = 128;
    std::uint32_t max_blocks = 0;  // block-table row stride (0: no page checks)
    std::uint32_t num_blocks = 0;  // physical page count of the cache (0: unknown)
    std::uint32_t fused_context_limit = 4096;
};

// One slot of the workspace: head_size accumulator floats plus the kPartialTail-float
// statistics tail, exactly the device's PartialOffset stride.
constexpr std::uint32_t record_stride(std::uint32_t head_size)
{
    return head_size + kPartialTail;
}
constexpr std::uint64_t record_offset(const Geometry &g, std::uint32_t token, std::uint32_t head,
                                      std::uint32_t split)
{
    return (std::uint64_t{token} * g.heads + head) * g.splits * record_stride(g.head_size) +
           std::uint64_t{split} * record_stride(g.head_size);
}
constexpr std::uint64_t workspace_floats(const Geometry &g)
{
    return record_offset(g, g.tokens, 0, 0);
}

// The fused/split decision, as the kernel's IsFused and build_decode_tiling both compute it.
constexpr bool is_fused(std::uint32_t context, const Geometry &g)
{
    return g.splits <= 1 || context <= g.fused_context_limit;
}

// How many splits of a SPLIT (non-fused) token carry live work. Mirrors build_decode_tiling:
// split s owns block range [s*bps, min((s+1)*bps, seq_blocks)) and is live exactly when that
// range covers context rows, so the live splits are a prefix of 0..splits-1.
std::uint32_t live_splits_of(std::uint32_t context, const Geometry &g);

// The context row range [begin, end) one live split covers, same arithmetic as the tiling.
void split_rows(std::uint32_t context, const Geometry &g, std::uint32_t split,
                std::uint32_t &row_begin, std::uint32_t &row_end);

// One slot's statistics tail, decoded. position and logical_page are -1 (the sentinel) when
// the slot carries no record: a zero-context token, an unmapped worker, or a dead split.
struct Record {
    float max_logit = 0.0f;
    float run_sum = 0.0f;
    std::int64_t position = -1;
    std::int64_t logical_page = -1;
    bool finite = false;      // lanes 0, 1, 2 and 8 are all finite
    bool zero_padding = false;  // lanes 3..7 and 9..15 are zero, as the packer leaves them
};
Record read_record(const float *workspace, const Geometry &g, std::uint32_t token,
                   std::uint32_t head, std::uint32_t split);

// The merged winner of one (token, head) decode step. Fused tokens contribute their single
// slot-0 record; split tokens contribute every live split's record and the winner is the
// highest maximum logit, the earliest split on ties -- the device's own earliest-wins rule.
struct Merged {
    float max_logit = 0.0f;
    std::int64_t position = -1;      // context index of the winner
    std::int64_t logical_page = -1;  // its logical page in the token's block-table row
    bool valid = false;      // a live record was found
    bool stale = false;      // a live slot still carried the sentinel (never written/corrupt)
    bool drifted = false;    // the winning position is outside [0, context)
    bool inconsistent = false;  // logical_page != position / block_size on the winner
};
Merged merge(const float *workspace, const Geometry &g, std::uint32_t token, std::uint32_t head,
             std::uint32_t context);

// One line per (token, head) decode step: the merged winner, its max logit, its logical and
// physical page (physical from `block_table`, nullable) and the row inside the page. Anomalies
// (a stale live slot, a position outside the context, a page/position disagreement) are
// flagged on their line rather than hidden, because drift hunting is what the probe is for.
void dump(const float *workspace, const Geometry &g, const std::int32_t *context_lens,
          const std::int32_t *block_table, std::FILE *out);

// The post-launch workspace contract, checked word by word: every record the geometry says
// the launch must have written is present and sane, and every other word still holds
// `fill_word`. This REPLACES the runners' old "the fused path never touches the workspace"
// byte-compare, which the probe deliberately supersedes; the fill-word compare everywhere
// outside the records keeps the same canary strength. Returns ok=false with the first
// offending word index and a static message on any violation. `position_floor/ceiling` narrow
// the accepted winner range (the isolated-tile launches of test_fused_decode_e2e --probe
// expect the winner inside the probed tile).
struct VerifyResult {
    bool ok = true;
    const char *what = "";
    std::size_t word = 0;
};
VerifyResult verify(const std::uint32_t *words, std::size_t word_count, const Geometry &g,
                    const std::int32_t *context_lens, const std::int32_t *block_table,
                    std::uint32_t fill_word, std::uint32_t position_floor = 0,
                    std::uint32_t position_ceiling = UINT32_MAX);

}  // namespace ascend_micro::fused_decode_telemetry

#endif  // TURBOQUANT_HOST_TELEMETRY_H
