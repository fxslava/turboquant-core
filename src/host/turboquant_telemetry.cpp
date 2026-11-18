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

#include "turboquant_telemetry.h"

#include <cmath>
#include <cstring>

namespace ascend_micro::fused_decode_telemetry {

namespace {

constexpr std::uint32_t CeilDiv(const std::uint32_t a, const std::uint32_t b)
{
    return b ? (a + b - 1) / b : 0;
}

float bits_float(const std::uint32_t bits)
{
    float value = 0.0f;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

}  // namespace

std::uint32_t live_splits_of(const std::uint32_t context, const Geometry &g)
{
    if (is_fused(context, g)) {
        // A fused token has one worker total: split 0, and only when it has context rows.
        return context ? 1 : 0;
    }
    const std::uint32_t seq_blocks = CeilDiv(context, g.block_size);
    const std::uint32_t blocks_per_split = CeilDiv(seq_blocks, g.splits);
    std::uint32_t live = 0;
    for (std::uint32_t split = 0; split < g.splits; ++split) {
        const std::uint32_t block_start = split * blocks_per_split;
        if (block_start >= seq_blocks) break;
        std::uint32_t block_end = block_start + blocks_per_split;
        if (block_end > seq_blocks) block_end = seq_blocks;
        const std::uint32_t rows_start = block_start * g.block_size;
        std::uint32_t rows_end = block_end * g.block_size;
        if (rows_end > context) rows_end = context;
        if (rows_end <= rows_start) continue;
        ++live;
    }
    return live;
}

void split_rows(const std::uint32_t context, const Geometry &g, const std::uint32_t split,
                std::uint32_t &row_begin, std::uint32_t &row_end)
{
    const std::uint32_t seq_blocks = CeilDiv(context, g.block_size);
    const std::uint32_t blocks_per_split = CeilDiv(seq_blocks, g.splits);
    const std::uint32_t block_start = split * blocks_per_split;
    std::uint32_t block_end = block_start + blocks_per_split;
    if (block_end > seq_blocks) block_end = seq_blocks;
    row_begin = block_start * g.block_size;
    row_end = block_end * g.block_size;
    if (row_end > context) row_end = context;
}

Record read_record(const float *workspace, const Geometry &g, const std::uint32_t token,
                   const std::uint32_t head, const std::uint32_t split)
{
    const float *tail = workspace + record_offset(g, token, head, split) + g.head_size;
    Record r;
    const float position = tail[kTelemetryPosLane];
    const float page = tail[kTelemetryPageLane];
    r.max_logit = tail[kTelemetryMaxLane];
    r.run_sum = tail[kPartialSumLane];
    r.finite = std::isfinite(r.max_logit) && std::isfinite(position) && std::isfinite(page) &&
               std::isfinite(r.run_sum);
    r.position = std::isfinite(position) ? static_cast<std::int64_t>(std::llround(position)) : -1;
    r.logical_page = std::isfinite(page) ? static_cast<std::int64_t>(std::llround(page)) : -1;
    r.zero_padding = true;
    for (std::uint32_t lane = kTelemetryPageLane + 1; lane < kPartialSumLane; ++lane)
        if (tail[lane] != 0.0f) r.zero_padding = false;
    for (std::uint32_t lane = kPartialSumLane + 1; lane < kPartialTail; ++lane)
        if (tail[lane] != 0.0f) r.zero_padding = false;
    return r;
}

Merged merge(const float *workspace, const Geometry &g, const std::uint32_t token,
             const std::uint32_t head, const std::uint32_t context)
{
    Merged out;
    const auto fold = [&](const Record &r, const bool expect_record) {
        if (r.position < 0) {
            // A live worker always publishes a record; the sentinel here means the slot was
            // never written. A zero-context token legitimately has nothing to report.
            if (expect_record) out.stale = true;
            return;
        }
        if (r.logical_page != r.position / static_cast<std::int64_t>(g.block_size))
            out.inconsistent = true;
        if (!out.valid || r.max_logit > out.max_logit) {
            out.max_logit = r.max_logit;
            out.position = r.position;
            out.logical_page = r.logical_page;
        }
        out.valid = true;
    };
    if (is_fused(context, g)) {
        fold(read_record(workspace, g, token, head, 0), context > 0);
    } else {
        const std::uint32_t live = live_splits_of(context, g);
        for (std::uint32_t split = 0; split < live; ++split)
            fold(read_record(workspace, g, token, head, split), true);
    }
    if (out.valid) out.drifted = out.position >= static_cast<std::int64_t>(context);
    return out;
}

void dump(const float *workspace, const Geometry &g, const std::int32_t *context_lens,
          const std::int32_t *block_table, std::FILE *out)
{
    for (std::uint32_t token = 0; token < g.tokens; ++token) {
        const std::uint32_t context = static_cast<std::uint32_t>(context_lens[token]);
        for (std::uint32_t head = 0; head < g.heads; ++head) {
            const Merged m = merge(workspace, g, token, head, context);
            std::fprintf(out, "tq-telemetry token=%u head=%u ctx=%u ", token, head, context);
            if (!m.valid) {
                std::fprintf(out, "no-record%s\n", m.stale ? " (stale live slot)" : "");
                continue;
            }
            const std::int64_t row = m.position % static_cast<std::int64_t>(g.block_size);
            std::fprintf(out, "max_logit=%.6g argmax=%lld page=%lld", m.max_logit,
                         static_cast<long long>(m.position),
                         static_cast<long long>(m.logical_page));
            if (block_table && m.logical_page >= 0 &&
                m.logical_page < static_cast<std::int64_t>(g.max_blocks)) {
                const std::int32_t physical =
                    block_table[static_cast<std::size_t>(token) * g.max_blocks + m.logical_page];
                std::fprintf(out, "(phys=%d)", physical);
            }
            std::fprintf(out, " row=%lld", static_cast<long long>(row));
            if (m.stale) std::fprintf(out, " STALE-LIVE-SLOT");
            if (m.drifted) std::fprintf(out, " DRIFT(pos>=ctx)");
            if (m.inconsistent) std::fprintf(out, " PAGE-MISMATCH");
            std::fprintf(out, "\n");
        }
    }
}

VerifyResult verify(const std::uint32_t *words, const std::size_t word_count, const Geometry &g,
                    const std::int32_t *context_lens, const std::int32_t * /*block_table*/,
                    const std::uint32_t fill_word, const std::uint32_t position_floor,
                    const std::uint32_t position_ceiling)
{
    const std::uint32_t stride = record_stride(g.head_size);
    const std::uint64_t records = std::uint64_t{g.tokens} * g.heads * g.splits;
    if (word_count < records * stride) {
        VerifyResult bad;
        bad.ok = false;
        bad.what = "the workspace is shorter than the telemetry record extent";
        bad.word = word_count;
        return bad;
    }
    const auto fail = [](const std::size_t word, const char *what) {
        VerifyResult bad;
        bad.ok = false;
        bad.what = what;
        bad.word = word;
        return bad;
    };
    for (std::uint64_t i = 0; i < word_count; ++i) {
        const std::uint64_t r = i / stride;
        if (r >= records) {
            if (words[i] != fill_word)
                return fail(i, "fill word clobbered past the telemetry record extent");
            continue;
        }
        const std::uint32_t lane = static_cast<std::uint32_t>(i % stride);
        const std::uint32_t split = static_cast<std::uint32_t>(r % g.splits);
        const std::uint64_t th = r / g.splits;
        const std::uint32_t head = static_cast<std::uint32_t>(th % g.heads);
        static_cast<void>(head);
        const std::uint32_t token = static_cast<std::uint32_t>(th / g.heads);
        const std::uint32_t context = static_cast<std::uint32_t>(context_lens[token]);
        const float v = bits_float(words[i]);
        if (is_fused(context, g) && (split != 0 || lane < g.head_size)) {
            // A fused token writes only the statistics tail of its split-0 slot; everything
            // else -- the slot's accumulator region and the duplicate splits' whole slots --
            // keeps the fill.
            if (words[i] != fill_word)
                return fail(i, "workspace written outside the fused telemetry record");
            continue;
        }
        if (lane < g.head_size) {
            // A split token's partials are written for every split, live and dead alike
            // (RunTask stages neutral partials for dead ones). Only finiteness is checkable.
            if (!std::isfinite(v))
                return fail(i, "a partial accumulator lane is not finite");
            continue;
        }
        const std::uint32_t tail = lane - g.head_size;
        // A fused token publishes one live record (split 0) unless its context is empty; a
        // split token publishes a live record per live split and the sentinel record per dead
        // one (RunTask stages neutral partials for dead splits).
        const bool live_record =
            is_fused(context, g) ? (context > 0) : (split < live_splits_of(context, g));
        if (live_record) {
            // A live record: lane checks against the probe contract.
            if (tail == kTelemetryMaxLane) {
                if (!std::isfinite(v))
                    return fail(i, "the max-logit lane of a live record is not finite");
            } else if (tail == kTelemetryPosLane) {
                if (!std::isfinite(v))
                    return fail(i, "the position lane of a live record is not finite");
                const std::int64_t pos = static_cast<std::int64_t>(std::llround(v));
                if (pos < static_cast<std::int64_t>(position_floor) ||
                    pos >= static_cast<std::int64_t>(position_ceiling) ||
                    pos >= static_cast<std::int64_t>(context))
                    return fail(i, "the winning position is outside the live context");
            } else if (tail == kTelemetryPageLane) {
                if (!std::isfinite(v))
                    return fail(i, "the page lane of a live record is not finite");
                const std::int64_t pos =
                    static_cast<std::int64_t>(std::llround(bits_float(words[i - 1])));
                const std::int64_t page = static_cast<std::int64_t>(std::llround(v));
                if (page != pos / static_cast<std::int64_t>(g.block_size))
                    return fail(i, "the record's page disagrees with its position");
                if (g.max_blocks && page >= static_cast<std::int64_t>(g.max_blocks))
                    return fail(i, "the record's logical page is outside the block table");
            } else if (tail == kPartialSumLane) {
                if (is_fused(context, g)) {
                    if (v != 0.0f)
                        return fail(i, "a fused record carries a run_sum lane");
                } else if (!(v > 0.0f) || !std::isfinite(v)) {
                    return fail(i, "a live split's run_sum lane is not positive and finite");
                }
            } else if (v != 0.0f) {
                return fail(i, "a telemetry record's padding lanes are not zero");
            }
            continue;
        }
        // A dead split of a non-fused token, or a zero-context fused task: the sentinel
        // record. The probe lanes must be exactly the sentinel and every other lane finite
        // (the accumulator is zeroed and the max is the negated-inf init, both finite).
        if (tail == kTelemetryPosLane || tail == kTelemetryPageLane) {
            if (v != kTelemetryInvalid)
                return fail(i, "a dead slot's probe lanes are not the sentinel");
        } else if (!std::isfinite(v)) {
            return fail(i, "a dead slot carries a non-finite lane");
        }
    }
    return VerifyResult{};
}

}  // namespace ascend_micro::fused_decode_telemetry
