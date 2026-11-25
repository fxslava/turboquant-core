#ifndef TURBOQUANT_HOST_TELEMETRY_H
#define TURBOQUANT_HOST_TELEMETRY_H

#include "turboquant_layout.h"

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

struct Geometry {
    std::uint32_t tokens = 0;
    std::uint32_t heads = 0;
    std::uint32_t splits = 1;
    std::uint32_t head_size = 256;
    std::uint32_t block_size = 128;
    std::uint32_t max_blocks = 0;
    std::uint32_t num_blocks = 0;
    std::uint32_t fused_context_limit = 4096;
};

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

constexpr bool is_fused(std::uint32_t context, const Geometry &g)
{
    return g.splits <= 1 || context <= g.fused_context_limit;
}

std::uint32_t live_splits_of(std::uint32_t context, const Geometry &g);

void split_rows(std::uint32_t context, const Geometry &g, std::uint32_t split,
                std::uint32_t &row_begin, std::uint32_t &row_end);

struct Record {
    float max_logit = 0.0f;
    float run_sum = 0.0f;
    std::int64_t position = -1;
    std::int64_t logical_page = -1;
    bool finite = false;
    bool zero_padding = false;
};
Record read_record(const float *workspace, const Geometry &g, std::uint32_t token,
                   std::uint32_t head, std::uint32_t split);

struct Merged {
    float max_logit = 0.0f;
    std::int64_t position = -1;
    std::int64_t logical_page = -1;
    bool valid = false;
    bool stale = false;
    bool drifted = false;
    bool inconsistent = false;
};
Merged merge(const float *workspace, const Geometry &g, std::uint32_t token, std::uint32_t head,
             std::uint32_t context);

void dump(const float *workspace, const Geometry &g, const std::int32_t *context_lens,
          const std::int32_t *block_table, std::FILE *out);

struct VerifyResult {
    bool ok = true;
    const char *what = "";
    std::size_t word = 0;
};
VerifyResult verify(const std::uint32_t *words, std::size_t word_count, const Geometry &g,
                    const std::int32_t *context_lens, const std::int32_t *block_table,
                    std::uint32_t fill_word, std::uint32_t position_floor = 0,
                    std::uint32_t position_ceiling = UINT32_MAX);

}

#endif
