#pragma once

// KV4/FP8 Cube decode: 64 keys per tile, up to eight query heads per AIV.
// The caller owns DMA, Fixpipe visibility, and inter-core synchronization.
// All arithmetic stays on Vector; state fields are replicated 32-byte blocks.
#include "routine_vector_online_softmax.hpp"

#if defined(__CCE_AICORE__) || defined(ASCEND_MICRO_ENABLE_NATIVE)
namespace ascend_micro {

// Two distinct sixty-fours. kCubeTileKeys is the tile's key count, which sets every row
// operand's element count and a reduction's mask. kCubeRepeatLanes is how many FP32 lanes one
// vector repeat covers, which sets the stride of a sweep across a head's channels. They are
// equal on this part and neither is the other.
constexpr std::uint32_t kCubeTileKeys    = 64;
constexpr std::uint32_t kCubeRepeatLanes = 64;

// The 32-byte blocks one tile row spans, which is a reduction's source repeat stride.
constexpr std::uint32_t kCubeTileBlocks = kCubeTileKeys * sizeof(float) / 32;

// DIRTY-DATA (D2): the finite sentinel written over a dead tail lane. It is FLT_MAX negative,
// so exp(sentinel - max) underflows to zero, and -- unlike the old Min against a ramp -- a
// SELECT writes it even when the lane's score is NaN, because it does not touch the score's
// value at all.
constexpr float kTailDeadLogit = -3.4028235e38f;

template <class Codec, class Operand>
struct NativeCubeSoftmaxEngine {
    AscendC::LocalTensor<float>   scores, key_scale, value_scale, q_inv, mask;
    AscendC::LocalTensor<float>   run_max, run_sum, tile_max, new_max, alpha;
    AscendC::LocalTensor<float>   prob_scale, part, row_reduce;
    AscendC::LocalTensor<Operand> probabilities;
    Codec&                        codec;
    std::uint32_t                 heads, valid;
    float                         operand_max, epsilon;
    AscendC::BinaryRepeatParams   by_vector, by_block;

    // The attention telemetry probe (ASCEND_TQ_DECODE_TELEMETRY, turboquant_layout.h). The
    // caller carves this arena out of a buffer that is dead for the whole tile loop and fills
    // probe_pos/probe_page with the sentinel and probe_ramp with 0..63 before the first tile;
    // probe_context_base is the context index of this tile's first row and probe_page_id the
    // physical cache page it was read from. argmax_probe() runs between update_max_and_alpha
    // and exponentiate_scores (vector_online_softmax_probabilities_probed), the only point
    // where the finished logits, their per-head maxima and the PRE-commit running maxima are
    // all live. With the macro off the phase compiles to nothing and the operands are unread.
    AscendC::LocalTensor<float>   probe_pos, probe_page, probe_ramp;
    AscendC::LocalTensor<float>   probe_bc, probe_diff, probe_sel;
    AscendC::LocalTensor<float>   probe_stage, probe_cand_pos, probe_cand_page;
    AscendC::LocalTensor<float>   probe_pred;  // 32 bytes, reinterpreted as the predicate
    std::uint32_t                 probe_context_base = 0, probe_page_id = 0;

    // Per-head running argmax fold. A head "improved" this tile when tile_max > run_max
    // (strictly: ties keep the EARLIER position, within a tile by the MIN reduce over the
    // ascending index ramp and across tiles by the strict comparison). The candidate position
    // is the tile's context base plus the first lane whose logit equals the tile maximum --
    // score == tile_max exactly, because the maximum is one of the lanes and x - x is +0.
    // Everything is vector instructions over replicated head blocks; no scalar readback, no
    // events, no GM. probe_bc is reused for both per-head broadcasts; probe_diff holds the
    // row difference first and the wide improvement test after the loop.
    __aicore__ inline void argmax_probe()
    {
#if defined(ASCEND_TQ_DECODE_TELEMETRY) && ASCEND_TQ_DECODE_TELEMETRY
        const std::uint32_t lanes = heads * kStateLanes;
        const AscendC::LocalTensor<std::uint8_t> predicate =
            probe_pred.template ReinterpretCast<std::uint8_t>();
        for (std::uint32_t head = 0; head < heads; ++head) {
            AscendC::Brcb(probe_bc, tile_max[head * kStateLanes], 1, {1, 8});
            AscendC::Sub(probe_diff, scores[head * kCubeTileKeys], probe_bc,
                         std::uint64_t{kCubeTileKeys});
            AscendC::Compares(predicate, probe_diff, 0.0f, AscendC::CMPMODE::EQ,
                              std::uint32_t{kCubeTileKeys});
            AscendC::Select(probe_sel, predicate, probe_ramp,
                            static_cast<float>(kCubeTileKeys),
                            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                            std::uint32_t{kCubeTileKeys});
            AscendC::ReduceRepeat<AscendC::ReduceType::MIN>(
                probe_stage, probe_sel, static_cast<std::int32_t>(kCubeTileKeys), 1, 1, 1,
                static_cast<std::int32_t>(kCubeTileBlocks), AscendC::ReduceOrder::ORDER_ONLY_VALUE);
            AscendC::Brcb(probe_bc, probe_stage, 1, {1, 8});
            AscendC::Adds(probe_cand_pos[head * kStateLanes], probe_bc, 0.0f,
                          std::uint64_t{kStateLanes});
        }
        AscendC::Adds(probe_cand_pos, probe_cand_pos, static_cast<float>(probe_context_base),
                      lanes);
        AscendC::Sub(probe_diff, tile_max, run_max, lanes);
        AscendC::Compares(predicate, probe_diff, 0.0f, AscendC::CMPMODE::GT, lanes);
        AscendC::Select(probe_pos, predicate, probe_cand_pos, probe_pos,
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, lanes);
        AscendC::Duplicate(probe_cand_page, static_cast<float>(probe_page_id), lanes);
        AscendC::Select(probe_page, predicate, probe_cand_page, probe_page,
                        AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, lanes);
#endif
    }

    template <AscendC::ReduceType Kind>
    __aicore__ inline void reduce(const AscendC::LocalTensor<float>& dst)
    {
        constexpr std::int32_t mask       = static_cast<std::int32_t>(kCubeTileKeys);
        constexpr std::int32_t rep_blocks = static_cast<std::int32_t>(kCubeTileBlocks);
        if constexpr (Kind == AscendC::ReduceType::SUM) {
            AscendC::ReduceRepeat<Kind>(row_reduce, scores, mask, heads, 1, 1, rep_blocks);
        } else {
            AscendC::ReduceRepeat<Kind>(row_reduce, scores, mask, heads, 1, 1, rep_blocks,
                                       AscendC::ReduceOrder::ORDER_ONLY_VALUE);
        }
        AscendC::PipeBarrier<PIPE_V>();
        vector_broadcast_block(dst, row_reduce);
    }

    __aicore__ inline void update_max_and_alpha()
    {
        AscendC::Mul(scores, scores, key_scale, std::uint64_t{kCubeTileKeys}, heads, by_vector);
        AscendC::Mul(scores, scores, q_inv, std::uint64_t{kCubeTileKeys}, heads, by_block);
        // DIRTY-DATA (D2): rows past `valid` in a partial last tile were read from the cache
        // page (the tail of the page, not the live context). Their key codes and scales are
        // dirty: 0xDEADBEEF bytes are finite garbage, but a poisoned fp32 scale is a NaN, and
        // Min's NaN behaviour is not specified. Overwrite every dead lane with a finite sentinel
        // through the lane predicate instead, so the MAX below and the online recurrence never
        // see the neighbor slots' data. Live lanes are copied through unchanged. `scores` holds
        // `heads` contiguous 64-lane rows and the 64-bit predicate is the same for each, so the
        // Select is per row: the first-n form only consumes one row's mask bits.
        if (valid < kCubeTileKeys) {
            const AscendC::LocalTensor<std::uint8_t> predicate =
                mask[kCubeTileKeys].template ReinterpretCast<std::uint8_t>();
            for (std::uint32_t head = 0; head < heads; ++head) {
                const std::uint32_t at = head * kCubeTileKeys;
                AscendC::Select(scores[at], predicate, scores[at], kTailDeadLogit,
                                AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                                std::uint32_t{kCubeTileKeys});
            }
        }
        reduce<AscendC::ReduceType::MAX>(tile_max);
        AscendC::Max(new_max, run_max, tile_max, heads * kStateLanes);
        AscendC::Sub(alpha, run_max, new_max, heads * kStateLanes);
    }

    __aicore__ inline void exponentiate_scores()
    {
        AscendC::Sub(scores, scores, new_max, std::uint64_t{kCubeTileKeys}, heads, by_block);
        AscendC::Exp(alpha, alpha, heads * kStateLanes);
        AscendC::Exp(scores, scores, heads * kCubeTileKeys);
    }

    __aicore__ inline void update_denominator()
    {
        reduce<AscendC::ReduceType::SUM>(part);
        AscendC::Mul(run_sum, run_sum, alpha, heads * kStateLanes);
        AscendC::Add(run_sum, run_sum, part, heads * kStateLanes);
    }

    // Denominator uses unscaled probabilities. Value scale belongs only in the numerator,
    // followed by the source's FP8 dynamic range map.
    __aicore__ inline void quantize_probabilities()
    {
        AscendC::Mul(scores, scores, value_scale, std::uint64_t{kCubeTileKeys}, heads, by_vector);
        // DIRTY-DATA (D2): the value scale of a dead tail row can be NaN (the scale plane past
        // `context_lens` is allocator garbage). exp(dead) is zero, but 0 * NaN is NaN, and the
        // MAX reduction below would then hand a NaN probability scale to every LIVE row of the
        // tile. SELECT the dead lanes to exactly zero after the multiply, so the reduction and
        // the operand cast only ever see the live rows. Per row, for the same reason as above.
        if (valid < kCubeTileKeys) {
            const AscendC::LocalTensor<std::uint8_t> predicate =
                mask[kCubeTileKeys].template ReinterpretCast<std::uint8_t>();
            for (std::uint32_t head = 0; head < heads; ++head) {
                const std::uint32_t at = head * kCubeTileKeys;
                AscendC::Select(scores[at], predicate, scores[at], 0.0f,
                                AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                                std::uint32_t{kCubeTileKeys});
            }
        }
        reduce<AscendC::ReduceType::MAX>(part);
        AscendC::Duplicate(prob_scale, operand_max, heads * kStateLanes);
        AscendC::Adds(part, part, epsilon, heads * kStateLanes);
        AscendC::Div(prob_scale, prob_scale, part, heads * kStateLanes);
        AscendC::Mul(scores, scores, prob_scale, std::uint64_t{kCubeTileKeys}, heads, by_block);
        codec.template CastToOperand<Operand>(probabilities, scores, heads * kCubeTileKeys);
    }
};

// A replicated head block broadcasts over each 64-channel chunk. No scalar
// extraction and no reduction side channel, including for ragged head counts.
__aicore__ inline void vector_cube_scale_rows(
    const AscendC::LocalTensor<float>& rows,
    const AscendC::LocalTensor<float>& scale, std::uint32_t heads,
    std::uint32_t channels, const AscendC::BinaryRepeatParams& repeat)
{
    for (std::uint32_t col = 0; col < channels; col += kCubeRepeatLanes)
        AscendC::Mul(rows[col], rows[col], scale, std::uint64_t{kCubeRepeatLanes}, heads, repeat);
}

__aicore__ inline void vector_cube_accumulate(
    const AscendC::LocalTensor<float>& acc,
    const AscendC::LocalTensor<float>& context,
    const AscendC::LocalTensor<float>& probability_scale,
    const AscendC::LocalTensor<float>& inverse, std::uint32_t heads,
    std::uint32_t channels, const AscendC::BinaryRepeatParams& repeat)
{
    AscendC::Duplicate(inverse, 1.0f, heads * kStateLanes);
    AscendC::Div(inverse, inverse, probability_scale, heads * kStateLanes);
    vector_cube_scale_rows(context, inverse, heads, channels, repeat);
    AscendC::Add(acc, acc, context, heads * channels);
}

// alpha = exp(run_max - new_max) decays what is already accumulated, before this tile's context
// lands. The new maximum becomes the running one in the same call, so the caller sees one stage
// and cannot leave the two halves of the step apart.
__aicore__ inline void vector_cube_decay_accumulator(
    const AscendC::LocalTensor<float>& acc,
    const AscendC::LocalTensor<float>& alpha,
    const AscendC::LocalTensor<float>& run_max,
    const AscendC::LocalTensor<float>& new_max, std::uint32_t heads,
    std::uint32_t channels, const AscendC::BinaryRepeatParams& repeat)
{
    vector_cube_scale_rows(acc, alpha, heads, channels, repeat);
    AscendC::Adds(run_max, new_max, 0.0f, heads * kStateLanes);
}

// Scores of a partial last tile are gated by this ramp: a ramp crossing zero at `valid`, huge
// and positive before it and huge and negative after. The caller turns its sign into the bit
// predicate the softmax SELECTs against; the ramp itself is kept because a lane index ramp is
// what names the live rows, not the scores (a dirty score can be NaN). `lanes` aliases the mask
// buffer's own tail as int32 scratch for the progression.
__aicore__ inline void vector_cube_tail_mask(
    const AscendC::LocalTensor<float>&        mask,
    const AscendC::LocalTensor<std::int32_t>& lanes,
    std::uint32_t valid, float midpoint, float gain)
{
    AscendC::ArithProgression(lanes, 0, 1, static_cast<std::int32_t>(kCubeTileKeys));
    AscendC::Cast(mask, lanes, AscendC::RoundMode::CAST_NONE, kCubeTileKeys);
    AscendC::Adds(mask, mask, midpoint - static_cast<float>(valid), kCubeTileKeys);
    AscendC::Muls(mask, mask, gain, kCubeTileKeys);
}

// The task's kv head picked out of every tile row's scale slot, K lanes first and V lanes
// behind, from byte offsets the caller built once for the task. The stored scale is the whole
// per-row scale (the writers' convention, turboquant_codec_mx.h): the score scale carries the
// bare attention scale and the inverse gain is 1 -- a non-uniform codebook would reconstruct
// inside the operand and leave nothing to divide out either.
__aicore__ inline void vector_cube_tile_scales(
    const AscendC::LocalTensor<float>&         key_scale,
    const AscendC::LocalTensor<float>&         value_scale,
    const AscendC::LocalTensor<float>&         scale_tile,
    const AscendC::LocalTensor<std::uint32_t>& offsets,
    float score_scale, float inverse_gain)
{
    AscendC::Gather(key_scale, scale_tile, offsets, 0, kCubeTileKeys);
    AscendC::Gather(value_scale, scale_tile, offsets[kCubeTileKeys], 0, kCubeTileKeys);
    AscendC::Muls(key_scale, key_scale, score_scale, kCubeTileKeys);
    AscendC::Muls(value_scale, value_scale, inverse_gain, kCubeTileKeys);
}

// The online softmax's close-out: 1 / (run_sum + eps) per head, then one scaling sweep over the
// accumulator's channels.
//
// UB FOOTPRINT IS A CONTRACT HERE. `sums` is a Brcb destination, and Brcb writes 64 floats from
// sums[head * kStateLanes] for every head, so it needs (heads - 1) * kStateLanes + 64 floats and
// NOT the heads * kStateLanes a lane count suggests. Callers static_assert that `sums` cannot
// reach `inverses` and that `inverses` cannot reach whatever follows it; neither operand may be
// narrowed to its lane count.
__aicore__ inline void vector_cube_normalize_heads(
    const AscendC::LocalTensor<float>& acc,
    const AscendC::LocalTensor<float>& run_sum,
    const AscendC::LocalTensor<float>& sums,
    const AscendC::LocalTensor<float>& inverses, std::uint32_t heads,
    std::uint32_t channels, float epsilon)
{
    const std::uint8_t                row_blocks = static_cast<std::uint8_t>(channels / kStateLanes);
    const AscendC::BinaryRepeatParams by_block{1, 1, 0, row_blocks, row_blocks, 1};
    for (std::uint32_t head = 0; head < heads; ++head)
        AscendC::Adds(run_sum[head * kStateLanes], run_sum[head * kStateLanes], epsilon, 1);
    for (std::uint32_t head = 0; head < heads; ++head)
        vector_broadcast_block(sums[head * kStateLanes], run_sum[head * kStateLanes]);
    AscendC::Duplicate(inverses, 1.0f, heads * kStateLanes);
    AscendC::Div(inverses, inverses, sums, heads * kStateLanes);
    vector_cube_scale_rows(acc, inverses, heads, channels, by_block);
}

// One split's workspace record per head is [accumulator (channels), run_max lane, run_sum lane]
// and `stride` is the tail's whole width, so a record stays 32-byte aligned however few lanes
// carry state. The zeroing and the two lane writes are one vector chain; ccec orders the
// overlap within the pipe.
__aicore__ inline void vector_cube_pack_partial_tails(
    const AscendC::LocalTensor<float>& tails,
    const AscendC::LocalTensor<float>& run_max,
    const AscendC::LocalTensor<float>& run_sum, std::uint32_t heads,
    std::uint32_t stride, std::uint32_t max_lane, std::uint32_t sum_lane)
{
    AscendC::Duplicate(tails, 0.0f, heads * stride);
    for (std::uint32_t head = 0; head < heads; ++head) {
        AscendC::Adds(tails[head * stride + max_lane], run_max[head * kStateLanes], 0.0f, 1);
        AscendC::Adds(tails[head * stride + sum_lane], run_sum[head * kStateLanes], 0.0f, 1);
    }
}

// Query quantization onto the operand grid, all of a subcore's rows at once. The per-head
// reciprocals stay in UB as replicated blocks, so nothing here waits on the vector unit and the
// softmax reads them as a block operand rather than through a scalar.
//
// ReduceMax AND NOT ReduceRepeat, deliberately: a query row is wider than one vector register,
// so the reduction cannot be a single repeat, and ReduceMax's library-internal scalar readback
// is harmless because the decode runs under a TPipe. The leaf recurrence in
// routine_vector_online_softmax.hpp may not use it -- a kernel that places its own LocalTensors
// has a null GetTPipePtr() and traps at GM 0x40. Do not lift this engine into such a kernel.
template <class Codec, class Operand>
struct NativeCubeQueryEngine {
    AscendC::LocalTensor<float>   query, magnitude, scratch;
    AscendC::LocalTensor<float>   row_max, amax, scale, inverse;
    AscendC::LocalTensor<Operand> operand;
    Codec&                        codec;
    std::uint32_t                 heads, channels;
    float                         operand_max, epsilon;
    AscendC::BinaryRepeatParams   by_block;

    __aicore__ inline void measure_magnitude()
    {
        AscendC::Abs(magnitude, query, heads * channels);
        for (std::uint32_t head = 0; head < heads; ++head)
            AscendC::ReduceMax<float>(row_max[head], magnitude[head * channels], scratch, channels, false);
    }

    __aicore__ inline void derive_scale()
    {
        vector_broadcast_block(amax, row_max);
        AscendC::Adds(amax, amax, epsilon, heads * kStateLanes);
        AscendC::Duplicate(scale, operand_max, heads * kStateLanes);
        AscendC::Duplicate(inverse, 1.0f, heads * kStateLanes);
        AscendC::Div(scale, scale, amax, heads * kStateLanes);
        AscendC::Div(inverse, inverse, scale, heads * kStateLanes);
    }

    __aicore__ inline void quantize()
    {
        vector_cube_scale_rows(query, scale, heads, channels, by_block);
        codec.template CastToOperand<Operand>(operand, query, heads * channels);
    }
};

template <class Engine>
__aicore__ inline void vector_cube_quantize_query(Engine& engine)
{
    engine.measure_magnitude();
    engine.derive_scale();
    engine.quantize();
}
} // namespace ascend_micro
#endif
