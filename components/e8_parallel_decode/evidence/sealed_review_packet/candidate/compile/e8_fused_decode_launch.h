
#ifndef HEADER_ACLRTLAUNCH_E8_FUSED_DECODE_FP32_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_E8_FUSED_DECODE_FP32_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_e8_fused_decode_fp32(uint32_t numBlocks, void* stream, void* gm_keys, void* gm_values, void* gm_offsets, void* gm_query, void* gm_output, void* gm_key_witness, void* gm_value_witness, void* gm_score_witness, uint32_t tiles, uint32_t last_rows, float scale);

inline uint32_t e8_fused_decode_fp32(uint32_t numBlocks, void* hold, void* stream, void* gm_keys, void* gm_values, void* gm_offsets, void* gm_query, void* gm_output, void* gm_key_witness, void* gm_value_witness, void* gm_score_witness, uint32_t tiles, uint32_t last_rows, float scale)
{
    (void)hold;
    return aclrtlaunch_e8_fused_decode_fp32(numBlocks, stream, gm_keys, gm_values, gm_offsets, gm_query, gm_output, gm_key_witness, gm_value_witness, gm_score_witness, tiles, last_rows, scale);
}

#endif
