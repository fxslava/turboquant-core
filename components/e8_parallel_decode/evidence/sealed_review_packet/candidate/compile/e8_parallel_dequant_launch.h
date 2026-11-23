
#ifndef HEADER_ACLRTLAUNCH_E8_PARALLEL_COMPONENT_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_E8_PARALLEL_COMPONENT_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_e8_parallel_component(uint32_t numBlocks, void* stream, void* labels, void* scales, void* bf16, void* int8);

inline uint32_t e8_parallel_component(uint32_t numBlocks, void* hold, void* stream, void* labels, void* scales, void* bf16, void* int8)
{
    (void)hold;
    return aclrtlaunch_e8_parallel_component(numBlocks, stream, labels, scales, bf16, int8);
}

#endif
