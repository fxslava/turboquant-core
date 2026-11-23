#pragma once
#include <cstdint>
#include "slot02/src/e8_packet_layout.hpp"

// Components consume PRE-ENCODED E8 labels and FP32 token scales.
// They do not accept BF16 activation tensors or implement an E8 quantizer.
// Each destination packet is one D256 plane; K and V are separate packets.
extern "C" void launch_e8_packet_v1_scatter(
    void* stream, std::uint8_t* labels, std::uint8_t* scale,
    std::uint8_t* packet, std::uint32_t row);

// Fixed minimal probe: strips=6, inputMode=2; 96 token rows, two packets/plane.
// Separate label/scales arrays are contiguous in token order. Output packet
// order follows token/64. The final32 rows remain untouched.
extern "C" void launch_e8_packet_v1_batch_encoded(
    void* stream, std::uint8_t* keyLabels, std::uint8_t* keyScales,
    std::uint8_t* valueLabels, std::uint8_t* valueScales,
    std::uint8_t* keyPackets, std::uint8_t* valuePackets,
    std::uint32_t strips, std::uint32_t inputMode);
