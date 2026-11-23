# Slot 02 — Verified E8 packet scatter

**PASS on CANN9.2.1 dav-3510 CAModel.** This is the user-requested encoded packet output checkpoint: single D256 token/plane, 64 uint16-label bytes plus one FP32 token scale. It does not quantize raw BF16 or execute fused attention.

## Contract and implementation

Each K or V plane is a separate **4352-byte** packet: LEuint16 labels `[64][32]` at bytes0..4095; LEFP32 scales `[64]` at4096..4351. Input labels64B, packet base32B aligned, scale4B aligned. Caller exclusively owns `(packet,row)`.

The kernel stages64 label bytes and one4-byte scale into disjoint UB regions0..63 and64..95. A documented MTE2_MTE3 pair precedes output DMA, and MTE3_S drains before return. DataCopyPad writes exactly4 scale bytes, preserving neighbouring rows. UB footprint96B; no SIMD/SIMT VF, Cube, cross-core events, scalar tensor-data loop or device PRINTF. Generated ISA includes the normal compiler epilogue BAR.ALL/DSB, disclosed in `device-isa.txt`; these are not source hot-loop barriers. No emitted LDK/STK spill instructions were found.

The shared immutable ABI header is `src/e8_packet_layout.hpp`, SHA256 **53c18584ae352e9f00677493450504ec013c5fbe970b9e2549251436ea8f5b71**. Slot07 copies this exact snapshot.

## Gates

| Gate | Result |
| --- | --- |
| AKA absolute gate, no suppressions needed | **0 fatal, 0 warnings** |
| New warnings relative to initial small-kernel baseline | **0**; initial2 unresolved brace-initializer-domain warnings were resolved with truthful layout annotations |
| CANN9.2.1 Release device/host/runner build | **PASS**, `-Wall -Wextra -Werror` |
| CAModel six writes: K/V × rows0,31,63 | **PASS** |
| Both full4352B packets checked after every write | **52,224 compared bytes, 0 mismatches** |
| Unowned rows, adjacent scales, input buffers, all GM guards | **PASS** |
| Saved device packet -> unchanged frozen E8 leaf on CPU | **1536 BF16 outputs, 0 BF16 ULP** |
| Physical NPU | Not run; reserved for user yellow-zone testing |

The measured simulator footer is **14,476 total ticks** for the complete six-launch runner. This is an aggregate session count, not a single-token latency or steady-state throughput result. Wall runtime35.737s.

`run/loaded_model_libraries.json` records live-mapped `dav_3510/lib/libpem_davinci.so`, SHA256 `1a6420fe8123e88a120f0bbfc76710968e6708599c77d80a7b1097aca06eed33`. `measurement_manifest.json` confirms exit0 and loaded_pem_verified=true. No callback helper was used. The generic verifier's cache template text does not establish warm-cache persistence for this functional runner.

The4 saved output and expected packet files and result JSON are under `run/`. Input labels/scales come from the immutable Wave1 BF16 fixtures; CPU reconstruction directly calls frozen Project/ShapeResidualBits with original header SHA256 `28f13039c8fec29ef91919a69a8314a60c9cb9aec3f5f3fc75b270f8e524cda2` and compares against archived BF16 golden bytes.

## Reproduction and provenance

Native source/build/run: `/home/dmin/work/checkpoint2_20261123/slot02/{src,build-r2,run}`. Run `bash device.sh build`, then `bash device.sh run` **only when the coordinator grants the lane**. Both commands take the same checkpoint-exclusive lock and shared compile/CAModel phase guard. The source and executable/device-object hashes are retained in `source.sha256` and `device-artifacts.sha256`.

The first strict compile, retained as `build.log`, rejected CANN's host wrapper handling of `std::uint32_t` and a missing host GM_ADDR alias. The verified revision uses the documented raw pointer/native uint32_t interface and retains strict flags; `build-r2.log` is the clean final build. SDK headers are system includes and generated overflow-status wrappers receive the existing scoped consume-parameter policy. No compiler warning was relaxed.

No verified Wave-v11, baseline source, frozen E8 leaf or registry slot was modified. The lane was released after this successful run. Remaining raw encoder and E8 fused-attention reader integration are outside this packet-output checkpoint.

