# Exact requested-path reachability audit

All references below are from this workstation's current source snapshot.
Verified Wave-v11 files and the frozen E8 export remain unchanged.

## Native/production Wave-v11 is not the new E8 packet reader

- `C:/ascend-micro-harness/tests/sim/kernel/fused_decode_opt_kernel.cpp:2`
  includes the production `turboquant_fused_decode.cpp` body directly.
  Lines26 and40 explicitly instantiate `TurboQuantMode::KV4_FP8`.
- `C:/ascend-micro-harness/tests/sim/kernel/fused_decode_opt_sim_common.hpp:1`
  is a host ACL/session/guard helper, with a namespace at line11 and
  `GuardedDeviceBuffer` at line46. It contains no E8 reconstruction call.
- `C:/ascend-micro-harness/kernels/ops/turboquant/vllm_integration_layer/op_kernel/turboquant_fused_decode.cpp:82`
  includes `common/turboquant_codec.h`; line85 includes the vector service;
  line170 selects `TurboQuantModeCodec<MODE,...>`.
- `.../op_kernel/common/turboquant_codec.h:4` includes `turboquant_codec_mx.h`.
  `turboquant_codec_mx.h:21–23` includes the four-bit dequant routine,
  `turboquant_codec_950.h` and mode definitions. Its line40 specializes
  `TurboQuantPlanes<KV4_FP8>`; lines583–585 alias KV3/KV4/KV5 codecs.
- `.../op_kernel/vector/turboquant_vector_service.h:35` includes the950 codec.
  Lines944–1171 stage/unpack the mode codec. The KV4 mode is explicitly4bits
  at `common/turboquant_mode.h:283`.
- `.../op_kernel/common/turboquant_codec_950.h:27` asserts `BITS==4` for that
  helper. It is not a two-bit E8 label decoder.

No `E8`/`e8` reference occurs in the requested native probe, common host helper
or production fused-kernel translation unit. Other production modes exist, but
none of the inspected closure connects to the frozen E8 leaves. Calling this
verified native path an E8 reader would be inaccurate. No changes were applied
to this protected closure.

```text
native fused_decode_opt_kernel.cpp
  -> production turboquant_fused_decode.cpp (KV4_FP8 instantiation)
  -> TurboQuantModeCodec / vector service
  -> four-bit codec/unpack, Cube attention
  [no E8 leaf call]
```

## New active E8 closure

```text
parallel_decode/src/e8_fused_decode.cpp
  -> e8_parallel_decode.hpp::Decode
  -> Project (8 independent coefficient registers)
  -> ShapeResidualBits
       Xor8 parity
       2 × Repair -> Max8 (each7 IMNMX, ISA depth3)
       independent masked corrections + integer distance selection
  -> 8 × StoreCoordinate<Value,C>
       (q2*0.5f)*scale -> BF16_RNE -> exact FP32 widen + witness
```

The separate public compatibility APIs are active, compiled device paths:

```text
parallel_component.cpp
  -> src/routines/routine_e8_bf16_dequant_simt.hpp::Dequant
     -> explicit Project/Shape adapters -> common R8 helpers -> 8 BF16 stores
  -> src/routines/routine_e8_int8_dequant_simt.hpp::Dequant
     -> same common R8 helpers -> 8 RNE+saturated INT8 stores
```

Both replacement public headers retain Arithmetic/CompactInt8, all four compact
table layouts, and LegacyNearest/ResidualBits interfaces. There is one uniform
whole-block distribution loop, no coordinate loop. The original leaf headers
under `C:/ascend-micro-harness/routines` are unchanged frozen host oracles;
they are not in these active GPU call closures. The compiled reader's AIC text
matches the frozen Slot09 AIC text byte for byte.
