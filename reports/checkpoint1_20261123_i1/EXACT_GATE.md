# Bounded exact production-entry gate

Prepared `tests/npu/test_wave_v11_exact.cpp` and target `test_wave_v11_exact_sim`.
It launches the integrated production `launch_fused_decode_opt`, without a
diagnostic kernel wrapper, exactly once: grid 1, six query heads, one KV head,
four 64-row tiles, a 16-row final tail, context 208, BF16 output. This reaches
the port's paired six-head path and depth-two ring wrap/drain.

The independent oracle method comes from harness commit d026b81's
`tests/sim/kernel/test_camodel_fused_decode_opt.cpp`: uniform keys make every
softmax weight exactly one, and an FP64 mean over tile values supplies the
reference. The old harness's odd affine reconstruction is adapted to core's
actual signed-nibble-times-scale contract: key nibble 3, scale 2, value nibbles
1..7, reconstructed values 2..14. The fixture checks the independent tile mean
against decoding its actual NZ-packed cache by live rows before any launch.

The device gate demands equality at each stored BF16 word (0 ULP). It also
checks every input allocation byte for byte, exterior guard bands, inactive
output rows, and the shared workspace telemetry contract. Unassigned payloads
carry FP32 quiet-NaN words. It makes no E8 or raw-query claim.

Validation completed without BiSheng or CAModel:

- CPU-only `test_wave_v11_fixture`: strict GCC 11.4 build and host fixture pass.
- Actual ACL-host runner branch: GCC `-std=c++17 -Wall -Wextra -Werror
  -fsyntax-only`, using CANN 9.2.1 headers, passes.
- Generated-wrapper policy: a temporary generated-wrapper example compiles
  with strict GCC flags after consuming `overflow_status`; repeat configure
  is idempotent. Only `auto_gen_*.cpp` sources can be changed. The previous
  `-Wno-error=unused-parameter` exception is removed, so new builds must be
  warning clean. Production kernel sources and toolkit installation remain
  untouched by this policy.

The dedicated target can be built after the coordinator grants the serialized
build lane. No device compile, simulation, exact device result, or new timing
has yet been produced. The native 34,935-tick reference remains historical;
this fixture differs and is a correctness gate rather than an exact timing
reproduction.
