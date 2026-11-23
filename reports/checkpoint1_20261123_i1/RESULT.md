# I1 integration result

The isolated Wave-v11 production decode integration passes its bounded CAModel
gate: **0 BF16 ULP**, four tiles, 208 live rows, six query heads, one KV head,
one production-entry launch. Every input allocation stayed byte-identical;
output and workspace guards, inactive output rows and telemetry all passed.
No nonempty model exception dump was produced. Exit status is 0.

The model reported **34,701 ticks** and 108,981 ms wall time. This is a new
independent uniform-key correctness fixture, not the native 34,935-tick fixture;
the difference is not evidence of a speedup. There was one launch and no
warmup or cache-persistence experiment. The verifier's generic cache-evidence
boilerplate in measurement_manifest.json does not describe this test's launch
count.

## Provenance and checks

- Frozen compiled source: 5231e2c, ext4
  `/home/dmin/work/checkpoint1_20261123_i1/device-source`; full source manifest
  SHA-256 `add2f41af87382d746168f3ff6bd3e9ba4f24b1836fb580d57c85208c1ce94f0`.
- The existing build completed naturally with exit 0. Its session 91942 was
  recovered; no duplicate build was issued. AIC/AIV device objects, shared
  library and exact runner all completed under strict compile flags.
- Generated wrappers consume overflow_status in all four entry points; the
  -Wno-error exception was removed. Strict device/runner flags are captured in
  device-aiv-flags.txt and runner-flags.txt. CANN's multiple-kernel generator
  recommendation and generated-project CMake notices are separate from C++
  compiler diagnostics. Full initial build stdout was not redirected; natural
  completion was recovered from the existing session.
- The serialized build and model phases each held the exclusive checkpoint
  lock and phase guard. No compiler/model overlap occurred.
- Live libpem mapping matched canonical CANN 9.2.1 dav_3510 SHA-256
  `1a6420fe8123e88a120f0bbfc76710968e6708599c77d80a7b1097aca06eed33`.
  See loaded_model_libraries.json, measurement_manifest.json, device-run.log,
  device-run.exit and device-artifacts.sha256.
- CPU CTest: 8/8 pass using the separate regenerated complete golden dataset.
  See CPU_GATE.md and cpu-ctest.log.
- Scoped AKA regression gate: 0 fatal and 0 new warning after exact fingerprint
  suppression of the frozen verified Wave-v11 backlog; raw counts remain
  explicit in AKA_GATE.json. No diagnostic category was broadly suppressed.
- The five production port files remain byte-identical to frozen 7c31a112.
  Existing Qwen telemetry fixes are preserved. Verified tag, original checkout,
  original harness probes and original fixture data remain untouched.

## Yellow-zone handoff

The branch is ready for user physical-NPU testing of this **bounded signed-INT4
production decode path**. This is not E8 compatibility evidence, or exhaustive
raw-query, multi-worker, Split-K, long-context or hardware performance coverage.
The exact runner has a matching `test_wave_v11_exact_npu` target for the user's
device checkout; the NPU leg itself has not been built or run here.

```sh
cmake -S . -B build-yellow -DRUN_MODE=npu -DTQ_BUILD_KERNELS=ON -DASCEND_HOME_PATH=/path/to/cann
cmake --build build-yellow --target test_wave_v11_exact_npu -j2
build-yellow/tests/test_wave_v11_exact_npu
```

The workstation's physical NPU remains inaccessible. All device results above
come from the canonical CAModel.
