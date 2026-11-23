# Scalar metadata hoist selected for yellow-zone testing

This branch adds only the verified Slot03 metadata-hoist source delta on top of
Wave-v11 integration commit3e57bfa. The fused-decode production files remain
unchanged. The uniform grinder retains its phases, math, UB and synchronization;
it calculates the first mapping before startup quantization and the successor
mapping after phase4, retaining addresses through phases5..7. Exactly one mapping
read per strip remains. No future strip is read beyond the tail.

The isolated strict CANN9.2.1 build and six-strip CAModel run passed. Total ticks
fall64266→63629 (-637,0.991%). The finite interior estimate8722→8720.5 is nearly
unchanged; this is primarily a fill/drain gain. Exact K/V, scales, input and cache
canaries pass; three reviews, baseline-relative AKA0fatal/0new warnings and all76
release checks pass. Cold scalar metadata dependencies remain and are explicitly
attributed in POST_TRACE_I2.md; no globally stall-free scalar claim is made.

The packaged source is byte-identical to the simulated source, SHA256
bcbdcc866230bc72bf47ba3ec4fd025e5a063cd9333a58ee558f37e2924e89ad.
The device build/run logs and artifact hashes are included. This is source
integration of the tested isolated probe, not a second production-entry CAModel
launch. The baseline fixture and model trace remain immutable in the harness.
All eight toolkit-free integration CPU gates pass on this branch, including the
complete generated golden-decode fixtures (cpu-ctest.log).

For physical yellow-zone testing, configure a fresh RUN_MODE=npu device build
with TQ_BUILD_KERNELS=ON using the local CANN path, build test_live_write_npu,
then run tests/test_live_write_npu --slices6 --strips-per-block6 (with spaces
between option names and values). The actual command is:

```
./build-npu/tests/test_live_write_npu --slices 6 --strips-per-block 6
```

Build and physical execution are the owner's next gate. No physical NPU timing
or correctness result is claimed here. WAVE1_SCHEDULING.md retains all four
ablation outcomes and the CPM model's predictive ranking failure.
