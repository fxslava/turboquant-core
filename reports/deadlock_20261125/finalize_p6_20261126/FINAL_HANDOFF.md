# P6 integration handoff — 2026-11-26

P6 is promoted from core into vLLM under the explicit timing waiver. **36 shared C++ files and one analyzer configuration match byte for byte; all shared kernel headers are comment-free.** P7 remains isolated and unpromoted.

Signed-off product commits: core `043bcc43aa2e5d6ffa1d97930e0a31145731322f`; vLLM `7ee3e928078fc7e5d6e1b191b3ee5b953c8e7fd8`. The verification evidence is retained separately in core. Unrelated pre-existing untracked artifacts are preserved.

## Runtime verification

| P6 case | Tiles | Raw ticks | Result |
| --- | ---: | ---: | --- |
| STRIP narrow uniform, matched primary | 6 | 63,649 | Exact parity, guards, mappings, events PASS |
| STRIP wide Lloyd | 1 | 31,276 | PASS |
| NZ narrow Lloyd | 4 | 147,436 | PASS |
| NZ wide uniform, two groups | 4 | 56,379 | PASS |

Every run completed naturally within its 900-second watchdog. The longest took 503.7 seconds. No run exceeded six tiles.

The primary is 20 ticks above the historical 63,629 cap and five below the matched modern Frozen03 result of 63,654. Its original `raw_cap_met=false` is preserved. The user explicitly waived that modern delta. There are zero new active-body ticks, identical 229 VF PCs per AIV and 549 matching HardEvent pairs. The separate original-host 63,630 control and 63,637 P6 failure remain recorded. **No counter was normalized.**

[Waiver](C:/Ascend/turboquant-core/reports/deadlock_20261125/finalize_p6_20261126/USER_WAIVER.json) · [Four functional records](C:/Ascend/turboquant-core/reports/deadlock_20261125/finalize_p6_20261126/FUNCTIONAL.json) · [Fresh Sol 6.2 sign-off](C:/Ascend/turboquant-core/reports/deadlock_20261125/finalize_p6_20261126/FINAL_SOL62_RELEASE_REVIEW.json) · [Final acceptance](C:/Ascend/turboquant-core/reports/deadlock_20261125/finalize_p6_20261126/FINAL_ACCEPTANCE.json)

## Final validation

| Gate | Result |
| --- | --- |
| Guarded attention, repeated after final checker correction | 606 passed |
| Sync/build selection, repeated after correction | 14 passed |
| C++ contract gates, repeated after correction | 7 passed |
| Core host CTest | 9 passed |
| Strict real-Torch binding syntax | 5 translation units passed |
| Deployed CANN syntax | 30 role checks passed |
| Analyzer | 12 deployed cases plus 4 independent device-profile comparisons: zero new diagnostics |
| Real Torch 2.10 CPU library | Six Meta and six Fake calls; 24 E8 rejection cases; three positive and six rejected tiling cases; zero device calls |

The first analyzer invocation reported an unmatched wide host launch because it received only the narrow translation unit. A separately reviewed checker-only correction submits both actual source files, binds reports by physical source path, and preserves all raw diagnostics. Narrow and wide results retain independent Frozen03 and Strict B references. No suppressions or candidate-derived baselines were added. The eight unaffected results and all 30 successful syntax checks were rebound; affected comparisons were rerun. Inherited diagnostics and frontend limitations remain documented rather than being described as zero total findings.

[Host results](C:/Ascend/turboquant-core/reports/deadlock_20261125/finalize_p6_20261126/FINAL_HOST_RESULTS.json) · [Final syntax/analyzer results](C:/Ascend/turboquant-core/reports/deadlock_20261125/finalize_p6_20261126/DEPLOYED_AKA_SYNTAX.json) · [Consumer correction and evidence](C:/Ascend/turboquant-core/reports/deadlock_20261125/finalize_p6_20261126/analyzer-hygiene/CONSUMER_HYGIENE_DELTA.json) · [Independent consumer review](C:/Ascend/turboquant-core/reports/deadlock_20261125/finalize_p6_20261126/analyzer-hygiene/FINAL_SOL62_CONSUMER_HYGIENE_REVIEW.json)

The prior discovery run also passed 905 distinct host pytest cases. These are separate from the final reruns above.

## ABI and deployment limits

The supported public C++/Python signatures, tensor metadata and strides, tiling layouts and cache wiring agree. The modern launch ABI remains 96 bytes; SIM envelopes remain 104 bytes. The SIM path retains its inherited SDK eight-byte scratch pair and adds no dispatch allocation or D2H copy. Modules use process lifetime and quiescent cleanup. Existing extreme standalone geometry limits are unchanged.

The verified E8 reader/dequantizer preserve the external 4,352-byte plane and internal 1,056-byte K stride. The unverified E8 encoder is excluded. Physical NPU execution and a full device-plugin build were not attempted. The unavailable remote incident ELF prevents attribution of its original `+0x2270` stall; no local canary reproduced a 900-second hang.

`setup.py` required no patch: its native CMake component/manifest installation already includes the existing target, and both compiler depfiles track the shared header. Its build-selection tests passed.

## Product files in these commits

Core:

- [cmake/NpuTargets.cmake](C:/Ascend/turboquant-core/cmake/NpuTargets.cmake)
- [cmake/SimTargets.cmake](C:/Ascend/turboquant-core/cmake/SimTargets.cmake)
- [src/host/turboquant_sim_binary_lifetime.cpp](C:/Ascend/turboquant-core/src/host/turboquant_sim_binary_lifetime.cpp)
- [src/turboquant/SOURCE_SYNC.json](C:/Ascend/turboquant-core/src/turboquant/SOURCE_SYNC.json)
- [src/turboquant/ingress/native/routine_reshape_batch_pipeline.hpp](C:/Ascend/turboquant-core/src/turboquant/ingress/native/routine_reshape_batch_pipeline.hpp)
- [src/turboquant/ingress/native/turboquant_reshape_batch.cpp](C:/Ascend/turboquant-core/src/turboquant/ingress/native/turboquant_reshape_batch.cpp)
- [src/turboquant/ingress/native/turboquant_reshape_batch_wide.cpp](C:/Ascend/turboquant-core/src/turboquant/ingress/native/turboquant_reshape_batch_wide.cpp)

vLLM:

- [CMakeLists.txt](C:/vllm-ascend/CMakeLists.txt)
- [csrc/attention/turboquant/README.md](C:/vllm-ascend/csrc/attention/turboquant/README.md)
- [csrc/attention/turboquant/SOURCE_SYNC.json](C:/vllm-ascend/csrc/attention/turboquant/SOURCE_SYNC.json)
- [csrc/attention/turboquant/ingress/native/routine_reshape_batch_pipeline.hpp](C:/vllm-ascend/csrc/attention/turboquant/ingress/native/routine_reshape_batch_pipeline.hpp)
- [csrc/attention/turboquant/ingress/native/turboquant_reshape_batch.cpp](C:/vllm-ascend/csrc/attention/turboquant/ingress/native/turboquant_reshape_batch.cpp)
- [csrc/attention/turboquant/ingress/native/turboquant_reshape_batch_wide.cpp](C:/vllm-ascend/csrc/attention/turboquant/ingress/native/turboquant_reshape_batch_wide.cpp)
- [tests/host/turboquant/check_analyzer.py](C:/vllm-ascend/tests/host/turboquant/check_analyzer.py)
- [tests/host/turboquant/check_kernels.py](C:/vllm-ascend/tests/host/turboquant/check_kernels.py)

Earlier session commits remain included: core G `d6c63d8`, vLLM G `9a0c30f8f` (source manifest, decode option header and Cube service); vLLM cache-spec `e19e9ecbe` (`vllm_ascend/core/kv_cache_interface.py` and `tests/ut/worker/test_turboquant_hybrid_alloc.py`). The original checkpoints are core `9822714` and vLLM `964190003`.

## Commands and receipts

The exact argv and logs are preserved in the linked receipts, including output directories and compiler flags. Final Python selectors were:

```powershell
python -B tests/ut/attention/run_turboquant_device_free.py -q
python -B -m pytest --noconftest -q -p no:cacheprovider tests/host/turboquant/test_source_sync.py tests/host/turboquant/test_build_selection.py
python scripts/check_turboquant_sync.py --core C:/Ascend/turboquant-core
```

The sealed `slots/P/r6/promotion/run_final_host.sh` executed `core`, `plugin-asserts` and `torch`; `run_final_host.ps1` ran the Windows suites. The initial deployed supplement completed CANN `-fsyntax-only -Werror`; [the analyzer-tail recipe](C:/Ascend/turboquant-core/reports/deadlock_20261125/finalize_p6_20261126/AKA_TAIL_ROOT_RECIPE.json) ran the corrected comparisons. [The release recipe](C:/Ascend/turboquant-core/reports/deadlock_20261125/slots/P/r6/promotion/USER_WAIVER_RELEASE.md) and [deployed syntax plan](C:/Ascend/turboquant-core/reports/deadlock_20261125/slots/P/r6/promotion/DEPLOYED_AKA_SYNTAX_PLAN.md) retain the full command vectors.
