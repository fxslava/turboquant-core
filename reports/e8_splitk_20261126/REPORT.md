# E8 Split-K implementation and local verification — 26 November 2026

This is a feature-branch implementation, not a verified main promotion. Local
host tests, CANN syntax and sim/NPU-target builds pass. The retained analyzer's
zero-new-diagnostics gate fails, and physical accuracy/timing remain uncollected.
No physical NPU or numerical CAModel workload was executed. The user explicitly
requested local WSL verification and a manual physical-server handoff.

## Changes

- Page-aligned splits 1–8, one MIX reader grid across sequence/KV-head/split,
  and query groups of six processed inside each block.
- FP32 partial accumulator/max/sum records and a separate AIV merge using the
  existing LM4 online-softmax routine. One split writes output directly without
  workspace or a reduction launch. Empty, one-tile and odd-tile slices are handled.
- Packet ABI, dequantization arithmetic and frozen oracle sources are unchanged.
  Expanded launch descriptors require rebuilding callers and reader together.
- Native timings come from ACL event timestamps, with a captured device-stream
  timeline by default and an explicitly labelled eager comparison. No host clock
  supplies device latency. Output uses `device_p50_us`, `device_p95_us`,
  `device_mean_us`, and median-based `device_effective_GB_s`.
- Witness GM writes are disabled for benchmarks. Explicit V-pipe fences preserve
  accumulator dependencies. E8 accuracy now requires cosine at least 0.99999.
- Split override, profiling/export script, strict measurement parser, CPU merge
  test, ABI checks, and build/test/performance guides are included.

The previous benchmark already used ACL events. The additional capture path
queues the full sample sequence before replay to avoid CPU submission gaps in
the device timeline. An event span still includes device scheduling and the
reader/reducer gap; msprof task durations supply the separate kernel view.

## Local gates

| Gate | Result |
| --- | --- |
| CPU CMake/CTest with generated fixtures | PASS: 11/11 |
| CLI / profiler parser / metric controls | PASS: 6 + 4 + 3 tests |
| Guarded device-free attention suite | PASS: 623 tests |
| Shared source build/sync suite | PASS: 14 tests |
| Shared byte/comment parity | PASS: 36 C/C++ files plus `reader_layout.aka` |
| CANN 9.2.1 dav-3510 strict syntax | PASS: core and plugin, host and device |
| Rebuilt CPU Torch bindings | PASS: five syntax units, link, Meta/FakeTensor/fullgraph; zero device launches |
| `build-sim`: `tq_sim_tests`, `tq_benchmarks` | PASS: build/link; 24 E8 decode catalog rows READY with split override 8 |
| `build-npu`: `tq_npu_tests`, `tq_benchmarks` | PASS: compile/link and catalog only |
| E8-disabled configuration | PASS: production runners build; E8 rows explicitly UNSUPPORTED |
| Python/bash syntax and Git whitespace | PASS |
| Plugin `format.sh ci` | UNAVAILABLE: pre-commit is not installed |
| Numerical CAModel / physical accuracy | NOT RUN |
| ACL event samples / msprof hardware counters | NOT COLLECTED |

`BUILD_ARTIFACTS.json` records build configurations, binary hashes and relevant
dynamic symbols, including the reader, reducer and ACL timing/capture APIs.
Parser unit inputs are explicitly synthetic fixtures, never hardware evidence.
The user prohibited remote SSH from WSL; no remote connection was made after
that clarification. A prior Windows probe to the configured mock SSH entry was
refused and supplied no hardware execution or evidence.

## Analyzer blocker

Pinned analyzer commit: `46f16829ee37b4a31961832595e96b280d7f053e`.
Fresh independent-reference comparisons pass for the eight unchanged ordinary
cases, then fail for E8: **12 new diagnostic occurrences with annotations and
17 without annotations**. The wider historical gate stops at that failure;
its later checks are not represented as passing.

The new findings include `AKA1004` classifications of `Reg::Max` register
operands as UB-only operations and confusion between the `output`/`offsets`
members of `VectorBuffers` and similarly named GM entry arguments. For example,
the reported GM-to-GM operations are `DataCopy(b.offsets, map, ...)` and
`DataCopy(destination, b.output, ...)`; `BindBuffers` maps those members to UB
through the typed `Ub<T>` helper. New reducer local `acc` operations also inherit
REG classifications from other functions. The analyzer's API lookup falls back
to an intrinsic's final unqualified name, and include-expanded diagnostics can
point to the wrapper include line instead of the underlying source.

These are evidence for inference limitations, not a waiver or a passing gate.
Raw fingerprints, multiplicities, summaries, frontend coverage and diagnostic
details are retained in `ANALYZER_SUMMARY.json` and
`ANALYZER_DIAGNOSTICS.json`; full raw reports are hash-indexed in
`RAW_EVIDENCE.json`. No analyzer source, baseline, suppression or severity was
changed to obtain a result. Resolve this blocker before main promotion.

## Physical-server handoff

Use [the exact build, accuracy and profiling commands](../../docs/E8_SPLITK.md)
inside `vllm_ascend_v030_dev` at `/workspace/turboquant-core`. The scripted
collection is `bash scripts/profile_e8_decode.sh 0 build-npu results/e8-splitk-run1`.
It covers split8, deepseek-v4-flash-64k and qwen35-gqa6 at split 8 with blocks
64/128, requires passing accuracy/guards, preserves every event sample, and
exports reader/reducer task durations, Block Dim and PipeUtilization counters.
Use a fresh result directory. Also compare qwen splits 1 and 8 at fixed geometry.

No speedup, saturation, numerical device parity or physical utilization claim
is established here. The receipt deliberately stores physical measurements as
null. Both main refs and `codex/swa-sink` are preserved; the synchronized plugin
changes live on a companion feature branch. Historical reports remain intact.
