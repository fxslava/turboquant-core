# E8 integration validation — 26 November 2026

This report records local host checks and compile-only device preparation for
Ascend950PR_9599 / dav-3510 with CANN 9.2.1. Physical NPU execution is excluded.

## Packet and promotion contract

For D=256, each 64-token K or V plane occupies 4,352 bytes: 4,096 bytes
of uint16 lattice labels followed by 64 FP32 row scales. The K+V allocation is
8,704 bytes per head tile. Device dequantization has zero external scratch;
the sealed reader retains its internal 1,056-byte K-channel stride.
The reader and SIMT dequantizer bodies remain unchanged. Their archived
287,101-tick and 14,252-tick results are historical, not fresh measurements;
`kernel/HISTORICAL_E8.json` pins the source equivalence and archived results.

Device encoder promotion requires fresh independent review and passing
16-row and 32-row CAModel canaries against CPU labels and scales. A timeout
or failed numerical gate leaves device ingress disabled. Prepared packets,
dequantization and decode remain separate from that promotion decision.

## Initial host verification

- The source inventory and byte/hash synchronization check passed.
- The guarded device-free attention suite passed 606 tests on
  Python 3.14 / Torch 2.14.0+cpu without importing the real `torch_npu`.
- The standalone CMake CPU build passed all nine CTest tests on WSL.
- The plugin C++ contract suite passed seven tests.
- The source-sync/build-selection suite passed 14 tests after repairing its
  isolated AST loader to include the `CMakeExtension` declaration used by
  `cmake_build_ext` annotations. The initial failure is retained in
  `baseline-host14.log`.

`format.sh ci` could not run in this environment: the checked-out shell script
uses CRLF line endings, and a normalized invocation reports that `pre-commit`
is not installed. Targeted Ruff checks are recorded separately.

## Expanded host verification

The final guarded attention run passed **623/623**, retaining all **606**
baseline test identities. The final sync/build-selection run passed **14/14**.
`HOST_VALIDATION.json` pins both JUnit receipts, the seven C++ contract tests,
the synchronized source inventory and changed tracked source hashes.
The post-format E8-focused run passed **32/32**. The source review's runner-V2
and sequence-parallel dummy-row defects have regression coverage: V2 rejects
E8 before allocation, and E8 metadata requires table/context rows only for
requests with live queries while still rejecting missing live metadata.
The fresh final source/host integration review passed and independently reran
all 623 guarded tests (`reviews/integration-final.txt`).

The new production host target brings CTest to **10/10**. Its matrix emits
216 validated rows covering 12 workload profiles, two block sizes, three
formats and three stages. The staged reference comparison passed **38**
dequant/decode rows: 36 for H=1, GQA=6 and Split-K=4 with both block sizes,
plus a ragged 131,072-token E8 case and a 32-sequence Lloyd-Max 3-bit case
with Split-K=16. Both additional cases had maximum BF16 ULP distance 1.
Three zero-vector/non-finite positive and negative metric controls also passed.
Its 18 ingress rows explicitly report unsupported because that reference
starts with packed cache data. Native-supported profile limits are emitted
by the host catalog, which identifies 78 native-supported rows; geometry
enumeration is not physical NPU validation.

The CPU E8 encoder matched the retained r3 oracle's UINT16 labels and FP32
scale bits for the first 16 and 32 rows, with zero mismatches. The fixture
and reference source hashes and input-basis conversion are recorded in
`python-committed/encoder-reference.json`. This is CPU parity, not device promotion.

The isolated real-Torch CPU binding check passed five strict syntax units,
a host-only link with throwing device seams, and Meta/FakeTensor/Dynamo
fullgraph packet dequantization. It blocks real device-module imports.
Expected Meta errors in `bindings-final.log` are negative controls proving
that the native writer rejects E8; the final JSON reports PASS and zero launches.

All 30 CANN 9.2.1 host/device syntax checks passed (plugin NZ 12, core NZ 12,
core strip 6). The final writer audit widened runtime GM products before
multiplication in both batch and scatter paths without changing their geometry
guards or layouts. Sixteen analyzer comparisons against independent sources
reported zero new diagnostics. Existing raw analyzer diagnostics and frontend
coverage limits remain visible in `kernel/ANALYZER_SUMMARY.json`; this is not
a claim of zero total diagnostics. The source and staged Git inventories
match for all 36 shared C++ files and the AKA sidecar.

The sealed encoder supervisor was admitted after fresh cross-review and final
review. Runtime completion and numerical comparisons are a separate gate;
the canary outcome section below records the actual results.

## Existing configuration and prepared packets

E8 uses `ENABLE_TURBOQUANT=1` and
`VLLM_ASCEND_TURBOQUANT_QUANT_MODE=2`. Select the registered reader with
`TURBOQUANT_KERNEL_STAGES=decode`; `none` retains portable decoding. These
are existing settings. Keep the existing `--kv-cache-dtype auto` default;
`turboquant_4bit_nc` belongs to the separate latent-cache implementation.
The E8 packet layout requires D=256 and BF16 attention, with block size 64
or 128. Each block contains one or two 64-token packets per KV head in each
of two planes.
The executable AST of `envs.py` is unchanged from the baseline: only its
existing quantization-mode documentation changed. No environment variable
declaration or production CLI flag was added.

The serving integration supports model runner V1. Model runner V2 is rejected
during E8 platform configuration because it has no E8 packet allocation or
CPU block-table metadata path. The standalone ACL benchmark is independent
of either serving runner.

`TurboQuantOpsAdapter.load_e8_packets` loads CPU-prepared K/V packet planes
outside replay. The backend can decode those packets without raw K/V input.
This prepared-cache interface does not enable an unverified raw NPU writer.

## Frozen source and concurrent work

The reviewed plugin snapshot is signed-off commit
`9d4f1d3b33726adef41473ff12fcfdc6b6b0afea`, tree
`ae7438ad07e80245597f166420959ff5edfdd1a4`, anchored at
`codex/e8-prepared-integration-20261126`. Concurrent hybrid-cache edits in the
live plugin checkout were preserved and excluded from this commit.
All 4,598 regular archive files matched the committed bytes. The 623-test,
32-test and 14-test gates and the retained encoder oracle were rerun against
that content-only snapshot (`python-committed/`).

The core source tree was frozen as
`f8ba371e056f8eceea590eac8d335c0592cf5ff3` through a private Git index.
Its toolkit-free snapshot passed 10 CTests, 216 matrix rows and the 78-row
native catalog. The snapshot-to-snapshot sync check passed
(`validation-frozen/`). `FROZEN_SOURCES.json` and `HOST_VALIDATION.json`
bind these inputs without reading unrelated live Python edits.

Raw gate receipts retain their exact bytes, including original line endings.
Task-scoped Git attributes prevent conversion of hashed evidence. Generated
compiler logs and reviewer text retain their original whitespace.

## Encoder review and admission

Fresh-context Sol 6.2 reviews rejected earlier runner/supervisor bundles for
resource-header setup, float backing storage, untouched shortlist counts,
compiled-region coverage, timeout cleanup, process identity and cancellation
handling. Those bundles were never executed in CAModel. Their immutable
verdicts and CPU-only negative controls remain under `reviews/` and `kernel/`.

The final v11 supervisor uses a native timer owner and rootless private PID
namespace. PIDFD identities, backend path/inode/hash verification and scoped
kernel teardown keep case processes and locks owned through retirement.
The approved launch uses the phase guard's exec path and `flock --no-fork`.
CPU controls cover TERM-resistant and detached descendants, interruption,
namespace refusal, delayed discovery and timeout status preservation.

The first stop event observed by the supervisor is sticky; the independent
absolute cutoff remains enforced. This does not promise sender chronology
for pending standard signals, whose delivery order is unspecified by
[Linux signal(7)](https://man7.org/linux/man-pages/man7/signal.7.html).

The unchanged compiled encoder retained fresh static review A4. The sealed
v11 bundle passed fresh cross-review B11 and fresh final review 11, admitting
only the pinned serialized command in `kernel/V11_CANARY_LAUNCH.json`.
Static admission establishes no numerical parity; actual results follow.

## Encoder canary outcomes

The admitted v11 packet is
`99866169988a2800ce1bb2722dd97be9763aa19e2536c9443d688141e97976a8`.
Each case uses a configured 900-second CLOCK_MONOTONIC deadline and a
15-second termination grace, with the global simulation lane and build/model
exclusion locks held until namespace retirement. No physical NPU workload
was launched.

The 16-row case returned status **124** before its first component comparison
completed. Its host wrapper measured **956.888 seconds**; the model exit
reported **306,432 partial ticks**, with last completed retirement tick
306,431. Label mismatch, scale mismatch and packet/workspace guard results
are **unknown**, represented by null in the receipt. They are not zero.
The case namespace was empty and all three runtime locks were available
after retirement (`kernel/canary/rows-16-receipt.json`).

A follow-up clock probe found WSL clock domains diverging: a nominal 12-second
monotonic interval measured 12.008 seconds in CLOCK_MONOTONIC, 12.413 in
CLOCK_MONOTONIC_RAW and 13.896 in WSL realtime; the Windows invocation measured
12.563 seconds. The configured deadline therefore did not enforce a strict
900-second host-wall limit. The observed overrun is retained explicitly in
the receipts rather than treated as a successful budget gate.

Both requested compilation and simulator alias use Ascend950PR_9599 / dav-3510.
The alias resolves to the pinned dav_3510 PEM directory, whose loaded path,
inode and SHA-256 were verified. The runtime driver stub identifies itself
as **Ascend950PR_9589**; this is retained separately from the requested target.
It is not evidence of execution on a physical 9599 device.

Both admitted cases returned status **124** without a completed comparison:

| Rows | Host wall seconds | Partial model stop ticks | Completed comparisons |
| --- | ---: | ---: | ---: |
| 16 | 956.888 | 306,432 | 0 |
| 32 | 957.270 | 298,486 | 0 |

For 32 rows, the full launch/retirement bracket measured 904.426 seconds on
WSL CLOCK_MONOTONIC, 957.460 on CLOCK_MONOTONIC_RAW, 958.476 on WSL realtime
and 957.260 on the Windows stopwatch. Its last complete retirement trace tick
was 298,474. Its namespace was empty and all three runtime locks were released.
Both receipts retain null numerical/guard results and partial ticks, not
completed encoder latency or zero-mismatch claims. `kernel/CANARY_RESULTS.json`
binds the final receipts; their raw run-evidence hashes were rechecked.

Device E8 writes remain gated. Neither probe established device numerical
parity or met a strict 900-second host-wall budget. The prepared 4,352-byte
packet paths, registered reader and packet dequantizer remain enabled through
the existing settings; no encoder source was promoted into ingress.

## Compiled server package

`build-e8-production/ascend950pr-cann921-x86_64.tar.gz` contains 11 physical
Linux x86-64 executables, seven matching kernel DSOs, one host profile catalog,
the staged benchmark script and an internal `SHA256SUMS` file.
Its SHA-256 is
`dba6a5833fe17ae50d59691b1146596afd65e4b84de1bc1fcfd772aac8ed4e77`.
The complete CANN physical target built successfully, followed by a hash-checked
refresh of all four writer libraries and relinking of their dependent runners.
The extracted file hashes and origin-relative DSO dependencies were verified.
Only the extracted CPU catalog was executed locally.

Native LM3 kernels and native E8 Split-K are unavailable. Their rows report
unsupported; the staged PyTorch benchmark compares LM4, LM3 and E8 through
unpack, gather, attention and Split-K merge on the server. LM4 and LM3 use
128-byte and 96-byte row payloads plus scales; E8 uses 4,352 bytes per 64-row
packet. Event timing
includes warmup and p50/p95/mean microseconds. Effective GB/s uses declared
logical bytes rather than measured HBM transactions. No local NPU performance
numbers are reported.

See `../../benchmarks/README.md` for full build, extraction, checksum and
`msprof` instructions. After extracting the archive into `build-e8-production`
and sourcing the server's CANN 9.2.1 environment, run from its parent directory:

```bash
./build-e8-production/tests/test_turboquant_production_npu --device 0
./build-e8-production/tests/bench_turboquant_production_npu --device 0 \
  --mode all --stage decode --profile qwen35-gqa6 --warmup 10 --iterations 50
python3 build-e8-production/benchmarks/benchmark_turboquant_production.py \
  --backend torch-npu --build-dir build-e8-production --device 0 \
  --mode all --stage decode --profile qwen35-gqa6 --verify
python3 build-e8-production/benchmarks/benchmark_turboquant_production.py \
  --backend torch-npu --build-dir build-e8-production --device 0 \
  --mode all --stage decode --profile qwen35-gqa6 --warmup 10 --iterations 50
```
