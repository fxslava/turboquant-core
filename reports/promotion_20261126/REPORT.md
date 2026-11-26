# Product audit and promotion, 26 November 2026

The requested product snapshots pass the host, strict syntax and independent-reference
analyzer gates below. Core promotion preserves its existing main history. Plugin
promotion is withheld because current upstream main removed the model runner required
by the verified serving implementation. No physical NPU or CAModel workload was run.

## History and staging audit

The complete range `b92cba0714f5b4ca11114ab70f7914afd0137a10..c50b7582571921ba96ee00001f81ffe27858aaa0`
contains seven commits and changes 259 paths. `AUDIT.json` records every commit and path.
The seven changes integrate the champions, categorize shared sources, checkpoint the
deadlock baseline, repair native NZ staging, restore Frozen03 with the wide fallback,
record the P6 waiver, and seal E8 packet/server validation.

`e29a9ebdcbd5` is not an ancestor of the requested product tip. Its 103-file
`components/e8_packet_writers` package belongs to the separate packet-writer branch;
there are no files from that directory in the requested product tree. Its writers
copy pre-encoded labels/scales and are not raw BF16 encoders. Their package evidence
must not be interpreted as promotion of raw device E8 ingress.

`47abf6526259` already migrated `components/e8_parallel_decode` into
`src/turboquant/common`, `src/turboquant/decode/e8`, and `tests/cpu/e8`.
The two empty directories left on the core checkout were pruned locally. Git tracks
neither empty directories nor any remaining `components` file, so a deletion commit
for that path would be artificial. Historical provenance paths inside SOURCE_SYNC.json
are retained intentionally.

`tests/npu/test_wave_champions.cpp`, `tests/npu/test_e8_packet.cpp`, and
`tests/npu/e8_packet.cpp` do not exist at the requested tip. The actual NPU runners
and all CMake target source paths were inspected. Both sim and NPU configurations
with device targets enabled generate successfully; no duplicate target is registered.
Strict host syntax checks cover 15 runner variants. The retained main finite-spike
dirty-tail regression was merged, its canonical include remains valid, and its runner
was syntax-checked again after the merge. No device target was executed.

There is one exact duplicate source pair in the core product: the rotation helper
under `decode/native/cube` and `ingress/native`. Both are actively included and
explicitly listed in the two synchronized source inventories. They are intentional
category-local dependencies, not staging leftovers. The E8 frozen files are independent
CPU oracle inputs, also retained. Unrelated upstream duplicate groups in the plugin
are inventoried, not pruned as part of a TurboQuant audit.

The plugin's unreferenced `tmp/turboquant_third_page_diagnostics.patch`,
`tools/nb_audit.py`, and root `ablation_results.json` were moved unchanged into
`reports/turboquant_archive_20261126` by dedicated cleanup commit
`57c7e68b3d4d65dc0743b6470b005f38cdea558d` on `codex/product-promotion-20261126`.
All three renames retain identical Git blobs. The capture probe remains because the
graph-capture RFC explicitly references it. No existing report was deleted.

## Verification evidence inventory

| Core directory | Files on disk | Tracked at product tip | Role |
| --- | ---: | ---: | --- |
| `reports/checkpoint1_20261123_i1` | 1 | 0 | Historical verification evidence |
| `reports/deadlock_20261125` | 2,542 | 37 | Candidate snapshots, reviews, model/host receipts and audit tools |
| `reports/wave_champions_20261125` | 5 | 5 | Champion provenance and verification summaries |
| `reports/e8_final_20261126` | 127 | 127 | Final host, compiler, analyzer, canary and packaging receipts |

There is no `reports/checkpoint2_*` directory in the requested core tree. Checkpoint2
provenance references external harness/WSL evidence. Plugin evidence consists of one
deadlock baseline file, ten host-refactor files, and five champion files, plus the new
archive. These directories are evidence, not CMake product inputs. Scripts and candidate
source copies inside reports are retained for reproducibility. Pre-existing untracked
reports were preserved and were not swept into the commit.

## Gates

| Check | Result |
| --- | --- |
| Source inventory, hashes, comments and byte parity | PASS: 36 shared C/C++ files and `reader_layout.aka` |
| Guarded device-free attention suite | PASS: 623 tests against the isolated requested plugin checkout |
| Sync and build-selection suite | PASS: 14 tests |
| Plugin toolkit-free C++ contracts | PASS: 7 tests |
| Core CMake CPU build and CTest | PASS: 10 tests with the complete generated fixture root |
| Real CPU Torch binding syntax/link/Meta/FakeTensor/fullgraph | PASS: 5 syntax units, throwing device seams, zero device launches |
| CANN 9.2.1 dav-3510 host/device syntax | PASS: plugin NZ 12, core NZ 12, core strip 6 |
| Core runner syntax | PASS: 15 variants, plus merged dirty-tail runner |
| sim/NPU target configuration with E8 enabled | PASS: both generate; no kernel build/run requested |
| Independent-reference analyzer comparisons | PASS: 12 ordinary and 4 explicit-device-macro batch pairs, zero new diagnostics |
| Pinned Python semantic goldens | PASS: Torch 2.10 CPU / NumPy 1.26.4 container |
| Benchmark accuracy metric controls | PASS: 3 tests |

The analyzer gate measures diagnostic deltas, not zero total diagnostics. Existing
fatal/warning counts and frontend coverage remain visible in `ANALYZER_SUMMARY.json`.
The pinned analyzer is `46f16829ee37b4a31961832595e96b280d7f053e`.

The initial default-root CTest run found six missing generated fixture directories.
The complete CPU generator was run into the audit workspace and selected through
`TQ_GOLDEN_DUMP_ROOT`; the full CTest suite then passed. README now documents this
fresh-checkout prerequisite. No tracked fixture was overwritten. A preliminary
bit-exact semantic-fixture check under Windows Torch 2.14 differed in an expected
attention output; the established pinned Torch 2.10 environment reproduces every
tracked fixture byte. Both receipts remain in raw evidence. This does not establish
cross-version bit identity.

Raw device E8 encoding remains disabled. Its historical 16/32-row canaries timed out
without completed numerical comparisons; no success was inferred from those receipts.
Native LM3 and native E8 Split-K remain unsupported as already documented.

## Promotion policy and remaining blocker

The user selected preserving existing upstream plugin main and porting TurboQuant.
A whole-branch merge has 432 conflicts. Isolating the actual product delta from its
clean release base `c09d3e20b9c45c80b34ae3781440d90904a44023` reduces this to nine:
CMake, cache-spec imports/registration, environment configuration, worker/zeroer
integration, and affected tests. The old runner and zeroer test have been deleted
upstream, rather than merely moved.

Upstream commit `c3c48aa994dd9af8b3f4daaea6a333294af95b3b` removed Model Runner V1.
`main:vllm_ascend/mrv2_utils.py` now always selects V2 and rejects explicit V1 requests.
The verified product routes allocation, cache carving and host block-table metadata
through `worker/model_runner_v1.py`; its E8 platform gate explicitly rejects V2.
Passing tests for the old product snapshot therefore cannot certify a main-compatible
serving port. Removing that guard or restoring obsolete upstream code would not be
a verified promotion.

The plugin main ref is preserved at `a1df499e0762621c2b9561e2ee66c498a70d0328`.
`codex/swa-sink` and its checkout are preserved at
`ec6771ff099b8d07f23e6bd581e3b23c12f78ab6`. The cleanup branch retains the verified
product bytes. A complete product-only patch and the precise conflict list are
retained under `C:/Ascend/turboquant-integration-work/promotion-20261126`.
Completing plugin promotion requires a functional V2 cache-allocation/metadata and
ownership port, with corresponding tests; that port is not represented as complete.

Core main promotion retains both the requested product and the prior main finite-spike
tail regression. Final main refs and the fresh main CTest/sync results are recorded
in the adjacent final receipt. Nothing was pushed to a remote.

`VERIFIED_SOURCES.json` pins the shared product bytes. `RAW_EVIDENCE.json` records
the external raw logs and their hashes; historical reports remain in place.
