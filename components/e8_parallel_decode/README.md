# Parallel E8 decoder component

Eight-coordinate reconstruction uses explicit registers, depth-three integer
parity/argmax trees and masked repairs. The public BF16/INT8 API shims retain
Arithmetic/CompactInt8, all four compact table layouts and both shaping modes.
Their only loop distributes independent E8 blocks. The separate fused reader
uses the same helper and has no SIMT branch/backedge.

The device source files are byte-identical to the isolated strict CANN9.2.1
build. `evidence/` includes actual ELF/resource/ISA certificates and a sealed
native artifact manifest. The native build recipe is retained separately as
`native_probe.CMakeLists.txt`. The portable component changes only a host helper
include from an absolute path to a local byte-identical copy.

CPU gates need C++17 and no toolkit:

```
cmake -S . -B build-cpu -DE8_PARALLEL_BUILD_DEVICE=OFF
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
```

All four gates pass: all65,536 labels,24 scales (12,582,912 BF16 and INT8 outputs
each), all four compact layouts, and both fused-reader host fixtures. Frozen
originals under `src/frozen` are CPU-only oracles, not the active GPU closure.

The bounded CAModel component passes4,096 BF16 words at0ULP and4,096 INT8 bytes
exactly, input preservation and512guard bytes. Signed FP32 max-subnormal,
min-normal and2^-132 scales pass. Total14,856 ticks covers both formats; it is
not a fused-attention timing result. No physical NPU result is claimed.

Uniform six-tile encoded-scatter→reader also passes:1,536 BF16 attention words
at0ULP,196,608 BF16 K/V words and2,304 FP32 QK words exact; all packet/input
guards/inactive output intact.47 cross waits and12 FP32 MMADs pass trace audit.
Total336,288ticks includes two scatter launches; reader frame330,879ticks.
Nonuniform and raw-writer composition remain separate pending gates.

Optional CANN libraries are off by default. Setting
`E8_PARALLEL_BUILD_DEVICE=ON` declares `e8_parallel_dequant` and `e8_fused_decode`
with strict flags. The latter is a cache-basis compatibility entry: pre-rotated
FP32 query in and BF16 cache-basis context out. It does not rotate a raw query
or inverse-transform output. A surrounding production adapter remains separate.

Actual runtime adapters check compiled SIMT requirements before submission:
12private bytes/thread for the component,8for the reader, zero external
divergence requirement. The native standalone runner, inputs, canaries and
canonical PEM identity are bound in the evidence; retain phase isolation when
rebuilding or simulating. Registered private-frame requirements are runtime
checked, not assumed from an undocumented default.

`ACTIVE_CLOSURE.md` explains why the verified Wave-v11
production/native probe is still a KV4 codec path and was not edited as E8.


Three fresh Sol 6.2 reviews passed the unchanged source/ISA packet. Their sealed
72-file packet, verdicts and admission gate are retained under `evidence/`.
No reviewer mutation ideas were implemented. `evidence/SCALAR_RUNTIME.md`
classifies the existing uniform trace and states the partial-baseline limits.

The recorded AKA gate has an explicit frontend coverage gap: component
`Dequant`/`DequantVf` and reader `DecodePacked` template calls were opaque to
that analyzer version. Zero accepted diagnostics do not prove those bodies.
Their active source and compiled ISA were independently audited in all three
fresh reviews. `evidence/sealed_review_packet/candidate/aka/frontend_coverage.json`
preserves this limitation. Reader raw diagnostics are 21 fatal/2 warnings;
the source-bound provenance/API fingerprint handling yields 0 fatal/0 warnings,
with inherited generic-event equivalence and negative controls retained.

Packaging preserves device source bytes. Git disables text conversion inside
this component. The local host include and portable CMake entry are the only
build-path adaptations. Optional device targets are a rebuild recipe; supplied
evidence describes the hashed original native objects, not a newly built ELF.
The package contains no unverified raw encoder implementation. Pending raw
composition paths and scope are documented in `NEXT_STEPS.md`.
