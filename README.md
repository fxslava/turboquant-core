# turboquant-core

Standalone repository for the audited TurboQuant production kernels, their runtime integration
layout, and the dedicated unit / hardware test fixtures.

Everything here is extracted **exclusively** from
`ascend-micro-harness/kernels/ops/turboquant/vllm_integration_layer`. No source is taken from
`vllm-ascend/csrc`, from the harness root (`include/`, `tests/`, `routines/`), or from the
`baseline/` and `fused_decode/` prototype trees.

Target: Ascend 950PR / arch35 / dav-3510, CANN 9.2.0-beta.2, `__NPU_ARCH__ == 3510`.

## Build and test

```bash
cmake -B build -DRUN_MODE=sim
cmake --build build -j16
```

```bash
./build/tests/test_contract_gates
```

```bash
./build/tests/test_layout_bridge
```

`ctest --test-dir build` runs both. They are toolkit-free — the layout contract, the host tiling
and the kernels' host-visible geometry and nothing else — so a clean configure plus build plus
`ctest` is about five seconds with any C++17 compiler, and needs no device.

The device legs are **off by default**, behind `TQ_BUILD_KERNELS`, because one `ascendc_library`
expands into nested ExternalProjects and compiles the whole translation unit five to seven times,
every one of them `BUILD_ALWAYS`. The option is load-bearing and not a convenience:
`ascendc_library` does **not** honour `EXCLUDE_FROM_ALL`, so a configure that merely declared the
kernels put the entire BiSheng build back into the default target.

```bash
cmake -B build -DRUN_MODE=sim -DTQ_BUILD_KERNELS=ON && cmake --build build --target tq_sim_kernels -j16
```

```bash
cmake -B build-npu -DRUN_MODE=npu -DTQ_BUILD_KERNELS=ON && cmake --build build-npu --target tq_npu_kernels -j16
```

`RUN_MODE=sim` compiles the three operators for CAModel and links the runners against
`libnpu_drv_camodel.so` / `libmodel_top.so`; `RUN_MODE=npu` compiles them with BiSheng for the
physical dav-3510 target and links `ascendcl`. Both legs compile the identical kernel sources
with the identical definitions, deliberately: CAModel evidence is only about the device binary if
they do. Every one of our own translation units is built with `-std=c++17 -Wall -Wextra -Werror`;
CANN's own headers are reclassified as system headers so they cannot spend that budget.

If the CANN toolkit is absent, configure still succeeds and says the device leg was skipped.

## Layout

```
include/turboquant/      normative contracts, no kernel toolchain required
  turboquant_layout.h            RFC 3.1 strip layout + the NZ-tiled cache, address functions
  turboquant_common.h            kernel-side pipe/event/broadcast helpers
  turboquant_wave_protocol.hpp   strip boundary, mode-2 channel map, UB/L1/L0 arena algebra
  turboquant_mode.h              quantisation mode contract and codebooks
  turboquant_types.h             AscendType
  turboquant_fused_decode_opt.hpp  decode launch ABI, UB map, flag budget
  turboquant_gemm_reduce.hpp       Cube GEMM reduction contract
  turboquant_splitk_contract.hpp   Split-K partial layout
src/
  prefill/turboquant_reshape_batch.cpp     MIX_AIC_1_2 BF16 prefill, mode-2 only
  prefill/routine_stage_page_zn_bf16.hpp
  scatter/turboquant_reshape_scatter.cpp   AIV-only single-token / sub-strip RMW writer
  scatter/routine_vector_rotate_butterfly.hpp
  decode/turboquant_fused_decode.cpp       three-zone modulo-scheduled fused decode
  decode/cube/{turboquant_cube_service,turboquant_gemm_reduce_service,turboquant_query_basis}.h
  decode/vector/{turboquant_vector_service.h,routine_vector_cube_softmax.hpp,
                 routine_vector_online_softmax.hpp}
  common/{turboquant_codec_950.h,turboquant_codec_mx.h,routine_turboquant_dequant4.hpp}
  host/turboquant_tiling.{h,cpp}           the decode's host tiling engine
tests/
  common/turboquant_test_common.hpp    failure harness, bit helpers, guarded buffers,
                                       the RFC 3.1 index maps
  common/turboquant_golden_oracle.hpp  reference integer quantiser, power-of-two scale
                                       extractor, FP64 attention reference
  common/turboquant_scatter_oracle.hpp the scatter writer's golden cache image
  common/rotate_butterfly_oracle.hpp   host model of the vector butterfly
  common/fused_decode_fixture.hpp      the decode's gold model, `expected` in double
  common/decode_tiling_gate.hpp        the host tiling equivalence gate
  common/camodel_session.hpp           device session and guarded GM allocations
  cpu/test_contract_gates.cpp          UB arena, host tiling, launch guards, mode-2 channels,
                                       marked-graph token balance, shared recurrence
  cpu/test_layout_bridge.cpp           NZ writer == decode reader, bijections, co-byte pairings
  npu/test_live_write.cpp              batch prefill + scatter RMW against the oracle
  npu/test_fused_decode_e2e.cpp        end-to-end decode against the FP64 reference
```

## What the extraction changed

**Prefill (`src/prefill/turboquant_reshape_batch.cpp`)**

* **Mode-2 only.** No `template <bool Mode2>`, no `bool Reuse`, no single-shot fallback entry
  point, no `reuse_mode4_flags` launcher argument. Cross-core synchronisation is the three
  mode-2 channels `kSync2PanelReady = 8`, `kSync2PanelFree = 9`, `kSync2PageDrained = 10`, each
  behind its `HardEvent::x_S` drain, single-sourced in `turboquant_wave_protocol.hpp`.
* **Seamless chunk pipeline.** The outer strip loop is gone. A block owns `tokensPerCore` tokens
  = `strips = tokensPerCore / 16` strips and runs them as one chunk stream of
  `total_chunks = strips * kChunks` columns. Column *j* stages chunk *j* and multiplies chunk
  *j − kIngressLookahead*, and **both indices cross strip boundaries**: the prologue ingress of
  strip *s + 1* is issued inside the Mmads of strip *s*, so MTE2 does not drain at the end of a
  16-token strip. The last strip is peeled (`StageNext` / `QuantNext` template argument) so no
  steady-state step carries a runtime predicate. The ring periods all divide `kChunks`
  (static_asserted), so a strip boundary never shifts which slot a chunk owns.
* **In-page strip offsets.** One scalar `slotMapping` read per strip gives
  `block_idx = slot0 / block_size` and `strip_bytes = ((slot0 % block_size) / 16) * kPackedBytes`;
  `StoreChunk` writes
  `(block_idx * num_kv_heads + kChunkHead<Chunk>) * head_bytes + strip_bytes`.
* The 100+ line essay header, the mode-4 flag-id retrospective, the CAModel trace diagnosis
  narrative and the sparse-sampling probe (`ASCEND_MICRO_SPARSE_SAMPLING`) are removed.

**Scatter (`src/scatter/turboquant_reshape_scatter.cpp`)**

* The verified RMW merge (`scatter_merge_token_vf`) and the in-register butterfly
  (`RotateImpl::RegGather`) are retained unchanged.
* The synthetic `launchProbe` buffer is removed: the parameter, its UB arena slot and its
  publication block are gone from the kernel, the `__global__` entry and the launcher.
* Narrative comments and the butterfly ablation table are removed.

**Fused decode (`src/decode/turboquant_fused_decode.cpp`)**

* The audited 3-zone modulo schedule, the Cube GEMM reduction
  (`ASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1`, on by default) and both the native FP16 and BF16
  instances are retained, along with the raw-query variants.
* `GM_ADDR` launch-site casts go through `AsGmAddr()` (`static_cast<uint8_t*>`), the one spelling
  that compiles under both the type-checked chevron and the generated `aclrtlaunch_*` stub.
* Host tiling moved to `src/host/turboquant_tiling.{h,cpp}`: the descriptors and dispatch records
  are declared in the header and `build_decode_tiling` is defined in a translation unit, rather
  than being an `inline` function in a device-side header.

## What is verified, and what is not

**Verified here, on this host, with no device:**

* `test_layout_bridge` — the NZ ingress address function and an independently re-derived decode
  reader name the same byte and the same nibble for every logical element, over four geometries;
  both the NZ and the strip maps are bijections onto their allocations; the co-byte pairings of
  the two contracts are structurally incompatible; both scale planes are bijections.
* `test_contract_gates` — the fused decode's UB arena `static_assert`s; the host tiling's
  dispatch odometer and descriptors against the arithmetic they replace over thirteen geometries
  plus five refusals; seven launch-guard admissions and refusals; the mode-2 channel map; the
  shared online-softmax phase order; and a **marked-graph token-balance audit of the seamless
  chunk pipeline** over 1, 2, 3, 4, 8, 17 and 64 strips — every ring stays inside its depth,
  every primed token reaches a terminal sink, and a 64-strip share costs exactly 64× the chunk
  work of a 1-strip share with **no extra prologue, drain or rendezvous per strip**, which is the
  seamless property stated as a count.

**Partly verified — compiled, never executed:**

* The **prefill** and **scatter** kernels build through BiSheng for CAModel
  (`RUN_MODE=sim`): `libtq_prefill_sim.so` and `libtq_scatter_sim.so` link. That exercises
  `-Wall -Werror` on the device legs and every `static_assert` in their device translation units
  — the UB and L1 arena bounds, the bank-skew strides, the ring-period divisibility and the
  mode-2 id budget of the refactored pipeline. It says nothing about what the kernels do when run.
* The **fused decode**'s `precompile` and `aic` legs compiled, so BiSheng accepts the source and
  its template instantiations for the AIC role. The build was stopped before `libtq_decode_sim.so`
  linked, so the AIV leg and the final link are unconfirmed; expect roughly 40 minutes for it.

**Not verified at all. Do not read these as done:**

* **The seamless chunk pipeline has no CAModel evidence.** Per
  `REFINEMENT_OF_COMBAT_KERNELS.md`, the baseline a pipeline refactor needs is: capacity-one
  token/queue model → compile-time UB checks → isolated logical-page head/middle/tail canaries on
  CAModel → depth and wrap tests → trace audit and census. The first two are done — the
  capacity-one model as `test_contract_gates`' token-balance audit, the compile-time UB checks as
  the prefill's and scatter's device `static_assert`s — and **the last three are not started**.
  The token-balance audit is a counting argument, **not** a deadlock proof and **not** a liveness
  or overlap claim: source order and balanced events do not establish overlap, which needs a
  trace with intersecting MTE2/Cube intervals.
* **No NPU is attached to this host**, so neither device runner has been executed. A bit-exact
  PASS from either would still say nothing about whether a cross-core announcement overtook the
  pipeline work it reports — both earlier revisions of the batch writer passed numerically while
  their traces showed a wait retiring before its set. Audit the trace.
* `tests/npu/*` are compiled only as a syntax check here (`g++ -fsyntax-only` against the ACL
  headers). They have never been linked against the kernel libraries.
