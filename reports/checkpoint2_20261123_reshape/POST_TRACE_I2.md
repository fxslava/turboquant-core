# Slot03 — Independent post-trace audit (I2 worker)

Read-only analysis of the existing six-strip CAModel run; no source, build or model changes. Slot03 is the scalar metadata hoist alone. Source arithmetic, UB and AIC object remain those established in the pre-simulator reviews.

## Outcome and critical-role shift

| Metric | Baseline | Slot03 | Change |
| --- | ---: | ---: | ---: |
| Model footer total | 64266 | 63629 | -637 (-0.991%) |
| Late interior estimate, two samples | 8722 | 8720.5 | -1.5 ticks/strip |
| Last egress interval, separate tail | 13256 | 13004 | -252 |
| AIC role frame end | 59573 | 59070 | -503 |
| AIV0 role frame end | 64033 | 63618 | -415 |
| AIV1 role frame end | 64257 | 63424 | -833 |

The final critical role **moves from AIV1 to AIV0**. The six global egress completion ticks are18694,24475,32571,41281,50012,63016; intervals5781,8096,8710,8731,13004. The finite interior8720.5 estimate excludes the separate tail and is not production throughput. The improvement is mainly in finite-probe fill/tail rather than a materially faster interior plateau.

Terminal pipeline observations make the role shift concrete:

| Role and event | Baseline | Slot03 |
| --- | ---: | ---: |
| AIV0 final Vector-ready scalar gate | 59109 | 58606 |
| AIV0 final quantization V_MTE3 publication | 62848 | 62436 |
| AIV0 final output DMA completion | 63426 | 63016 |
| AIV1 final Vector-ready scalar gate | 58736 | 57699 |
| AIV1 final quantization V_MTE3 publication | 63090 | 62292 |
| AIV1 final output DMA completion | 63665 | 62837 |

AIV1's final gate advances1037ticks but its gate-to-quant span grows4354→4593ticks; AIV0's corresponding span grows3739→3830ticks. Changed composite spans and the critical-role flip prevent explaining the gain by adding isolated load savings.

## All twelve metadata miss chains are proved

Each row below is one direct scalar GM32 mapping load, a DCache MISS at the same PC within one tick, and a repeated CCU ZEROEXT consuming the **same destination register**. The six GM addresses advance64B per strip, matching `slot_mapping[strip*16]`. This is dependency evidence, not an inference from popped gaps alone. Exact PCs, IDs, registers, addresses, attempts and operands are in post_trace_i2_evidence.json.

| Strip | Baseline AIV0 load / dependent stall | Slot03 AIV0 load / dependent stall | Baseline AIV1 load / dependent stall | Slot03 AIV1 load / dependent stall |
| --- | --- | --- | --- | --- |
| 0 | 10008 /416 | 1447 /515 | 1723 /664 | 1535 /532 |
| 1 | 18822 /438 | 14985 /447 | 13809 /434 | 11919 /482 |
| 2 | 24633 /500 | 21326 /370 | 22553 /512 | 19964 /393 |
| 3 | 33361 /406 | 28694 /410 | 31300 /421 | 28682 /519 |
| 4 | 42089 /406 | 37416 /467 | 40028 /465 | 37404 /459 |
| 5 | 50829 /517 | 46138 /519 | 48769 /468 | 46126 /512 |

Baseline metadata retry sums are2683/2964ticks on AIV0/1; Slot03 sums2728/2897ticks. All twelve cold dependency chains remain. Their combined elapsed sum changes only5647→5625ticks, while model time improves637ticks: hoisting changes **when** those misses block scalar issue and how other engines overlap them. These per-role sums must not be added as a kernel savings estimate.

## Baseline-relative retry admission audit

No new long unaccounted scalar retry class was found. All candidate repeated CCU issue attempts map to a typed popped instruction; the mixed-case `ALIGNv2` mnemonic was matched by ID+PC to avoid a false unknown-DMA category.

- The existing initial scalar frame store (ST_XD_XN_IMM using X30) blocks585/617ticks versus baseline582/614; both are inherited ABI-frame prologue paths, not tensor-data stores. The frame base/extent change is already recorded by review3.json.
- Existing cleanup OR retries are246/245ticks versus247/245; they remain outside the grinder. AIC's initial CMP786ticks and cleanup OR244ticks are unchanged.
- Apart from the twelve proved mapping consumers and explicit WAIT_FLAG fences, every other scalar retry is at most6ticks and is a register/address arithmetic or fixed-frame instruction. Candidate source adds no scalar tensor-data loop.
- Scalar fence retry counts become47/48 versus48/49 on AIV0/1 (a wait no longer retries); sums38145/46079 versus37927/46233 reflect altered dependency arrival. AIC retains26 retried scalar fences and its sum drops12757→12231. These sums overlap across roles and are not additive speedup estimates.
- New generated fixed-frame LD_IO/ST_IO instructions were disclosed in pre-simulator review3: +12 static LD_IO.b32, +8 LD_IO.b64, +9 ST_IO.b64; no new scalar WAIT_FLAG route or branch count, and no additional vector LDK/STK spills. Actual long retry census above accounts for them rather than assuming they are free.

This audit establishes that the *repeated-issue* scalar stalls are accounted relative to baseline. It does not claim every fetch/decode gap is explained or that scalar issue is globally stall-free.

## Publication, correctness and model residual

Source-pipe-matched release audit independently checks all76 cross-wait observations, including the two AIC epilogue waits using their actual FIXP_S drain. **Zero early-release observations**. See post_trace_i2_release.json. The root trace census retains549HardEvent sets/549waits with zero residual; ready/quorum/join counts remain balanced.

The existing runner passed both49152-byte caches, all768 FP32 scales,65536 untouched cache bytes and1024 guard bytes. These integer-BF16 fixtures establish bit-exact quantization parity for this probe, not arbitrary BF16 encoder quality or physical NPU timing.

Checkpoint1's conditional metadata-only CPM estimate was63850ticks. Slot03 observes63629, a **-221tick residual (-0.346%)**. Predicted saving416 vs measured637 differs by221. The direction is correct for this finite probe; retained baseline composite weights and an assumed AIV1 critical suffix do not predict the new role-specific final spans exactly. Baseline replay was a consistency check, not independent predictive validation. No post-hoc refit is presented as an advance prediction.
