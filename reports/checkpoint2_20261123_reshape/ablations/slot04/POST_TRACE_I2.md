# Slot04 — independent post-trace audit

Read-only analysis of the existing six-strip CAModel trace. No source changes, builds, simulations or post-hoc model refits.

Slot04 moves the AIC next-page prefetch into the FIX_S interval before ready publication. Its AIV object is byte-identical to baseline (SHA256 f053658a261238afc73fb6e1f5dce63e3f7a2afcb7f0b857c818c91a71b7f043).

Total 64715 ticks versus64266 baseline: **+449 regression**. Conditional Checkpoint1 prediction63593, observed residual **+1122**. First global egress is already658 ticks later. Finite late-interior estimate8709 versus8722; separate last interval12973 versus13256. The slightly shorter interior fails to recover the startup loss; this is not a production throughput measurement.

Final critical role is **AIV1**. All terminal times below are absolute trace ticks; frame ends use the root frame log and are distinct from last retired instruction timestamps.

| Event | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| AIC final dual-AIV join FIXP_S drain | 59281 | 59820 | +539 |
| AIC frame end | 59573 | 60114 | +541 |
| AIV0 final ready Vector-drain gate | 59109 | 59143 | +34 |
| AIV0 final quant V_MTE3 publication | 62848 | 63439 | +591 |
| AIV0 final output DMA completion | 63426 | 64020 | +594 |
| AIV0 frame end | 64033 | 64619 | +586 |
| AIV1 final ready Vector-drain gate | 58736 | 59648 | +912 |
| AIV1 final quant V_MTE3 publication | 63090 | 63491 | +401 |
| AIV1 final output DMA completion | 63665 | 64061 | +396 |
| AIV1 frame end | 64257 | 64700 | +443 |

The AIC final join/drain ends before the final AIV quantization/publication and output suffix; it is not evidence that output is complete. The root pipe-matched cross-release audit passes76/76 observations with no early release. Cache/scale/untouched-byte/guard gates pass in the original runner.

## Every metadata chain and baseline-relative retry census

| Strip | Baseline AIV0 load/stall | Candidate AIV0 load/stall | Baseline AIV1 load/stall | Candidate AIV1 load/stall |
| --- | --- | --- | --- | --- |
| 0 | 10008/416 | 10543/473 | 1723/664 | 1779/612 |
| 1 | 18822/438 | 19911/359 | 13809/434 | 13859/368 |
| 2 | 24633/500 | 25279/456 | 22553/512 | 23204/446 |
| 3 | 33361/406 | 33991/461 | 31300/421 | 31935/443 |
| 4 | 42089/406 | 42703/496 | 40028/465 | 40647/510 |
| 5 | 50829/517 | 51427/509 | 48769/468 | 49372/380 |

All twelve chains are direct GM32 load → same-register ZEROEXT, DCache MISS at the load PC, and CCU consumer retries every tick. Metadata retry sums AIV0/AIV1 are2754/2759 versus2683/2964; combined5513 versus5647. Per-role retry sums overlap and cannot be added as total-kernel cost.

Other scalar CCU retries longer than6ticks:

| Role | Instruction class / first attempt / elapsed |
| --- | --- |
| cubecore0 | CMP / 741 / 666 |
| cubecore0 | OR / 59829 / 246 |
| veccore0 | ST_XD_XN_IMM / 740 / 629 |
| veccore0 | OR / 64035 / 247 |
| veccore0 | ST_XD_XN_IMM / 1375 / 97 |
| veccore1 | ST_XD_XN_IMM / 737 / 736 |
| veccore1 | OR / 64074 / 245 |

The newly exposed second frame-store retry is accounted precisely in ../post_abi_ldp_i2.json: LDP.b64 X10,X4,[PARA_BASE+16] issues731, DCache MISS732, and ST [X30-1408],X4 repeats1375..1472 (98 attempts,97 elapsed); LDP retirement1473. The store consumes the outstanding load destination; this is an implicit scalar register dependency on a cold kernel-argument load. The retirement event is bookkeeping after final issue, not an exact scoreboard-release timestamp. Existing source/object provenance rules out a new tensor-data loop, but does not make the newly exposed elapsed cost zero.

The first frame-store and cleanup OR stalls are inherited control/ABI classes. All other nonmetadata/nonfence scalar retries are at most6ticks, and every attempt maps to a typed popped instruction. New/unaccounted retry classes:0; newly exposed accounted ABI dependency above is explicitly retained.

## Original dynamic scalar census

Counts below come from the original popped instructions, not source-line counts. Fixed-frame IO includes compiler ABI traffic. Explicit-fence retry elapsed is overlapping per-role observation, not additive kernel cost.

| Role | Baseline LD_IO/ST_IO dynamic counts | Candidate LD_IO/ST_IO dynamic counts | Baseline scalar-fence retries/count elapsed | Candidate scalar-fence retries/count elapsed |
| --- | --- | --- | --- | --- |
| cubecore0 |  |  | 26/12757 | 26/13070 |
| veccore0 |  |  | 48/37927 | 48/38436 |
| veccore1 |  |  | 49/46233 | 49/47179 |

Both candidates retain the root original protocol census: 549 HardEvent sets/549 waits, zero residual, and 76 pipe-matched cross-wait releases with no early release. No wait/quorum protocol was removed. Complete per-engine retry classes, instruction IDs and operands remain in the JSON evidence.

## Strict performance selection

Root selection: **REJECT**. In addition to total-time regression, the newly exposed ABI-load-to-frame-store stall is a new implicit scalar stall (97ticks in04,89ticks in05). Its source-backed explanation accounts for the mechanism; it does not make the candidate pass the strict no-new-stall performance gate. Functional/canary/protocol PASS is separate. No source change or re-run is implied by this classification.

## Delivery gaps and model limitation

- cubecore0: CCU-silent intervals >100ticks: baseline1, candidate0; elapsed sums389→0. These are not repeated-issue stalls.
- veccore0: CCU-silent intervals >100ticks: baseline34, candidate33; elapsed sums16276→16020. These are not repeated-issue stalls.
- veccore1: CCU-silent intervals >100ticks: baseline15, candidate15; elapsed sums7945→7530. These are not repeated-issue stalls.

The CCU-silent intervals are instruction-delivery observations; the logs do not identify an instruction-cache/fetch mechanism. They remain separate from the source-backed load/register retry chains. Baseline already has many such gaps; no causal subtraction or new cache-miss claim is made.

The advance CPM ranking misses this candidate by1122ticks. Retaining baseline composite weights omitted changed startup contention, kernel-argument availability, metadata arrival and the measured role-specific suffix. The retained dual-AIV protocol and earlier AIC staging do not establish the hoped-for asynchronous-AIC benefit. The observed terminal chain and original census explain where the model assumptions fail, without refitting the prediction after measurement.

Evidence: post_trace_i2_evidence.json; ../post_abi_ldp_i2.json; ../post_terminal_i2.json; ../post_trace_i2_ccu_quiet.json; analysis/trace_summary.json; original trace/CCU/DCache logs. Correct selected03 AKA evidence is aka_scalar.json, not the abandoned DMA prototype gate.
