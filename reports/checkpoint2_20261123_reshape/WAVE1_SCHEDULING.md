# Wave 1 scheduling selection — six-strip CAModel probes

All cases use the same depth-four INT4 pipeline, six strips, 16 tokens/strip,
four KV heads and D256. This is not the new E8 encoder timing comparison.
Physical NPU execution is deferred to the workstation owner.

| Case | Total ticks | Delta | Late interior ticks/strip | Final retiring role |
|---|---:|---:|---:|---|
| Baseline | 64266 | — | 8722 | AIV1 |
| 03: metadata hoist | 63629 | -637 (-0.991%) | 8720.5 | AIV0 |
| 04: sync reorder | 64715 | +449 (+0.699%) | 8709 | AIV1 |
| 05: crossover | 65213 | +947 (+1.474%) | 8699.5 | AIV0 |
| 06: owner-credit async | 64552 | +286 (+0.445%) | 8557 | AIV1 |

The interior estimates are the median of the last two interior egress intervals;
the separately compiled tail is excluded. Two samples from one cold launch do
not establish asymptotic throughput. Slot03 wins total time; its improvement is
mainly fill/drain. Slot06 has the best interior estimate but loses total time.
Slots04 and06 remain pure ablations: measured inherited metadata misses are
retained, while new or unaccounted scalar stalls are rejected.

Every numerical probe passed exact K/V outputs (49152 bytes per plane), all768
FP32 scales, 65536 untouched cache bytes and1024 exterior guard bytes. All four
implementations completed three pre-simulator reviews, strict builds and the
baseline-relative AKA gate (zero fatal / zero new warnings). Source-pipe-matched
release audits pass76 waits for03/04/05; the owner-aware06 audit passes150.
Post-trace audits establish all twelve scalar metadata dependency chains in
each case and account for generated fixed-frame scalar I/O. Slots04/05 expose
an additional kernel-argument LDP-to-frame-store dependency of97/89 ticks;
the exact registers, DCache MISS and CCU attempts are retained. No new
unaccounted repeated-issue class remains. CCU-silent delivery gaps are reported
separately without inventing an instruction-cache explanation.

## The original CPM ranking was not predictively validated

| Implemented case | Conditional pre-run estimate | Observed | Residual |
|---|---:|---:|---:|
| 03 | 63850 | 63629 | -221 |
| 04 | 63593 | 64715 | +1122 |
| 05 | 63177 | 65213 | +2036 |

The baseline graph replayed its trace exactly, which checks construction
consistency. It did not independently validate fixed composite weights under
retiming. Slot03 changes the final role from AIV1 to AIV0, while04/05 alter
admission and observed queue spans. The changed costs invalidate additive
baseline saving estimates. A single trace cannot separately identify MTE2
startup and HBM contention; no such independent calibration is claimed.
Measured reports and raw instruction/CCU evidence are retained by slot. No
post-hoc refit is labeled a prediction.

## Selection boundary

Package03 for yellow-zone testing. Retain06's bounded, verified ownership graph
for a possible metadata-hoist crossover in Wave2; do not promote it as a total
tick winner. No Wave2 implementation or additional timing launch is authorized
by this selection report itself.

Evidence: slot03/POST_TRACE_I2.md; slot03/analysis/trace_summary.json;
slot04/POST_TRACE_I2.md; slot05/POST_TRACE_I2.md;
slot04/analysis/trace_summary.json; slot05/analysis/trace_summary.json;
slot06/MEASURED_RESULT.md and its bound protocol/dynamic audit JSONs.
