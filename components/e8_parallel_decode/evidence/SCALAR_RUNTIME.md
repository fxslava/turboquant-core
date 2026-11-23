# Uniform reader scalar runtime classification

This is a read-only audit of the already verified uniform run. No new model was
launched. The participating reader frame is 330,879 ticks; the 336,288-tick
process total includes two preceding encoded-scatter launches. The old Slot09
reader trace was terminated and is only a partial comparison window, with no
completed numerical verdict or full-kernel timing baseline.

Repeated CCU issues measure elapsed retry intervals, not an exact hardware
service-time decomposition. The sums below overlap different cores and engines
and must not be added to predict kernel duration or possible savings.

| Route or interval | AIC | AIV0 | AIV1 |
|---|---:|---:|---:|
| FIXP to scalar event0 | 13 waits / 328,528 ticks | — | — |
| MTE2 to scalar event0 | — | 1 / 512 | 1 / 368 |
| MTE3 to scalar event0 | — | 24 / 317,862 | 24 / 315,735 |
| SDK Vector to scalar event7 | — | 7 / 7,029 | 7 / 9,380 |
| DSB #0 | — | 13 / 274 | 13 / 274 |
| DSB #2 | — | 7 / 294 | 7 / 294 |

Five static SDK event7 set/wait pairs match the old reader ISA, including the
same API-generated boundaries. They execute seven times per AIV in the complete
six-tile run. In chronological order the observed retry lengths are:

- AIV0: 533, 530, 529, 526, 514, 2,919, 1,478 ticks.
- AIV1: 513, 529, 529, 505, 436, 5,333, 1,535 ticks.

The partial baseline covers only the first two event7 waits: AIV0 531/506,
AIV1 530/512 ticks. The corresponding candidate deltas are +2/+24 and -17/+17.
The remaining waits have no dynamic baseline coverage. Their inherited static
origin does not erase the measured wait cost or establish unchanged duration.

Matching relative scalar PCs and covered ordinal occurrences shows the same
prologue store retries: AIC 490, AIV0 472, AIV1 487 ticks. AIV ADD retries at
0x3b4, 0x690, 0x1334 and 0xb74 are respectively 12, 11, 9 and 12 ticks in both
traces where covered. The 0x3b4 consumer follows the scalar packet-offset load
at 0x3a8 with two independent issues between them; load-use waiting therefore
remains possible. Retry records alone do not distinguish cache latency from
all other producer constraints. No cold-GM causal claim is made.

DSBs absent from the popped log are typed using their exact static instruction
and relative PC. Two final 2-tick retries are explicit BAR.ALL. The terminated
baseline has an unpopped MTE3-to-scalar wait at 0xdec, censored at 9,041/11,191
ticks; those are lower bounds, not completed wait lengths. Final cleanup OR
retries (AIC 244, AIV0 247, AIV1 245 ticks) have no baseline runtime coverage.

The independently reviewed AIV prefix retains 2,461 ordered scalar operations;
only VF target relocations and encoded VF lengths differ. Larger decoder VF
bodies and observable SDK waits can change runtime. This partial comparison
cannot prove a global absence of new implicit stalls or a speedup. CCU-silent
gaps are not classified as arithmetic, GM-witness-store or instruction-fetch
cost without an independent event witness.

`SCALAR_RETRY_CLASSIFICATION.json` binds source logs by SHA256 and records typed
retry intervals. The package copy omits the redundant scalar-popped inventory;
the complete report remains at the recorded workstation path. The classifier
is retained under evidence/tools for replay against the existing native logs.
