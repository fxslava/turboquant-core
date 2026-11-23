# Slot07 — Verified E8 encoded batch packet transport

**PASS:** one CANN9.2.1 dav-3510 CAModel launch, exactly **six16-row strips** with a four-slot ring. All **17,408 output bytes** match; both K/V packet planes, all input buffers, all GM guards, and the unused last32 rows of the second packet are preserved.

Inputs are already-encoded LEuint16 E8 labels and FP32 token scales. This is a clean batched **packet transport**, not raw BF16 quantization. Separate K/V planes each use two4352B packets. Input mode2 means encoded E8 packet input in this separate experimental entry point; production quant_mode dispatch is unchanged.

## Gates and immutable revision

- Strict CANN9.2.1 Release build: PASS with `-Wall -Wextra -Werror` for device, host wrapper and runner.
- AKA: raw0fatal/0warnings; no category suppressions. The rejected runtime-ID attempt and its8warnings are retained separately.
- All3 pre-simulator reviews: PASS, recorded in PRE_SIM_REVIEWS.md.
- Lightweight CPU token/ownership/canary model: PASS; ring wraps and drains with zero residual tokens.
- Live canonical model library mapping: verified libpem SHA256 `1a6420fe8123e88a120f0bbfc76710968e6708599c77d80a7b1097aca06eed33`, exit0.
- Source and artifact SHA256 manifests both rechecked after the run.

Native revision: `/home/dmin/work/checkpoint2_20261123/slot07/src`. Built kernel source SHA256 `aff1d34e3ae279c85dd48f78663bb2b923de3ff5d969acd79789721e35042db9`. The unchanged Slot02 packet ABI header SHA256 is `53c18584ae352e9f00677493450504ec013c5fbe970b9e2549251436ea8f5b71`.

The six identical compile-time phases use eventIDs0,1,2,3,0,1. Four disjoint skewed rings occupy9216B UB. The generated grinder PCs[0x184,0x60c] contain24 input DMAs,24 output DMAs, no branch, no scalar tensor load and no explicit scalar-pipeline wait. Prime/drain control loops and compiler BAR.ALL/DSB epilogue remain outside it.

## Measured schedule and scalar issue audit

The single model footer reports **4576 total ticks** and10.987s wall runtime. Four MTE3 output transfers per strip complete as follows:

| Strip | Ring slot | All four outputs retired | Delta from previous strip |
| --- | --- | ---: | ---: |
| 0 | 0 | 2325 | — |
| 1 | 1 | 2861 | 536 |
| 2 | 2 | 2883 | 22 |
| 3 | 3 | 3444 | 561 |
| 4 | 0 | 3467 | 23 |
| 5 | 1 | 3971 | 504 |

The two post-fill intervals average263.5ticks/strip. Their23/504 variation is retained; **stationary steady-state throughput is not established** by two samples. The average of all five completion intervals is329.2ticks. Total/6 includes prologue/epilogue and is not a steady-state metric.

Despite zero explicit scalar waits/loads inside the grinder, the popped trace shows two scalar issue gaps: **524ticks** between PC0x1f8 and0x200 (ticks1183..1707), and **527ticks** between PC0x3fc and0x400 (1773..2300). These are implicit stalls and remain causally unclassified. Descriptor/queue admission is a hypothesis, not a verified explanation. No claim of zero implicit scalar stalls or improvement over the raw BF16 baseline is made.

`strip_trace.json` retains all48 DMA instruction IDs, PCs, popped/retired timestamps and completion deltas; popped-to-retired spans may include queueing and are not isolated hardware latency measurements. `run/` retains raw trace, packet outputs, independent expected packets, mapping evidence and packet_result.json. The generic mapping verifier's cache template text does not establish cache persistence here.

## Queue status and scope

The device lane was released immediately after the successful footer. No further build or simulation is authorized for this worker without a coordinator grant. Raw BF16 encoder and actual E8 fused attention are separate tracks. No physical NPU, verified Wave-v11 source or baseline source was touched.

## Performance gate follow-up: INCOMPLETE

CCU evidence supersedes the initial popped-gap hypothesis. ADD atPC0x200/id141 first reaches D1 at1706, issues once at1707 and executes at1707. MOVI atPC0x400/id269 first reaches D1 at2299, issues once at2300 and executes at2300. Neither target has repeated issue attempts or a recorded general-hazard retry. CCU logs no intervening instruction events within either delivery gap.

ADD reads an existing packet baseX1 and constant4096 inX23, and overwritesX12 previously used as a DMA descriptor. MOVI writes immediate4512 toX28 and has no data source operand. The retained evidence therefore establishes **pre-D1 delivery gaps**, not a demonstrated pending ADD/MOVI scoreboard or MTE event stall. A separate DMA id270 retry at2300..2301 follows the second gap and does not explain its preceding527ticks.

The boundaries0x200/0x400 suggest cold instruction delivery, but no instruction-fetch/cache miss reason is logged. The exact cause is unproven and there is no matched compact-grinder baseline. Consequently the **performance gate remains incomplete**, while CAModel packet correctness remains PASS. These gaps are not relabelled as explained baseline cost. See ccu_gap_classification.json for exact CCU line numbers and stage records. A compact-grinder source revision must be reviewed before any new model grant; the tested source stays immutable.
