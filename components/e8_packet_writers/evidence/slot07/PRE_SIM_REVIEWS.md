# Slot07 pre-simulator reviews

Scope: encoded-label+FP32-scale batch transport, not raw BF16 quantization. Six strips ×16 rows, four UB slots. K/V packets are distinct. Exact immutable Slot02 ABI SHA25653c18584ae352e9f00677493450504ec013c5fbe970b9e2549251436ea8f5b71.

## Review1 — ownership, buffer bounds and token balance

Four ring regions: keylabels[0,4224), keyscales[4224,4608), valuelabels[4608,8832), valuescales[8832,9216). Each label-slot stride1056B holds1024B; each scale-slot stride96B holds64B. Every base/stride is32B aligned. Static budget9216B is below221184B.

For each slot, MTE3_MTE2 free credit is primed once, consumed before reads, and returned only after both K/V output DMAs. MTE2_MTE3 ready credit is set after all four input transfers and consumed before outputs. The feedback free edge also protects ready-ID reuse. Six phases use IDs0,1,2,3,0,1; phase4/5 exercise wrap. Epilogue consumes one final free credit per slot, then MTE3_S drains output. Lightweight capacity-one token/byte model PASS, zero residual tokens, last32 rows of the second packet preserved.

## Review2 — API and scalar issue order

Grounded in local PipeSim DataCopyPad GMToUB/UBToGM descriptions and intra_core_synchronization_capability_overview.md, plus950PR vector skill. Batched copies are aligned DataCopy, uint16 count512 =1024B and FP32 count16 =64B. GM label/scale payloads and destinations remain32B aligned for these counts.

The grinder consists of six identical compile-time strip bodies, with constant IDs. The only dispatch guard is before the grinder. No per-strip if/switch, scalar tensor load/loop, GetValue/SetValue, device PRINTF, VF transition, Cube work or cross-core quorum exists. The S pipeline calculates no tensor values and performs no cold GM metadata read. Only MTE3_S at the final drain waits for completion on S; all in-flight dependencies are MTE2/MTE3-local. This pure-DMA path has no AIC to decouple. The generated ISA remains to be reviewed after strict compilation.

Rejected attempt: the runtime strip%4 source was not built or simulated. Its four unresolved-ID AKA3010 warnings caused four cascading AKA2010 warnings. It is retained as rejected-runtime-ids.cpp and aka-rejected-runtime-ids.json. No warning category was suppressed. The constant-ID revision has raw0fatal/0warnings.

## Review3 — final strict build and instruction audit PASS

CANN9.2.1 Release compile passed device, host wrapper and runner with -Wall -Wextra -Werror and the existing generated-wrapper consume-parameter policy. The final raw AKA result is0fatal/0warnings, no code-category suppressions. The compiled source SHA256 is aff1d34e3ae279c85dd48f78663bb2b923de3ff5d969acd79789721e35042db9; source.sha256 recheck passed. Artifact hashes are in device-artifacts.sha256.

Generated device-isa.txt shows all six strip bodies in PC[0x184,0x60c], with24 input DMAs and24 output DMAs. There is no jump, scalar LD/LD_IO, LDK/STK or WAIT_FLAG.*.S in that interval. The four-iteration credit-prime and credit-drain control loops are outside the grinder; the only explicit scalar completion wait is MTE3_S after the grinder at0x628. Compiler BAR.ALL/DSB at0x62c/0x634 are the disclosed epilogue. No per-strip runtime branching, cold scalar data load, data loop or device printf exists. Both K/V packet offsets cross the tile boundary atstrip4, with the independent CPU row oracle preserving the second tile tail.

All3 reviews PASS. This revision is READY_FOR_COORDINATOR_CAMODEL_GRANT; no simulation has run yet. CPU/static/build evidence alone does not establish device packet correctness.


## Post-run observation (not a pre-simulation assumption)

The approved single CAModel launch passed all17408 packet bytes and guards at4576total ticks. The source/ISA absence of explicit S waits does not prove an unstalled scalar pipeline: actual popped trace has524/527tick S-issue gaps. These are retained and unclassified; zero implicit scalar stalls is not claimed. See README.md and strip_trace.json. Source and artifact manifests rechecked unchanged after simulation.
