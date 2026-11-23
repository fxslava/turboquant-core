# E8 decoder eight-coordinate audit and replacement

Frozen originals and the verified Wave-v11 source/tag were not edited. The new
active implementation is isolated in `parallel_decode/src`; exact patches and
source/artifact manifests accompany this report.

| Original source | Status before replacement | New active source |
|---|---|---|
| BF16 leaf `Project`, lines87–106 | Already eight independent register expressions for label extraction and G2 transform | Shared `e8_parallel_decode.hpp::Project`, same expressions |
| BF16 leaf lines39,116–169,180–227,254 | Coordinate loops; parity, argmax, distance and tie state carried across coordinates despite pragma-unroll | Explicit I8 registers, XOR/Sum/Max trees, masked repair/select and eight stores |
| INT8 sibling lines39,117–170,181–228,265 | Same coordinate loops; final RNE+saturation conversion | Same shared decoder, eight independent PhysicalInt8 stores |
| Slot09 `DecodeBlock` line52 | Eight coordinate stores after frozen sequential shaping | Eight `StoreCoordinate<Value,C>` calls after shared R8 decoder |
| Slot09 `PackQuery` line74 | Eight independent slice-copy iterations, not lattice reconstruction | Eight explicit `PackQuerySlice<Step>` calls |
| Native Wave-v11 tests/sim/kernel and production fused decoder | Inspected path is signed INT4, not this E8 ABI; codec950 asserts BITS==4 | Protected files untouched |

The public BF16 and INT8 replacement headers under `src/routines` retain the
existing Arithmetic/CompactInt8, four table layouts and LegacyNearest/ResidualBits
APIs. Compact projection has eight explicit two-table additions. The only
remaining public-VF loop distributes whole 8D blocks; there is no coordinate
loop in its device call closure. Frozen originals are host-only numerical oracles.

The exact repair key is `(abs(r)<<8)|(nonnegative<<7)|(nonnegative?7-index:index)`.
Its depth-three maximum tree preserves the frozen scan's first-positive versus
last-negative tie rule. Both cosets use independent masked repairs; parity is
an explicit XOR tree, distances an integer sum tree, and coset selection is
eight masked register selects. No extra UB LUT is required for Arithmetic.
One warp still owns a D256 row: 32 threads, one 16-bit E8 label/eight coordinates
per thread. Eight independent register expressions are not a one-cycle promise.

Strict CPU tests pass all 65,536 labels against frozen LegacyNearest,
ResidualBits and INT8 reconstruction, all four Compact layouts, and 24 scales:
12,582,912 BF16 words and 12,582,912 INT8 outputs, zero mismatches. Signed
max-FP32-subnormal, min-normal, 2^-132 and true-min scales are included.

Fresh strict device/host build-r3 passes `-Wall -Wextra -Werror`. Source AKA is
raw0/0 for the component and baseline-relative0fatal/0new warnings for the reader
with the documented source-bound raw-pointer provenance repair and independent
I1 generic-Sync warning equivalence. Negative controls remain failing.

Compiled ISA contains20 independent argmax trees: seven IMNMX instructions and
an exact producer-graph depth of3 each. Reader SIMT functions have no branches;
public-VF branches only implement entry and whole-block distribution. No
coordinate backedge remains. Source Sum8 is balanced, but the compiler fuses
distance arithmetic into IMUL/IMAD groups; an exact three-cycle distance claim
is not made. Component Rmax93, reader Rmax22/106/113; bounded private frames12/8
bytes per thread, divergence requirement0. Small STK/LDK counts are disclosed.
Actual UB arenas are14400/156416 bytes at base0; reader AIC text is byte-identical
to the frozen reader, including all Cube/data-publication logic.

Granted component CAModel passes4096 BF16 words at0ULP,4096 INT8 bytes exactly,
all inputs and512guard bytes, including signed FP32/BF16 subnormal-boundary
scales. Total14856ticks covers both public output formats in one component;
it is not fused-attention performance. Canonical dav_3510 PEM mapping verified,
three HardEvent routes balanced, no nonempty exception dump. Runtime capacity
queries return1152private bytes/thread and1024divergence bytes/warp.

The separately granted uniform six-tile fused-reader gate passes. It checks
actual encoded-scatter→reader packets,1,536 BF16 attention words at0ULP,
196,608 exact BF16 K/V witness words,2,304 exact FP32 Cube QK words,208,896 full
cache-afterimage bytes,1,536 exterior guard bytes and1,024 inactive output bytes.
All47 cross waits pass the bounded source/pipe-marker audit; twelve actual
FP32 MMADs include the word statically misdecoded as a VAG alias. SDK-reserved
VEC→SCALAR event7 is inherited: exact five static pairs are unchanged, with
seven balanced pairs perAIV at runtime. No new manual event7 was introduced.
Total336,288ticks includes two encoded-scatter launches plus the reader;
the participating reader frame is330,879ticks. No speedup comparison is made
to the aborted old reader, and no asymptotic throughput is inferred.

The nonuniform and raw-writer composition remain separate pending gates.
This does not imply raw-query rotation or inverse output-basis production
integration. No physical NPU result is claimed.


## Analyzer call-closure limitation

The historical AKA frontend did not resolve component Dequant/DequantVf or the
reader DecodePacked template instances. Raw zero diagnostics for the component
and normalized zero diagnostics for the reader cover the resolved subset, not
those leaf bodies. Fresh independent reviews cover their source/ISA; the exact
frontend coverage inventory and raw/accepted diagnostics are sealed alongside.
Later analyzer changes do not retroactively strengthen this recorded gate.
