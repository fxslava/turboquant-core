# Review 3: source admission and compiled ISA

No coordinate loop exists in the active parallel header, public BF16/INT8
replacement headers, component kernel or fused reader. No device PRINTF or
scalar tensor-data loop was added. The reader retains only uniform paired-tile
grinders and a separate tail. Explicit SIMD/SIMT DataSyncBarrier bracketing and
V_MTE3 publication are retained; the component additionally drains output DMA
through MTE3_S. The component's two public SIMT VFs share the Vector engine and
do not claim concurrent compute.

Source AKA: component raw 0 fatal/0 warning; reader 0 fatal/0 new warning with
the source-hash-bound raw-Cube-origin repair, original exact Reg API exceptions,
and independently archived I1 generic-Sync warning equivalence. Its negative
controls still detect illegal TPipe/A1 allocations and overlapping raw Cube
views. No diagnostic category or global checker is disabled.

CPU controls pass strict -Wall -Wextra -Werror, with only frozen third-party
oracle headers treated as system headers. Fresh build-r3 passes strict device
and host builds. The initial missing explicit BF16 include and the legacy CANN
incremental text-object relink failure are preserved in separate failure logs.

ELF positively reserves static UB at base zero: component 14,400 bytes, reader
156,416. Reader AIC instruction bytes are identical to the frozen reader.
The three reader SIMT functions contain no branch; public BF16/INT8 VFs contain
only entry and whole-block distribution branches. Twenty independent argmax
trees each contain seven IMNMX instructions with a measured producer-graph
depth of three. No coordinate backedge exists. Register maxima are 93 for
the component and 22/106/113 for the reader. Bounded private frames are 12/8
bytes per thread respectively; TLV10 divergence requirement is zero. The
few private STK/LDK sites are disclosed, not described as spill-free.

Host-only strict relinks add actual-resource runtime checks before any launch,
preserving device object hashes. Component CAModel passes all 4,096 BF16 words
and 4,096 INT8 bytes, input preservation and 512 guard bytes, including signed
FP32/BF16 subnormal-boundary scales. Runtime capacity queries return 36,864
bytes per warp (1,152 per thread) and 1,024 divergence bytes per warp.
No FTZ mismatch is observed in these bounded controls. Fused-reader parity
remains a separate device gate. Independent register operations are not
claimed to execute in one cycle.
