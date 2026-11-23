# Review 2: API, layout and source ownership

The reader's packet ABI, 32-thread D256-row mapping, two AIV owners, two ring
slots, scalar issue order, CrossCore events, DMA, Mmad/Fixpipe, recurrence and
peeled tail are unchanged. Only the pure register decoder and its eight stores
are replaced. PackQuery's independent eight slices are explicit calls.
Reader UB remains 156,416 bytes; L1 282,816; L0A/B/C 16/64/16 KiB. Actual ELF
reservation and new register/stack counts remain mandatory after compilation.

Both requested public dequant headers are active replacements under src/routines.
Arithmetic and CompactInt8 (AoS/SoA/PaddedSoA/Blocked) retain their interfaces;
Shape and ShapeResidualBits use the balanced exact helper. Frozen originals
are separate CPU oracles. The only remaining public-VF loop distributes whole
eight-dimensional blocks uniformly; it does not iterate coordinates.

The new bounded BF16+INT8 component calls both public APIs on 16 D256 rows,
with 128 threads (32 per row), 512 labels, 16 scales and 4,096 outputs/format.
One referenced static arena is 14,400 bytes, with offsets 0/1280/1600/10048,
256-byte read pads and no dynamic/TPipe allocation. A 32-byte dummy table view
occupies label padding, not live label/scale/output storage; Arithmetic does
not read it. UB+reserve leaves the 128 KiB DataCache cap.

24 CPU scales include positive/negative max FP32 subnormal, min FP32 normal,
2^-132 and true-min values. Public API tests exhaust all 65,536 labels and all
four compact layouts; independent frozen Shape/ResidualBits/INT8 outputs agree.
Hardware FMUL/F2F subnormal behavior is still a pending device observation.
