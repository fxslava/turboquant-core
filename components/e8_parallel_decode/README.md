# Verified Wave-3 E8 decoder

The sealed `review_packet_v5/candidate/source` closure is copied from
`C:/ascend-micro-harness/reports/checkpoint2_20261123/wave3/w3_e8_dec`.
The arithmetic, tie rules, reader and internal K padding are unchanged.
`SOURCE_MANIFEST.json` in the integration report records the imported hashes.
The `frozen/` routines are independent CPU oracles, not dispatch candidates.

The measured component is one 16-row frame: 512 UINT16 labels and 16 FP32
scales produce 4096 BF16 values and 4096 INT8 values. The measured reader
supports one KV task, six cache-basis FP32 query heads padded to 16 rows,
4..256 even tiles, and a 1..64-row final tile. It writes cache-basis BF16
output and mandatory caller-owned K/V/score witnesses. Allocate the UINT64
byte-offset table rounded up to four entries for the 32-byte DMA read.

Each K or V GM packet is 4352 bytes: 2048 little-endian UINT16 labels in
bytes 0..4095 and 64 FP32 scales in bytes 4096..4351. The effective rate is
two bits per coordinate, not a two-bit scalar label alphabet. The 1056-byte
K stride exists only inside UB; GM packets retain their external ABI.

The regular core CTest run includes both exhaustive decoder parity and UB
layout tests. `TQ_BUILD_E8_KERNELS=OFF` is the default. A deployment checkout
can explicitly enable it to configure `tq_e8_kernels`; it was not enabled here.
No encoder, raw-query composition or new device performance result is claimed.
