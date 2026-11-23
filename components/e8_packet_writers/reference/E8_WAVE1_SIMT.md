# Frozen Wave 1 SIMT E8 dequant

These are byte-identical Agent06 BF16 and Agent08 INT8 headers. Provenance and exact hashes are in `e8_wave1_export.json`; the Phase0 registry commit is `cf94dd09c4e7721f006a3eb43633a1dda5d9658c`. Native slot trees are source snapshots without `.git`; that commit is in the destination harness repository.

## Select the measured specialization explicitly

Header defaults are128 threads and LegacyNearest. The frozen winners require512 threads and ResidualBits:

```cpp
#include "routine_e8_bf16_dequant_simt.hpp"
#include "routine_e8_int8_dequant_simt.hpp"

namespace bf = ascend_micro::e8;
namespace i8 = ascend_micro::e8_int8_simt;
// The caller supplies aligned/padded LocalTensor views and DMA synchronization.
bf::Dequant<bfloat16_t, 8192, bf::Backend::SIMT,
            bf::Representation::Arithmetic, 512, bf::TableLayout::AoS,
            bf::Shaping::ResidualBits>(indices, scales, bf16Output, scratch, table);
i8::Dequant<int8_t, 8192, i8::Backend::SIMT,
            i8::Representation::Arithmetic, 512, i8::TableLayout::AoS,
            i8::Shaping::ResidualBits>(indices, scales, int8Output, scratch, table);
```

Indices are LE uint16 per8 coordinates:32 labels/64B per D256 row. Scales are a separate FP32 value per row. Both base pointers are32B aligned. Selected scratch and table sizes are zero; their LocalTensor parameters are unused in this specialization. `DequantFootprint` specifies each caller-owned allocation and256B full-register read pad. The leaf allocates no hidden UB and owns no DMA/cross-core flags. Caller stages input, establishes MTE2-to-vector readiness, joins vector-to-MTE3 before output DMA, and drains output DMA before reuse/return. Historical standalone wrappers use32768B static UB; this is not an enclosing production-kernel budget.

The decoded canonical doubled coordinates are scaled as `(q2*0.5f)*scale`. BF16 output uses RNE; INT8 output uses RNE and saturation[-128,127]. The frozen leaf includes no inverse Hadamard or channel equalization. Do not chain leaf wrappers and claim register-only fusion.

| Frozen standalone result,8192 elements | Postwarmup ticks | Ticks/element |
|---|---|---:|
| BF16 Agent06 | 9736 /9815 |1.19329833984375|
| INT8 Agent08 | 8011 /7857 |0.968505859375|

CANN9.2.1/dav_3510 evidence is archived under `../reports/e8_wave1/`. These timings apply only to the selected source/configuration; physical peak VREG is unknown. No new device build or CAModel run was used for this export.

## Verify and regenerate the golden vectors

Verification needs only Python3 standard library:

```sh
python3 -B routines/generate_e8_wave1_goldens.py --verify-only
```

For exact regeneration use NumPy2.2.6 and a new or empty native WSL directory:

```sh
python3 -B routines/generate_e8_wave1_goldens.py --output /home/dmin/work/e8_wave1_regenerated
```

Use the checkout path appropriate to the home workstation. Builds and CAModel stay in native WSL directories and the prescribed Docker image. The wrapper stages the unchanged oracle in a temporary directory beside the requested output; diagnostics never write beside the checked-out source. The original oracle is also directly usable with `--fixture-output` and `--stress-fixture`; it writes a report relative to its own location, so prefer the portable wrapper.

`e8_wave1_goldens/manifest.json` describes all23 original payloads and their SHA-256 hashes. The generator reproduces PCG64 seed9503510, the original NumPy FP32 RMS reduction and exact finite-codebook reference with smallest-label ties. The BF16 and INT8 physical golden files are `bf16_dequant.bf16le` and `int8_dequant.i8`; `*_golden_unrounded.f32le` holds the corresponding reconstruction before output conversion. Auxiliary `*_aux_unscaled_z.i8` / `*_aux_scales.f32le` are the distinct q2-plus-scale/2 ABI, not the selected physical-output ABI.

Reproduction validates every payload record; report timings, Python version and source paths are machine-dependent. These frozen Wave1 fixtures are independent of the later Wave2 rotation/equalization and fullfinite BF16 quant golden revision.
