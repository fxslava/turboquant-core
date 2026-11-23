# E8 packet writer components

This isolated component package adds **pre-encoded E8 label/scale transport** alongside the integrated Wave-v11 baseline at3e57bfa. Existing decode, prefill, scatter, dispatch and benchmark sources are unchanged.

Each plane is a **4352-byte D256 packet**: LEuint16 labels `[64 tokens][32 E8 vectors]` at0..4095 and FP32 token scales `[64]` at4096..4351. K and V use separate packets. `e8_packet_writers.hpp` declares the new experimental launch symbols.

| Component | Functional evidence | Performance status |
| --- | --- | --- |
| Slot02 single-token label+scale scatter | CANN9.2.1 strict build; raw AKA0fatal/0warnings; six CAModel writes atrows0,31,63 forK/V;52,224 packet-byte comparisons and guards PASS | Six-launch session14476ticks; not single-token latency |
| Slot07 six-strip encoded batch transport | CANN9.2.1 strict build; raw AKA0fatal/0warnings;3 pre-simulator reviews; one6-strip/ring4 model run;17,408 bytes, inputs, guards and untouched tail PASS | **Gate incomplete**:524/527tick pre-D1 delivery gaps lack a verified cause; no speedup acceptance |

The implementations under `slot02/src` and `slot07/src`, including their original CMake and wrapper policy files, are **byte-identical to the tested native snapshots**. `source_provenance.json` and `verify_package.py` verify21 source files. Complete frozen23-payload E8 fixtures and unchanged BF16/INT8 leaf headers retain their original hashes. Compressed active-core CCU/instruction/cache traces, output packets, manifests, ISA dumps and gate reports are retained under `evidence/`.

The Slot02 saved model packet bytes reconstruct1536 BF16 values at0ULP through the unchanged frozen E8 leaf on CPU. That check is packet/leaf evidence; it does not execute fused attention or establish raw BF16 encoding parity.

## Toolkit-free checks

```sh
cmake -S components/e8_packet_writers -B build-e8-packets -DCMAKE_BUILD_TYPE=Release
cmake --build build-e8-packets -j2
ctest --test-dir build-e8-packets --output-on-failure
```

These bounded tests exercise all65536 labels, all64 row placements, K/V isolation, scale payload bits, guards, packet-to-frozen-leaf fixture parity, the six-strip token model and snapshot hashes. SDK dependencies are absent from the default configuration. The unchanged CANN leaf headers are system includes so their device-only unroll pragmas do not become GNU C++ warnings; new host sources retain `-Wall -Wextra -Werror`.

## Optional isolated CAModel build targets

With `TQ_E8_CONFIGURE_CAMODEL=ON` and an explicit `ASCEND_HOME_PATH`, the component CMake exposes `e8_packet_scatter_camodel` and `e8_packet_batch_camodel`. They are excluded from default builds and use separate child build directories. They never launch a model automatically. Configure using CANN9.2+ environment and Release mode. Use the coordinator's shared build/CAModel phase guard plus checkpoint lock for **every device build or simulation**; never build while CAModel is running. This packaging change does not trigger another device run.

Original verified native roots are `/home/dmin/work/checkpoint2_20261123/slot02` and `slot07`. Their source/artifact manifests and live canonical dav3510 libpem mapping evidence are archived. The model library SHA256 is `1a6420fe8123e88a120f0bbfc76710968e6708599c77d80a7b1097aca06eed33`.

## Explicit remaining work

The launchers accept **encoded uint16 labels and FP32 scales**, not BF16 activations. Raw BF16 normalization/rotation/exact E8 encoding and the actual E8 fused-attention reader belong to separate integration work. Slot07 has a fixed six-strip minimal probe, not arbitrary production batching. Its performance gate must remain incomplete until the delivery gaps are explained or a compact-grinder source revision is reviewed and separately validated. This package neither modifies production `quant_mode` nor claims a raw quantizer, production attention parity or physical NPU timing.
