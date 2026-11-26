# TurboQuant production validation

The production matrix uses synthetic D256 paged-cache fixtures with 64- and
128-token blocks. The Qwen3.5-27B, GLM-4-9B and DeepSeek-V4-Flash profile labels
describe workload shapes exercised with generated KV and query data. The CPU
gate covers 12 profiles at both block sizes, with 1, 4, 16 and 32 sequences,
2K through 128K contexts, ragged tails, GQA ratios 4, 6, 8 and 16, the H=1 case,
and Split-K values 1, 4, 8 and 16.

## Build

On Linux with the CANN 9.2.1 toolkit and Ascend950PR driver installed, run from
the turboquant-core checkout.

```bash
source /usr/local/Ascend/cann-9.2.1/set_env.sh
cmake -S . -B build-e8-production -DRUN_MODE=npu -DSOC_VERSION=Ascend950PR_9599 \
  -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.1 \
  -DTQ_BUILD_KERNELS=ON -DTQ_BUILD_E8_KERNELS=ON
cmake --build build-e8-production --target tq_npu_kernels test_turboquant_production_host -j6
./build-e8-production/tests/test_turboquant_production_host
./build-e8-production/tests/test_turboquant_production_host --list --stage decode
```

## Run on an Ascend950PR server

For the compiled Linux x86-64 archive, extract it into the build directory and
verify the file digests. Source the server's CANN 9.2.1 environment before
launching the physical binaries.

```bash
mkdir -p build-e8-production
tar -xzf ascend950pr-cann921-x86_64.tar.gz -C build-e8-production
(cd build-e8-production && sha256sum -c SHA256SUMS)
source /usr/local/Ascend/cann-9.2.1/set_env.sh
```

```bash
./build-e8-production/tests/test_turboquant_production_npu --device 0
./build-e8-production/tests/bench_turboquant_production_npu --device 0 \
  --mode all --stage decode --profile qwen35-gqa6 --warmup 10 --iterations 50
msprof --application="./build-e8-production/tests/bench_turboquant_production_npu --device 0 --mode e8 --stage decode --profile h1 --block-size 64 --warmup 10 --iterations 50"
python3 benchmarks/benchmark_turboquant_production.py --backend torch-npu \
  --build-dir build-e8-production --device 0 --mode all --stage decode --profile qwen35-gqa6 --verify
python3 benchmarks/benchmark_turboquant_production.py --backend torch-npu \
  --build-dir build-e8-production --device 0 --mode all --stage decode --profile qwen35-gqa6
```

From the vLLM Ascend checkout, use
`python3 benchmarks/scripts/benchmark_turboquant_production.py --core-root /path/to/turboquant-core`
with the same benchmark filters and an absolute `--build-dir` for the
synchronized build.

Use `--mode`, `--stage`, `--profile` and `--block-size` to select rows.
A native invocation with only unsupported selections fails before initializing
ACL. Native LM4 uses the shipping scatter writer and fused decoder, whose
decode contract supports up to four KV heads and eight splits. The native
ingress case writes up to two pages per sequence, with partial writes in the
ragged profiles. Native E8 applies the sealed 16-row dequantizer to both
planes of one block for every KV head. Its fused reader accepts 4–256 even
tiles. The harness submits one E8 reader launch per six query heads sharing
the same KV head, pads the final group and checks only live head outputs.
It serializes these launches on one stream. The E8 path uses host-packed
input and has no native Split-K merge. The matrix reports gated E8 device
ingress and unavailable native LM3 kernels as unsupported.

The staged PyTorch reference compares all three codecs through dequantization
and FP32 attention, using 128-byte LM4 rows, 96-byte LM3 rows and sealed
4,352-byte E8 packets. It combines page waves and Split-K partials on one
stream. PyTorch NPU events measure staged reference performance, including
the unpack, gather, matmul and merge operations.
The same reference supports `--backend torch-cpu` for device-free verification.
With CPU PyTorch installed, run
`python3 tests/cpu/test_turboquant_benchmark_metrics.py` from the source checkout
to check zero-vector and non-finite accuracy handling.

The native event timer reports p50, p95 and mean microseconds after warmup.
The decode GB/s numerator is the logical bytes of the unique live K and V
tiles, including scales and row padding of the final tile. It excludes query,
output, workspace, E8 witness traffic and rereads for padded E8 head groups.
Ingress counts BF16 input reads and logical packed-code/scale writes.
Standalone dequantization counts label and scale reads and BF16/INT8 writes
for the native E8 component. The staged dequantization reference counts one
block of compressed inputs and FP32 outputs. The GB/s field measures
effective bandwidth from that numerator and event latency.

The native accuracy binary compares BF16 context output to FP64 attention
and reports maximum BF16 ULP distance, RMSE and cosine similarity. It checks
every fused E8 K/V dequant witness and standalone E8 BF16 output against the
CPU golden. It validates representative packed labels and scales on both
planes and checks guard bands on every GM allocation. The LM4 writer check
compares every packed byte and scale, including unwritten cache regions.
