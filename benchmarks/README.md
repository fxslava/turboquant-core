# Measuring TurboQuant performance

Use `bench_turboquant_production_npu` for the native LM4/E8 matrix: ingress,
packet dequantization and decode wherever the selected kernel supports the shape.
`bench_qwen35_decode` is a separate fixed Qwen3.5-shaped decode/Split-K sweep.
Both use synthetic KV/query fixtures. They measure operator latency, not model
loading, full-model generation or serving tokens/s.

## Build and locate the benchmarks

Run from the repository root on Linux with an Ascend950PR, its driver and
CANN 9.2.1. WSL is suitable for the host build and compile-only checks.

```bash
unset ASCEND_TOOLKIT_LATEST_HOME TOOLCHAIN_HOME CMAKE_PREFIX_PATH SOC_VERSION
source /usr/local/Ascend/cann-9.2.1/set_env.sh
cmake -S . -B build-npu -DCMAKE_BUILD_TYPE=Release -DRUN_MODE=npu \
  -DSOC_VERSION=Ascend950PR_9599 \
  -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.1 \
  -DTQ_BUILD_KERNELS=ON -DTQ_BUILD_E8_KERNELS=ON
cmake --build build-npu --target tq_help
cmake --build build-npu --target tq_benchmarks test_turboquant_production_npu -j6
```

| Binary | Purpose |
| --- | --- |
| `build-npu/bin/bench_turboquant_production_npu` | Configurable native production timing |
| `build-npu/bin/test_turboquant_production_npu` | Same production matrix with numerical verification forced on |
| `build-npu/bin/bench_qwen35_decode` | Fixed Qwen3.5 decode/Split-K sweep |
| `build-npu/bin/test_turboquant_production_host` | CPU verification and shape catalog; build with `tq_host_tests` |

All executables are in `bin/`. Build targets compile artifacts without executing them. In a fresh
device configure E8 follows `TQ_BUILD_KERNELS`; an existing cache may retain
`TQ_BUILD_E8_KERNELS=OFF`. The production pair is always built; E8 rows are disabled when the option is OFF.
See [the test/build guide](../tests/README.md) for the complete target map.

## Verify first, then time the same case

```bash
./build-npu/bin/test_turboquant_production_npu --list --mode all --stage decode
./build-npu/bin/test_turboquant_production_npu --device 0 \
  --mode lm4 --stage decode --profile qwen35-gqa6 --block-size 128
./build-npu/bin/test_turboquant_production_npu --device 0 \
  --mode e8 --stage decode --profile h1 --block-size 64
```

`--list` prints geometry and `READY`/`UNSUPPORTED` without initializing ACL.
The accuracy executable checks numerical outputs, packed bytes/scales and guards.
The timing executable reports `MEASURED`; use its `--verify` switch or the
accuracy executable for numerical verification. Guard checks still run in timing
mode. A selection containing only unsupported physical cases exits nonzero.

```bash
mkdir -p results
set -o pipefail
./build-npu/bin/bench_turboquant_production_npu --device 0 \
  --mode lm4 --stage decode --profile qwen35-gqa6 --block-size 128 \
  --warmup 20 --iterations 100 | tee results/lm4-decode.log
./build-npu/bin/bench_turboquant_production_npu --device 0 \
  --mode lm4 --stage ingress --profile qwen35-gqa4 --block-size 128 \
  --warmup 20 --iterations 100 | tee results/lm4-ingress.log
./build-npu/bin/bench_turboquant_production_npu --device 0 \
  --mode e8 --stage dequant --profile h1 --block-size 64 \
  --warmup 20 --iterations 100 | tee results/e8-dequant.log
./build-npu/bin/bench_turboquant_production_npu --device 0 \
  --mode e8 --stage decode --profile h1 --block-size 64 \
  --warmup 20 --iterations 100 | tee results/e8-decode.log
```

`--iterations` accepts 1–512. Omitting `--block-size` selects both 64 and 128.
`--profile all --mode all --stage all` enumerates the complete matrix, including
unsupported rows. `--list` gives all 12 profile names and exact geometry.
GLM/DeepSeek profile labels describe workload shapes, not model accuracy.

## Read and compare the measurements

ACL stream events after warmup report `p50_us`, `p95_us`, `mean_us`,
`effective_bytes` and `effective_GB_s`. Allocation, input upload and CPU checks
are outside the timed launch. E8 decode serializes a reader launch for each
group of up to six query heads sharing a KV head; its case timing includes all
those launches.

Effective GB/s is `effective_bytes / (mean_us * 1000)` in decimal units. Decode
counts unique live K/V tiles and scales, including final-tile row padding. It
excludes query/output/workspace, E8 witness traffic and repeated reads caused
by padded head groups. Ingress counts BF16 reads and packed/scale writes.
Native E8 dequantization counts labels/scales read and BF16/INT8 outputs written.
This is a logical-byte rate, not measured HBM bandwidth.

For A/B comparisons, hold profile, block size, codec, stage, device, warmup and
iterations constant. Avoid competing device jobs. Record the commit, CANN and
compiler versions, SoC and device power/clock configuration. Repeat complete
runs and compare p50 and p95 across them. Record profiler runs separately
because profiling changes overhead:

```bash
msprof --application="./build-npu/bin/bench_turboquant_production_npu --device 0 --mode lm4 --stage decode --profile qwen35-gqa6 --block-size 128 --warmup 20 --iterations 100"
```

## Kernel support

| Codec | Native ingress | Native dequantization | Native decode |
| --- | --- | --- | --- |
| LM4 | Scatter writer, including ragged writes | No standalone kernel | At most 4 KV heads and 8 splits, subject to launch validation |
| LM3 | Unsupported | Unsupported | Unsupported |
| E8 | Raw device encoder remains gated | Sealed packet component | Prepared packets; even 4–256 tiles per sequence; no Split-K merge |

E8 uses D256, 4,352-byte packets per 64-token K or V plane and host-prepared
labels/scales. Native ingress writes up to two pages per sequence.
`--list` is authoritative for a particular profile/shape.

## Staged PyTorch comparison and CPU checks

The staged reference includes unpacking, gather, FP32 attention and Split-K
merge on one stream. It is a different workload from the fused native kernel.
LM4 uses 128-byte packed rows, LM3 96-byte rows, and E8 sealed packets.

```bash
cmake --build build-npu --target test_turboquant_production_host -j6
python3 benchmarks/benchmark_turboquant_production.py --backend torch-npu \
  --build-dir build-npu --device 0 --mode all --stage decode \
  --profile qwen35-gqa6 --block-size 128 --verify
python3 benchmarks/benchmark_turboquant_production.py --backend torch-npu \
  --build-dir build-npu --device 0 --mode all --stage decode \
  --profile qwen35-gqa6 --block-size 128 --warmup 20 --iterations 100
```

PyTorch NPU events time this staged implementation. `--backend torch-cpu`
runs it without a device, with host-clock timing. `--backend host --list`
prints the CPU shape catalog; `--backend host` verifies it. Neither host mode
measures NPU performance. Staged ingress is unsupported. If the host catalog
was built separately, provide `--host-build-dir /absolute/path/to/build-host`.
The wrapper also dispatches native `--backend npu` and `--backend npu-test`.

For a small CPU check after building `tq_host_tests` in `build-host`:

```bash
python3 benchmarks/benchmark_turboquant_production.py --backend torch-cpu \
  --build-dir build-host --mode all --stage dequant --profile h1 \
  --block-size 64 --verify
python3 tests/cpu/test_turboquant_benchmark_metrics.py
```

The vLLM Ascend wrapper under `benchmarks/scripts` accepts `--core-root` plus
the same options; pass an absolute `--build-dir` for the matching core build.

## Existing Qwen sweep and runner microbenchmarks

```bash
./build-npu/bin/bench_qwen35_decode --static
./build-npu/bin/bench_qwen35_decode --verify --device 0
./build-npu/bin/bench_qwen35_decode --run --device 0 > results/qwen35.csv
```

`--static` only enumerates plans. `--verify` checks short-context cases;
`--run` performs those checks and then the full fixed sweep. The sweep uses
10 warmups and 50 event samples per supported geometry and emits CSV with
min/p50/mean/p99 latency and effective KV bandwidth. `--block-size`,
`--batch-size` and `--split-k` filter verification, not a `--run` sweep.
Use the production benchmark for selectable timing profiles and sample counts.

The legacy correctness runners also support event microbenchmarks after their
own correctness checks, using fixed fixtures:

```bash
./build-npu/bin/test_live_write_npu --op both --perf 50
./build-npu/bin/test_live_write_nz_npu --op scatter --perf 50
./build-npu/bin/test_nz_ingress_canary_npu --perf 50
./build-npu/bin/test_fused_decode_e2e_npu --perf 50
```

Keep these rows separate from production-matrix measurements. CAModel builds
are for bounded correctness and trace investigations. The Qwen simulator binary
allows `--static` and `--canary`, and refuses a performance sweep. Simulator
ticks and host wall time are not physical NPU latency.

## Existing compiled archive

The historical `ascend950pr-cann921-x86_64.tar.gz` archive retains its original
`tests/bench_*` layout. Extract into a separate directory, verify `SHA256SUMS`
and source the matching CANN environment. Use its archived paths directly or
the Python launcher, which accepts both layouts. Rebuild from current source
for `bin/` paths; use executables and DSOs from the same build.
