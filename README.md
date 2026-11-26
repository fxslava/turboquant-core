# TurboQuant core

Standalone Ascend950PR / dav-3510 kernels, host contracts, correctness tests
and performance harnesses. Shared product sources live under `src/turboquant`
and are mirrored into vLLM Ascend through `SOURCE_SYNC.json`.

- [Container and CMake build guide](docs/BUILD.md): release image, clean environment,
  host/sim/NPU builds and exact binary paths.
- [Build and test guide](tests/README.md): CPU fixtures, CMake target groups,
  host tests, physical NPU tests and bounded CAModel checks.
- [Performance guide](benchmarks/README.md): the production LM4/E8 benchmark,
  Qwen3.5 decode sweep, staged PyTorch comparison, profiling and metrics.
- [E8 packet/decoder contract](tests/cpu/e8/README.md): sealed layout and
  independent CPU oracle provenance.

## Quick CPU validation

Use Linux or WSL with CMake 3.16+, a C++17 compiler and Python with CPU torch.
No CANN installation or NPU is needed for this path.

```bash
cmake -S . -B build-host -DCMAKE_BUILD_TYPE=Release \
  -DTQ_BUILD_KERNELS=OFF -DTQ_BUILD_E8_KERNELS=OFF
cmake --build build-host --target tq_generate_goldens
cmake --build build-host --target tq_check_host -j8
```

Fixtures are generated under `build-host/golden-dumps`; override
`TQ_GOLDEN_DUMP_ROOT` to reuse a complete fixture set. Select a torch-enabled
interpreter with `-DPython3_EXECUTABLE=/absolute/path/to/python`. Without torch,
build `tq_host_tests` and run `ctest --test-dir build-host -L host -LE golden`
for the nine tests that do not consume generated dumps.

## Quick physical benchmark build

On an Ascend950PR server with the matching driver and CANN 9.2.1:

```bash
unset ASCEND_TOOLKIT_LATEST_HOME TOOLCHAIN_HOME CMAKE_PREFIX_PATH SOC_VERSION
source /usr/local/Ascend/cann-9.2.1/set_env.sh
cmake -S . -B build-npu -DRUN_MODE=npu -DSOC_VERSION=Ascend950PR_9599 \
  -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.1 \
  -DTQ_BUILD_KERNELS=ON -DTQ_BUILD_E8_KERNELS=ON
cmake --build build-npu --target tq_help
cmake --build build-npu --target tq_benchmarks test_turboquant_production_npu -j6
./build-npu/bin/test_turboquant_production_npu --device 0 \
  --mode lm4 --stage decode --profile qwen35-gqa6 --block-size 128
./build-npu/bin/bench_turboquant_production_npu --device 0 \
  --mode lm4 --stage decode --profile qwen35-gqa6 --block-size 128 \
  --warmup 20 --iterations 100
```

There are two benchmark executables: `bench_turboquant_production_npu` for
profile-selected native ingress/dequant/decode measurements, and
`bench_qwen35_decode` for the fixed Qwen-shaped decode/Split-K sweep. New
builds place both in `<build>/bin`; old server archives retain their
original `<build>/tests/bench_*` paths. The Python benchmark wrapper accepts
both layouts.

## Build organization

`tq_host_tests` builds CPU tests. `tq_tests` additionally builds the selected
runtime's correctness runners. `tq_kernels` builds configured device libraries,
and `tq_benchmarks` builds device benchmarks. `tq_help` prints the available
groups and active configuration. Device compilation is explicitly enabled by
`TQ_BUILD_KERNELS`; E8 follows that option in a fresh cache and can be disabled
independently. Device CTest registration is a separate opt-in through
`TQ_REGISTER_DEVICE_TESTS=ON`. Building a group never launches device tests.

Kernel registration uses the CANN 9.2.1 ASC language interface. Host sources
use strict warnings; device SDK warnings remain warnings. CPU tests and E8
device targets are declared in separate CMake modules. Use separate
host/sim/NPU build directories. See the build guide for the SDK architecture
flag and clean container commands.

## Product scope and verification

Native LM4 supports the current scatter/decoder contracts. The sealed E8
dequantizer and fused reader consume CPU-prepared packets. Raw device E8
encoding remains gated; native LM3 and native E8 Split-K are unsupported.
The performance guide lists shape limits and what each timing includes.

The latest promotion audit records CPU, syntax and analyzer checks in
[`reports/promotion_20261126`](reports/promotion_20261126/REPORT.md). These
checks do not constitute physical NPU performance measurements. Historical
CAModel and candidate receipts remain under `reports/`; they are verification
evidence and are not compiled as product inputs.

Check mirrored sources from the vLLM Ascend checkout with:

```bash
python scripts/check_turboquant_sync.py --core /absolute/path/to/turboquant-core
```

`src/turboquant` holds the canonical packet ABI, native ingress/decode kernels
and E8 reader. `tests/common` contains host fixtures/oracles, `tests/cpu`
contains toolkit-free gates, `tests/npu` contains device runners and canaries,
and `benchmarks` contains the native and staged performance harnesses.
