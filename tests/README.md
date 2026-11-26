# Building and running tests

Run from the repository root on Linux. On Windows use WSL for the GNU C++/CANN
build. Host tests need C++17 and CMake 3.16+. Complete golden fixtures also
need Python with CPU PyTorch. Native/device builds use CANN 9.2.1 targeting
Ascend950PR_9599 / dav-3510; physical tests need the matching Ascend server.

Commands using `ctest --test-dir` require CTest 3.20+. With 3.16–3.19,
run CTest inside the build directory instead, for example
`(cd build-host && ctest -L host --output-on-failure)`.

## Build targets

```bash
cmake -S . -B build-host -DTQ_BUILD_KERNELS=OFF -DTQ_BUILD_E8_KERNELS=OFF
cmake --build build-host --target tq_help
```

| Target | Builds or executes |
| --- | --- |
| `tq_host_tests` | All 11 CPU C++ executables |
| `tq_generate_goldens` | Generates complete fixtures with `Python3_EXECUTABLE` and CPU torch |
| `tq_check_host` | Builds host tests, then executes `ctest -L host` |
| `tq_tests` | Builds host tests and selected-runtime device tests, if configured |
| `tq_kernels` | Builds configured device libraries only |
| `tq_benchmarks` | Builds device benchmarks; available with `TQ_BUILD_KERNELS=ON` |
| `tq_npu_tests` / `tq_sim_tests` | Builds only the selected runtime's correctness runners |
| `tq_npu_benchmarks` / `tq_sim_benchmarks` | Builds only the selected runtime's benchmarks |
| `tq_e8_kernels` | Builds the sealed E8 reader and packet dequantizer |
| `tq_npu_kernels` / `tq_sim_kernels` | Selected-runtime device libraries only |
| `tq_npu_production` | Production accuracy/timing pair; E8 is optional |
| `tq_golden_sim` | Simulator golden-comparison runners |

Only `tq_check_host` runs tests; `tq_generate_goldens` runs fixture preparation.
Other groups build artifacts without launching them. Outputs are `<build>/bin` and `<build>/lib`. Keep host, sim and NPU build directories
separate. A device configuration's default SDK `ALL` build can compile expensive
device libraries; use explicit targets.

## Complete CPU suite

Select a Python with CPU torch installed; add
`-DPython3_EXECUTABLE=/absolute/path/to/python` to choose it explicitly.
`tq_generate_goldens` exists when CMake finds a Python interpreter.

```bash
cmake -S . -B build-host -DCMAKE_BUILD_TYPE=Release \
  -DTQ_BUILD_KERNELS=OFF -DTQ_BUILD_E8_KERNELS=OFF
cmake --build build-host --target tq_generate_goldens
cmake --build build-host --target tq_host_tests -j8
ctest --test-dir build-host -L host --output-on-failure
# After fixture preparation, this builds and runs the same suite:
cmake --build build-host --target tq_check_host -j8
```

Fixtures default to `<build>/golden-dumps`. To reuse a complete existing set,
configure `-DTQ_GOLDEN_DUMP_ROOT=/absolute/path/to/golden-dumps` and skip generation.
Host and device golden tests share that path. The tracked `golden/dumps` has
only the original synthetic subset; it cannot satisfy the complete oracle gate.
Missing `manifest.txt` errors mean fixture preparation is incomplete.

Without torch or complete fixtures, build the C++ tests and run the ten gates
that do not read generated dumps:

```bash
cmake --build build-host --target tq_host_tests -j8
ctest --test-dir build-host -L host -LE golden --output-on-failure
```

| CPU test | Coverage |
| --- | --- |
| `contract_gates` | Geometry, layouts and scheduling contracts |
| `layout_bridge` | Packed writer/reader addresses |
| `telemetry_dump` | Telemetry ABI and host parsing |
| `qwen35_decode` | Qwen-shaped decode planning |
| `decode_reduce` | Split-K/reduction coverage and padding |
| `e8_splitk` | Page-aligned E8 splits, FP32 merge, odd/empty tails and launch validation |
| `qwen35_telemetry` | Telemetry isolation from Split-K statistics |
| `golden_decode_oracle` | C++ oracle against all eight generated decode cases |
| `turboquant_production_host` | 216 profile/format/block/stage validation rows |
| `e8_decode_cpu_check` | Exhaustive sealed E8 decoder arithmetic |
| `e8_layout_check` | Packet and UB mapping |

List tests with `ctest --test-dir build-host -N`; select one with `-R '^NAME$'`.
Additional Python checks are independent of the C++ build:

```bash
python3 tests/cpu/test_benchmark_cli.py
python3 tests/cpu/test_turboquant_benchmark_metrics.py
python3 tests/cpu/check_semantic_goldens.py
```

The last check reproduces tracked fixture bytes; use the pinned CPU Torch
2.10 / NumPy 1.26.4 environment for that bit-exact check. Different Torch
versions can change floating-point reference bytes.

## Physical NPU suite

```bash
unset ASCEND_TOOLKIT_LATEST_HOME TOOLCHAIN_HOME CMAKE_PREFIX_PATH SOC_VERSION
source /usr/local/Ascend/cann-9.2.1/set_env.sh
cmake -S . -B build-npu -DRUN_MODE=npu -DSOC_VERSION=Ascend950PR_9599 \
  -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.1 \
  -DTQ_BUILD_KERNELS=ON -DTQ_BUILD_E8_KERNELS=ON \
  -DTQ_REGISTER_DEVICE_TESTS=ON
cmake --build build-npu --target tq_generate_goldens
cmake --build build-npu --target tq_npu_tests -j6
ctest --test-dir build-npu -N -L npu
ctest --test-dir build-npu -L npu --output-on-failure
```

Device CTest registration is **off by default**, even in a device build.
`TQ_REGISTER_DEVICE_TESTS=ON` opts in; it requires `TQ_BUILD_KERNELS=ON`.
Device tests are serialized and the legacy runners use device 0. CTest does
not compile executables. Building a device library does not verify its numerical
behavior; executing its correctness tests on the server does.

| Test / executable | Coverage |
| --- | --- |
| `npu_live_write` / `test_live_write_npu` | Strip-layout batch and scatter writers |
| `npu_live_write_nz` / `test_live_write_nz_npu` | NZ scatter writer |
| `npu_nz_ingress` / `test_nz_ingress_canary_npu` | Batch writer's NZ layout |
| `npu_fused_decode` / `test_fused_decode_e2e_npu` | Decode ladder and Split-K |
| `npu_dirty_tail`, `npu_dirty_blocktable` | Finite spikes, empty workers and invalid block-table entries |
| `npu_golden_*` / `test_golden_*_npu` | Prefill strip/NZ, scatter and decode versus generated dumps |
| `npu_production` / `test_turboquant_production_npu` | Native-supported matrix rows, E8 witnesses and guards |

For example, run only the tail regression with
`ctest --test-dir build-npu -R '^npu_dirty_tail$' --output-on-failure`.
The production test supports E8 enabled or disabled. Unsupported rows are explicit;
appearing in the host catalog does not verify a physical row. Prepared E8
packet tests do not verify the gated raw device encoder.

## CAModel build and bounded tests

```bash
unset ASCEND_TOOLKIT_LATEST_HOME TOOLCHAIN_HOME CMAKE_PREFIX_PATH SOC_VERSION
source /usr/local/Ascend/cann-9.2.1/set_env.sh
cmake -S . -B build-sim -DRUN_MODE=sim -DSOC_VERSION=Ascend950PR_9599 \
  -DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.1 \
  -DTQ_BUILD_KERNELS=ON -DTQ_BUILD_E8_KERNELS=OFF \
  -DTQ_REGISTER_DEVICE_TESTS=ON \
  -DTQ_GOLDEN_DUMP_ROOT="$PWD/build-host/golden-dumps"
cmake --build build-sim --target tq_sim_tests -j6
ctest --test-dir build-sim -N -L sim
```

Execute model workloads individually under your simulator admission/exclusive-lane
procedure. For an admitted bounded decode canary:

```bash
ctest --test-dir build-sim -R '^sim_qwen35_decode$' --output-on-failure --timeout 900
```

Simulator CTests include dirty tail/block-table gates, small writer probes,
the decode smoke ladder, Qwen reduction/tail canaries and golden comparisons.
The registered golden decode selects `decode_ctx128`; choose other `--case`
values explicitly when their model budget is admitted. CTest provides the
simulator loader environment and separate artifact directories under
`<build>/device-test-artifacts/<test-name>`.
A timeout is a failed/incomplete test. Syntax, host parity and balanced event
counts do not prove runtime liveness or performance. No model run is needed
for the CPU suite.

## Troubleshooting

- No `tq_benchmarks`: reconfigure a device directory with `TQ_BUILD_KERNELS=ON`.
  `tq_help` describes the active configuration.
- Missing production targets: use a fresh build directory with the current source.
  The production pair is available with E8 on or off.
- CTest cannot find a binary: first build `tq_host_tests`, `tq_npu_tests` or
  `tq_sim_tests` for the intended label.
- No device CTests listed: opt in with `TQ_REGISTER_DEVICE_TESTS=ON`.
- Missing golden manifests: run `tq_generate_goldens` or select a complete
  `TQ_GOLDEN_DUMP_ROOT`.
- Driver-symbol link errors: install the Ascend driver libraries as well as
  toolkit headers and source the matching CANN environment.
- Shared-library load errors: use a matching build's executable and `lib/`
  tree, CANN environment, SoC and runtime.

See [the performance guide](../benchmarks/README.md) for timing commands,
supported shapes, bandwidth definitions and A/B comparison procedure.
