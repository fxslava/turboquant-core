# CANN 9.2.1 build guide

Target: `SOC_VERSION=Ascend950PR_9599`, architecture `dav-3510`.
Release image: `quay.io/ascend/vllm-ascend:v0.30.0-a5`.
Use the image's bundled `/usr/local/Ascend/cann-9.2.1`; no toolkit overlay.
Commands run from the repository root in Linux/WSL.

## Fast CPU build

This path has no CANN headers, libraries or compiler dependency:

```bash
cmake -B build -DTQ_BUILD_KERNELS=OFF -DTQ_BUILD_E8_KERNELS=OFF
cmake --build build --target tq_host_tests -j16
./build/bin/test_turboquant_production_host --list --mode all --stage all
```

For all 10 CPU CTests, first generate complete fixtures with a torch-enabled
Python, then run the host check target:

```bash
cmake --build build --target tq_generate_goldens
cmake --build build --target tq_check_host -j16
```

To reuse fixtures, configure `-DTQ_GOLDEN_DUMP_ROOT=/absolute/fixture/path`.
To select the Python used for generation, set `Python3_EXECUTABLE` at configure.
Without torch, use `ctest --test-dir build -L host -LE golden` for nine gates.

## WSL container wrapper

The tracked wrapper mounts source read-only at `/workspace/turboquant-core`
and persistent ext4 build storage at `/workspace/build` (the working directory).
Choose an empty build root to avoid reusing old compiler caches:

```bash
export TQ_BUILD_ROOT="$HOME/turboquant-build-cann921"
bash scripts/wsl_docker.sh cmake -S /workspace/turboquant-core -B build-host \
  -DTQ_BUILD_KERNELS=OFF -DTQ_BUILD_E8_KERNELS=OFF
bash scripts/wsl_docker.sh cmake --build build-host --target tq_host_tests -j16
bash scripts/wsl_docker.sh ./build-host/bin/test_turboquant_production_host \
  --list --mode all --stage all
```

The inner entrypoint unsets `ASCEND_TOOLKIT_LATEST_HOME`, `TOOLCHAIN_HOME`,
`CMAKE_PREFIX_PATH` and `SOC_VERSION`, then sources CANN 9.2.1 `set_env.sh`.
Pass the SoC to CMake explicitly. No physical devices are mounted by this
build wrapper. It uses the already-pulled image and disables network access.

## CAModel build

```bash
bash scripts/wsl_docker.sh cmake -S /workspace/turboquant-core -B build-sim \
  -DRUN_MODE=sim -DSOC_VERSION=Ascend950PR_9599 \
  -DTQ_BUILD_KERNELS=ON -DTQ_BUILD_E8_KERNELS=ON
bash scripts/wsl_docker.sh cmake --build build-sim --target tq_sim_tests tq_benchmarks -j8
bash scripts/wsl_docker.sh ./build-sim/bin/test_turboquant_production_sim \
  --list --mode all --stage all
```

`--list` enumerates supported rows and exits without launching kernels. Actual
CAModel execution is separate and follows the simulation-lane admission policy.
Build success and a catalog listing are not numerical/performance measurements.

## Physical-target builds, with or without E8

These commands compile physical NPU artifacts inside the image. The catalog
smoke test needs no physical device. Actual numerical/timing runs need the
matching Ascend950PR server and driver environment.

```bash
bash scripts/wsl_docker.sh cmake -S /workspace/turboquant-core -B build-npu \
  -DRUN_MODE=npu -DSOC_VERSION=Ascend950PR_9599 \
  -DTQ_BUILD_KERNELS=ON -DTQ_BUILD_E8_KERNELS=ON
bash scripts/wsl_docker.sh cmake --build build-npu --target tq_npu_tests tq_benchmarks -j8
bash scripts/wsl_docker.sh ./build-npu/bin/test_turboquant_production_npu \
  --list --mode all --stage all

bash scripts/wsl_docker.sh cmake -S /workspace/turboquant-core -B build-npu-noe8 \
  -DRUN_MODE=npu -DSOC_VERSION=Ascend950PR_9599 \
  -DTQ_BUILD_KERNELS=ON -DTQ_BUILD_E8_KERNELS=OFF
bash scripts/wsl_docker.sh cmake --build build-npu-noe8 --target tq_benchmarks \
  test_turboquant_production_npu -j8
bash scripts/wsl_docker.sh ./build-npu-noe8/bin/test_turboquant_production_npu \
  --list --mode all --stage all
```

The production test and benchmark exist in both configurations. With E8 off,
E8 device rows report `UNSUPPORTED reason=E8 kernels disabled at build time`.
Native LM4 rows remain available. Native LM3 and raw E8 device ingress remain
unsupported regardless of the E8 build option.

For a native Linux server, run the same CMake commands directly after:

```bash
unset ASCEND_TOOLKIT_LATEST_HOME TOOLCHAIN_HOME CMAKE_PREFIX_PATH SOC_VERSION
source /usr/local/Ascend/cann-9.2.1/set_env.sh
```

Use `-S .` instead of the container source path, and add
`-DASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.1` when configuring.

## Outputs and targets

All executables are in `<build>/bin/`; shared libraries are in `<build>/lib/`.

| Target | Result |
| --- | --- |
| `tq_host_tests` | 10 CPU contract/production executables |
| `tq_check_host` | Builds and runs host CTests |
| `tq_sim_kernels` / `tq_npu_kernels` | Selected-runtime device libraries only |
| `tq_sim_tests` / `tq_npu_tests` | Selected-runtime correctness runners |
| `tq_tests` | Host and configured device test executables |
| `tq_benchmarks` | Production and Qwen benchmark executables |
| `tq_kernels` | All configured device libraries |
| `tq_help` | Active configuration and target help |

Examples: `build-host/bin/test_contract_gates`,
`build-sim/bin/test_turboquant_production_sim`,
`build-npu/bin/test_turboquant_production_npu`,
`build-npu/bin/bench_turboquant_production_npu`, and
`build-npu/bin/bench_qwen35_decode`.
The Python launcher searches `bin/` first and still accepts historical
`tests/` and `benchmarks/` archive layouts.

## Architecture and warning policy

Root CMake owns the cached SoC and forces `ASCEND_SOC_VERSION` to match it.
The release image's legacy `ascendc.cmake` selects `dav-c310` for 950-class
products, even with those variables set. This project therefore uses the
bundled `asc-config.cmake` ASC language interface, mapping the supported SoC
to the SDK's `CMAKE_ASC_ARCHITECTURES=dav-3510`.

Generated `flags.make` files contain the supported driver flag
`--npu-arch=dav-3510`. The legacy spelling `--cce-aicore-arch=dav-3510` is not
the ASC driver's generated flag. No manual `--cce-aicore-arch` or `__DAV_*__`
override is applied by this project. Inspect generated flags and driver
commands rather than accepting a configured SoC string alone.

`tq_add_ascend_kernel` owns kernel registration. Device warnings remain
warnings, including CANN internal-header warnings. Host launch code uses
`-Wall -Wextra -Werror`; exact E8 floating-point options are target-local.
The SDK's language/runtime setup is isolated in a child directory so host
contracts do not inherit CANN link libraries. Host-only adapters generate ASC
launch stubs for the unchanged E8/canary device sources.

See [test execution](../tests/README.md) and
[performance measurement](../benchmarks/README.md) for numerical gates and timing.
