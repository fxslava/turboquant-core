---
name: turboquant-sim-build
description: Build turboquant-core host gates and Ascend950PR dav-3510 device targets inside the v0.30.0-a5 release image with its bundled CANN 9.2.1.
---

# WSL build and simulator configuration

Use `quay.io/ascend/vllm-ascend:v0.30.0-a5` with the bundled
`/usr/local/Ascend/cann-9.2.1`. No host-toolkit overlay is needed.
Canonical checkout: `C:/Ascend/turboquant-core`, mounted in WSL at
`/mnt/c/Ascend/turboquant-core`.

Use the tracked `scripts/wsl_docker.sh` wrapper. Its inner entrypoint resets
`ASCEND_TOOLKIT_LATEST_HOME`, `TOOLCHAIN_HOME`, `CMAKE_PREFIX_PATH` and
`SOC_VERSION`, then sources the bundled CANN 9.2.1 environment.
The source is mounted read-only at `/workspace/turboquant-core`; ext4 build
storage is mounted at `/workspace/build`. Override `TQ_BUILD_ROOT` for an
isolated build. Run commands through `bash scripts/wsl_docker.sh` from WSL.

```bash
bash scripts/wsl_docker.sh cmake -S /workspace/turboquant-core -B build-sim \
  -DRUN_MODE=sim -DSOC_VERSION=Ascend950PR_9599 \
  -DTQ_BUILD_KERNELS=ON -DTQ_BUILD_E8_KERNELS=ON
bash scripts/wsl_docker.sh cmake --build build-sim --target tq_sim_tests -j8
bash scripts/wsl_docker.sh ./build-sim/bin/test_turboquant_production_sim \
  --list --mode all --stage all
```

All executables are in `bin/`; libraries are in `lib/`. `tq_sim_kernels`
builds libraries, while `tq_sim_tests` builds correctness runners. The
production runner also exists when E8 device libraries are disabled and
reports those codec rows as unsupported.

Root CMake owns `SOC_VERSION` and synchronizes `ASCEND_SOC_VERSION`. This
image's legacy `ascendc.cmake` selects the old c310 backend. Use the bundled
ASC language integration selected by the project, which derives dav-3510
from the SoC. Do not inject architecture flags, `__DAV_*__` macros, preload
simulator DSOs, reuse old CMake caches or purge arbitrary object files.

The `--list` gate initializes no device and is safe for build verification.
Actual CAModel workloads require the environment's admission and exclusive
simulation-lane protocol. Preserve exact permits, hashes and historical
receipts; changing a launcher does not renew an old execution permit.
Do not run a model workload merely to check an image tag or build migration.

See `docs/BUILD.md`, `tests/README.md` and `benchmarks/README.md` in the
canonical checkout for full commands and measurement limits.