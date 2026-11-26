# CANN 9.2.1 build migration verification

All four requested configurations pass compilation and non-launching catalog checks
inside `quay.io/ascend/vllm-ascend:v0.30.0-a5`, image `sha256:4f78ac408902e0f63baf2294da86bfde62211788492cffb8f54026d9238214bc`.
Toolkit: `/usr/local/Ascend/cann-9.2.1`; SoC: `Ascend950PR_9599`.
No physical NPU or CAModel workload was executed.

## Results

| Configuration | Exit code | READY rows | UNSUPPORTED rows |
| --- | ---: | ---: | ---: |
| host | 0 | 78 | 138 |
| sim | 0 | 78 | 138 |
| npu | 0 | 78 | 138 |
| npu-noe8 | 0 | 42 | 174 |

Every catalog has 216 rows. Host `device=READY` describes the general device
geometry contract; it is not a physical execution result. Sim/NPU catalogs use
`status=READY`. E8-off retains 42 native LM4 rows; all E8 device rows explicitly
report `E8 kernels disabled at build time`.

- Full CPU suite: **10/10 CTests passed**, including freshly generated complete goldens.
- Launcher tests: **5 passed**; bandwidth/accuracy metric controls: **3 passed**.
- The existing plugin build/sync regression suite also passed **14 tests**.
- Source sync passed: **36 shared C/C++ files plus `reader_layout.aka`** unchanged.
- All **81 unique build input hashes** match the final workspace source.
- Generated kernel flags select `dav-3510`; device groups exclude `-Werror`,
  host groups retain `-Wall -Wextra -Werror`, and host runners retain `-O3`.

The requested literal `--cce-aicore-arch=dav-3510` check is replaced by the
supported ASC driver flag `--npu-arch=dav-3510`. The image's legacy
`ascendc.cmake` maps 950 products to `dav-c310` and does not consume
`ASCEND_SOC_VERSION`; setting that variable alone cannot repair its mapping.
The project uses the bundled `asc-config.cmake`/ASC language API instead.
No architecture flag or `__DAV_*__` macro is injected manually. The target
architecture passes; the requested legacy flag spelling is not emitted.
- E8-off binaries have no E8 reader/dequantizer dynamic dependencies.

`VERIFICATION.json` contains generated flags, binary hashes, paths and catalog
counts. Raw config/build/catalog/CTest/unit outputs are alongside this report.

## SDK interface adjustment

The image's legacy `ascendc.cmake` maps 950-class SoCs to `dav-c310` and does
not consume `ASCEND_SOC_VERSION`. Setting that cache variable alone cannot
produce the requested legacy flag. Root CMake still owns and synchronizes both
SoC cache values, but device registration uses the bundled `asc-config.cmake`
ASC language interface. The SoC maps to `CMAKE_ASC_ARCHITECTURES=dav-3510`.

The supported generated flag is **`--npu-arch=dav-3510`**, rather than the
requested literal `--cce-aicore-arch=dav-3510`. The architecture-intent gate
passes; that exact legacy spelling is not emitted. No manual architecture
flag or `__DAV_*__` macro is injected. The installed SDK modules were copied
for inspection under the external evidence directory's `sdk-cmake/`.

The SDK's framework setup also clears CXX Release/Debug flags and omits the
ASC SYSTEM include marker. The integration restores the caller's CXX flags
and marks CANN includes as system headers. Host options are grouped to prevent
CMake deduplication across host/device sections. Four small host adapters emit
ASC launch glue while leaving canonical device sources unchanged. Their warning
suppression covers only compiler-injected parameters in the empty host stubs.

## Configuration migration and cleanup

Updated **116 active WSL launcher/configuration/guide files**,
the registry `C:/Ascend/ascend-wsl-registry.json`, and the core/native-WSL Kimi
build guide. `WSL_MIGRATION.json` lists exact WSL paths and before/after SHA256.
All migrated shell/Python/JSON files pass syntax/parse checks; no stale image
tag, old image ID or `/opt/e8/cann` overlay remains in those active files.
Startup wrappers reset the four requested stale variables before sourcing
bundled CANN 9.2.1. No old simulation approval was renewed.

Frozen snapshots, approval/session records, old prototype guides and report
evidence retain their original environment references. The retained-reference
inventory is external at `wsl-old-image-refs.json`. Originals of modified WSL
files and the registry are backed up there under `backups/`.

Removed the unused legacy `CannWrapperWarnings.cmake`,
`CannDeviceWrapperTargets.cmake`, and `turboquant_sim_binary_lifetime.cpp`.
CPU and device registration are separate, shared helpers own kernel/runner
setup, production runners exist with E8 on or off, and normal binaries/libraries
use `bin/` and `lib/`. The optional vendored-decode probe retains its isolated
`probe/lib` output to avoid overwriting the product decode library.

## Commands and build locations

Every command below ran through the tracked `scripts/wsl_docker.sh` wrapper,
which mounts source read-only and uses ext4 build storage. All three roots were
absent before the first configure. Physical configurations ran in independent
directories; the sim build finished before its final rebuild and smoke checks.

```bash
# TQ_BUILD_ROOT=/home/dmin/tq-release-final2-20261126
cmake -S /workspace/turboquant-core -B build-host -DTQ_BUILD_KERNELS=OFF -DTQ_BUILD_E8_KERNELS=OFF
cmake --build build-host --target tq_host_tests -j16
./build-host/bin/test_turboquant_production_host --list --mode all --stage all
cmake --build build-host --target tq_generate_goldens
cmake --build build-host --target tq_check_host -j16
cmake -S /workspace/turboquant-core -B build-sim -DRUN_MODE=sim -DSOC_VERSION=Ascend950PR_9599 -DTQ_BUILD_KERNELS=ON -DTQ_BUILD_E8_KERNELS=ON
cmake --build build-sim --target tq_sim_tests tq_benchmarks -j8
./build-sim/bin/test_turboquant_production_sim --list --mode all --stage all

# TQ_BUILD_ROOT=/home/dmin/tq-release-npu-20261126
cmake -S /workspace/turboquant-core -B build-npu -DRUN_MODE=npu -DSOC_VERSION=Ascend950PR_9599 -DTQ_BUILD_KERNELS=ON -DTQ_BUILD_E8_KERNELS=ON
cmake --build build-npu --target tq_npu_tests tq_benchmarks -j4
./build-npu/bin/test_turboquant_production_npu --list --mode all --stage all

# TQ_BUILD_ROOT=/home/dmin/tq-release-noe8-20261126
cmake -S /workspace/turboquant-core -B build-noe8 -DRUN_MODE=npu -DSOC_VERSION=Ascend950PR_9599 -DTQ_BUILD_KERNELS=ON -DTQ_BUILD_E8_KERNELS=OFF
cmake --build build-noe8 --target tq_benchmarks test_turboquant_production_npu -j4
./build-noe8/bin/test_turboquant_production_npu --list --mode all --stage all
```

The sim tests and benchmarks were rebuilt after the final host-option changes;
the full CPU suite, Python checks and launcher smoke check were then rerun.
Earlier failed/interrupted attempts remain external and are not promotion gates.
External evidence root:
`C:/Ascend/turboquant-integration-work/cann921-build-refactor-20261126`.

## Exact source changes and status

`MODIFIED_SOURCE_FILES.txt` lists the 30 source/configuration/documentation
paths changed by this refactor; report artifacts are recorded in `SHA256SUMS`.
The existing main worktree was clean before promotion. Pre-existing untracked
evidence in the canonical checkout was retained and was not swept into this
commit. Post-commit/main status and hashes are recorded externally in
`final-status.log`.
