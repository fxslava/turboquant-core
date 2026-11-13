---
name: turboquant-sim-build
description: Build turboquant-core kernels and run CAModel sim tests (test_fused_decode_e2e_sim --probe etc.) in the WSL docker CANN 9.2.0-beta.2 image, including the 9P build corruption workaround and the exact working runtime environment. Use when building BiSheng/AscendC device legs or running simulator tests for this repo on the Windows host without an NPU.
---

# turboquant-core build + CAModel run recipe (WSL docker, no NPU)

Target: Ascend 950PR / dav-3510, CANN `9.2.0-beta.2`, image
`ascendai/cann:9.2.0-beta.2-910b-ubuntu22.04-py3.12` (present in the WSL docker).
The repo lives at `F:\turboquant-core` = `/mnt/f/turboquant-core` in WSL; the
existing `build/` was configured with the repo mounted at `/workspace`, so keep
that mount point or the CMake cache breaks.

## Build

From the Windows host (Git Bash):

```bash
wsl -- bash -lc 'docker run --rm -v /mnt/f/turboquant-core:/workspace -w /workspace \
  ascendai/cann:9.2.0-beta.2-910b-ubuntu22.04-py3.12 \
  bash -lc "cmake --build build --target tq_decode_sim -j12"'
```

Fresh configure (only if `build/` is gone): `cmake -B build -DRUN_MODE=sim
-DTQ_BUILD_KERNELS=ON` inside the same container. `ASCEND_HOME_PATH` defaults to
`/usr/local/Ascend/cann-9.2.0-beta.2`, which this image has.

### Pitfall: the aggregate `tq_sim_kernels` target is flaky over /mnt/f

CANN's `ascendc_library` expands each kernel into nested ExternalProjects. The
`merge_aic_obj_text` / `merge_aiv_obj_text` steps run
`ld.lld -m aicorelinux -Ttext=0 file.o -o file.o` **in place**. Over the 9P
mount this intermittently truncates the object; the next build then dies with
`ld.lld: error: ...cpp.o: unknown file type`. It is NOT a source error.

Workarounds:
- Build the one target you need (`tq_decode_sim`, `tq_prefill_sim`,
  `tq_scatter_sim`) instead of the aggregate, retry on failure, prefer `-j1`
  when it misbehaves.
- If a preprocess `.o` is already corrupted, delete it
  (`build/tq_<k>_sim_preprocess-prefix/src/*-build/CMakeFiles/{aic,aiv}_obj.dir/.../*.o`)
  and rebuild — the inner make regenerates it.
- Runner binaries can be (re)linked without re-triggering the kernel
  ExternalProjects: `gmake -C build test_fused_decode_e2e_sim/fast -j4`
  (the `/fast` target skips dependency re-checks; needs `build/lib/libtq_decode_sim.so`
  and `build/libturboquant_host.a`, both already present).
- **The `/fast` route also works for NEW runner targets** added after the last
  full configure: re-run `cmake -B build -DRUN_MODE=sim -DTQ_BUILD_KERNELS=ON`
  (cheap, no kernel rebuild), then `gmake -C build <new_runner>/fast -j4`.
  Building a new runner with `cmake --build build --target <new_runner>`
  instead re-enters the kernel ExternalProjects and can corrupt the preprocess
  `.o` files again (see above) even though the kernels did not change.
- Runners dlopen the kernel `.so`; rebuilding `libtq_decode_sim.so` alone is
  enough to pick up kernel-side changes — no runner relink needed.

## Run on CAModel (the recipe that works)

Stock environment, sourced `set_env.sh`, simulator lib dir on
`LD_LIBRARY_PATH`, scratch CWD. **No `LD_PRELOAD`, no `CAMODEL_*` vars, no
cannsim wrapper needed.**

```bash
wsl -- bash -lc 'docker run --rm -v /mnt/f/turboquant-core:/workspace -w /workspace \
  ascendai/cann:9.2.0-beta.2-910b-ubuntu22.04-py3.12 bash -lc "
    set +u; source /usr/local/Ascend/cann-9.2.0-beta.2/set_env.sh; set -u
    t=/usr/local/Ascend/cann-9.2.0-beta.2
    export LD_LIBRARY_PATH=/workspace/build/lib:\$t/tools/simulator/Ascend950PR_9599/lib:\$t/lib64:\$t/devlib:\${LD_LIBRARY_PATH:-}
    rm -rf /tmp/probe_run && mkdir -p /tmp/probe_run && cd /tmp/probe_run
    /workspace/build/tests/test_fused_decode_e2e_sim --probe"'
```

Expected: PEM model init, one `probe gh 16 raw-query/cube ...` line, `PASS
fused decode opt --probe`, ~40 s wall. Sim artifacts (dumps, `profile_*_log*.toml`)
land in CWD — run from a scratch dir, not the repo root.

### Golden-dump comparison runners

`tests/npu/test_golden_dump.cpp` builds four runners
(`test_golden_prefill_strip_sim`, `test_golden_prefill_nz_sim`,
`test_golden_scatter_sim`, `test_golden_decode_sim`) that compare every kernel
against the PyTorch golden (`golden/turboquant_golden.py`) via the dumps in
`golden/dumps/` (regenerate on the Windows host: `python
golden/generate_dumps.py`). They take `--dump-root <dir>` (default
`golden/dumps` relative to CWD) and, for the decode, `--ulp-budget N`. The
helper `build/run_golden_sim.sh [runner ...]` runs them inside the container
with the exact working environment above and a scratch CWD per runner:

```bash
wsl -- bash -lc 'docker run --rm -v /mnt/f/turboquant-core:/workspace -w /workspace \
  ascendai/cann:9.2.0-beta.2-910b-ubuntu22.04-py3.12 bash build/run_golden_sim.sh'
```

Build them with the `/fast` route: `gmake -C build test_golden_prefill_strip_sim/fast
test_golden_prefill_nz_sim/fast test_golden_scatter_sim/fast
test_golden_decode_sim/fast -j4`.

### Runtime pitfalls (all observed, do not repeat)

- **Do not `LD_PRELOAD` the simulator libs.** `tools/simulator/Ascend950PR_9599`
  is a symlink to `x86_64-linux/simulator/dav_3510`, whose `lib/` and
  `camodel/` subdirs carry DIFFERENT builds of same-SONAME libs
  (`libstars.so`, `libpem_davinci.so`, ...). Preloading the `camodel/`
  variants poisons config init (`Config file is not in
  hard_code_toml_cfg_map`, then SIGSEGV in `stars_top::config_subtask` →
  `stars_kickstart_ctrl::associate` on a null shared_ptr). With plain
  `LD_LIBRARY_PATH` the libs' own NEEDED/RPATH resolution picks the right set.
- `DRVSTUB_LOG ... soc_version:Ascend950PR_9589` is normal (drvstub default)
  even for a 9599 build — not an error.
- `cannsim`/`npusim record` (`python3 $CANN/python/site-packages/cannsim/main.py
  record <app> -s Ascend950DT`) also works but needs `LD_PRELOAD=libgomp.so.1`
  for `libParallelScheduler.so`; the plain recipe above avoids that entirely.
- The old `build/tests/*_sim` binaries are runnable as-is after a kernel
  change once `libtq_*_sim.so` is rebuilt — but flags added to the test
  SOURCE (e.g. `--probe`) require the `/fast` relink above.
- The driver log line `halGetSocVersion soc_version:Ascend950PR_9589` plus
  `Config file is found, path is .../dav_3510/lib/config_stars.json` are the
  GOOD path; `cannot be found` or `hard_code_toml_cfg_map` mean the
  environment is wrong, not the kernel.

## Diagnosing inside the container

`docker run --rm` starts fresh from the image every time — `apt-get install`
does not persist across invocations. Chain install + use in ONE command:
`apt-get update -q && apt-get install -y -q gdb` then run (network works).
`gdb -batch -ex run -ex bt --args <bin> --probe` with env set via `-ex "set
environment ..."` backtraces simulator crashes. Write helper scripts to files
under `build/` and execute them — nested quoting through
`git-bash → wsl → docker bash -lc` eats `$vars`.
