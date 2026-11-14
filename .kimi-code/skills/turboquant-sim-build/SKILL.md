---
name: turboquant-sim-build
description: Build turboquant-core kernels and run CAModel sim tests (test_live_write_nz_sim --probe, test_fused_decode_e2e_sim --probe etc.) in WSL with the natively installed CANN 9.2.0-beta.2, including the merge-step .o purge that every rebuild needs. Use when building BiSheng/AscendC device legs or running simulator tests for this repo on the Windows host without an NPU.
---

# turboquant-core build + CAModel run recipe (WSL, optional docker, no NPU)

Target: Ascend 950PR / dav-3510, CANN `9.2.0-beta.2`, installed NATIVELY in WSL at
`/usr/local/Ascend/cann-9.2.0-beta.2`. Runs can also go through docker with the
`quay.io/ascend/vllm-ascend:v0.26.0rc1-a5` image (its own CANN is 9.1.0, which is
too old — the native 9.2.0-beta.2 is bind-mounted over it; see "The docker route"
below). Both routes work; the docker one is what the golden-dump flow uses.

The repo lives at `C:\Ascend\turboquant-core` = `/mnt/c/Ascend/turboquant-core` in WSL. Its
`build/` was configured on that path. Editing happens on the Windows side; for build+run speed
work on a native-ext4 rsync copy:

```bash
wsl -- bash -lc 'rsync -a --exclude=build --exclude=.git \
  /mnt/c/Ascend/turboquant-core/ ~/tqwork/turboquant-core/'
```

Configure once: `cmake -B build -DRUN_MODE=sim -DTQ_BUILD_KERNELS=ON` in the copy (needs
`set_env.sh` sourced or `ASCEND_HOME_PATH` default resolves anyway).

## Build: `build/wsl_build.sh <target> [kernels...]`

Checked into the repo (`build/` is gitignored, so copy it back if lost). It rsyncs the tree,
then builds. **The purge it does is mandatory, not optional:**

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

```bash
wsl -- bash -lc 'bash /mnt/c/Ascend/turboquant-core/build/wsl_build.sh test_live_write_nz_sim'
```

- Runner relinks re-trigger the kernel ExternalProjects too (BUILD_ALWAYS) — the purge is
  needed even when only test sources changed.
- The decode kernel (`tq_decode_sim`) takes ~40 min; prefill ~3 min; scatter ~15 s.
- Concurrent CAModel runs in separate processes work fine.

## Run on CAModel: `build/wsl_run_camodel.sh <binary> [args...]`

Stock environment, sourced `set_env.sh`, simulator lib dir on `LD_LIBRARY_PATH`, scratch CWD
under /tmp (sim dumps land there). **No `LD_PRELOAD`, no `CAMODEL_*` vars, no cansim
wrapper.** Runs the binaries from the native copy's `build/tests/`.

```bash
wsl -- bash -lc 'bash /mnt/c/Ascend/turboquant-core/build/wsl_run_camodel.sh \
  test_live_write_nz_sim --probe'
```

Expected gates:
- `test_live_write_nz_sim --probe`: one 16-token run crossing the strip boundary of one NZ
  tile; `... (--probe, head 0): PASS`, ~40 s wall.
- `test_live_write_nz_sim`: the full NZ ladder (strip-crossing run, single token, one whole
  64-row tile as four runs, the sequential accumulation ladder, the zero-row branch).
- `test_live_write_sim --op scatter [--mode sweep]`: the strip writer, unchanged.
- `test_nz_ingress_canary_sim`: the NZ prefill, one 64-row tile.
- `test_fused_decode_e2e_sim --probe`: one `probe gh 16 raw-query/cube ...` line, `PASS fused
  decode opt --probe`, ~40 s wall.

### Golden-dump comparison runners

`tests/npu/test_golden_dump.cpp` builds four runners
(`test_golden_prefill_strip_sim`, `test_golden_prefill_nz_sim`,
`test_golden_scatter_sim`, `test_golden_decode_sim`) that compare every kernel
against the PyTorch golden (`golden/turboquant_golden.py`) via the dumps in
`golden/dumps/` (regenerate with `python golden/generate_dumps.py`; needs torch —
run it inside the container, which carries a CPU torch, with
`TORCH_DEVICE_BACKEND_AUTOLOAD=0` so torch_npu is not auto-loaded). They take
`--dump-root <dir>` (default `golden/dumps` relative to CWD) and, for the decode,
`--ulp-budget N`.

### The docker route (vllm-ascend image + native CANN mount)

`build/wsl_docker.sh <command...>` runs a command inside
`quay.io/ascend/vllm-ascend:v0.26.0rc1-a5` with:

- the ext4 copy mounted AT ITS OWN PATH (`$HOME/tqwork/turboquant-core`), so the
  configured CMake build tree's absolute paths stay valid;
- the WSL-native CANN 9.2.0-beta.2 bind-mounted over the image's CANN 9.1.0 at
  the same path (the kernels target dav-3510 / Ascend950PR, which only
  9.2.0-beta.2 carries);
- the WSL `/usr/bin/ccache` bind-mounted in (the build was configured with it as
  the compiler launcher and the image does not carry it), with `CCACHE_DIR` on
  the ext4 volume so the warm cache survives `--rm` runs;
- `build/docker_inner.sh` as the entrypoint chain: clean PATH, then the 9.2
  `set_env.sh`.

`build/run_golden_sim.sh <runner> [args...]` is the CAModel runner inside the
container (same environment rules as `wsl_run_camodel.sh`), and
`build/docker_build.sh <target> [kernel...]` is the docker-side analogue of
`wsl_build.sh` (same mandatory preprocess `.o` purge, then the build):

```bash
wsl -- bash -lc 'bash ~/tqwork/turboquant-core/build/wsl_docker.sh \
  bash build/run_golden_sim.sh test_golden_decode_sim'
wsl -- bash -lc 'bash ~/tqwork/turboquant-core/build/wsl_docker.sh \
  bash build/docker_build.sh tq_decode_sim'
```

Build them with the `/fast` route: `gmake -C build test_golden_prefill_strip_sim/fast
test_golden_prefill_nz_sim/fast test_golden_scatter_sim/fast
test_golden_decode_sim/fast -j4`.

### Runtime pitfalls (all observed, do not repeat)

- **Do not `LD_PRELOAD` the simulator libs.** `tools/simulator/Ascend950PR_9599` is a symlink
  to `x86_64-linux/simulator/dav_3510`, whose `lib/` and `camodel/` subdirs carry DIFFERENT
  builds of same-SONAME libs (`libstars.so`, `libpem_davinci.so`, ...). Preloading the
  `camodel/` variants poisons config init (`Config file is not in hard_code_toml_cfg_map`,
  then SIGSEGV in `stars_top::config_subtask`). With plain `LD_LIBRARY_PATH` the libs' own
  NEEDED/RPATH resolution picks the right set.
- `DRVSTUB_LOG ... soc_version:Ascend950PR_9589` is normal (drvstub default) even for a 9599
  build — not an error.
- The driver log line `halGetSocVersion soc_version:Ascend950PR_9589` plus `Config file is
  found, path is .../dav_3510/lib/config_stars.json` are the GOOD path; `cannot be found` or
  `hard_code_toml_cfg_map` mean the environment is wrong, not the kernel.

## DataCopyPad UB->GM stride units (hard-won, cost a debug cycle)

From the vendor docs (`C:\PipeSim\docs\ascendc_api\SIMD-API\basic_api\memory_vector_compute\
data_move\DataCopyPad_UBToGM.md`): `blockLen` is in BYTES both sides, `dstStride` is in BYTES
(GM), but `srcStride` is in **32-byte DataBlocks** (UB). Both are the gap AFTER a block, and
the UB side advances `roundup32(blockLen) + srcStride*32`. Passing bytes for srcStride (e.g.
96 instead of 3) makes blocks 1+ read kilobytes past the intended data — CAModel UB is
zero-filled, so the symptom is rows 1+ of a run landing as 0x00 with block 0 correct.

## Diagnosing inside WSL

Write helper scripts to files under `build/` and execute them — nested quoting through
`git-bash → wsl → bash -lc` eats `$vars`. `gdb` is available natively.
