---
name: turboquant-sim-build
description: Build turboquant-core kernels and run CAModel sim tests (test_live_write_nz_sim --probe, test_fused_decode_e2e_sim --probe etc.) in WSL with the natively installed CANN 9.2.0-beta.2, including the merge-step .o purge that every rebuild needs. Use when building BiSheng/AscendC device legs or running simulator tests for this repo on the Windows host without an NPU.
---

# turboquant-core build + CAModel run recipe (native WSL, no docker, no NPU)

Target: Ascend 950PR / dav-3510, CANN `9.2.0-beta.2`, installed NATIVELY in WSL at
`/usr/local/Ascend/cann-9.2.0-beta.2` (the docker image the old recipe used is not on this
host; do not go looking for it).

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

CANN's `ascendc_library` runs `merge_obj_text.sh`, which does
`ld.lld -m aicorelinux -Ttext=0 file.o -o file.o` **in place**, turning the preprocess object
REL -> EXEC. The merge target is phony inside a BUILD_ALWAYS ExternalProject, so EVERY build
that touches a kernel target re-runs it, and on an already-merged EXEC input ld.lld dies with
`ld.lld: error: ...cpp.o: unknown file type`. This is NOT a source error and NOT a 9P
filesystem problem (the old "9P truncation" theory was wrong — it reproduces on ext4). The
script deletes the preprocess `.o` of the kernels named (default: the ones the target depends
on) so the inner make regenerates a REL object. Kernel keys: `prefill prefill_nz scatter
scatter_nz decode`.

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

## Runtime pitfalls (all observed, do not repeat)

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
