# Batched sealed E8 decode and NPU measurements

The packet ABI and label/dequantization arithmetic remain sealed. The reader
accepts `split_k=1..8`, divides context on 64/128-token page boundaries, and
handles empty, single-tile and odd-tile slices. Each slice supports at most
256 tiles of 64 tokens. A 64K context with eight splits uses 128 tiles per
slice. `--split-k` overrides the selected profile's default.

One MIX reader launch has `Block Dim = batch * kv_heads * split_k` logical
blocks. Each block handles its query groups of six heads internally. For
`qwen35-gqa6` and `split8` at eight splits, the reader grid is 128. For
`deepseek-v4-flash-64k` it is 32, with two query groups per block. Logical
block count is not a measurement of physical core utilization.

With one split, output is written directly and no workspace is allocated or
accessed. With multiple splits, each padded head writes 256 FP32 accumulated
values, an eight-lane max block and an eight-lane sum block. A separate AIV
kernel uses LM4's `NativeMergeEngine` / `vector_merge_partials` routine, then
normalizes and rounds to BF16. Kernel ordering on the same stream publishes
the partials before reduction. Workspace bytes are
`lanes * query_groups * 6 * split_k * 272 * sizeof(float)`.

## Local WSL gates

Use the image and environment wrapper in [BUILD.md](BUILD.md). These gates
need no physical NPU:

```bash
bash scripts/wsl_docker.sh cmake -S /workspace/turboquant-core -B build-host \
  -DTQ_BUILD_KERNELS=OFF -DTQ_BUILD_E8_KERNELS=OFF
bash scripts/wsl_docker.sh cmake --build build-host --target tq_host_tests tq_generate_goldens -j12
bash scripts/wsl_docker.sh ctest --test-dir build-host -L host --output-on-failure

bash scripts/wsl_docker.sh cmake -S /workspace/turboquant-core -B build-sim \
  -DRUN_MODE=sim -DSOC_VERSION=Ascend950PR_9599 \
  -DTQ_BUILD_KERNELS=ON -DTQ_BUILD_E8_KERNELS=ON
bash scripts/wsl_docker.sh cmake --build build-sim --target tq_sim_tests tq_benchmarks -j8
bash scripts/wsl_docker.sh ./build-sim/bin/test_turboquant_production_sim \
  --list --mode e8 --stage decode --split-k 8
```

The 11 CPU CTests include a new E8 test for page coverage, 64-bit addressing,
empty workers, FP32 workspace merges against an independent unsplit
long-double oracle, guards and launch validation. Build/catalog checks do
not execute CAModel kernels or establish device numerical accuracy.
Simulator timing runs are rejected; the accuracy runner remains available
for separately admitted bounded model workloads.

## Physical-server accuracy and timing

Run inside `quay.io/ascend/vllm-ascend:v0.30.0-a5` on an Ascend950PR server with its driver
and the current source checkout. These commands are a server handoff;
no SSH connection or physical execution is required for local development.
Rebuild the reader and runners together; the expanded launch descriptor is
not binary-compatible with archived executables built against the old header.

```bash
cd /workspace/turboquant-core
unset ASCEND_TOOLKIT_LATEST_HOME TOOLCHAIN_HOME CMAKE_PREFIX_PATH SOC_VERSION
source /usr/local/Ascend/cann-9.2.1/set_env.sh
cmake -S . -B build-npu -DRUN_MODE=npu -DSOC_VERSION=Ascend950PR_9599 \
  -DTQ_BUILD_KERNELS=ON -DTQ_BUILD_E8_KERNELS=ON
cmake --build build-npu --target tq_npu_tests tq_benchmarks -j8

./build-npu/bin/test_turboquant_production_npu --device 0 --mode e8 --stage decode \
  --profile qwen35-gqa6 --split-k 8 --block-size 128
# Exercise odd split lengths and the one-split direct-output path as well.
for splits in 1 3 5 8; do
  ./build-npu/bin/test_turboquant_production_npu --device 0 --mode e8 --stage decode \
    --profile qwen35-gqa6 --split-k "$splits" --block-size 64
done
./build-npu/bin/bench_turboquant_production_npu --device 0 --mode e8 --stage decode \
  --profile qwen35-gqa6 --split-k 8 --block-size 128 \
  --warmup 20 --iterations 100 --timing graph --samples

# Accuracy, unprofiled device timing, then hardware profiling for all three
# requested profiles at both block sizes. Select a fresh output directory.
bash scripts/profile_e8_decode.sh 0 build-npu results/e8-splitk-run1
```

The script stops on accuracy, collection, export or parsing failures. Its
`measurement.json` includes every ACL-event sample from both unprofiled and
profiled runs and the exported reader/reducer rows with `Task Duration(us)`,
`Block Dim`, `Mix Block Dim`, `aiv_vec_ratio` and `aic_mte2_ratio`. Missing
per-task counters remain null; missing entire required metric groups cause
an error. Raw logs and CSV paths/row numbers are retained.

To compare qwen at one and eight splits, repeat the accuracy and timing
commands with `--split-k 1` and `--split-k 8`, keeping all other arguments
fixed. A 64K unsplit context exceeds the sealed per-slice capacity; compare
four versus eight splits for that profile.

## What the measurements mean

`--timing graph` is the native default. Warmup and initialization happen
before capture. At least one untimed initialization launch occurs even
with `--warmup 0`. The entire sample sequence is captured as
`event-start → reader → optional reduction → event-stop`, then submitted
with one `aclmdlRIExecuteAsync`. External timestamp events remain observable
after replay. `aclrtEventElapsedTime` supplies all samples; no host clock or
host synchronization interval supplies the reported device latency.
Capture or timing errors fail the benchmark without a wall-clock fallback.
Verification witness writes are disabled in timing mode.
Output fields are `device_p50_us`, `device_p95_us`, `device_mean_us` and
`device_effective_GB_s`; bandwidth uses the device median. E8 accuracy runs
fail when cosine is below 0.99999 or a GM guard changes.
Vector accumulator dependencies retain explicit V-pipe fences when the
witness-copy synchronization is skipped.

The event span includes device scheduling, event boundaries and the gap
between reader and reduction. It is labelled `captured_device_stream`, not
claimed as an instruction-only duration. `--timing events` provides an
explicit eager comparison labelled `eager_device_stream`, where CPU
submission gaps can appear. `msprof` task durations provide the separate
kernel execution view. Profiler instrumentation can change timing; keep
profiled and unprofiled results separate. CSV rows may include warmup tasks
and are not paired to event samples by index.

Pipe ratios are per-core cycle fractions, not additive percentages of
event latency. Block Dim describes launch geometry, not measured active
core occupancy. No speedup or utilization claim is established by a build,
catalog listing, or synthetic parser test.

API references: [ACL stream capture](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_1782.html),
[external event recording](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/runtimeapi/aclcppdevg_03_2187.html),
[device event timestamps](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/others/acldevg/runtime_doc_dev_0023.html),
[logical block dimensions](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00021.html),
and [hardware pipe metrics](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_performance_data.md).
