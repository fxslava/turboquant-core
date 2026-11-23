# Wave-v11 integration — Checkpoint 1

Branch: codex/wave-v11-integration. Base: b92cba0714f5b4ca11114ab70f7914afd0137a10. Port commit: b3d6058 (cherry-pick of frozen 7c31a112c5e2250e55dbe37c3355079f7ab76629). Managed isolated worktree: C:/Users/Admin/.codex/worktrees/wave-v11-integration/turboquant-core.

The five-file Wave-v11 port applied without conflict. The existing Qwen35 telemetry changes remain. Verified tag, main checkout and harness probes were not modified. The actual available port is KV4_FP8 signed-nibble decode, not E8.

## CPU validation

GCC 11.4.0, CMake 3.22.1 on Ubuntu-22.04. All CPU targets compile with -std=c++17 -Wall -Wextra -Werror, using TQ_BUILD_KERNELS=OFF; no BiSheng or simulator was launched.

Initial CTest result: 5/7 pass. The unchanged b92cba0 baseline was built separately and reproduced both failures exactly:

- telemetry_dump: range override [200,205) was incorrectly tested against a three-token fixture whose other token has live winners outside that range. The test now restricts the override check to token 0 and its full workspace extent; production verification and all nine corruption checks are unchanged.
- golden_decode_oracle: six declared fixture directories are absent (decode_serving_ctx272, decode_serving_ctx316, decode_real_ctx64, decode_real_ctx128, decode_real_ctx200, decode_real_ctx1055). The repository's two available cases are decode_ctx128 and decode_ctx200. Missing cases remain an explicit gate failure; no skip or weakened budget was introduced.

After the test fixture correction, all six non-golden CTest entries pass. Both available golden cases also pass when invoked individually. Their stage0 CPU oracle is exactly 0 FP32 ULP; stage1 uses the existing combined absolute/relative comparison and is not an exact device result. The full seven-entry suite remains incomplete because six golden case directories are absent.

Commands (builds on ext4, source in isolated worktree):

```powershell
wsl -d Ubuntu-22.04 -- cmake -S /mnt/c/Users/Admin/.codex/worktrees/wave-v11-integration/turboquant-core -B /home/dmin/work/checkpoint1_20261123_i1/cpu-build -DRUN_MODE=sim -DTQ_BUILD_KERNELS=OFF -DTQ_COMPILER_CACHE=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
wsl -d Ubuntu-22.04 -- cmake --build /home/dmin/work/checkpoint1_20261123_i1/cpu-build -j 4
wsl -d Ubuntu-22.04 -- ctest --test-dir /home/dmin/work/checkpoint1_20261123_i1/cpu-build --output-on-failure -E golden_decode_oracle
wsl -d Ubuntu-22.04 -- /home/dmin/work/checkpoint1_20261123_i1/cpu-build/tests/test_golden_decode_oracle --case /mnt/c/Users/Admin/.codex/worktrees/wave-v11-integration/turboquant-core/golden/dumps/decode_ctx128
wsl -d Ubuntu-22.04 -- /home/dmin/work/checkpoint1_20261123_i1/cpu-build/tests/test_golden_decode_oracle --case /mnt/c/Users/Admin/.codex/worktrees/wave-v11-integration/turboquant-core/golden/dumps/decode_ctx200
```

Baseline CPU build: /home/dmin/work/checkpoint1_20261123_i1/baseline-cpu-build, using /mnt/c/Ascend/turboquant-core at unchanged b92cba0. Baseline CTest failed the same telemetry assertion and six missing fixture loads.

## Harness reference reconciliation

User-supplied harness commit d026b816461f16ca50d3fca0582c741f43255087 is present. Its tests/sim/kernel/fused_decode_opt_kernel.cpp includes the shared production integration kernel and wraps isolated/reduction probes. Its common.hpp supplies guarded ACL buffers. Its test_camodel_fused_decode_opt.cpp has an independent stored-word BF16 zero-ULP oracle with configurable tiles, heads and tail. A bounded candidate command geometry is --dtype bf16 --tiles 4 --heads 6 --tail-rows 16 (208 live rows, paired six-head path). This source has no --wave-v11-tiles native runner interface. It recovers an exact fixture suitable for a later isolated port, not the missing signed native binary.

The harness integration kernel predates the port's paired schedule and differs from the five-file tag snapshot; its op_host/turboquant_tiling.cpp is a five-line static-assert shim. It must not overwrite the full host descriptor builder already in core. The port's paired register SIMD helper and early-ingress schedule remain the intended implementation.

## Remaining device and analyzer gates

Device strict compilation and CAModel verification await the coordinator's serialized lane after the reshape baseline. No fresh device 0-BF16-ULP or timing result is claimed.

The retained K analyzer report records zero raw delta, but 71 raw fatal AKA1004 and 65 warnings remain. Search found analyzer support for saved-baseline suppression and inline ignores, but no standard Wave-v11 suppression/baseline file in the available core or harness scripts. No new broad suppression was invented and no legacy warnings were chased. A suppressed zero-fatal and zero-new-warning run remains pending recovery of that policy or coordinator-defined equivalent baseline handling.

