# Decode kernel vs the PyTorch golden: the incompatibility, pinned down

Handoff for the session that fixes the fused decode. Everything below was
produced by the golden-dump comparison harness added in the same change:

* `golden/turboquant_golden.py` — standalone copy of vllm-ascend's
  `turboquant_dry_run.py` golden path (codec, NZ/strip cache writers, fp64
  decode reference). No vLLM / torch_npu dependency.
* `golden/generate_dumps.py` — writes `golden/dumps/` (gitignored):
  deterministic integer-valued BF16-exact fixtures, kernel operands (LCG
  `pi_signs`, BF16 codec tables, `h16`) and the golden's expected images.
* `tests/npu/test_golden_dump.cpp` — four CAModel runners
  (`test_golden_prefill_strip_sim`, `test_golden_prefill_nz_sim`,
  `test_golden_scatter_sim`, `test_golden_decode_sim`). Writers compared bit
  for bit; the decode against the fp64 golden with `--ulp-budget` (default 8).

Run: `python golden/generate_dumps.py`, then
`bash build/run_golden_sim.sh` inside the CANN container (see
`.kimi-code/skills/turboquant-sim-build/SKILL.md`).

## What is already verified compatible (do not "fix" these)

* **Scatter writer**: bit-identical to the golden, both steps of an RMW ladder.
* **Prefill writer, both layouts** (strip and `TURBOQUANT_INGRESS_NZ`): both
  packed planes bit-identical at `block_size = 128` (in-page strip offsets
  covered — the `DRY_RUN_COMPATIBILITY.md` §7.1 gap). The scale plane matches
  except 8 floats on all-zero rows: the kernel floors absmax at 1.0
  (`AscendC::Max(abs_max, abs_max, one, ...)`,
  `src/prefill/turboquant_reshape_batch.cpp:658`) and stores 2^-6, while the
  golden takes its explicit `absmax == 0` branch and stores 1.0. Codes are 0
  on both sides, dequant is 0 either way — immaterial, counted and named by
  the runner. Decide once whether the contract is the floor or the branch and
  align one side; until then the runner accepts exactly that pair.
* **The decode's raw-query rotation prologue**: reproduces the golden's
  rotated query **bit for bit** (`query_rot prologue 65536/65536 bytes
  bit-exact` on both raw-query legs). Pi's structure, the sign operand and the
  butterfly stage order are NOT the problem.

## Defect 1: the dequantisation convention is not the writers'

The fused decode reads a cache row as

    value = (code + 0.5) * stored_scale / kGain,   kGain = 2.9832882881

* the `+0.5`: `kSignedLevelOffset = kNibbleSignShift - kAffineBias = 8 - 7.5`
  in `ComputeNibbleOperands` (`src/common/turboquant_codec_mx.h:372,435-441`),
  the affine KV4_FP8 centroid grid of `turboquant_mode.h`;
* the `/kGain`: `vector_cube_tile_scales` multiplies the gathered scale lanes
  by `scoreScale_ = scale / Codec::kGain` (scores) and `invGain_ = 1 / kGain`
  (values) — `src/decode/vector/routine_vector_cube_softmax.hpp:143-154`,
  `src/decode/vector/turboquant_vector_service.h:636-637`.

The writers and the PyTorch golden store `code * scale` with
`scale = 2^(e-2) / sqrt(D)` and no gain — i.e. the decode's reading of a
writer-produced cache is wrong by a per-row centroid shift plus a uniform
~2.983x. Measured on CAModel (`decode_ctx128`, smooth softmax): against the
fp64 golden the stage-0 output is off by up to ~34k bf16 ULP; against the same
reference evaluated under the kernel's own convention (`kernel_stage*` tensors
in the dump) it agrees to within the pipeline's fp8 rounding (worst abs ~2.1,
scale ratio 0.9968).

The kernel's own e2e fixture encodes the centroid convention (oracle values
`2c+1` at `scale = kGain * 2`, `tests/common/fused_decode_fixture.hpp`), so
`test_fused_decode_e2e_sim` cannot see this. The two test oracles are the two
conventions.

**Fix choice** (pick one, both sides must meet): either teach the decode's
codec the writers' convention (drop the `+0.5` offset and the kGain folds for
KV4_FP8), or change the writers + golden to store centroid codes and
kGain-folded scales. The dry run documents `q = rint(y * 2^(2-e))`,
`scale = 2^(e-2) / sqrt(D)` as the contract, so the decode is the side that
drifted — but note the KV4_FP8 mode traits were designed around the centroid
grid, so this is a contract decision, not a mechanical edit.

## Defect 2: `kUnrotated` output stage is 16x the model-basis context

`TurboQuantQueryBasis::FinishHeads` applies the egress butterfly with scale
**1.0f** (`src/decode/cube/turboquant_query_basis.h:106`):

    rotate_butterfly_gather_vf<RotateDirection::Egress>(acc, acc, signs, 1.0f, heads)

The accumulator is `Pi . context / sqrt(D)`, so `Pi^T` alone leaves a
`sqrt(D) = 16` factor: the stage-1 output measured on CAModel is 15.95x the
kernel-convention model-basis context (least-squares over 16k channels). The
fold belongs either there (`inv_sqrt_len_` is already a member) or in the
V dequant path; the dry run launches production at `output_stage = 1` and
expects the model basis, so as it stands the raw-query entry's stage-1 output
is 16x what its caller consumes.

Note the old fixture's `RawQueryBasis::apply` folds `inv_sqrt_len` into BOTH
directions (DHD involution, `tests/common/fused_decode_fixture.hpp:200-215`),
which is why `test_fused_decode_e2e_sim --reproduce` could not see this
either: its stage-1 leg has no oracle (`reference_bfloat16` refuses out of
[1,15]).

## Residual to expect after both fixes

Even with the convention aligned, the decode pipeline rounds its Cube
operands to fp8 e4m3 (the `kv4fp8` operand), including the softmax
probabilities. With random fp32 data that leaves a few-percent per-channel
residual against an fp64 reference — the observed worst abs ~2 on O(10)
values. The existing e2e fixture hides it by construction (constant-per-tile
levels, fp8-exact). When re-gating, either budget for it or make the dump
fixtures fp8-exact (the generator can round `query_rot` to e4m3-representable
values and choose cache levels that convert exactly).

## State of the gate

`test_golden_prefill_strip_sim`, `test_golden_prefill_nz_sim` and
`test_golden_scatter_sim` PASS. `test_golden_decode_sim` FAILS against the
golden by design and prints both comparisons (golden vs KV4_FP8 convention)
plus the prologue's bit-exactness, per case (`decode_ctx128`, `decode_ctx200`).
After the fix it should pass against the **golden** comparison with no test
change; keep the kernel-convention print as the diagnostic.
