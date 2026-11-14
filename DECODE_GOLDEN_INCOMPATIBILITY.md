# Decode kernel vs the PyTorch golden: the incompatibility, pinned down and FIXED

This was the handoff for the session that fixed the fused decode; it now records
the diagnosis and the fix. The comparison harness:

* `golden/turboquant_golden.py` — standalone copy of vllm-ascend's
  `turboquant_dry_run.py` golden path (codec, NZ/strip cache writers, fp64
  decode reference). No vLLM / torch_npu dependency.
* `golden/generate_dumps.py` — writes `golden/dumps/` (gitignored):
  deterministic integer-valued BF16-exact fixtures, kernel operands (LCG
  `pi_signs`, BF16 codec tables, `h16`) and the golden's expected images.
* `golden/simulate_decode_fp8.py` — a torch simulation of the decode pipeline's
  two e4m3 operand roundings (query and softmax probabilities). Calibrated
  against the pre-fix kernel's measured residual (sim 2.042 / ratio 0.996723 vs
  the device's ~2.1 / 0.9968 against `kernel_stage0`); predicts the post-fix
  residual to within 0.02%.
* `tests/npu/test_golden_dump.cpp` — four CAModel runners
  (`test_golden_prefill_strip_sim`, `test_golden_prefill_nz_sim`,
  `test_golden_scatter_sim`, `test_golden_decode_sim`). Writers compared bit
  for bit; the decode against the fp64 golden with `--ulp-budget` (default 8)
  OR-ed with `--rel-budget` (default 0.08 of the row's max magnitude) — see
  "The residual after both fixes" for why a pure ULP budget cannot gate this
  pipeline.

Run: `python golden/generate_dumps.py` (needs torch; inside the vllm-ascend
container with `TORCH_DEVICE_BACKEND_AUTOLOAD=0`), then the runners via
`build/wsl_docker.sh bash build/run_golden_sim.sh <runner>` (see
`.kimi-code/skills/turboquant-sim-build/SKILL.md`, "The docker route").

## What is verified compatible (unchanged by the fix)

* **Scatter writer**: bit-identical to the golden, both steps of an RMW ladder.
* **Prefill writer, both layouts** (strip and `TURBOQUANT_INGRESS_NZ`): both
  packed planes bit-identical at `block_size = 128`. The scale plane matches
  except 8 floats on all-zero rows: the kernel floors absmax at 1.0
  (`AscendC::Max(abs_max, abs_max, one, ...)`,
  `src/prefill/turboquant_reshape_batch.cpp:658`) and stores 2^-6, while the
  golden takes its explicit `absmax == 0` branch and stores 1.0. Codes are 0
  on both sides, dequant is 0 either way — immaterial, counted and named by
  the runner. Decide once whether the contract is the floor or the branch and
  align one side; until then the runner accepts exactly that pair.
* **The decode's raw-query rotation prologue**: reproduces the golden's
  rotated query **bit for bit** (`query_rot prologue 65536/65536 bytes
  bit-exact` on every raw-query leg, before and after the fix).

## Defect 1 (FIXED): the dequantisation convention was not the writers'

The fused decode read a cache row as

    value = (code + 0.5) * stored_scale / kGain,   kGain = 2.9832882881

* the `+0.5` came from `kSignedLevelOffset = kNibbleSignShift - kAffineBias =
  8 - 7.5` in `ComputeNibbleOperands` (`src/common/turboquant_codec_mx.h`);
* the `/kGain` from `scoreScale_ = scale / Codec::kGain` (scores) and
  `invGain_ = 1 / kGain` (values) in
  `src/decode/vector/turboquant_vector_service.h`.

The writers and the PyTorch golden store `code * scale` with
`scale = 2^(e-2) / sqrt(D)` and no gain — the dry run documents
`q = rint(y * 2^(2-e))`, `scale = 2^(e-2) / sqrt(D)` as the contract, so the
decode was the side that had drifted, and it is the side that moved:

* `ComputeNibbleOperands` drops the offset entirely: the widened signed nibble
  IS the code, and every code in [-8, 7] is an exact e4m3 value, so the
  operand cast adds no rounding of its own.
* The score scale is the bare attention scale and the value scale 1
  (`scoreScale_ = scale`, `invGain_ = 1`). `Codec::kGain` and
  `TurboQuantCodebookGain` are gone — nothing in the read path folds a gain
  any more, for any codebook (a non-uniform one reconstructs inside the
  operand, which is what `TurboQuantLloydMaxTraits::kGain == 1` says).

Test-side alignment: the e2e fixture (`tests/common/fused_decode_fixture.hpp`)
encoded the centroid convention (oracle values `2c+1` at scale `kGain * 2`),
which is why `test_fused_decode_e2e_sim` could not see the defect. It now
stores scale 2.0 with oracle `2*level` (value levels in [1, 7] keep the
expected values inside `reference_bfloat16`'s [1, 15]) and passes at its
exact budget of 2 bf16 ULP — measured worst 0 ULP on every smoke rung.
`tests/common/turboquant_golden_oracle.hpp`'s `affine_offset()` restatement of
the rule now returns 0.

## Defect 2 (FIXED): `kUnrotated` output stage was 16x the model-basis context

`TurboQuantQueryBasis::FinishHeads` applied the egress butterfly with scale
**1.0f**. The accumulator is `Pi . context / sqrt(D)`, so `Pi^T` alone left a
`sqrt(D) = 16` factor. The fold now lives where
`routine_vector_rotate_butterfly.hpp`'s header always said it did ("The decode
egress passes kRotOrthoScale"): the egress call gets `inv_sqrt_len_`
(`src/decode/cube/turboquant_query_basis.h`). The stage-1 legs of both golden
cases now land at the same few-percent-of-row-max residual as stage 0.

Note the old fixture's `RawQueryBasis::apply` folds `inv_sqrt_len` into BOTH
directions (DHD involution, `tests/common/fused_decode_fixture.hpp`), which is
why `test_fused_decode_e2e_sim --reproduce` could not see this either: its
stage-1 leg has no oracle (`reference_bfloat16` refuses out of [1,15]).

## The residual after both fixes, and how the gate is re-based

The decode pipeline rounds its Cube operands to fp8 e4m3 — the query (per-head
448/amax dynamic map) and the softmax probabilities (per-tile 448/max map).
Against an fp64 reference that is a real, unavoidable residual, and with the
original random integer query it measured ~21% of the row max worst-case
(dominated by the query operand; the e2e fixture hides all of it by
construction: constant-per-tile levels, fp8-exact).

A pure bf16-ULP budget cannot express that residual (it explodes on
near-zero channels), and no non-toy query is e4m3-exact AFTER the kernel's
data-dependent scaling — so the fixture moved to the one design that is:

* `query_rot` is **+-16 per channel** (LCG signs, per head). The kernel's
  per-head scale is then 448/16 = 28 exactly and every query operand lane is
  +-448, the e4m3 max: the query quantisation rounds nothing.
* The raw query is the inverse rotation of that image — an integer vector in
  [-256, 256] (every entry a signed sum of 256 +-1s), exactly representable in
  BF16, and the prologue's fp32 butterflies stay integer-exact, so the
  bit-exact `query_rot` check still applies. The generator asserts both.

What remains is the probability-operand rounding alone: measured on CAModel
0.3-0.4% of the row max on `decode_ctx128` and 2.7-3.6% on `decode_ctx200`
(the ragged 8-row tail), against the simulation's 0.2% / 3.5%. The runner
therefore passes a channel on `--ulp-budget` (default 8 bf16 ULP) OR on
`--rel-budget` (default 0.08 of the row's max |golden|), and prints both
residuals plus the scale ratio per leg. The old convention sits ~3.1x off in
scale ratio and ~100% of the row max away — the budget catches it with an
order of magnitude of margin, and every defect class this runner exists for
(addressing, masking, scale-lane, factor, and convention bugs) is far outside
it.

## State of the gate (all PASS)

* `test_golden_prefill_strip_sim`, `test_golden_prefill_nz_sim`,
  `test_golden_scatter_sim`: bit-identical to the golden (the named zero-row
  scale convention counted, as before).
* `test_golden_decode_sim`: PASSES against the golden on both cases and all
  three legs, with the kernel-convention print kept as the diagnostic — it
  now shows the kernel does NOT implement `(code + 0.5) * scale / kGain`.
* `test_fused_decode_e2e_sim --probe`: PASS. `--camodel-smoke`: every rung at
  worst 0 bf16 ULP against the updated fixture, but the smoke ladder then
  fails its own `ragged_rungs >= 3` bookkeeping assertion (it has 2 ragged
  rungs) — a pre-existing ladder defect, not the decode.
