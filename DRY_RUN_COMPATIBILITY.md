# `turboquant_dry_run.py` against the kernels in this repository

Subject: `C:\vllm-ascend\vllm_ascend\attention\turboquant_dry_run.py` and `turboquant_layout.py`,
and the binding that reaches these kernels,
`C:\vllm-ascend\csrc\attention\turboquant\torch_bindings.cpp`, read against
`src/prefill/turboquant_reshape_batch.cpp`, `src/scatter/turboquant_reshape_scatter.cpp` and
`src/decode/turboquant_fused_decode.cpp` at `c5b1259`.

`/c/vllm-ascend/csrc/attention/turboquant/op_kernel/` vendors its own copy of all three kernels.
They agree with this repository on every fact below — the BF16 table operand, the in-page strip
offsets, the BF16-only decode instantiations, the planar scale plane — so the binding is bound
against kernels that behave as described here.

The dry run's **golden path is unaffected by any of this**: it never calls a kernel. Everything
below is about what happens when `TURBOQUANT_KERNEL_STAGES` switches a stage on.

Verdict: **the two ingress writers are compatible with the dry run's cache model and its ingestion
planner; the fused decode is not compatible with either writer and cannot be, because it reads a
different cache layout.** One defect was fatal, and it was fatal on both sides of the seam at once.

---

## 0. Status

Fixed (code, this round):

| | what | where |
|---|---|---|
| **§2.1** | the rotation operand was INT8 on both sides and BF16 on the device — fatal, plus a 16 KB over-read | `_codec_tables`, `check_tables`, three test assertions |
| **§2.4** | `mode_tables` was a null pointer | `torch_bindings.cpp` |
| **§2.5** | a staged kernel's geometry was only checked per call, from inside an attention forward | `_check_kernel_geometry` |
| **§3** | three stale docstrings claiming constraints the writers no longer have | `turboquant_dry_run.py` |

Already correct in the binding before this round, and left alone — my first pass asserted these
were missing and was wrong. They are listed because the reasoning in §2.2, §2.3 and §5 is still
worth having, and because the Python side had no equivalent until §2.5:

* `num_kv_heads == 4` and `head_size == 256` on the batch writer; `head_size == 256` on the
  scatter writer.
* a bf16-only refusal on the decode, with a comment naming the silent-no-op failure mode exactly.
* every clause of `valid_launch` pre-checked before the launcher can throw.
* `block_size % 16 == 0` rather than `== 16`, with a comment explaining what changed in the kernel.
* `VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS`, an opt-in host-side audit of the batch writer's
  strip-alignment contract.

Not fixed, deliberately — §6 (`tokens_per_core` pinned to 16, a measurement rather than a
correctness question) and the kernel-side self-defence suggested in §2.2.

---

## 1. What already agrees, and is worth not breaking

**The RFC 3.1 strip layout is a match, on all three planes.** `include/turboquant/turboquant_layout.h`
states the writers' contract as

```
key   nibble = ((block*H + head)*S + token)*d + channel
value nibble = ((block*H + head)*S)*d + (token/16)*(16*d) + channel*16 + (token % 16)
scale index  = ((kv*P + block)*H + head)*S + token
```

and that is, element for element, `TurboQuantValueLayout.CHANNEL_MAJOR` as the dry run documents
it and as `_golden_gather_pages_strip` inverts it. The scale plane is
`(2, num_blocks, num_kv_heads, block_size)` fp32 on both sides, which is exactly
`turboquant_scale_cache_shape`.

That includes the one thing the dry run's module docstring said it did *not* model:

> The scale plane is modelled at the staged kernels' layout … not the burst-padded per-token slot
> the AIV scatter writer uses.

**That caveat was stale.** The current scatter writer computes `((kv*P + block)*H + head)*S + token`
and `DataCopyPad`s one float per token into it ([`:493`](src/scatter/turboquant_reshape_scatter.cpp:493)
for the key plane, [`:534`](src/scatter/turboquant_reshape_scatter.cpp:534) for the value plane).
There is no burst-padded per-token slot any more; the two agree. The padded slot still exists, but
it belongs to the *decode's* NZ cache (`NzScaleSlotFloats`,
[turboquant_layout.h:149](include/turboquant/turboquant_layout.h:149)) — §4. The module docstring
now says this.

**Since then:** the scatter writer has grown an NZ build of its own —
`TURBOQUANT_INGRESS_NZ=1` emits the NZ-tiled cache directly (no strip RMW: a byte of the NZ
cache is wholly owned by one (token, kv head)), gated by `test_live_write_nz_sim --probe` /
the NZ ladder. The strip build remains the default and is unchanged.

Also matching: `TURBOQUANT_STRIP_TOKENS` 16 = `kStripTokens`/`kStripHeight`; the
`TURBOQUANT_STAGE_*` values 0/1/2 = `TurboQuantOutputStage::{kRotatedBasis, kUnrotated, kGated}`;
`TURBOQUANT_DECODE_SPLITS` 1 and `TURBOQUANT_DECODE_MAX_HEADS_PER_TASK` 16 are both inside what
`valid_launch` admits; `_h16_operand` is a `[16,16]` **fp16** ±1 matrix, which is what
`turboquant_query_basis.h:94` reads it as and the `kRotateQH16Elements = 256` budget expects;
negative slots are padding and skipped by both the scatter writer
([`:662`](src/scatter/turboquant_reshape_scatter.cpp:662)) and the golden writer.

**`turboquant_plan_strip_ingestion` is correct against the current batch writer** — see §3.

---

## 2. The defects

### 2.1 The rotation operand was INT8 on both sides and BF16 on the device — fatal (FIXED)

`_codec_tables()` returned `torch.int8`, `[2*(D/32), 32, 32]`, and its docstring called it "the
Cube's resident **INT8** operand". `check_tables` in the binding agreed — `TORCH_CHECK(scalar_type()
== at::kChar)` — and so did the comment at its launch site ("the bytes are the INT8 blocks checked
above") and three assertions in `test_turboquant_dispatch_seam.py`. Five places, one belief.

The kernel reads that pointer as **bfloat16**:

```cpp
// src/prefill/turboquant_reshape_batch.cpp:542
ascend_micro::stage_blocks_nz<kTableBlocks, kGroup>(
    reinterpret_cast<__gm__ const bfloat16_t*>(tables), kL1Table);
```

`stage_blocks_nz<Blocks, Dim>` reads `Blocks * Dim * Dim` *bfloat16_t* elements
([routine_stage_page_zn_bf16.hpp:148](src/prefill/routine_stage_page_zn_bf16.hpp:148)) — 16384 of
them at `D = 256`, so 32768 bytes, which is what `kTableBytes = kTableElems * kElemBytes` reserves
in L1 with `kElemBytes = 2` ([`:24`](src/prefill/turboquant_reshape_batch.cpp:24),
[`:31`](src/prefill/turboquant_reshape_batch.cpp:31)). The dry run's tensor was 16384 bytes. A
staged `reshape_and_cache` would therefore

* feed the Cube ±1 bytes reinterpreted as bf16 — a wrong rotation, not a slightly wrong one, and
* read 16 KB past the end of the operand.

Nothing in the signature could have caught it: the launcher's parameter is declared
`const std::int32_t*` and the kernel `reinterpret_cast`s it, so the declared pointer type says
nothing about the element type and neither side of the cast is checkable. The authority is the
operator's own runner, which converts the oracle's int8 blob on the way up
([test_live_write.cpp:176](tests/npu/test_live_write.cpp:176)): `tb[i] = to_bf16(table_blob[i])`
into a buffer sized `table_blob.size() * sizeof(std::uint16_t)`.

**Fixed.** `_codec_tables` returns `TURBOQUANT_TABLE_DTYPE` (`torch.bfloat16`); `check_tables`
requires `at::kBFloat16` and names the byte count in its message; the test assertions now pin the
dtype *and* the byte count, which is the quantity the kernel's L1 reservation is sized in — the
element count alone is what passed while the operand was wrong. Every entry is ±1, exact in BF16,
so the conversion is a representation change and the operand stays bit-identical to the oracle's.

Why it survived: the decode takes the same blob as `rot_tables`, and for `KV4_FP8` **never
dereferences it** — see §2.4. So a decode that runs is no evidence that this operand is right.

### 2.2 The prefill writer is hard-specialised to 4 KV heads × 256 channels, with no kernel-side guard

`kHeads = 4` and `kHeadSize = 256` are compile-time constants
([`:15`](src/prefill/turboquant_reshape_batch.cpp:15),
[`:16`](src/prefill/turboquant_reshape_batch.cpp:16)); `kChunks = 2*kHeads` = 8 chunk phases are
unrolled by template argument, and `kChunkHead<Chunk>` is `chunk % kHeads`. The `numKvHeads` and
`headSize` runtime arguments drive addressing only
([`:697`](src/prefill/turboquant_reshape_batch.cpp:697),
[`:211`](src/prefill/turboquant_reshape_batch.cpp:211)).

**The kernel checks neither.** The only early returns are the strip-range guards
([`:690`](src/prefill/turboquant_reshape_batch.cpp:690),
[`:692`](src/prefill/turboquant_reshape_batch.cpp:692)) and the launcher checks only the strip share
([`:750`](src/prefill/turboquant_reshape_batch.cpp:750)). At `num_kv_heads = 8` it would silently
read one head's rows and write another head's page — the same class of failure the scatter writer
refuses on the device at [`:606`](src/scatter/turboquant_reshape_scatter.cpp:606).

The binding **already** refuses both (`TORCH_CHECK(a.num_kv_heads == 4)`,
`TORCH_CHECK(a.head_size == kDecodeHeadSize)`), so this is not reachable through `torch.ops`. Worth
adding anyway, for the same reason the scatter writer has it: a launcher is not the only caller a
kernel can have, and the device-side check is what protects the runners.

**Suggested, not done:** `if (numKvHeads != kHeads || headSize != kHeadSize) return;` in the kernel
body.

### 2.3 The whole stack is BF16-only, and the decode's dispatch does not say so

Only two kernel instances exist ([`:695`](src/decode/turboquant_fused_decode.cpp:695),
[`:696`](src/decode/turboquant_fused_decode.cpp:696)), both `bfloat16_t`. `ascend_type_of` returns
`AscendType::FP16` for `element_type == 0` — which is `Launch`'s **default**
([turboquant_fused_decode_opt.hpp:130](include/turboquant/turboquant_fused_decode_opt.hpp:130)) —
and then both `_impl`s guard the launch with `if (type == AscendType::BF16)`
([`:738`](src/decode/turboquant_fused_decode.cpp:738),
[`:770`](src/decode/turboquant_fused_decode.cpp:770)) and **fall off the end without launching and
without raising**. The README's claim that "both the native FP16 and BF16 instances are retained"
does not hold for this tree. Same for ingress: both writers take `bfloat16_t` activations only.

The binding **already** refuses a non-bf16 query and passes `AscendType::BF16` unconditionally,
with a comment that names the failure mode precisely ("would therefore leave `output` whatever
`at::empty` gave it and report success: the worst available failure mode").

What was stale was the dry run's own account of it: `_kernel_fused_decode`'s docstring said
`element_type` "is read off the output dtype", which is what the binding stopped doing. **Fixed**,
along with a load-time refusal (§2.5).

### 2.4 `rot_tables` and `mode_tables` are dead for `KV4_FP8` (mode_tables FIXED)

`rot_tables` reaches `TurboQuantModeCodec::Init`
([turboquant_query_basis.h:103](src/decode/cube/turboquant_query_basis.h:103)) and `mode_tables`
reaches the decode's own codec. For `KV4_FP8` the traits are affine and `GATHER_LUT` is off, so
`ConstTableWords` returns 0 ([turboquant_codec_mx.h:119](src/common/turboquant_codec_mx.h:119)) and
**neither table is read at all**.

That is why §2.1's dtype error was invisible in the decode, and the dry run's docstring now says so
at both ends.

`mode_tables` was also missing from `_kernel_fused_decode`'s list of inputs "built inside the op" —
it is a fifth, not one of four — and the binding passed `/*mode_tables=*/nullptr`. A null pointer
*is* safe here, but only because of the mode: `SetGlobalBuffer(nullptr)` is harmless exactly as long
as nothing reads it, and switching the mode or turning `GATHER_LUT` on is a template argument away,
after which it becomes a null dereference on the device — an AIV fault with no line attached.

**Fixed.** The binding allocates `kMinBurstBytes` of zeros, which is what the operator's own fixture
passes (`modes.Upload(std::vector<std::uint32_t>(8, 0))`), and the docstring now lists all six
binding-built inputs with the reasoning for this one.

### 2.5 A staged kernel's geometry was only checked per call (FIXED)

Every constraint in §2.2, §2.3 and §5 was checked by the binding and by nothing else. That check is
the one that matters — a direct caller cannot skip it — but it arrives from inside an attention
forward, one decode step into a server that has already loaded weights and sized a KV cache from
`turboquant_page_size_bytes`, and it names a tensor shape rather than the model.

**Fixed.** `AscendTurboQuantDryRunBackendImpl._check_kernel_geometry` runs at layer construction and
refuses, per staged stage: `head_size != 256` (all three kernels), `num_kv_heads != 4` (either
writer), `num_kv_heads` outside 1..4 or a partial GQA group (decode), and a non-bf16 model dtype.
The message names the stage and `TURBOQUANT_KERNEL_STAGES`, so the remedy is in the error.

It is a refusal, not a warn-and-fall-back, and the difference from how a *missing* registration is
handled is deliberate: a kernel that is absent has a golden model behind it and the run stays
correct, while a kernel that is present and mis-specialised writes one head's rows into another
head's page with every canary intact. The new tests cover it, including that a golden run — and a
run naming only the four stages with no kernel behind them — is constrained by none of it.

`block_size` is not checked there: it belongs to the cache the model runner allocates, is not known
at layer construction, and the binding checks it per call.

---

## 3. Three stale docstrings that understated what the writers accept (FIXED)

Documentation-only — the planner's *code* was already right, and so was the binding — but they were
the first thing anyone wiring this up would read, and they described a stack more constrained than
it is.

`TurboQuantValueLayout`'s docstring said:

> the scatter writer carries a strip index inside the page and takes any of them, while the batch
> writer writes one strip at the page base with no in-page offset and therefore needs
> `block_size == 16`.

`CalculateStripAddr` ([`:203`](src/prefill/turboquant_reshape_batch.cpp:203)) derives both the page
and the strip within it from the strip's own `slot_mapping[strip*16]`:

```cpp
const std::uint32_t block_idx      = slot0 / block_size;
const std::uint32_t strip_in_block = (slot0 % block_size) / kTokens;
```

and all three plane offsets carry `strip_in_block`. The batch writer takes **any** `block_size` that
is a multiple of 16, the same as the scatter writer. `_kernel_reshape_and_cache_a8`'s docstring
likewise named the shapes `(num_blocks, 16, H, D/2)` and `(2, num_blocks, H, 16)`; they are
`(num_blocks, block_size, H, D/2)` and `(2, num_blocks, H, block_size)`.

This matters more than a docstring usually would, because `block_size == 16` is **incompatible with
the decode** (§4 — it needs a power-of-two multiple of 64). Under the stale reading the three
operators have no common page size at all; under the current code `block_size = 128` — what
`tests/npu/test_fused_decode_e2e.cpp` uses, and a normal vLLM page — satisfies all three. That is
the single most useful fact in this document, and it was hidden by a docstring.

**Caveat, and it is a real one:** no test exercises it.
[test_live_write.cpp:74](tests/npu/test_live_write.cpp:74) pins `kBlockSize = 16`, so
`strip_in_block` is always 0 in every run that has ever been made, and the prefill writer's in-page
strip offset is **code without evidence**. It should be the first thing a new rung covers. All three
docstrings now say so, and the third — `tokens_per_core` as "the kernel's own compile-time 16-token
strip" — is corrected in §6.

---

## 4. The decode cannot read either writer's cache. This is structural.

The dry run states this, in capitals, in `_kernel_fused_decode`:

> THIS OP CANNOT READ A CACHE EITHER WRITER WROTE, not directly.

**That is accurate and still accurate**, and the binding's own file header argues it at length and
reaches the same conclusion by a different route (the token-parity permutation that `DeInterleave`
would deposit in L1). Confirmed against `turboquant_layout.h` and gated by
`tests/cpu/test_layout_bridge.cpp` ("the co-byte pairings of the two contracts are structurally
incompatible"). The axes, each independent:

| | strip cache (both writers) | NZ-tiled cache (decode) |
|---|---|---|
| **byte order** | row-major page: `((block*H + head)*S + token)*d/2` | 64-row tile fractal: `NzTiledPackedByte`, `[tile][head][32-byte column group][tile row]` ([turboquant_layout.h:101](include/turboquant/turboquant_layout.h:101)) |
| **nibble pairing** | adjacent channels `(2c, 2c+1)` share a byte | channels `(c, c + d/2)` of **one token** share a byte ([`:131`](include/turboquant/turboquant_layout.h:131)) |
| **value plane token axis** | channel-major inside a 16-token strip | token-major, same as the key plane |
| **scale plane** | `(2, P, H, S)`, plane-major | `slot * NzScaleSlotFloats(H) + kv*H + head` — slot-major, padded to an 8-float block ([`:149`](include/turboquant/turboquant_layout.h:149), [`:155`](include/turboquant/turboquant_layout.h:155)) |
| **page size** | multiple of 16 | power-of-two multiple of **64** ([turboquant_fused_decode_opt.hpp:89](include/turboquant/turboquant_fused_decode_opt.hpp:89)) |

So it is five axes, not four. The dry run's own remedy —
`turboquant_nz_ingress.turboquant_repack_strip_to_nz_tiled` — is the right one for gating the
decode, and its header note that the real fix is a write-time NZ ingress kernel in the harness is
the right conclusion. **There is no such kernel in this repository.** `src/prefill/` and
`src/scatter/` both write the strip layout and nothing else.

Practical reading: `TURBOQUANT_KERNEL_STAGES=all` is not a configuration that can work.
`reshape_and_cache,scatter` (strip cache, golden reader) and `decode` (NZ cache, repacked) are each
reachable; together they are not.

---

## 5. Decode launch constraints

`valid_launch` ([turboquant_fused_decode_opt.hpp:82](include/turboquant/turboquant_fused_decode_opt.hpp:82))
refuses the following, and the launcher's response is a thrown `std::invalid_argument` from inside
an op. The binding **already** pre-checks every one of them, which is the right place for it; §2.5
adds the three a layer can see at construction.

* `grid > 32` — one co-resident wave, which the reduction's `SyncAll` needs.
* **`kv_heads > 4`** — the constraint a deployed GQA model is most likely to hit first; 8 is common.
* `heads % kv_heads != 0`.
* **`dim != 256`** — `kHeadSize`.
* `block_size % 64 != 0`, or `block_size` not a power of two.
* `splits > 8`, `heads_per_task > 16` — both satisfied by the dry run's constants.
* task coverage: `grid * tasks_per_block >= tokens*kv_heads*splits*ceil(group/chunk)` and
  `grid * reduce_tasks_per_block >= tokens*heads`.

Raw-query entry, additionally ([`:759`](src/decode/turboquant_fused_decode.cpp:759)):
`grid * prologue_vectors_per_block >= tokens * heads`, `prologue_cube_chunk_vectors ∈ {0, 16}`,
`output_stage <= 2`. The binding derives all three and picks 16 only when a block's share holds that
many query vectors, which is also what decides whether `h16` is read at all.

Note that the dry run calls the **raw-query** entry (it passes `query`, `pi_signs`, `h16`, `gate`,
`output_stage`), while `tests/npu/test_fused_decode_e2e.cpp` exercises the **pre-rotated** entry
`launch_fused_decode_opt`. The raw-query query prologue — `InitBasis` + `RotateQuery`, the
`query_rot` fp32 image, the Cube H16 chunk path — therefore has **no end-to-end test in this
repository at all**. The dry run is the first caller that would use it.

---

## 6. Prefill `tokens_per_core`: correct but wasteful (NOT CHANGED)

`_kernel_reshape_and_cache_a8`'s docstring said `tokens_per_core` "is the kernel's own compile-time
16-token strip". It is not compile-time — it is the width of one grid block's share, and the
seamless chunk pipeline runs `tokens_per_core / 16` strips as one uninterrupted chunk stream so MTE2
does not drain at a strip boundary. **The docstring is fixed; the value is not.**

Deriving it as 16 is valid — `WholeStripShare(16)` holds and `TotalStrips(T, 16) = T/16` blocks — but
it is one strip per block: the stream is one strip long and the refactor buys nothing. The binding
already documents this as a deliberate hold ("nothing has measured which share width is fastest on
the part, and a one-strip share is the geometry the goldens were taken at"), which is the right call
with no device attached — raising it is a measurement, not a correctness change. The only constraint
is that `num_tokens` stay a whole multiple of it, which the launcher throws on.

`test_live_write --perf --strips-per-block S` is the measurement that would settle it, available as
soon as a device is.

---

## 7. Priority of what remains

1. **§3 caveat — the prefill's in-page strip offset has no test.** `block_size > 16` is now the
   geometry that makes the whole stack coherent, and `strip_in_block` has never been non-zero in a
   run. A rung at `block_size = 128` is the highest-value thing left.
2. **§5 caveat — the raw-query decode prologue has no end-to-end test**, and it is the entry the dry
   run uses.
3. **§2.2 — kernel-side self-defence in the prefill writer**, matching the scatter writer's.
4. **§6 — measure `tokens_per_core`** once a device is attached.
5. **§4 — a write-time NZ ingress kernel.** Until it exists, the writers and the decode cannot share
   a cache, and `turboquant_repack_strip_to_nz_tiled` is a reference conversion rather than a path.
