"""Generate synthetic golden dumps for the CAModel comparison tests.

For every kernel in this repository this writes a case directory under
``golden/dumps/`` holding a line-based ``manifest.txt`` and one raw
little-endian ``.bin`` per tensor:

* ``prefill/``   -- 8 whole 16-token strips over 128-token pages, with the
                    expected cache image in BOTH layouts (the strip build and
                    the TURBOQUANT_INGRESS_NZ build of the batch writer).
* ``scatter/``   -- two sequential AIV scatter launches into one strip of one
                    page, the second a read-modify-write of bytes the first
                    left half-written. Expected images after each step.
* ``decode_ctx128/`` and ``decode_ctx200/`` -- the fused decode over an
                    NZ-tiled cache written by this golden model, one aligned
                    page and one ragged 8-row tail. Expected fp32 output in
                    the model basis (stage 1) and the rotated basis (stage 0).

All activations are small integers, exact in BF16, and every rotation
intermediate is an exact integer in FP32 (|Pi x| <= 256 * 128 < 2^24), so the
packed planes and scale floats of the writers are BIT-comparable against the
kernels -- there is no tolerance anywhere in an ingestion comparison. The
decode's fp32 accumulation is compared against this model's float64 reference
with a bf16 ULP budget, the one place a tolerance is honest.

Run:  python golden/generate_dumps.py [output_root]     (default golden/dumps)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from turboquant_golden import (  # noqa: E402
    TURBOQUANT_KERNEL_KV_HEADS,
    TURBOQUANT_KERNEL_HEAD_SIZE,
    TURBOQUANT_PLANES,
    TURBOQUANT_STRIP_TOKENS,
    TurboQuantEmulator,
    build_codec_tables,
    decode_reference,
    h16_operand,
    nz_scale_slot_floats,
    write_nz_cache,
    write_strip_cache,
)

HEADS = TURBOQUANT_KERNEL_KV_HEADS
HEAD_SIZE = TURBOQUANT_KERNEL_HEAD_SIZE
BLOCK_SIZE = 128
DTYPE_MAP = {
    "bf16": (torch.bfloat16, 2),
    "f16": (torch.float16, 2),
    "u16": (torch.uint16, 2),
    "f32": (torch.float32, 4),
    "i32": (torch.int32, 4),
    "i8": (torch.int8, 1),
    "u8": (torch.uint8, 1),
}

# The canary fills the C++ runners initialise their device payloads with, as
# the byte and the word. Unwritten lanes must read back as exactly this.
GUARD_WORD = 0xDEADBEEF
# The scatter RMW poison of the existing CAModel test, kept for parity.
POISON_WORD = 0x7FC0BEEF


# The softmax scale of the decode fixtures: small enough that scores stay
# O(1) and the online softmax is smooth, so the comparison measures the
# kernel's arithmetic rather than argmax flips.
DECODE_SOFTMAX_SCALE = 1.0 / 1024.0


def lcg_ints(count: int, seed: int, lo: int, hi: int) -> list[int]:
    """Deterministic pseudo-random integers in [lo, hi], no RNG dependency."""
    state = seed & 0xFFFFFFFF
    out = []
    span = hi - lo + 1
    for _ in range(count):
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        out.append(lo + int((state >> 8) % span))
    return out


def make_activations(rows: int, seed: int) -> torch.Tensor:
    """``[rows, H, D]`` integer-valued fp32 activations, exact in BF16.

    Mostly small magnitudes with every eighth row full-scale, so the rotated
    panel spreads over several scale exponents and the INT4 clamp is taken;
    row 0 is all zero, which exercises the explicit absmax == 0 branch.
    """
    values = lcg_ints(rows * HEADS * HEAD_SIZE, seed, -8, 8)
    x = torch.tensor(values, dtype=torch.float32).reshape(rows, HEADS, HEAD_SIZE)
    full = lcg_ints(rows * HEADS * HEAD_SIZE, seed ^ 0x9E3779B9, -128, 128)
    full = torch.tensor(full, dtype=torch.float32).reshape(rows, HEADS, HEAD_SIZE)
    x[3::4] = full[3::4]
    x[0] = 0.0
    return x


def to_bf16_bits(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> bf16 bit patterns as uint16, asserting exactness (integers)."""
    narrowed = x.to(torch.bfloat16)
    widened = narrowed.to(torch.float32)
    if not bool((widened == x).all()):
        raise ValueError("activation is not exact in BF16")
    return narrowed.view(torch.uint16)


class DumpWriter:
    def __init__(self, case_dir: Path) -> None:
        self.case_dir = case_dir
        case_dir.mkdir(parents=True, exist_ok=True)
        for old in case_dir.glob("*.bin"):
            old.unlink()
        self.lines: list[str] = []

    def comment(self, text: str) -> None:
        self.lines.append(f"# {text}")

    def line(self, key: str, *values: object) -> None:
        self.lines.append(" ".join([key, *(str(v) for v in values)]))

    def tensor(self, name: str, tensor: torch.Tensor, dtype: str) -> None:
        torch_dtype, _ = DTYPE_MAP[dtype]
        flat = tensor.contiguous().reshape(-1)
        if dtype == "bf16":
            raw = flat.to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
        elif dtype == "f16":
            raw = flat.to(torch.float16).view(torch.uint16).numpy().tobytes()
        elif dtype == "u16":
            raw = flat.to(torch.uint16).numpy().astype("<u2").tobytes()
        elif dtype == "f32":
            raw = flat.to(torch.float32).numpy().astype("<f4").tobytes()
        elif dtype == "i32":
            raw = flat.to(torch.int32).numpy().astype("<i4").tobytes()
        elif dtype == "i8":
            raw = flat.to(torch.int8).numpy().astype("i1").tobytes()
        elif dtype == "u8":
            raw = flat.to(torch.uint8).numpy().astype("u1").tobytes()
        else:  # pragma: no cover
            raise ValueError(dtype)
        assert torch_dtype is not None
        path = self.case_dir / f"{name}.bin"
        path.write_bytes(raw)
        self.line("tensor", name, dtype, flat.numel(), path.name)

    def write(self, kernel: str) -> None:
        manifest = self.case_dir / "manifest.txt"
        header = [f"kernel {kernel}"]
        manifest.write_text("\n".join(header + self.lines) + "\n")


def guard_bytes(num_bytes: int) -> torch.Tensor:
    word = GUARD_WORD.to_bytes(4, "little")
    return torch.frombuffer(bytearray(word * (num_bytes // 4)), dtype=torch.uint8).clone()


def guard_floats(count: int, word: int = GUARD_WORD) -> torch.Tensor:
    raw = word.to_bytes(4, "little") * count
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone().view(torch.float32)


def poison_bytes(num_bytes: int) -> torch.Tensor:
    word = POISON_WORD.to_bytes(4, "little")
    return torch.frombuffer(bytearray(word * (num_bytes // 4)), dtype=torch.uint8).clone()


def poison_floats(count: int) -> torch.Tensor:
    return guard_floats(count, POISON_WORD)


def audit_quantisation(tag: str, codes: torch.Tensor, scales: torch.Tensor) -> None:
    """Fail the generation if the fixture does not exercise the codec."""
    covered = torch.unique(codes).numel()
    clamped = int((codes == 7).sum())
    zero_rows = int((codes.abs().amax(dim=-1) == 0).sum())
    exponents = torch.unique(scales.view(torch.int32) >> 23).numel()
    print(
        f"    {tag}: nibble levels covered {covered}/16, clamped codes {clamped}, "
        f"zero rows {zero_rows}, distinct scale exponents {exponents}"
    )
    if covered < 16:
        raise ValueError(f"{tag}: the fixture does not exercise all sixteen nibbles")
    if clamped == 0:
        raise ValueError(f"{tag}: the INT4 upper clamp is never taken")
    if zero_rows == 0:
        raise ValueError(f"{tag}: the absmax == 0 branch is never taken")
    if exponents < 3:
        raise ValueError(f"{tag}: the scale exponent barely varies")


def quantise_planes(
    emulator: TurboQuantEmulator, key: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    codes_k, scales_k = emulator.quantize(emulator.rotate(key))
    codes_v, scales_v = emulator.quantize(emulator.rotate(value))
    audit_quantisation("key", codes_k, scales_k)
    audit_quantisation("value", codes_v, scales_v)
    return codes_k, scales_k, codes_v, scales_v


def generate_prefill(root: Path, emulator: TurboQuantEmulator) -> None:
    """8 whole strips, strip s on physical page 2s+1 of a 128-token page."""
    strips = 8
    tokens = strips * TURBOQUANT_STRIP_TOKENS
    num_blocks = 2 * strips + 2  # pages 1,3,...,15 written; the rest canary
    slots_list: list[int] = []
    for s in range(strips):
        base = (2 * s + 1) * BLOCK_SIZE
        slots_list.extend(base + t for t in range(TURBOQUANT_STRIP_TOKENS))
    slots = torch.tensor(slots_list, dtype=torch.int64)

    key = make_activations(tokens, seed=0xC0FFEE)
    value = make_activations(tokens, seed=0xBADC0DE)
    codes_k, scales_k, codes_v, scales_v = quantise_planes(emulator, key, value)

    packed_bytes = HEAD_SIZE // 2
    plane_bytes = num_blocks * BLOCK_SIZE * HEADS * packed_bytes
    scale_floats = num_blocks * BLOCK_SIZE * nz_scale_slot_floats(HEADS)

    # The strip image, and the NZ image, both over the guard fill.
    strip_key = guard_bytes(plane_bytes).view(torch.int8)
    strip_value = guard_bytes(plane_bytes).view(torch.int8)
    strip_scale = guard_floats(TURBOQUANT_PLANES * num_blocks * HEADS * BLOCK_SIZE)
    write_strip_cache(
        codes_k, codes_v, scales_k, scales_v, slots,
        strip_key, strip_value, strip_scale, num_blocks, BLOCK_SIZE, HEADS, HEAD_SIZE,
    )

    nz_key = guard_bytes(plane_bytes).view(torch.int8)
    nz_value = guard_bytes(plane_bytes).view(torch.int8)
    nz_scale = guard_floats(scale_floats)
    write_nz_cache(
        codes_k, codes_v, scales_k, scales_v, slots,
        nz_key, nz_value, nz_scale, HEADS, HEAD_SIZE,
    )

    dump = DumpWriter(root / "prefill")
    dump.comment("8 whole 16-token strips; strip s -> physical page 2s+1; block_size 128.")
    dump.comment("tokens_per_core 16: one strip per grid block, in-page strip offsets 0..7.")
    for key_, val in (
        ("num_tokens", tokens), ("num_kv_heads", HEADS), ("head_size", HEAD_SIZE),
        ("block_size", BLOCK_SIZE), ("num_blocks", num_blocks), ("tokens_per_core", 16),
    ):
        dump.line("geom", key_, val)
    dump.line("float", "inv_sqrt_len", repr(emulator.inv_sqrt_len))
    dump.tensor("key", to_bf16_bits(key), "u16")
    dump.tensor("value", to_bf16_bits(value), "u16")
    dump.tensor("slot_mapping", slots.to(torch.int32), "i32")
    dump.tensor("pi_signs", emulator.pi_signs, "f32")
    dump.tensor("tables", build_codec_tables(emulator).view(torch.uint16), "u16")
    dump.tensor("expected_key_strip", strip_key.view(torch.uint8), "u8")
    dump.tensor("expected_value_strip", strip_value.view(torch.uint8), "u8")
    dump.tensor("expected_scale_strip", strip_scale, "f32")
    dump.tensor("expected_key_nz", nz_key.view(torch.uint8), "u8")
    dump.tensor("expected_value_nz", nz_value.view(torch.uint8), "u8")
    dump.tensor("expected_scale_nz", nz_scale, "f32")
    dump.write("prefill")
    print(f"  prefill: {tokens} tokens, {strips} strips, {num_blocks} blocks, page 2s+1")


def generate_scatter(root: Path, emulator: TurboQuantEmulator) -> None:
    """Two sequential launches into one strip of page 3, the second an RMW."""
    num_blocks = 8
    tokens_a, tokens_b = 6, 4
    page = 3
    slots_a = torch.tensor([page * BLOCK_SIZE + 2 * t for t in range(tokens_a)], dtype=torch.int64)
    slots_b = torch.tensor([page * BLOCK_SIZE + 2 * t + 1 for t in range(tokens_b)], dtype=torch.int64)

    key_a = make_activations(tokens_a, seed=0x5CA77E2)
    value_a = make_activations(tokens_a, seed=0x5CA77E3)
    key_b = make_activations(tokens_b, seed=0x5CA77E4)
    value_b = make_activations(tokens_b, seed=0x5CA77E5)
    codes_ka, scales_ka, codes_va, scales_va = quantise_planes(emulator, key_a, value_a)
    codes_kb, scales_kb, codes_vb, scales_vb = quantise_planes(emulator, key_b, value_b)

    packed_bytes = HEAD_SIZE // 2
    plane_bytes = num_blocks * BLOCK_SIZE * HEADS * packed_bytes
    scale_floats = 2 * num_blocks * HEADS * BLOCK_SIZE

    key_img = poison_bytes(plane_bytes).view(torch.int8)
    value_img = poison_bytes(plane_bytes).view(torch.int8)
    scale_img = poison_floats(scale_floats)
    init_key, init_value, init_scale = key_img.clone(), value_img.clone(), scale_img.clone()

    write_strip_cache(
        codes_ka, codes_va, scales_ka, scales_va, slots_a,
        key_img, value_img, scale_img, num_blocks, BLOCK_SIZE, HEADS, HEAD_SIZE,
    )
    step1_key, step1_value, step1_scale = key_img.clone(), value_img.clone(), scale_img.clone()
    write_strip_cache(
        codes_kb, codes_vb, scales_kb, scales_vb, slots_b,
        key_img, value_img, scale_img, num_blocks, BLOCK_SIZE, HEADS, HEAD_SIZE,
    )

    dump = DumpWriter(root / "scatter")
    dump.comment("AIV scatter RMW ladder: step A writes even token lanes of page 3's first")
    dump.comment("strip, step B the odd lanes sharing their bytes. blockDim 4, one request,")
    dump.comment("item_begin 0, item_count 4 (all four KV heads).")
    for key_, val in (
        ("num_requests", 1), ("num_kv_heads", HEADS), ("head_size", HEAD_SIZE),
        ("block_size", BLOCK_SIZE), ("num_blocks", num_blocks),
        ("tokens_step_a", tokens_a), ("tokens_step_b", tokens_b),
        ("block_dim", HEADS), ("item_begin", 0), ("item_count", HEADS),
    ):
        dump.line("geom", key_, val)
    dump.line("float", "inv_sqrt_len", repr(emulator.inv_sqrt_len))
    dump.tensor("key_a", to_bf16_bits(key_a), "u16")
    dump.tensor("value_a", to_bf16_bits(value_a), "u16")
    dump.tensor("slots_a", slots_a.to(torch.int32), "i32")
    dump.tensor("key_b", to_bf16_bits(key_b), "u16")
    dump.tensor("value_b", to_bf16_bits(value_b), "u16")
    dump.tensor("slots_b", slots_b.to(torch.int32), "i32")
    dump.tensor("pi_signs", emulator.pi_signs, "f32")
    dump.tensor("init_key", init_key.view(torch.uint8), "u8")
    dump.tensor("init_value", init_value.view(torch.uint8), "u8")
    dump.tensor("init_scale", init_scale, "f32")
    dump.tensor("expected_key_step1", step1_key.view(torch.uint8), "u8")
    dump.tensor("expected_value_step1", step1_value.view(torch.uint8), "u8")
    dump.tensor("expected_scale_step1", step1_scale, "f32")
    dump.tensor("expected_key_step2", key_img.view(torch.uint8), "u8")
    dump.tensor("expected_value_step2", value_img.view(torch.uint8), "u8")
    dump.tensor("expected_scale_step2", scale_img, "f32")
    dump.write("scatter")
    print(f"  scatter: page {page}, steps {tokens_a}+{tokens_b} tokens, RMW on the odd lanes")


def generate_decode(root: Path, emulator: TurboQuantEmulator, context: int, case: str) -> None:
    """One request of `context` tokens, the cache written by this golden model."""
    requests = 1
    group_heads = 16
    num_heads = HEADS * group_heads
    pages = (context + BLOCK_SIZE - 1) // BLOCK_SIZE
    physical_pages = 2 * requests * pages + 2
    block_table = [2 * p + 1 for p in range(pages)]

    slots = torch.tensor(
        [block_table[s // BLOCK_SIZE] * BLOCK_SIZE + s % BLOCK_SIZE for s in range(context)],
        dtype=torch.int64,
    )
    key = make_activations(context, seed=0xDEC000 + context)
    value = make_activations(context, seed=0xDEC100 + context)
    codes_k, scales_k, codes_v, scales_v = quantise_planes(emulator, key, value)

    packed_bytes = HEAD_SIZE // 2
    plane_bytes = physical_pages * BLOCK_SIZE * HEADS * packed_bytes
    scale_floats = physical_pages * BLOCK_SIZE * nz_scale_slot_floats(HEADS)
    nz_key = guard_bytes(plane_bytes).view(torch.int8)
    nz_value = guard_bytes(plane_bytes).view(torch.int8)
    nz_scale = guard_floats(scale_floats)
    write_nz_cache(
        codes_k, codes_v, scales_k, scales_v, slots,
        nz_key, nz_value, nz_scale, HEADS, HEAD_SIZE,
    )

    # The raw query: integer-valued, exact in BF16, no zero channels so the
    # scores vary. The pre-rotated image is what the raw-query prologue
    # writes: Pi q * inv_sqrt_len, in fp32.
    query = torch.tensor(
        lcg_ints(num_heads * HEAD_SIZE, 0xDEC200 + context, -8, 8), dtype=torch.float32
    ).reshape(1, num_heads, HEAD_SIZE)
    query += (query == 0).float()
    query_rot = emulator.rotate(query) * emulator.inv_sqrt_len

    block_tables = torch.tensor([block_table], dtype=torch.int64)
    context_lens = torch.tensor([context], dtype=torch.int32)
    # The softmax scale. Production uses 1/sqrt(D); here a smaller value keeps
    # the scores O(1), so the comparison measures the kernel's arithmetic
    # instead of which side of a knife-edge argmax a one-ULP score difference
    # lands on. It is a launch parameter, identical on both sides.
    softmax_scale = DECODE_SOFTMAX_SCALE
    expected1, expected0 = decode_reference(
        query, nz_key, nz_value, nz_scale, block_tables, context_lens,
        emulator, num_heads, HEADS, BLOCK_SIZE, softmax_scale,
    )
    # The same reference under the KV4_FP8 mode's own codec convention
    # ((code + 0.5) * scale / kGain): a diagnostic, so the runner can NAME the
    # convention a disagreeing kernel implemented rather than reporting a bare
    # mismatch. See decode_reference for why this is not the asserted contract.
    kernel1, kernel0 = decode_reference(
        query, nz_key, nz_value, nz_scale, block_tables, context_lens,
        emulator, num_heads, HEADS, BLOCK_SIZE, softmax_scale, kernel_convention=True,
    )

    dump = DumpWriter(root / case)
    dump.comment(f"fused decode: one request, context {context}, {pages} page(s), "
                 f"physical table {block_table} of {physical_pages}.")
    dump.comment("4 KV heads x 16 group heads = 64 query heads, splits 1, heads_per_task 16.")
    for key_, val in (
        ("requests", requests), ("context", context), ("num_heads", num_heads),
        ("num_kv_heads", HEADS), ("head_size", HEAD_SIZE), ("block_size", BLOCK_SIZE),
        ("max_blocks", pages), ("physical_pages", physical_pages),
        ("splits", 1), ("heads_per_task", group_heads),
    ):
        dump.line("geom", key_, val)
    dump.line("float", "scale", repr(softmax_scale))
    dump.line("float", "inv_sqrt_len", repr(emulator.inv_sqrt_len))
    dump.tensor("raw_query", to_bf16_bits(query.reshape(-1)), "u16")
    dump.tensor("query_rot", query_rot, "f32")
    dump.tensor("key_cache", nz_key.view(torch.uint8), "u8")
    dump.tensor("value_cache", nz_value.view(torch.uint8), "u8")
    dump.tensor("scale_cache", nz_scale, "f32")
    padded_table = torch.full(((pages * 4 + 31) // 32 * 8,), -1, dtype=torch.int32)
    padded_table[:pages] = torch.tensor(block_table, dtype=torch.int32)
    dump.tensor("block_table", padded_table, "i32")
    dump.tensor("context_lens", context_lens, "i32")
    dump.tensor("pi_signs", emulator.pi_signs, "f32")
    dump.tensor("h16", h16_operand().view(torch.uint16), "u16")
    dump.tensor("rot_tables", build_codec_tables(emulator).view(torch.uint16), "u16")
    dump.tensor("mode_tables", torch.zeros(8, dtype=torch.int32), "i32")
    dump.tensor("expected_stage1", expected1, "f32")
    dump.tensor("expected_stage0", expected0, "f32")
    dump.tensor("kernel_stage1", kernel1, "f32")
    dump.tensor("kernel_stage0", kernel0, "f32")
    dump.write("decode")
    tail = context % 64
    print(f"  {case}: context {context}, {pages} page(s), tail {tail or 64} rows, "
          f"physical_pages {physical_pages}")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "dumps"
    torch.manual_seed(0)  # belt and braces; the fixtures are LCG-driven anyway
    emulator = TurboQuantEmulator(HEAD_SIZE, "cpu")
    print(f"writing golden dumps to {root}")
    generate_prefill(root, emulator)
    generate_scatter(root, emulator)
    generate_decode(root, emulator, 128, "decode_ctx128")
    generate_decode(root, emulator, 200, "decode_ctx200")
    meta = {
        "head_size": HEAD_SIZE,
        "num_kv_heads": HEADS,
        "block_size": BLOCK_SIZE,
        "pi_seed_note": "pi_signs come from the LCG of turboquant_golden.py",
    }
    (root / "dumps.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("done")


if __name__ == "__main__":
    main()
