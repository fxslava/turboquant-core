"""TurboQuant PyTorch golden model, standalone copy for dump generation.

Copied from ``vllm-ascend/vllm_ascend/attention/turboquant_dry_run.py`` (the
``_golden_*`` half of ``TurboQuantOpsAdapter`` plus the codec it builds on),
stripped of its vLLM backend, torch_npu and the ``torch.ops`` seam so it runs
on a plain CPU torch. The arithmetic contract is unchanged:

    Pi = (H_{D/32} (x) I_32) . diag(pi_signs) . (I_{D/32} (x) H_32)
    Pi[c1*32 + c0][g*32 + i] = H8[c1][g] . pi_signs[g*32 + c0] . H32[c0][i]

with ``H`` the unnormalised Sylvester Hadamard. ``Pi^T Pi = D I``: the
un-rotation is the transpose. Per rotated row, with ``2^e <= absmax < 2^(e+1)``:

    scale = 2^(e - 2) * 1/sqrt(D)      stored fp32, one per (token, head)
    q     = min(rint_even(y * 2^(2 - e)), 7)     signed INT4, low nibble first

Both powers of two come out of the fp32 exponent field, so the single
round-to-nearest-even is the only rounding in the path and nibbles and scales
are bit-comparable against a kernel rather than tolerance-comparable.

The cache writers produce two layouts, matching the two builds of the kernels
in this repository:

* the RFC 3.1 strip cache (``tq_prefill_sim`` / ``tq_scatter_sim``)::

      key   nibble = ((block*H + head)*S + token)*d + channel
      value nibble = ((block*H + head)*S)*d + (token//16)*(16*d) + channel*16 + (token % 16)
      scale index  = ((kv*P + block)*H + head)*S + token

* the NZ-tiled cache (``tq_prefill_nz_sim``, and what the fused decode reads):
  ``[64-row tile][kv head][32-byte column group][tile row][byte]`` via
  ``NzTiledPackedByte``, one byte carrying channels ``(c, c + D/2)`` of ONE
  token, low nibble first; scales slot-major,
  ``slot * NzScaleSlotFloats(H) + kv * H + head``.

The decode reference gathers the NZ cache, dequantises and un-rotates it and
runs a plain softmax attention in float64 -- the same numbers the fused decode
computes in fp32 in the rotated basis, because the rotation cancels:
``Pi q . Pi k = D q . k`` and the two ``1/sqrt(D)`` folds compose to one.
"""

from __future__ import annotations

import math
from typing import Final

import torch

# ---------------------------------------------------------------------------
# The layout contract. Shared with include/turboquant/turboquant_layout.h.
# ---------------------------------------------------------------------------

TURBOQUANT_GROUP: Final[int] = 32
TURBOQUANT_PACK_FACTOR: Final[int] = 2
TURBOQUANT_PLANES: Final[int] = 2
TURBOQUANT_KEY_PLANE: Final[int] = 0
TURBOQUANT_VALUE_PLANE: Final[int] = 1
TURBOQUANT_STRIP_TOKENS: Final[int] = 16

# NZ-tiled cache geometry: kCubeTileRows / kOperandC0 / kFp32PerBlock.
TURBOQUANT_NZ_TILE_ROWS: Final[int] = 64
TURBOQUANT_NZ_OPERAND_C0: Final[int] = 32
TURBOQUANT_NZ_FP32_PER_BLOCK: Final[int] = 8

# scale = 2^(e - TURBOQUANT_HEADROOM_BITS): the largest channel lands on +-4.
TURBOQUANT_HEADROOM_BITS: Final[int] = 2

# Seed of the +-1 diagonal of Pi.
TURBOQUANT_PI_SEED: Final[int] = 0x5F3759DF

# The sign vector comes from an explicit LCG (numerical-recipes constants), so
# every host produces the identical vector without an RNG dependency.
_LCG_MULTIPLIER: Final[int] = 1664525
_LCG_INCREMENT: Final[int] = 1013904223
_LCG_MODULUS_MASK: Final[int] = 0xFFFFFFFF
_LCG_SIGN_BIT: Final[int] = 16

# The signed INT4 grid.
TURBOQUANT_NIBBLE_MIN: Final[int] = -8
TURBOQUANT_NIBBLE_MAX: Final[int] = 7
TURBOQUANT_NIBBLE_MASK: Final[int] = 0xF
TURBOQUANT_NIBBLE_LEVELS: Final[int] = 16

# The IEEE-754 binary32 fields the quantiser reads directly.
_FP32_MANTISSA_BITS: Final[int] = 23
_FP32_EXPONENT_MASK: Final[int] = 0xFF
_FP32_EXPONENT_BIAS: Final[int] = 127
_FP32_MAX_BIASED_EXPONENT: Final[int] = 254

# The kernels are compile-time specialised to this geometry.
TURBOQUANT_KERNEL_HEAD_SIZE: Final[int] = 256
TURBOQUANT_KERNEL_KV_HEADS: Final[int] = 4

# The KV4_FP8 mode's codebook gain, from include/turboquant/turboquant_mode.h
# (TurboQuantModeTraits<TurboQuantMode::KV4_FP8>::kGain). The fused decode
# folds it out of its dequantised operands; see decode_reference.
TURBOQUANT_KV4_FP8_GAIN: Final[float] = 2.9832882881

# The Cube's resident h16 sign matrix order and the rotation operand's dtype.
TURBOQUANT_H16_ORDER: Final[int] = 16
TURBOQUANT_TABLE_DTYPE: Final[torch.dtype] = torch.bfloat16


def turboquant_pi_signs(head_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """The deterministic +-1 diagonal of ``Pi`` for ``head_size`` channels.

    An explicit LCG, seeded by ``TURBOQUANT_PI_SEED + head_size``.
    """
    state = (TURBOQUANT_PI_SEED + head_size) & _LCG_MODULUS_MASK
    bits: list[float] = []
    for _ in range(head_size):
        state = (state * _LCG_MULTIPLIER + _LCG_INCREMENT) & _LCG_MODULUS_MASK
        bits.append(1.0 if (state >> _LCG_SIGN_BIT) & 1 else -1.0)
    return torch.tensor(bits, dtype=torch.float32, device=device)


def _exp2_fp32(exponent: torch.Tensor) -> torch.Tensor:
    """``2^exponent`` as fp32, assembled from the exponent field.

    Exact for every exponent a normal binary32 can hold; the biased exponent is
    clamped into the normal range rather than allowed to wrap.
    """
    biased = torch.clamp(exponent.to(torch.int32) + _FP32_EXPONENT_BIAS, 1, _FP32_MAX_BIASED_EXPONENT)
    return (biased << _FP32_MANTISSA_BITS).contiguous().view(torch.float32)


def _sylvester_hadamard(order: int, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    """``H[i][j] = (-1)^popcount(i & j)``, the unnormalised Sylvester Hadamard."""
    if order <= 0 or order & (order - 1):
        raise ValueError(f"Hadamard order must be a positive power of two, got {order}")
    index = torch.arange(order, device=device, dtype=torch.int64)
    common = index.unsqueeze(1) & index.unsqueeze(0)
    parity = torch.zeros_like(common)
    for bit in range(order.bit_length()):
        parity = parity ^ ((common >> bit) & 1)
    return torch.where(parity.bool(), -1.0, 1.0).to(dtype)


class TurboQuantEmulator:
    """Native-PyTorch model of the TurboQuant codec, in the kernels' arithmetic."""

    def __init__(
        self,
        head_size: int,
        device: torch.device | str = "cpu",
        pi_signs: torch.Tensor | None = None,
        group: int = TURBOQUANT_GROUP,
    ) -> None:
        if head_size % group:
            raise ValueError(f"head_size {head_size} must be a multiple of the group width {group}")
        groups = head_size // group
        if groups & (groups - 1):
            raise ValueError(f"head_size / group = {groups} must be a power of two")
        self.head_size: Final[int] = head_size
        self.group: Final[int] = group
        self.groups: Final[int] = groups
        self.device: Final = device
        self.inv_sqrt_len: Final[float] = 1.0 / math.sqrt(float(head_size))
        self.pi_signs: torch.Tensor = (
            turboquant_pi_signs(head_size, device)
            if pi_signs is None
            else pi_signs.to(device=device, dtype=torch.float32)
        )
        if self.pi_signs.numel() != head_size:
            raise ValueError(f"pi_signs has {self.pi_signs.numel()} entries, expected {head_size}")
        self._rotation: torch.Tensor | None = None

    @property
    def rotation(self) -> torch.Tensor:
        """``Pi`` as a dense ``[D, D]`` fp32 matrix, built once and cached."""
        if self._rotation is None:
            eye_group = torch.eye(self.group, device=self.device, dtype=torch.float32)
            eye_groups = torch.eye(self.groups, device=self.device, dtype=torch.float32)
            h_inner = _sylvester_hadamard(self.group, self.device, torch.float32)
            h_outer = _sylvester_hadamard(self.groups, self.device, torch.float32)
            inner = torch.kron(eye_groups, h_inner)  # I_{D/32} (x) H_32
            outer = torch.kron(h_outer, eye_group)  # H_{D/32} (x) I_32
            self._rotation = outer @ torch.diag(self.pi_signs) @ inner
        return self._rotation

    def rotation64(self) -> torch.Tensor:
        """``Pi`` in float64, for the decode reference's exact un-rotation."""
        return self.rotation.to(torch.float64)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """``y = Pi x`` over the last axis, unnormalised, in fp32."""
        return x.to(torch.float32) @ self.rotation.transpose(0, 1)

    def inverse_rotate(self, y: torch.Tensor) -> torch.Tensor:
        """``x = Pi^T y / sqrt(D)``, the egress direction, fp32."""
        return (y.to(torch.float32) @ self.rotation) * self.inv_sqrt_len

    def quantize(self, rotated: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Dynamic per-row INT4 quantisation of a rotated panel.

        ``rotated`` is ``[..., D]``; returns ``(codes, scales)`` with codes INT8
        in ``[-8, 7]`` and one fp32 scale per row. The exponent is read straight
        out of the fp32 exponent field, which is what the kernel does.
        ``torch.round`` is round-half-to-even, matching the kernel's CAST_RINT.
        """
        rotated_fp32 = rotated.to(torch.float32)
        absmax = rotated_fp32.abs().amax(dim=-1, keepdim=True).contiguous()
        zero_row = absmax == 0.0
        biased_exponent = (absmax.view(torch.int32) >> _FP32_MANTISSA_BITS) & _FP32_EXPONENT_MASK
        exponent = biased_exponent - _FP32_EXPONENT_BIAS
        ones = torch.ones_like(absmax)
        pow_scale = _exp2_fp32(exponent - TURBOQUANT_HEADROOM_BITS)
        inv_scale = _exp2_fp32(TURBOQUANT_HEADROOM_BITS - exponent)
        codes = torch.round(rotated_fp32 * inv_scale)
        codes = torch.clamp(codes, TURBOQUANT_NIBBLE_MIN, TURBOQUANT_NIBBLE_MAX)
        codes = torch.where(zero_row, torch.zeros_like(codes), codes)
        scales = torch.where(zero_row, ones, pow_scale * self.inv_sqrt_len)
        return codes.to(torch.int8), scales.squeeze(-1)

    def dequantize(self, codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """``codes * scales`` in fp32: the rotated activation, ``Pi x / sqrt(D)``."""
        return codes.to(torch.float32) * scales.to(torch.float32).unsqueeze(-1)

    @staticmethod
    def pack(codes: torch.Tensor) -> torch.Tensor:
        """Pack INT4 codes two to a byte along the last axis, low nibble first."""
        if codes.shape[-1] % TURBOQUANT_PACK_FACTOR:
            raise ValueError(f"cannot pack an odd channel count {codes.shape[-1]}")
        nibbles = codes.to(torch.int16) & TURBOQUANT_NIBBLE_MASK
        pairs = nibbles.reshape(*nibbles.shape[:-1], nibbles.shape[-1] // TURBOQUANT_PACK_FACTOR, 2)
        packed = pairs[..., 0] | (pairs[..., 1] << 4)
        return packed.to(torch.uint8).view(torch.int8)

    @staticmethod
    def unpack(packed: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`pack`: ``[..., D/2]`` bytes -> ``[..., D]`` INT8 codes."""
        raw = packed.view(torch.uint8).to(torch.int16)
        low = raw & TURBOQUANT_NIBBLE_MASK
        high = (raw >> 4) & TURBOQUANT_NIBBLE_MASK
        pair = torch.stack((low, high), dim=-1)
        signed = torch.where(pair >= 8, pair - TURBOQUANT_NIBBLE_LEVELS, pair)
        return signed.reshape(*packed.shape[:-1], packed.shape[-1] * TURBOQUANT_PACK_FACTOR).to(torch.int8)


# ---------------------------------------------------------------------------
# The Cube's resident operands.
# ---------------------------------------------------------------------------


def build_codec_tables(emulator: TurboQuantEmulator) -> torch.Tensor:
    """The Cube's resident rotation operand, ``2 * (D/32)`` row-major 32x32 BF16 blocks.

    ``tables[s * groups + g][c0][i] = (s ? -1 : +1) * pi_signs[g*32 + c0] * H32[c0][i]``:
    the sign diagonal and the inner Hadamard factor, pre-composed, with the
    outer factor supplied by which block the kernel picks. BF16, because
    ``stage_blocks_nz`` reads it as ``bfloat16_t``; every entry is +-1, exact in
    BF16, so this is a representation change rather than a rounding.
    """
    h_inner = _sylvester_hadamard(emulator.group, emulator.device, torch.float32)
    signs = emulator.pi_signs.reshape(emulator.groups, emulator.group)
    positive = signs.unsqueeze(-1) * h_inner.unsqueeze(0)
    tables = torch.stack((positive, -positive), dim=0)
    return tables.reshape(TURBOQUANT_PLANES * emulator.groups, emulator.group, emulator.group).to(
        TURBOQUANT_TABLE_DTYPE
    )


def h16_operand(device: torch.device | str = "cpu") -> torch.Tensor:
    """The Cube's resident ``[16, 16]`` fp16 sign matrix (Sylvester Hadamard)."""
    return _sylvester_hadamard(TURBOQUANT_H16_ORDER, device, torch.float16)


# ---------------------------------------------------------------------------
# The NZ-tiled cache write and read-back, flat byte-plane form.
# ---------------------------------------------------------------------------


def nz_scale_slot_floats(num_kv_heads: int) -> int:
    """``NzScaleSlotFloats`` of turboquant_layout.h."""
    lanes = TURBOQUANT_PACK_FACTOR * num_kv_heads
    return -(-lanes // TURBOQUANT_NZ_FP32_PER_BLOCK) * TURBOQUANT_NZ_FP32_PER_BLOCK


def nz_byte_offsets(slots: torch.Tensor, num_kv_heads: int, packed_bytes: int) -> torch.Tensor:
    """``NzTiledPackedByte`` for every byte of every (token, head), vectorised.

    ``slots`` is int64 of the GLOBAL row (``page * block_size + token``).
    Returns ``[..., H, groups, 32]`` int64 byte offsets into the flat packed
    plane, where ``groups = packed_bytes / 32``.
    """
    tile_rows = TURBOQUANT_NZ_TILE_ROWS
    c0 = TURBOQUANT_NZ_OPERAND_C0
    groups = packed_bytes // c0
    device = slots.device
    tile_row = slots % tile_rows
    base = (slots - tile_row) * (num_kv_heads * packed_bytes) + tile_row * c0
    heads = torch.arange(num_kv_heads, device=device) * (tile_rows * packed_bytes)
    group = torch.arange(groups, device=device) * (tile_rows * c0)
    byte = torch.arange(c0, device=device)
    return base[..., None, None, None] + heads.view(-1, 1, 1) + group.view(-1, 1) + byte


def _nz_weave(codes: torch.Tensor, head_size: int) -> torch.Tensor:
    """``[T, H, D]`` codes -> ``[T, H, D/2 / 32, 32]`` woven packed bytes."""
    packed_bytes = head_size // TURBOQUANT_PACK_FACTOR
    low, high = codes[..., :packed_bytes], codes[..., packed_bytes:]
    woven = torch.stack((low, high), dim=-1).reshape(*codes.shape[:-1], head_size)
    return TurboQuantEmulator.pack(woven).reshape(*codes.shape[:-1], -1, TURBOQUANT_NZ_OPERAND_C0)


def write_nz_cache(
    codes_key: torch.Tensor,
    codes_value: torch.Tensor,
    key_scales: torch.Tensor,
    value_scales: torch.Tensor,
    slots: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
) -> None:
    """Write both packed planes and the slot-major scale records of the NZ cache.

    ``codes_*`` ``[T, H, D]`` INT8, ``*_scales`` ``[T, H]`` fp32, ``slots``
    ``[T]`` int64 global row. The byte planes are flat int8 tensors of
    ``num_blocks * block_size * H * D/2`` and the scale plane a flat fp32 tensor
    of ``num_blocks * block_size * NzScaleSlotFloats(H)``.
    """
    packed_bytes = head_size // TURBOQUANT_PACK_FACTOR
    offsets = nz_byte_offsets(slots, num_kv_heads, packed_bytes)
    for codes, cache in ((codes_key, key_cache), (codes_value, value_cache)):
        cache.view(-1)[offsets] = _nz_weave(codes, head_size).to(cache.dtype)

    slot_floats = nz_scale_slot_floats(num_kv_heads)
    lanes = torch.arange(num_kv_heads, device=slots.device)
    records = slots[:, None] * slot_floats + lanes
    flat = scale_cache.view(-1)
    flat[records] = key_scales.to(scale_cache.dtype)
    flat[records + num_kv_heads] = value_scales.to(scale_cache.dtype)


def gather_nz_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    block_tables: torch.Tensor,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Read an NZ cache back into token-major codes and scales.

    ``block_tables`` is ``[B, max_blocks]`` int64. Returns
    ``(codes_key, scales_key, codes_value, scales_value)`` with codes
    ``[B, S_pad, H, D]`` INT8 and scales ``[B, S_pad, H]`` fp32, the exact
    inverse of :func:`write_nz_cache`.
    """
    packed_bytes = head_size // TURBOQUANT_PACK_FACTOR
    pages = block_tables.to(torch.int64)
    batch_size, max_blocks = pages.shape
    within = torch.arange(block_size, dtype=torch.int64)
    slots = (pages.unsqueeze(-1) * block_size + within).reshape(batch_size, max_blocks * block_size)

    offsets = nz_byte_offsets(slots, num_kv_heads, packed_bytes)

    def read_plane(cache: torch.Tensor) -> torch.Tensor:
        gathered = cache.view(-1)[offsets]
        packed = gathered.reshape(batch_size, -1, num_kv_heads, packed_bytes)
        woven = TurboQuantEmulator.unpack(packed)
        codes = torch.cat((woven[..., 0::2], woven[..., 1::2]), dim=-1)
        return codes

    slot_floats = nz_scale_slot_floats(num_kv_heads)
    lanes = torch.arange(num_kv_heads)
    records = slots[..., None] * slot_floats + lanes
    flat_scales = scale_cache.view(-1)

    return (
        read_plane(key_cache),
        flat_scales[records],
        read_plane(value_cache),
        flat_scales[records + num_kv_heads],
    )


# ---------------------------------------------------------------------------
# The RFC 3.1 strip cache write.
# ---------------------------------------------------------------------------


def write_strip_cache(
    codes_key: torch.Tensor,
    codes_value: torch.Tensor,
    key_scales: torch.Tensor,
    value_scales: torch.Tensor,
    slots: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
) -> None:
    """Write both planes of the strip cache, the value plane read-modify-write.

    ``codes_*`` ``[T, H, D]`` INT8; ``slots`` ``[T]`` int64 global row. The key
    plane is ``(num_blocks, H, block_size, D/2)`` token-major; the value plane is
    channel-major inside a 16-token strip; the scale plane is
    ``(2, num_blocks, H, block_size)`` fp32. Unwritten lanes of a touched strip
    keep the bytes they had -- the read-modify-write the AIV scatter writer
    performs in registers.
    """
    if block_size % TURBOQUANT_STRIP_TOKENS:
        raise ValueError(f"block_size {block_size} must be a multiple of {TURBOQUANT_STRIP_TOKENS}")
    packed_bytes = head_size // TURBOQUANT_PACK_FACTOR
    strips_per_page = block_size // TURBOQUANT_STRIP_TOKENS

    pages = torch.div(slots, block_size, rounding_mode="floor")
    offsets = slots - pages * block_size

    key_pages = key_cache.view(num_blocks, num_kv_heads, block_size, packed_bytes)
    key_pages[pages, :, offsets] = TurboQuantEmulator.pack(codes_key)

    # The value plane: unpack the touched pages, set one token lane of every
    # channel, pack back. Tokens 2t and 2t+1 share a byte, so the untouched
    # lane's nibble survives -- that is the read-modify-write.
    strips = torch.div(offsets, TURBOQUANT_STRIP_TOKENS, rounding_mode="floor")
    lanes = offsets - strips * TURBOQUANT_STRIP_TOKENS
    value_pages = value_cache.view(
        num_blocks, num_kv_heads, strips_per_page, head_size, TURBOQUANT_STRIP_TOKENS // TURBOQUANT_PACK_FACTOR
    )
    touched, inverse = torch.unique(pages, return_inverse=True)
    panel = TurboQuantEmulator.unpack(value_pages[touched])
    panel[inverse, :, strips, :, lanes] = codes_value
    value_pages[touched] = TurboQuantEmulator.pack(panel)

    scale_planes = scale_cache.view(TURBOQUANT_PLANES, num_blocks, num_kv_heads, block_size)
    scale_planes[TURBOQUANT_KEY_PLANE][pages, :, offsets] = key_scales.to(scale_cache.dtype)
    scale_planes[TURBOQUANT_VALUE_PLANE][pages, :, offsets] = value_scales.to(scale_cache.dtype)


# ---------------------------------------------------------------------------
# The fused decode's reference: dense attention over the dequantised cache,
# in float64.
# ---------------------------------------------------------------------------


def decode_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    emulator: TurboQuantEmulator,
    num_heads: int,
    num_kv_heads: int,
    block_size: int,
    scale: float,
    kernel_convention: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The fused decode's gold output, computed in float64.

    Gathers the NZ cache, dequantises it and runs plain softmax attention in
    the rotated basis (where the rotation cancels: ``Pi q . Pi k = D q . k``
    and the two ``1/sqrt(D)`` folds compose to one). Returns
    ``(expected_stage1, expected_stage0)`` fp32 ``[B, N, D]``: the model-basis
    context (the raw-query entry's ``kUnrotated`` output) and the
    rotated-basis context (the pre-rotated entry's output).

    ``query`` is ``[B, N, D]`` fp32/fp64 in the model's basis (the raw query,
    widened from its bf16 storage exactly as the device widens it).

    ``kernel_convention=False`` is the TurboQuant arithmetic contract the
    writers and this model share: a dequantised value is ``code * scale``.
    ``True`` is the convention the KV4_FP8 mode traits describe and the fused
    decode's own fixture encodes: ``(code + 0.5) * scale / kGain`` with the
    same gain folded out of the score scale. It exists so a comparison can
    NAME which convention a kernel implemented instead of reporting a bare
    mismatch; it is not the contract the golden asserts.
    """
    if kernel_convention:
        bias, gain = 0.5, TURBOQUANT_KV4_FP8_GAIN
    else:
        bias, gain = 0.0, 1.0
    codes_k, scales_k, codes_v, scales_v = gather_nz_cache(
        key_cache, value_cache, scale_cache, block_tables, block_size, num_kv_heads, emulator.head_size
    )
    group = num_heads // num_kv_heads
    rotation = emulator.rotation64()
    inv_sqrt = emulator.inv_sqrt_len

    def dense(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        rotated = (codes.to(torch.float64) + bias) * scales.to(torch.float64).unsqueeze(-1) / gain
        return rotated

    dense_key = dense(codes_k, scales_k)
    dense_value = dense(codes_v, scales_v)

    batch = query.shape[0]
    padded_len = dense_key.shape[1]
    positions = torch.arange(padded_len, dtype=torch.int64)
    live = positions.unsqueeze(0) < context_lens.to(torch.int64).unsqueeze(1)
    assert bool(live.any()), "a decode reference over an empty context"

    # Attention in the rotated domain, which is where the kernel contracts:
    # q_rot = Pi q / sqrt(D), K/V as dequantised above. The stage-0 context
    # is the accumulator itself; the stage-1 (unrotated) output is Pi^T of it,
    # which composes to the model basis: Pi^T Pi / D == I.
    q_rot = (query.to(torch.float64) @ rotation.transpose(0, 1)) * inv_sqrt
    out_stage0 = torch.zeros(batch, num_heads, emulator.head_size, dtype=torch.float64)
    for b in range(batch):
        length = int(context_lens[b].item())
        k = dense_key[b, :length].repeat_interleave(group, dim=1)  # [L, N, D]
        v = dense_value[b, :length].repeat_interleave(group, dim=1)
        scores = torch.einsum("nd,lnd->nl", q_rot[b], k) * scale
        scores = scores - scores.amax(dim=-1, keepdim=True)
        probs = torch.exp(scores)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        out_stage0[b] = torch.einsum("nl,lnd->nd", probs, v)
    out_stage1 = (out_stage0 @ rotation) * inv_sqrt
    return out_stage1.to(torch.float32), out_stage0.to(torch.float32)
