"""Simulate the fused decode's fp8-rounded pipeline against a golden dump.

Calibrates against the doc's measurement: the OLD convention ((code+0.5)*scale/kGain
with gain folded) simulated with the kernel's two e4m3 roundings should land near the
doc's observed kernel-vs-kernel_stage residual (worst abs ~2.1). Then the NEW
convention (code*scale, no folds) tells us what the fixed kernel SHOULD score against
expected_stage*. If the device output is far off the simulation, the device has a
residual defect beyond the convention.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from turboquant_golden import (
    TurboQuantEmulator,
    decode_reference,
    gather_nz_cache,
    TURBOQUANT_KERNEL_HEAD_SIZE,
    TURBOQUANT_KERNEL_KV_HEADS,
)

GAIN = 2.9832882881
EPS = 1e-20  # TurboQuantCodec4::kEps, per the codec headers
FP8_MAX = 448.0


def fp8e4m3(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float8_e4m3fn).to(torch.float64)


def read_tensor(case: Path, name: str, dtype: torch.dtype) -> torch.Tensor:
    import re

    manifest = (case / "manifest.txt").read_text()
    m = re.search(rf"^tensor {name} \S+ (\d+) (\S+)$", manifest, re.M)
    count, file = int(m.group(1)), m.group(2)
    raw = (case / file).read_bytes()
    t = torch.frombuffer(bytearray(raw), dtype=dtype).clone()
    assert t.numel() == count, (name, t.numel(), count)
    return t


def simulate(case: str, bias: float, gain: float, round_query: bool = True, round_probs: bool = True) -> None:
    case_dir = Path(case)
    query_rot = read_tensor(case_dir, "query_rot", torch.float32).reshape(1, -1, 256).to(torch.float64)
    key_cache = read_tensor(case_dir, "key_cache", torch.int8)
    value_cache = read_tensor(case_dir, "value_cache", torch.int8)
    scale_cache = read_tensor(case_dir, "scale_cache", torch.float32)
    block_table = read_tensor(case_dir, "block_table", torch.int32).to(torch.int64).unsqueeze(0)
    context_lens = read_tensor(case_dir, "context_lens", torch.int32)
    expected0 = read_tensor(case_dir, "expected_stage0", torch.float32)
    expected1 = read_tensor(case_dir, "expected_stage1", torch.float32)
    kernel0 = read_tensor(case_dir, "kernel_stage0", torch.float32)
    kernel1 = read_tensor(case_dir, "kernel_stage1", torch.float32)

    geom = dict(
        [line.split()[1:3] for line in (case_dir / "manifest.txt").read_text().splitlines() if line.startswith("geom")]
    )
    num_heads, kv_heads, block_size = int(geom["num_heads"]), int(geom["num_kv_heads"]), int(geom["block_size"])
    context = int(context_lens[0])
    scale = float([l for l in (case_dir / "manifest.txt").read_text().splitlines() if l.startswith("float scale")][0].split()[2])
    pages = block_table.shape[1]
    while pages > 1 and block_table[0, pages - 1] < 0:
        pages -= 1
    block_table = block_table[:, :pages]

    codes_k, scales_k, codes_v, scales_v = gather_nz_cache(
        key_cache, value_cache, scale_cache, block_table, block_size, kv_heads, 256
    )
    group = num_heads // kv_heads
    # The kernel's K operand is (code + bias) and the score scale carries softmax/gain;
    # equivalently the dequantised key below. The V GEMM, though, runs on the raw
    # (code + bias) operand and the stored value scale enters ONLY through the prob
    # operand's scaling -- so the two are kept separate here on purpose.
    k = ((codes_k[0, :context].to(torch.float64) + bias) * scales_k[0, :context].to(torch.float64).unsqueeze(-1) / gain)
    v_operand = (codes_v[0, :context].to(torch.float64) + bias)
    v_scale = scales_v[0, :context].to(torch.float64) / gain
    k = k.repeat_interleave(group, dim=1)
    v_operand = v_operand.repeat_interleave(group, dim=1)
    v_scale = v_scale.repeat_interleave(group, dim=1)

    q = query_rot[0]  # [N, D] fp64
    # Kernel query leg: per-head amax scale to the fp8 max, e4m3 rounding, later undone.
    amax = q.abs().amax(dim=-1, keepdim=True)
    q_scale = FP8_MAX / (amax + EPS)
    q_hat = fp8e4m3(q * q_scale) / q_scale if round_query else q

    scores = torch.einsum("nd,lnd->nl", q_hat, k) * scale
    # Online softmax with running max (fp64 stands in for the kernel's fp32; the fp8
    # roundings are the effect under test).
    tiles = (context + 63) // 64
    run_max = torch.full((num_heads,), float("-inf"), dtype=torch.float64)
    run_sum = torch.zeros(num_heads, dtype=torch.float64)
    acc = torch.zeros(num_heads, 256, dtype=torch.float64)
    for t in range(tiles):
        rows = slice(64 * t, min(64 * (t + 1), context))
        s = scores[:, rows]
        tile_max = s.amax(dim=-1)
        new_max = torch.maximum(run_max, tile_max)
        alpha = torch.exp(run_max - new_max)
        p = torch.exp(s - new_max.unsqueeze(-1))
        run_sum = run_sum * alpha + p.sum(dim=-1)
        # Prob operand: scaled by the row's value scale, max-mapped to 448, e4m3.
        pv = p * v_scale[rows].t()
        part = pv.amax(dim=-1, keepdim=True)
        prob_scale = FP8_MAX / (part + EPS)
        p_hat = fp8e4m3(pv * prob_scale) / prob_scale if round_probs else pv
        ctx = torch.einsum("nl,lnd->nd", p_hat, v_operand[rows])
        acc = acc * alpha.unsqueeze(-1) + ctx
        run_max = new_max
    out0 = acc / (run_sum + EPS).unsqueeze(-1)

    emulator = TurboQuantEmulator(256, "cpu")
    rotation = emulator.rotation64()
    out1 = (out0 @ rotation) * emulator.inv_sqrt_len

    def report(name: str, got: torch.Tensor, want: torch.Tensor) -> None:
        d = (got - want.to(torch.float64)).abs()
        w = want.to(torch.float64).abs()
        rel = d / w.clamp_min(1e-30)
        ratio = (got * want).sum() / (want.to(torch.float64) ** 2).sum()
        # The honest shape of an fp8-operand residual: relative on ordinary channels,
        # absolute near zero. Report both tails and the ULP-style count over 1% / 5%.
        big = w > 0.05 * w.max()
        rowmax = w.reshape(w.shape[0], -1).amax(dim=-1)  # per (head) row
        rel_row = (d.reshape(w.shape[0], -1) / rowmax.unsqueeze(-1)).max()
        print(f"    {name}: worst abs {d.max():.4g}, scale ratio {ratio:.6f}, "
              f"rowmax {w.max():.4g}, worst |d|/rowmax {rel_row:.5f}")
        print(f"        worst rel {rel.max():.4g}; on channels >5% of max ({int(big.sum())}): "
              f"worst rel {rel[big].max():.4g}, over 1%: {int((rel[big] > 0.01).sum())}, "
              f"over 3%: {int((rel[big] > 0.03).sum())}; "
              f"worst abs on the rest: {d[~big].max():.4g}")

    print(f"  case {case_dir.name}, bias {bias}, gain {gain}:")
    report("stage0 vs golden ", out0, expected0.reshape(out0.shape))
    report("stage1 vs golden ", out1, expected1.reshape(out1.shape))
    report("stage0 vs krn conv", out0, kernel0.reshape(out0.shape))
    report("stage1 vs krn conv", out1, kernel1.reshape(out1.shape))


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "dumps"
    for case in ("decode_ctx128", "decode_ctx200"):
        simulate(str(root / case), 0.5, GAIN)  # the OLD kernel convention
        simulate(str(root / case), 0.0, 1.0)  # the writers'/golden convention, both roundings
        simulate(str(root / case), 0.0, 1.0, round_probs=False)  # query rounding only
        simulate(str(root / case), 0.0, 1.0, round_query=False)  # prob rounding only
        simulate(str(root / case), 0.0, 1.0, round_query=False, round_probs=False)  # neither
