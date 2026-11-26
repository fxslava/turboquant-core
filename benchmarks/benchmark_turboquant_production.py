"""Run the TurboQuant production matrix or compare staged PyTorch reference codecs."""

from __future__ import annotations

import argparse
import math
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIM = 256
TILE_ROWS = 64
LABEL_BYTES = 4096
PACKET_BYTES = 4352
PATTERN_PAGES = 5
REFERENCE_WAVE_PAGES = 16
LM4_CENTERS = (
    -2.75,
    -2.0,
    -1.625,
    -1.25,
    -0.9375,
    -0.6875,
    -0.375,
    -0.125,
    0.125,
    0.375,
    0.6875,
    0.9375,
    1.25,
    1.625,
    2.0,
    2.75,
)
EXECUTABLES = {
    "host": "test_turboquant_production_host",
    "npu": "bench_turboquant_production_npu",
    "npu-test": "test_turboquant_production_npu",
}
MODES = ("lm4", "lm3", "e8")
STAGES = ("ingress", "dequant", "decode")


def cli_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=(*EXECUTABLES, "torch-cpu", "torch-npu"), default="host"
    )
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-e8-production")
    parser.add_argument(
        "--host-build-dir", type=Path, help="Host profile catalog, if built separately"
    )
    parser.add_argument("--mode", choices=("all", *MODES), default="all")
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument(
        "--profile", default="all", help="Profile name from --backend host --list"
    )
    parser.add_argument("--block-size", type=int, choices=(64, 128))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print applicable profiles and native launch limits",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Run CPU golden and canary checks"
    )
    return parser


def executable(build, backend):
    name = EXECUTABLES[backend]
    directories = ("bin", "benchmarks", "tests") if backend == "npu" else ("bin", "tests")
    for directory in directories:
        candidate = (build / directory / name).resolve()
        if candidate.is_file():
            return str(candidate)
    target = {
        "host": "tq_host_tests",
        "npu": "tq_benchmarks",
        "npu-test": "tq_npu_tests",
    }[backend]
    raise FileNotFoundError(
        f"{name} is missing in {build}. Build it with: "
        f"cmake --build {build} --target {target}. "
        "NPU targets require -DTQ_BUILD_KERNELS=ON -DRUN_MODE=npu. "
        "Set -DTQ_BUILD_E8_KERNELS=ON to include E8 device rows."
    )


def command_options(args):
    options = [
        "--mode",
        args.mode,
        "--stage",
        args.stage,
        "--profile",
        args.profile,
        "--device",
        str(args.device),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
    ]
    if args.block_size is not None:
        options += ["--block-size", str(args.block_size)]
    if args.list:
        options.append("--list")
    if args.verify:
        options.append("--verify")
    return options


def catalog(args):
    options = [
        "--list",
        "--mode",
        "lm4",
        "--stage",
        "decode",
        "--profile",
        args.profile,
    ]
    if args.block_size is not None:
        options += ["--block-size", str(args.block_size)]
    host = executable(args.host_build_dir or args.build_dir, "host")
    result = subprocess.run(
        [host, *options], check=True, capture_output=True, text=True
    )
    for line in result.stdout.splitlines():
        fields = dict(word.split("=", 1) for word in line.split() if "=" in word)
        yield (
            fields["profile"],
            {
                key: int(fields[key])
                for key in (
                    "block",
                    "seqs",
                    "context",
                    "kv_heads",
                    "group",
                    "split_k",
                    "ragged",
                )
            },
        )


def gaussian_lm3():
    centers = [-2.2, -1.3, -0.7, -0.2, 0.2, 0.7, 1.3, 2.2]
    density_scale = 1 / math.sqrt(2 * math.pi)
    for _ in range(1000):
        boundaries = [(a + b) / 2 for a, b in zip(centers, centers[1:])]
        densities = [
            0,
            *(density_scale * math.exp(-0.5 * value**2) for value in boundaries),
            0,
        ]
        cumulative = [
            0,
            *(0.5 * math.erfc(-value / math.sqrt(2)) for value in boundaries),
            1,
        ]
        centers = [
            (densities[i] - densities[i + 1]) / (cumulative[i + 1] - cumulative[i])
            for i in range(len(centers))
        ]
    return centers


def e8_codebook(torch):
    label = torch.arange(65536, dtype=torch.int32)
    a = 2 * ((label >> 14) & 1) + ((label >> 15) & 1) - 2
    b, c, d, e, f, g, h = [((label >> shift) & 3) - 2 for shift in range(0, 14, 2)]
    projection = torch.stack(
        [
            a,
            a + 2 * b,
            a - 2 * b + 2 * c,
            a - 2 * c + 2 * d,
            a - 2 * d + 2 * e,
            a - 2 * e + 2 * f,
            a - 2 * f + 2 * g + 2 * h,
            a - 2 * g + 2 * h,
        ],
        dim=-1,
    )
    folded = (projection + 4) & 7
    parity = (((projection + 4) >> 3) & 1).sum(dim=-1) & 1
    integer = folded - 4
    half = integer + torch.where(integer < 0, 4, -4)
    half_parity = parity ^ ((integer < 0).sum(dim=-1) & 1)

    def repair(residual, needs_repair):
        index = torch.arange(8, dtype=torch.int32)
        positive = residual >= 0
        priority = (
            residual.abs() * 256
            + positive * 128
            + torch.where(positive, 7 - index, index)
        )
        preferred = priority.argmax(dim=-1)
        delta = torch.where(positive, -8, 8)
        return residual + delta * needs_repair[:, None] * (index == preferred[:, None])

    integer, half = repair(integer, parity), repair(half, half_parity)
    i_distance, h_distance = (integer**2).sum(dim=-1), (half**2).sum(dim=-1)
    use_half = (h_distance < i_distance) | (
        (h_distance == i_distance) & (half[:, 0] < integer[:, 0])
    )
    return torch.where(use_half[:, None], half, integer).float() * 0.5


def sample_codes(torch, block, heads, mode, plane):
    rows = torch.arange(PATTERN_PAGES * block, dtype=torch.int64).reshape(
        PATTERN_PAGES, 1, block, 1
    )
    kv = torch.arange(heads, dtype=torch.int64).reshape(1, heads, 1, 1)
    width = DIM // 8 if mode == "e8" else DIM
    channel = torch.arange(width, dtype=torch.int64)
    mask = 0xFFFFFFFF
    value = (
        1 + rows * 0x9E3779B9 + kv * 0x85EBCA6B + plane * 0xC2B2AE35 + channel
    ) & mask
    value = ((value ^ (value >> 16)) * 0x7FEB352D) & mask
    value = ((value ^ (value >> 15)) * 0x846CA68B) & mask
    levels = {"lm4": 16, "lm3": 8, "e8": 65536}[mode]
    return (value ^ (value >> 16)) % levels, (
        (2 + (rows + kv + plane) % 13) / 32
    ).float()


def pack_pattern(torch, mode, codes, scales):
    if mode == "e8":
        pages, heads, block, _ = codes.shape
        tiles = block // TILE_ROWS
        labels = (
            codes.to(torch.uint16)
            .contiguous()
            .view(torch.uint8)
            .reshape(pages, heads, tiles, LABEL_BYTES)
        )
        raw_scales = (
            scales.contiguous()
            .view(torch.uint8)
            .reshape(pages, heads, tiles, PACKET_BYTES - LABEL_BYTES)
        )
        return torch.cat((labels, raw_scales), dim=-1)
    width = 4 if mode == "lm4" else 3
    cluster = codes.reshape(*codes.shape[:-1], -1, 8)
    word = ((cluster << (torch.arange(8) * width)).sum(dim=-1)).unsqueeze(-1)
    return ((word >> (torch.arange(width) * 8)) & 255).to(torch.uint8).flatten(-2)


class Reference:
    def __init__(self, torch, device, mode, profile):
        self.torch, self.device, self.mode, self.profile = torch, device, mode, profile
        block, heads = profile["block"], profile["kv_heads"]
        self.lengths = [
            profile["context"] - (1 + (seq * 13 + 7) % 63 if profile["ragged"] else 0)
            for seq in range(profile["seqs"])
        ]
        self.num_pages = [math.ceil(length / block) for length in self.lengths]
        self.total_pages = sum(self.num_pages)
        self.pattern = [
            sample_codes(torch, block, heads, mode, plane) for plane in range(2)
        ]
        centers = (
            e8_codebook(torch)
            if mode == "e8"
            else torch.tensor(LM4_CENTERS if mode == "lm4" else gaussian_lm3())
        )
        self.lut_cpu = centers
        self.lut = centers.to(device)
        self.packed = []
        for codes, scales in self.pattern:
            packed = pack_pattern(torch, mode, codes, scales)
            if mode == "e8":
                stored_labels = (
                    packed[..., :LABEL_BYTES]
                    .contiguous()
                    .view(torch.uint16)
                    .reshape_as(codes)
                )
                stored_scales = (
                    packed[..., LABEL_BYTES:]
                    .contiguous()
                    .view(torch.float32)
                    .reshape_as(scales)
                )
                if not torch.equal(
                    stored_labels.to(torch.int64), codes
                ) or not torch.equal(stored_scales, scales):
                    raise RuntimeError(
                        "E8 packet labels or scales disagree with CPU golden"
                    )
            repeats = (math.ceil(self.total_pages / PATTERN_PAGES),) + (1,) * (
                packed.ndim - 1
            )
            payload = packed.repeat(repeats)[: self.total_pages].to(device)
            scale_payload = (
                None
                if mode == "e8"
                else scales.repeat(repeats)[: self.total_pages].to(device)
            )
            self.packed.append((payload, scale_payload))
        self.shifts = (
            torch.arange(8, dtype=torch.int32) * (4 if mode == "lm4" else 3)
        ).to(device)
        self.query = [self.make_query(seq).to(device) for seq in range(profile["seqs"])]

    def make_query(self, seq):
        p, torch = self.profile, self.torch
        heads = p["kv_heads"] * p["group"]
        values = torch.arange(heads)
        query = torch.zeros((heads, DIM))
        query[values, (seq * 31 + values * 37) % DIM] = (8 + (seq + values) % 7) * 0.5
        return query.reshape(p["kv_heads"], p["group"], DIM)

    def dequant(self, plane, first, last):
        torch, p = self.torch, self.profile
        packed, scales = self.packed[plane]
        payload = packed[first:last]
        if self.mode == "e8":
            labels = (
                payload[..., :LABEL_BYTES]
                .reshape(*payload.shape[:-1], -1, 2)
                .to(torch.int32)
            )
            code = labels[..., 0] | (labels[..., 1] << 8)
            samples = self.lut.index_select(0, code.flatten()).reshape(
                last - first, p["kv_heads"], p["block"], DIM
            )
            rows = payload[..., LABEL_BYTES:].contiguous().view(torch.float32)
            return (
                (samples * rows.reshape(*samples.shape[:-1], 1))
                .to(torch.bfloat16)
                .float()
            )
        width = self.shifts.shape[0] * payload.shape[-1] // DIM
        clusters = payload.reshape(*payload.shape[:-1], -1, width).to(torch.int32)
        word = clusters[..., 0]
        for byte in range(1, width):
            word = word | (clusters[..., byte] << (8 * byte))
        code = ((word.unsqueeze(-1) >> self.shifts) & ((1 << width) - 1)).reshape(
            *payload.shape[:-1], DIM
        )
        samples = self.lut.index_select(0, code.flatten()).reshape(code.shape)
        return samples * scales[first:last]

    def decode(self):
        torch, p = self.torch, self.profile
        results = []
        physical_end = self.total_pages
        for seq, (length, pages) in enumerate(zip(self.lengths, self.num_pages)):
            q = self.query[seq]
            max_score = total = accumulator = None
            split_pages = math.ceil(pages / p["split_k"])
            for split_begin in range(0, pages, split_pages):
                split_end = min(pages, split_begin + split_pages)
                for begin in range(split_begin, split_end, REFERENCE_WAVE_PAGES):
                    end = min(split_end, begin + REFERENCE_WAVE_PAGES)
                    rows = min(length, end * p["block"]) - begin * p["block"]
                    keys, values = (
                        self.dequant(plane, physical_end - end, physical_end - begin)
                        .flip((0,))
                        .transpose(0, 1)
                        .reshape(p["kv_heads"], -1, DIM)[:, :rows]
                        for plane in range(2)
                    )
                    score = (q @ keys.transpose(-1, -2)) / math.sqrt(DIM)
                    local_max = score.amax(dim=-1, keepdim=True)
                    updated_max = (
                        local_max
                        if max_score is None
                        else torch.maximum(max_score, local_max)
                    )
                    weight = torch.exp(score - updated_max)
                    local_total = weight.sum(dim=-1, keepdim=True)
                    local_accumulator = weight @ values
                    if max_score is None:
                        total, accumulator = local_total, local_accumulator
                    else:
                        shrink = torch.exp(max_score - updated_max)
                        total, accumulator = (
                            total * shrink + local_total,
                            accumulator * shrink + local_accumulator,
                        )
                    max_score = updated_max
            results.append((accumulator / total).to(torch.bfloat16))
            physical_end -= pages
        return torch.stack(results)

    def golden(self, stage):
        torch, p = self.torch, self.profile
        planes = []
        for codes, scales in self.pattern:
            centers = self.lut_cpu[codes].reshape(*scales.shape[:-1], DIM)
            values = centers * scales
            if self.mode == "e8":
                values = values.to(torch.bfloat16).float()
            planes.append(values)
        if stage == "dequant":
            return torch.stack([value[:1] for value in planes])
        k, v = [
            value.transpose(0, 1).reshape(p["kv_heads"], -1, DIM).double()
            for value in planes
        ]
        expected = []
        physical_end = self.total_pages
        for seq, (length, pages) in enumerate(zip(self.lengths, self.num_pages)):
            counts = torch.zeros((PATTERN_PAGES, p["block"]), dtype=torch.float64)
            for page in range(pages):
                rows = min(p["block"], length - page * p["block"])
                counts[(physical_end - 1 - page) % PATTERN_PAGES, :rows] += 1
            logits = (
                self.make_query(seq).double() @ k.transpose(-1, -2) / math.sqrt(DIM)
            )
            weights = logits.exp() * counts.flatten()[None, None, :]
            expected.append((weights @ v) / weights.sum(dim=-1, keepdim=True))
            physical_end -= pages
        return torch.stack(expected)

    def effective_bytes(self, stage):
        p = self.profile
        rows = (
            p["block"]
            if stage == "dequant"
            else sum(math.ceil(n / TILE_ROWS) * TILE_ROWS for n in self.lengths)
        )
        payload = {"lm4": 128, "lm3": 96, "e8": 64}[self.mode]
        written = 4 * DIM if stage == "dequant" else 0
        return rows * p["kv_heads"] * 2 * (payload + 4 + written)


def verify_output(torch, actual, expect, stage):
    actual = actual.cpu()
    if not actual.numel() or actual.shape != expect.shape:
        raise RuntimeError("reference output shape disagrees with CPU golden")
    a, b = actual.double().flatten(), expect.double().flatten()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise RuntimeError("reference output contains a non-finite value")
    delta = a - b
    rmse = delta.square().mean().sqrt().item()
    a_norm, b_norm = a.norm().item(), b.norm().item()
    if a_norm and b_norm:
        cosine = (a @ b).item() / (a_norm * b_norm)
    else:
        cosine = float(a_norm == b_norm)

    def ordered(x):
        bits = x.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, 0x8000 - (bits & 0x7FFF), 0x8000 + bits)

    ulp = (ordered(a) - ordered(b)).abs().max().item()
    if (
        cosine < 0.99999
        or delta.norm().item() > 0.003 * b_norm
        or (stage == "dequant" and ulp != 0)
    ):
        raise RuntimeError(
            f"reference accuracy failed, RMSE={rmse}, cosine={cosine}, BF16 ULP={ulp}"
        )
    return f"rmse={rmse:.8g} cosine={cosine:.9f} max_bf16_ulp={ulp}"


def measure(torch, use_npu, args, launch):
    for _ in range(args.warmup):
        launch()
    if use_npu:
        torch.npu.synchronize()
        events = [
            (torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True))
            for _ in range(args.iterations)
        ]
        for start, end in events:
            start.record()
            launch()
            end.record()
        events[-1][1].synchronize()
        samples = [start.elapsed_time(end) * 1000 for start, end in events]
    else:
        samples = []
        for _ in range(args.iterations):
            started = time.perf_counter_ns()
            launch()
            samples.append((time.perf_counter_ns() - started) / 1000)
    if any(duration <= 0 or not math.isfinite(duration) for duration in samples):
        raise RuntimeError("benchmark produced an invalid elapsed time")
    ordered = sorted(samples)
    return (
        statistics.median(ordered),
        ordered[math.ceil(0.95 * len(ordered)) - 1],
        statistics.mean(ordered),
    )


def torch_reference(args):
    profiles = list(catalog(args))
    modes = MODES if args.mode == "all" else (args.mode,)
    stages = STAGES if args.stage == "all" else (args.stage,)
    if args.list:
        for name, profile in profiles:
            print(name, profile, "staged_reference_stages=dequant,decode")
        return 0
    if stages == ("ingress",):
        raise ValueError("the staged reference supports --stage dequant or decode")
    import torch

    use_npu = args.backend == "torch-npu"
    if use_npu:
        import torch_npu  # noqa: F401

        device = f"npu:{args.device}"
        torch.npu.set_device(device)
    else:
        device = "cpu"
        torch.set_num_threads(1)
    with torch.inference_mode():
        for name, profile in profiles:
            geometry = " ".join(f"{key}={value}" for key, value in profile.items())
            for mode in modes:
                fixture = Reference(torch, device, mode, profile)
                for stage in stages:
                    label = f"implementation=staged_reference mode={mode} stage={stage} profile={name} {geometry}"
                    if stage == "ingress":
                        print(
                            f"{label} status=UNSUPPORTED reason=reference_begins_with_prepacked_cache",
                            flush=True,
                        )
                        continue

                    def launch():
                        if stage == "dequant":
                            return torch.stack(
                                [fixture.dequant(plane, 0, 1) for plane in range(2)]
                            )
                        return fixture.decode()

                    if args.verify:
                        report = verify_output(
                            torch, launch(), fixture.golden(stage), stage
                        )
                        print(f"{label} status=PASS {report}", flush=True)
                    else:
                        p50, p95, mean = measure(torch, use_npu, args, launch)
                        size = fixture.effective_bytes(stage)
                        print(
                            f"{label} status=MEASURED p50_us={p50:.3f} p95_us={p95:.3f} mean_us={mean:.3f} "
                            f"effective_bytes={size} effective_GB_s={size / mean / 1000:.3f}",
                            flush=True,
                        )
    return 0


def main():
    parser = cli_parser()
    args = parser.parse_args()
    if args.warmup < 0 or not 1 <= args.iterations <= 512 or args.device < 0:
        parser.error(
            "warmup and device must be nonnegative and iterations must be in [1, 512]"
        )
    try:
        if args.backend in EXECUTABLES:
            return subprocess.run(
                [executable(args.build_dir, args.backend), *command_options(args)]
            ).returncode
        return torch_reference(args)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
