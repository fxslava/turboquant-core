# SPDX-License-Identifier: Apache-2.0
"""Exercise the TurboQuant CPU oracle fixtures without compiling or loading kernels."""

import hashlib
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "golden"))
from generate_dumps import generate_decode, generate_prefill, generate_scatter
from turboquant_golden import TurboQuantEmulator


def project(labels):
    a = 2 * ((labels >> 14) & 1) + ((labels >> 15) & 1) - 2
    b, c, d, e, f, g, h = [((labels >> shift) & 3) - 2 for shift in range(0, 14, 2)]
    return np.stack(
        (a, a + 2*b, a - 2*b + 2*c, a - 2*c + 2*d, a - 2*d + 2*e,
         a - 2*e + 2*f, a - 2*f + 2*g + 2*h, a - 2*g + 2*h),
        axis=1,
    )


def residuals(projected):
    shifted = projected + 4
    integer = (shifted & 7) - 4
    half = np.where(integer < 0, integer + 4, integer - 4)
    parity = np.bitwise_xor.reduce((shifted >> 3) & 1, axis=1)
    return integer, half, parity, parity ^ np.logical_xor.reduce(integer < 0, axis=1)


def frozen_shape(projected):
    integer, half, parity, half_parity = residuals(projected)
    rows = np.arange(len(projected))
    for candidate, correction in ((integer, parity), (half, half_parity)):
        index = np.zeros(len(candidate), dtype=np.int32)
        best = np.full(len(candidate), -1, dtype=np.int32)
        positive = np.zeros(len(candidate), dtype=bool)
        for coord in range(8):
            value = candidate[:, coord]
            absolute = np.abs(value)
            replace = (absolute > best) | ((absolute == best) & ~positive)
            index[replace] = coord
            best[replace] = absolute[replace]
            positive[replace] = value[replace] >= 0
        candidate[rows, index] += correction * np.where(positive, -8, 8)
    distance, half_distance = (np.sum(candidate**2, axis=1) for candidate in (integer, half))
    use_half = (half_distance < distance) | ((half_distance == distance) & (half[:, 0] < integer[:, 0]))
    return np.where(use_half[:, None], half, integer)


def parallel_shape(projected):
    integer, half, parity, half_parity = residuals(projected)
    rows, coordinates = np.arange(len(projected)), np.arange(8)
    distances, fixed = [], []
    for candidate, correction in ((integer, parity), (half, half_parity)):
        positive = candidate >= 0
        keys = (np.abs(candidate) << 8) | (positive << 7) | np.where(positive, 7 - coordinates, coordinates)
        index = np.argmax(keys, axis=1)
        key = keys[rows, index]
        distances.append(np.sum(candidate**2, axis=1) + correction * (64 - 16 * (key >> 8)))
        candidate[rows, index] += correction * np.where(key & 128, -8, 8)
        fixed.append(candidate)
    distance, half_distance = distances
    integer, half = fixed
    use_half = (half_distance < distance) | ((half_distance == distance) & (half[:, 0] < integer[:, 0]))
    return np.where(use_half[:, None], half, integer)


def bf16_bits(values):
    bits = values.view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def check_e8():
    labels = np.arange(65536, dtype=np.int32)
    projected = project(labels)
    expected, actual = frozen_shape(projected), parallel_shape(projected)
    np.testing.assert_array_equal(actual, expected)
    scales = (1, -1, 0.5, -0.5, 0.1, 127, 2**-126 - 2**-149, -2**-126 + 2**-149,
              2**-126, -2**-126, 2**-132, -2**-132, 0.0, -0.0, 16, 0.03125)
    for scale in scales:
        reference = expected.astype(np.float32) * np.float32(0.5) * np.float32(scale)
        candidate = actual.astype(np.float32) * np.float32(0.5) * np.float32(scale)
        np.testing.assert_array_equal(bf16_bits(candidate), bf16_bits(reference))
        np.testing.assert_array_equal(np.rint(candidate.clip(-128, 127)), np.rint(reference.clip(-128, 127)))
    owners, rows, channels = np.indices((2, 32, 256))
    src = (channels // 8) * 1056 + rows * 32 + channels % 8 * 4
    assert np.all(src % 1056 < 1024)
    np.testing.assert_array_equal(
        owners * 1024 + src // 1056 * 2048 + src % 1056,
        channels // 8 * 2048 + (owners * 32 + rows) * 32 + channels % 8 * 4,
    )
    bank = ((np.arange(32)[:, None] * 1056 + np.arange(8) * 4) // 32) % 16
    for column in bank.T:
        np.testing.assert_array_equal(np.bincount(column, minlength=16), np.full(16, 2))
    return {"labels": len(labels), "scales": len(scales), "coordinate_scale_pairs": actual.size * len(scales),
            "key_addresses": src.size}


def check_native():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    emulator = TurboQuantEmulator(256, "cpu")
    fixture_names = ("prefill", "scatter", "decode_ctx128", "decode_ctx200")
    recorded = ROOT / "golden/dumps"
    with TemporaryDirectory(prefix="tq-cpu-golden-") as work:
        generated = Path(work)
        generate_prefill(generated, emulator)
        generate_scatter(generated, emulator)
        generate_decode(generated, emulator, 128, fixture_names[2])
        generate_decode(generated, emulator, 200, fixture_names[3])
        compared = {}
        for name in fixture_names:
            folder = recorded / name
            tensors = {
                line.split()[-1]
                for line in (folder / "manifest.txt").read_text().splitlines()
                if line.startswith("tensor ")
            }
            binaries = {path.name for path in folder.glob("*.bin")}
            if not tensors or tensors != binaries:
                raise AssertionError(f"Recorded fixture inventory differs: {name}")
            for filename in sorted(tensors):
                relative = Path(name) / filename
                original = (recorded / relative).read_bytes()
                candidate = (generated / relative).read_bytes()
                if candidate != original:
                    raise AssertionError(f"CPU fixture changed: {relative}")
                compared[relative.as_posix()] = hashlib.sha256(candidate).hexdigest()
    return compared


if __name__ == "__main__":
    evidence = {"e8_python_model": check_e8(), "native_python_fixtures": check_native(),
                "torch": torch.__version__, "numpy": np.__version__, "cpp_builds": 0, "device_launches": 0}
    print(json.dumps(evidence, indent=2))
