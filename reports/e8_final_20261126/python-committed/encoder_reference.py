"""Verify the committed Python encoder against retained CPU fixture bytes."""

import hashlib
import json
import sys
from pathlib import Path

repo = Path.cwd()
sys.path[:0] = [str(repo), str(repo / "tests/ut/attention")]
import stub_vllm

stub_vllm.install()

import numpy as np
import torch

from vllm_ascend.attention import turboquant_reference_cache

fixture_root = Path(sys.argv[1])
destination = Path(sys.argv[2])
names = ("six_strip_rotated.f32le", "six_labels_oracle.u16le", "six_scales.f32le")
raw = np.fromfile(fixture_root / names[0], dtype="<f4").reshape(-1, 256)
labels = np.fromfile(fixture_root / names[1], dtype="<u2").reshape(-1, 32)
scales = np.fromfile(fixture_root / names[2], dtype="<u4")
assert len(raw) == len(labels) == len(scales)
actual_labels, actual_scales = turboquant_reference_cache.encode_e8_rotated(torch.from_numpy(raw[:32]) * 16)
cases = []
for rows in (16, 32):
    cases.append({
        "rows": rows,
        "label_mismatches": int(np.count_nonzero(actual_labels[:rows].numpy() != labels[:rows])),
        "scale_mismatches": int(np.count_nonzero(actual_scales[:rows].numpy().view(np.uint32) != scales[:rows])),
    })
source = Path(turboquant_reference_cache.__file__)
data = {
    "fixture_root": str(fixture_root),
    "retained_fixture_root": "/home/dmin/work/checkpoint2_20261123/parallel_encoder/raw/fixture",
    "input_basis": "Retained rows are in cache basis. Multiply by 16 exactly in FP32 to supply the unnormalized producer output expected by encode_e8_rotated.",
    "fixture_sha256": {name: hashlib.sha256((fixture_root / name).read_bytes()).hexdigest() for name in names},
    "source": str(source.relative_to(repo)),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "cases": cases,
    "device_launches": 0,
}
result = json.dumps(data, indent=2) + "\n"
destination.write_text(result, encoding="utf-8")
print(result)
assert all(case["label_mismatches"] == case["scale_mismatches"] == 0 for case in cases)
