"""Bounded CPU checks for the staged benchmark's BF16 accuracy gate."""

import runpy
import unittest
from pathlib import Path

import torch

BENCHMARK = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "benchmark_turboquant_production.py"
)
VERIFY_OUTPUT = runpy.run_path(str(BENCHMARK))["verify_output"]


class ReferenceMetricsTest(unittest.TestCase):
    def test_two_zero_vectors_are_exact(self):
        zero = torch.zeros(8, dtype=torch.bfloat16)
        for stage in ("dequant", "decode"):
            report = VERIFY_OUTPUT(torch, zero, zero.double(), stage)
            self.assertIn("rmse=0 cosine=1.000000000 max_bf16_ulp=0", report)

    def test_only_one_zero_vector_fails(self):
        zero = torch.zeros(8, dtype=torch.bfloat16)
        perturbed = zero.clone()
        perturbed[0] = 1.0
        for actual, expected in ((zero, perturbed), (perturbed, zero)):
            with self.assertRaisesRegex(RuntimeError, "cosine=0.0"):
                VERIFY_OUTPUT(torch, actual, expected.double(), "decode")

    def test_non_finite_value_fails(self):
        zero = torch.zeros(8, dtype=torch.bfloat16)
        bad = zero.clone()
        bad[0] = float("nan")
        for actual, expected in ((zero, bad), (bad, zero)):
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                VERIFY_OUTPUT(torch, actual, expected.double(), "decode")


if __name__ == "__main__":
    unittest.main()
