"""Synthetic fixtures test parsing only; these are not NPU measurements."""

import csv
import runpy
import tempfile
import unittest
from pathlib import Path

SUMMARY = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/summarize_e8_profile.py"))


class ProfileSummaryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_rows(self, rows):
        path = self.root / "op_summary_fixture.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as target:
            writer = csv.DictWriter(target, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def test_keeps_reader_and_reducer_separate_and_na_null(self):
        self.write_rows([
            {"Op Name": "e8_fused_decode_fp32", "Task Duration(us)": "12", "Block Dim": "128", "aiv_vec_ratio": "0.5", "aic_mte2_ratio": "0.25"},
            {"Op Name": "e8_splitk_reduce_fp32", "Task Duration(us)": "2", "Block Dim": "96", "aiv_vec_ratio": "0.75", "aic_mte2_ratio": "N/A"},
        ])
        rows = SUMMARY["op_rows"](self.root)
        self.assertEqual(len(rows), 2)
        self.assertIsNone(rows[1]["aic_mte2_ratio"])
        self.assertEqual(rows[0]["block_dim"], "128")
        self.assertEqual(rows[0]["task_duration_us"], 12)

    def test_missing_counter_group_is_an_error(self):
        self.write_rows([{"Op Name": "e8_fused_decode_fp32", "Task Duration(us)": "12",
                          "Block Dim": "128", "aiv_vec_ratio": "N/A", "aic_mte2_ratio": "0.25"}])
        with self.assertRaisesRegex(ValueError, "did not export aiv_vec_ratio"):
            SUMMARY["op_rows"](self.root)

    def test_missing_counters_fail_instead_of_becoming_zero(self):
        self.write_rows([{"Op Name": "e8_fused_decode_fp32", "Task Duration(us)": "1", "Block Dim": "8"}])
        with self.assertRaisesRegex(ValueError, "did not export"):
            SUMMARY["op_rows"](self.root)

    def test_retains_actual_event_order_and_rejects_truncation(self):
        path = self.root / "events.log"
        text = "mode=e8 stage=decode timing_source=aclrtEventElapsedTime samples=2 device_samples_us=12,11 status=MEASURED"
        path.write_text(text, encoding="utf-8")
        self.assertEqual(SUMMARY["event_cases"](path)[0]["aclrt_event_samples_us"], [12, 11])
        path.write_text(text.replace("samples=2", "samples=3"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "all samples"):
            SUMMARY["event_cases"](path)


if __name__ == "__main__":
    unittest.main()
