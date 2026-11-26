"""Check benchmark selection across new builds and existing server archives."""

import runpy
import tempfile
import unittest
from pathlib import Path

CLI = runpy.run_path(
    str(
        Path(__file__).resolve().parents[2]
        / "benchmarks/benchmark_turboquant_production.py"
    )
)


class ExecutableSelectionTest(unittest.TestCase):
    def test_forwards_native_split_and_device_timing_options(self):
        args = CLI["cli_parser"]().parse_args(["--split-k", "8", "--samples"])
        options = CLI["command_options"](args)
        self.assertEqual(options[options.index("--split-k") + 1], "8")
        self.assertEqual(options[options.index("--timing") + 1], "graph")
        self.assertIn("--samples", options)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.build = Path(self.directory.name)

    def binary(self, directory, backend):
        path = self.build / directory / CLI["EXECUTABLES"][backend]
        path.parent.mkdir(exist_ok=True)
        path.touch()
        return str(path.resolve())

    def test_prefers_benchmark_directory_over_old_archive_layout(self):
        self.binary("tests", "npu")
        expected = self.binary("benchmarks", "npu")
        self.assertEqual(CLI["executable"](self.build, "npu"), expected)

    def test_prefers_bin_directory_for_every_native_backend(self):
        for backend in CLI["EXECUTABLES"]:
            with self.subTest(backend=backend):
                self.binary("tests", backend)
                self.binary("benchmarks", backend)
                expected = self.binary("bin", backend)
                self.assertEqual(CLI["executable"](self.build, backend), expected)

    def test_supports_existing_archive_layout(self):
        expected = self.binary("tests", "npu")
        self.assertEqual(CLI["executable"](self.build, "npu"), expected)

    def test_accuracy_and_host_backends_select_test_binaries(self):
        for backend in ("host", "npu-test"):
            with self.subTest(backend=backend):
                self.binary("benchmarks", backend)
                expected = self.binary("tests", backend)
                self.assertEqual(CLI["executable"](self.build, backend), expected)

    def test_missing_binary_reports_the_build_target(self):
        for backend, target in (
            ("host", "tq_host_tests"),
            ("npu", "tq_benchmarks"),
            ("npu-test", "tq_npu_tests"),
        ):
            with (
                self.subTest(backend=backend),
                self.assertRaisesRegex(FileNotFoundError, target),
            ):
                CLI["executable"](self.build, backend)


if __name__ == "__main__":
    unittest.main()
