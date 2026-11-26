"""Run committed plugin host-contract tests with device-module imports blocked."""

from pathlib import Path
import sys

root = Path.cwd()
sys.path[:0] = [str(root), str(root / "tests/ut/attention")]
import stub_vllm

stub_vllm.install()

import pytest

raise SystemExit(pytest.main([
    "--noconftest",
    "--import-mode=importlib",
    "-p", "no:cacheprovider",
    "-q",
    "tests/host/turboquant/test_source_sync.py",
    "tests/host/turboquant/test_build_selection.py",
    *sys.argv[1:]
]))
