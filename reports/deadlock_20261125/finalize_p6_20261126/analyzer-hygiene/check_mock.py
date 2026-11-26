"""Exercise the checker source join with mock analyzer output."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("consumer_check", HERE / "check_analyzer.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def report(path, diagnostics=(), *, complete=True):
    return {
        "source": {"path": str(path), "had_parse_errors": not complete, "parse_error_locations": []},
        "kernels": [{"name": "stub"}],
        "summary": {"total": len(diagnostics)},
        "diagnostics": list(diagnostics),
    }


def rejects(document, inputs, message):
    try:
        checker.primary_source_report(document, inputs)
    except RuntimeError as error:
        assert message in str(error), error
    else:
        raise AssertionError(f"accepted malformed response: {document}")


def exercise_join():
    narrow, wide, unrelated = map(Path, ("/narrow.cpp", "/wide.cpp", "/wrong.cpp"))
    records = [report(wide), report(narrow)]
    assert checker.primary_source_report({"files": records}, [narrow, wide]) == records[1]
    for wrong in ([report(narrow)], [report(narrow), report(narrow)], [report(narrow), report(unrelated)]):
        rejects({"files": wrong}, [narrow, wide], "sources differ")
    rejects({"files": [report(narrow), report(wide, complete=False)]}, [narrow, wide], "incomplete parse")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def exercise_checker(root):
    marker = {"code": "AKA3002", "severity": "INFO", "hardware_domain": "-", "message": "mock companion finding"}
    plugin, native, reshape, e8, output = [root / name for name in ("plugin", "native", "reshape", "e8", "out")]
    batch = "turboquant_reshape_batch"
    tq = plugin / "csrc/attention/turboquant"
    narrow = tq / "ingress/native" / (batch + ".cpp")
    wide = tq / "ingress/native" / (batch + "_wide.cpp")
    old = reshape / "src/prefill" / (batch + ".cpp")
    for source in (narrow, wide, old):
        write(source, "int source;\n")
    write(e8 / "e8_fused_decode.cpp", "// @ascend-layout: name=mock\n")
    write(tq / "decode/e8/reader_layout.aka", "@ascend-layout: name=mock\n")
    checked = ("reshape", "reshape_nz", "reshape_wide", "reshape_wide_nz")
    cases = [arg for case in checked for arg in ("--case", case)]
    args = [str(HERE / "check_analyzer.py"), "--analyzer", str(root), "--native", str(native),
            "--reshape", str(reshape), "--e8", str(e8), "--out", str(output), *cases]
    invocations = []

    def fake_run(command, **_):
        assert "--ignore-inline-pragmas" in command and "--suppress" not in command
        files = command[3:command.index("--chip")]
        invocations.append(files)
        units = [report(file, (marker,) if len(files) == 2 and file == files[1] else ()) for file in files]
        payload = {"files": units[::-1]} if len(units) > 1 else units[0]
        Path(command[-1]).write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch.object(checker, "ROOT", plugin), patch.object(sys, "argv", args), patch.object(subprocess, "run", fake_run):
        checker.main()
    assert invocations == [
        [str(old)], [str(narrow), str(wide)],
        [str(output / "reshape-nz-verified.cpp")],
        [str(output / "reshape-nz-integrated.cpp"), str(output / "reshape_wide-nz-integrated.cpp")],
        [str(old)], [str(wide)],
        [str(output / "reshape_wide-nz-verified.cpp")], [str(output / "reshape_wide-nz-integrated.cpp")],
    ]
    for case in ("reshape", "reshape_nz"):
        raw = json.loads((output / f"{case}-integrated.json").read_text())
        assert raw["files"][0]["diagnostics"] == [marker] and raw["files"][1]["diagnostics"] == []
    summaries = json.loads((output / "summary.json").read_text())
    assert set(summaries) == set(checked)
    assert all(row["new"] == [] for row in summaries.values())
    assert all("launch_companion_sha256" in summaries[case] for case in ("reshape", "reshape_nz"))


if __name__ == "__main__":
    exercise_join()
    with TemporaryDirectory(prefix="tq-analyzer-source-join-") as directory:
        exercise_checker(Path(directory))
    print("PASS mocked source order, complete source set, raw preservation, and scoped STRIP/NZ invocations")
