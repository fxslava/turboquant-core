"""Bind completed P6 syntax and rerun just the affected analyzer comparisons."""

import argparse
import copy
import json
from pathlib import Path
import re
import sys


HERE = Path(__file__).resolve().parent
PROMOTION = HERE.parent / "slots/P/r6/promotion"
sys.path.insert(0, str(PROMOTION))
import deployed_aka_and_syntax as gate
from post_host import snapshot
from promote_p6 import check, git, json_data, read, rec, write_new


DEPLOYED = Path("/home/dmin/work/deadlock_20261125/slotP/r6/promotion-final-r1")
APPLIED = DEPLOYED / "PROMOTION_APPLIED.json"
APPLIED_SHA = "6c73337efe9229f41f885351a144198392a243cbd0c8ce76a0625858ce396664"
HOST = DEPLOYED / "final-host/FINAL_HOST_RESULTS.json"
HOST_SHA = "28fe079c8703066de0e567f75bb05ad4114776a720b7dae7d7004c8c6cc941bd"
PREVIOUS = DEPLOYED / "deployed-aka-syntax"
FROZEN = PREVIOUS / "frozen-and-unaffected"
CONSUMER = "tests/host/turboquant/check_analyzer.py"
UNAFFECTED = ("decode", "vector", "softmax", "scatter", "scatter_nz", "e8_component", "e8_reader", "e8_reader_unannotated")
NARROW = ("reshape", "reshape_nz")
WIDE = ("reshape_wide", "reshape_wide_nz")
INCLUDE = re.compile(rb'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)
FAILURE = HERE / "AKA_HOST_SCOPE_FAILURE_AUDIT.json"
FAILURE_SHA = "9d8727f6fba3c403059741622edc94e9458480c859dc70088fd0eb5d7e62eb56"
HYGIENE = HERE / "analyzer-hygiene/CONSUMER_HYGIENE_DELTA.json"
HYGIENE_SHA = "922331ee5fe5dce362c78493cb4f8cb44cb8e63f1395222a3426c8155a08264c"


def pinned(path, sha):
    found = rec(path)
    assert found["sha256"] == sha, found
    return found


def require_original_inputs():
    return [pinned(path, digest) for path,digest in gate.PINS.items()]


def require_analyzer():
    assert git(gate.ANALYZER, "rev-parse", "HEAD") == gate.ANALYZER_COMMIT
    assert not git(gate.ANALYZER, "status", "--short", "-uno")


def existing_syntax():
    evidence = {}
    for label, (count, _) in gate.SYNTAX_LAYOUTS.items():
        file = PREVIOUS / label / "syntax.json"
        rows = read(file)
        repo = gate.PLUGIN if label == "plugin-nz" else gate.CORE
        assert len(rows) == len({(r["source"],r["role"]) for r in rows}) == count
        for row in rows:
            assert row["exit"] == 0 and {"-Werror", "-fsyntax-only"} <= set(row["command"])
            assert row["command"][0] == str(gate.BISHENG)
            assert {"host":"--cce-host-only","device":"--cce-aicore-only"}[row["role"]] in row["command"]
            assert ("-DTURBOQUANT_INGRESS_NZ=1" in row["command"]) == (label != "core-strip")
            pinned(repo / row["source"], row["source_sha256"])
        evidence[label] = {"count": count, "syntax": rec(file), "driver_log": rec(PREVIOUS / f"{label}.log"),
                           "compiler_logs": [rec(Path(row["log"])) for row in rows],
                           "checked_source_files": [pinned(repo / row["source"],row["source_sha256"]) for row in rows]}
    return evidence


def source_dependencies(source, seen):
    path = source.resolve(strict=True)
    if path in seen:
        return
    seen[path] = rec(path)
    for name in INCLUDE.findall(path.read_bytes()):
        include = path.parent / name.decode()
        if include.is_file():
            source_dependencies(include, seen)


def existing_unaffected():
    summary = read(FROZEN / "summary.json")
    cases = {}
    for case in UNAFFECTED:
        entry = summary[case]
        assert entry["new"] == []
        refs = []
        raw = []
        closure = {}
        for variant, expected in zip(("verified", "integrated"), entry["source_sha256"], strict=True):
            path = FROZEN / f"{case}-{variant}.json"
            report = read(path)
            assert not report["source"]["had_parse_errors"] and report["kernels"]
            assert report["source"]["parse_error_locations"] == []
            source = Path(report["source"]["path"])
            pinned(source, expected)
            source_dependencies(source, closure)
            refs.append(rec(path))
            raw.append(report)
        before, after = map(gate.fingerprint, raw)
        assert not after - before and sum((before - after).values()) == entry["resolved_occurrences"]
        cases[case] = {"zero_new": True, "raw": refs, "resolved_quoted_include_closure": list(closure.values())}
    return {"summary": rec(FROZEN / "summary.json"), "log": rec(PREVIOUS / "frozen-and-unaffected.log"),
            "cases": cases, "retained_narrow_failures": {case: summary[case]["new"] for case in NARROW}}


def verify_refs(value):
    if isinstance(value, dict):
        if set(value) == {"path", "bytes", "sha256"}:
            check(value)
        else:
            for nested in value.values():
                verify_refs(nested)
    elif isinstance(value, list):
        for nested in value:
            verify_refs(nested)


def freeze(path):
    applied = read(APPLIED)
    before = snapshot(applied)
    assert read(HOST)["source_before_and_after"] == before
    original = before["extra_product_bytes"]["plugin"][CONSUMER]
    syntax = existing_syntax()
    unaffected = existing_unaffected()
    preimage = path.with_name("CHECK_ANALYZER_PREIMAGE.py")
    write_new(preimage, check(original).read_bytes())
    archive = rec(preimage)
    assert archive["sha256"] == original["sha256"]
    before["extra_product_bytes"]["plugin"][CONSUMER] = archive
    value = {"schema": "p6-aka-tail-freeze-v1", "applied": pinned(APPLIED, APPLIED_SHA),
             "completed_host": pinned(HOST, HOST_SHA), "snapshot_with_archived_consumer_preimage": before,
             "unchanged_syntax": syntax, "unchanged_analyzer": unaffected,
             "failed_overall_analyzer_audit": pinned(FAILURE, FAILURE_SHA),
             "baseline_pins": require_original_inputs(), "analyzer_commit": gate.ANALYZER_COMMIT}
    verify_refs(value)
    write_new(path, json_data(value))
    print(json.dumps(rec(path)), flush=True)


def current_snapshot(frozen, consumer):
    assert check(consumer) == gate.PLUGIN / CONSUMER
    applied = copy.deepcopy(read(APPLIED))
    row = applied["exact_product_paths"]["plugin"][CONSUMER]
    row["after_sha256"], row["after_bytes"] = consumer["sha256"], consumer["bytes"]
    current = snapshot(applied)
    expected = copy.deepcopy(frozen["snapshot_with_archived_consumer_preimage"])
    expected["extra_product_bytes"]["plugin"][CONSUMER] = consumer
    assert current == expected
    return current


def compare_deployed(out):
    shim = out / "strict-reference/src/prefill/turboquant_reshape_batch.cpp"
    write_new(shim, f'#include "{gate.STRICT}"\n'.encode())
    records = {}
    for label, baseline, cases in (("frozen-narrow-joined", gate.FROZEN_ROOT, NARROW),
                                   ("strict-wide", shim.parents[2], WIDE)):
        target = out / label
        command = [sys.executable, "-B", gate.CHECKERS / "check_analyzer.py", "--analyzer", gate.ANALYZER,
                   "--native", gate.NATIVE, "--reshape", baseline, "--e8", gate.E8, "--out", target]
        for case in cases:
            command.extend(("--case", case))
        execution = gate.execute(gate.bounded(2400, command), out / f"{label}.log")
        summary = target / "summary.json"
        results = read(summary)
        assert set(results) == set(cases)
        for case,row in results.items():
            assert row["new"] == []
            assert all(item["functions"] > 0 and item["parse_errors"] == [] for item in row["frontend"])
            if case in NARROW:
                assert len(row["launch_companion_sha256"]) == 64
        records[label] = {**execution, "summary": rec(summary),
                          "all_raw_reports": [rec(x) for x in sorted(target.glob("*-*.json"))]}
    companion_checks = {}
    for narrow, wide in zip(NARROW, WIDE, strict=True):
        joined = out / f"frozen-narrow-joined/{narrow}-integrated.json"
        pair = read(joined)["files"]
        actual_wide = out / f"strict-wide/{wide}-integrated.json"
        standalone = read(actual_wide)
        assert len(pair) == 2
        expected_wide = (gate.PLUGIN / gate.PLUGIN_TQ / f"{gate.BATCH}_wide.cpp" if narrow == "reshape"
                         else joined.with_name("reshape_wide-nz-integrated.cpp"))
        assert Path(pair[1]["source"]["path"]) == expected_wide
        assert gate.fingerprint(pair[1]) == gate.fingerprint(standalone)
        companion_checks[wide] = {"joined": rec(joined), "standalone": rec(actual_wide), "equal_raw_fingerprints": True}
    return {"wide_reference_include_only": rec(shim), "comparisons": records, "companion_fingerprints": companion_checks}


def run(frozen_file, frozen_sha, consumer_sha, output):
    frozen_ref = pinned(frozen_file, frozen_sha)
    frozen = read(frozen_file)
    assert frozen["applied"] == pinned(APPLIED, APPLIED_SHA) and frozen["completed_host"] == pinned(HOST, HOST_SHA)
    verify_refs(frozen)
    require_analyzer()
    assert require_original_inputs() == frozen["baseline_pins"]
    assert existing_syntax() == frozen["unchanged_syntax"]
    assert existing_unaffected() == frozen["unchanged_analyzer"]
    consumer = pinned(gate.PLUGIN / CONSUMER, consumer_sha)
    preimage = frozen["snapshot_with_archived_consumer_preimage"]["extra_product_bytes"]["plugin"][CONSUMER]
    assert preimage["sha256"] != consumer["sha256"]
    delta_ref = pinned(HYGIENE, HYGIENE_SHA)
    delta = read(HYGIENE)
    check(delta["proposed_postimage"])
    assert delta["path"] == CONSUMER and delta["prior_reviewed_postimage"]["sha256"] == preimage["sha256"]
    assert delta["proposed_postimage"]["sha256"] == consumer_sha
    before = current_snapshot(frozen, consumer)
    assert output.is_absolute() and output.parent == DEPLOYED and output.resolve() == output
    output.mkdir(exist_ok=False)
    deployed = compare_deployed(output)
    independent = gate.check_independent_batch(output)
    assert len(independent) == 4 and current_snapshot(frozen, consumer) == before
    require_analyzer()
    verify_refs(frozen)
    result = output / "DEPLOYED_AKA_SYNTAX.json"
    write_new(result, json_data({
        "status": "PASS_DEPLOYED_CANN_SYNTAX_AND_ZERO_NEW_AKA",
        "candidate": "P/r6", "original_application": frozen["applied"],
        "retained_pre_consumer_fix_host_pass": frozen["completed_host"], "freeze":frozen_ref,
        "consumer_delta": {"path": CONSUMER, "before_archived": preimage, "after": consumer, "review_packet": delta_ref},
        "source_before_and_after": before, "analyzer_commit":gate.ANALYZER_COMMIT,
        "strict_cann_syntax_30_retained": frozen["unchanged_syntax"],
        "unaffected_analyzer_8_retained":frozen["unchanged_analyzer"],
        "fresh_affected_deployed_analyzer": deployed,"fresh_independent_batch_pairs": independent,
        "compiler_invocations_in_tail": 0, "model_invocations_in_tail":0,
    }))
    print(json.dumps(rec(result)), flush=True)


def main():
    if not __debug__:
        raise SystemExit("evidence verification requires regular Python")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run"))
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--freeze-sha256")
    parser.add_argument("--consumer-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        require_analyzer()
        freeze(args.freeze)
    else:
        if None in (args.freeze_sha256, args.consumer_sha256, args.output):
            parser.error("running requires the frozen receipt, exact installed consumer and a fresh output")
        run(args.freeze, args.freeze_sha256, args.consumer_sha256, args.output)


if __name__ == "__main__":
    main()
