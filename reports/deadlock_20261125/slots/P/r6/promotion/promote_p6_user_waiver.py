"""Finalize exact P/r6 bytes using the explicit waiver of its measured 20-tick delta."""

import argparse
import json
from pathlib import Path

import promote_p6 as base

HERE = Path(__file__).resolve().parent
RUNS = base.STAGED.parent / "canary-r11-r6-pair/runs"
MATCHED = RUNS / base.FIXTURES[0] / "result.json"
MATCHED_SHA = "c301c5e4d7117f810b3beaa57eb6d46ab1538a4f103f15f25133c02f4d9dbcd6"
WAIVER = base.REPORTS / "finalize_p6_20261126/USER_WAIVER.json"
WAIVER_SHA = "ca0e507c5a16a7f76788f8a2c654b6416fad89f847b5f997969d24b20e2bafd9"
MODERN_AUDIT = base.R6 / "canary/P6_BATCH_TIMING_AUDIT.json"
MODERN_AUDIT_SHA = "3f30c2255871d617bf3a825f796deb8f4d1d040c16eaa17235e60d1e1fbef330"
BASE_SCRIPT_SHA = "01ee0d5048ec1606554cb323745c51c78967f088f39f308e4311055a929910e4"
SETUP = base.PLUGIN / "setup.py"
SETUP_SHA = "27032a2cae757dba5b2a473cc26285004f0abc91ab0154c44b3f7f641edae75a"
CAP, ACTUAL = 63629, 63649


def digest_value(value):
    return base.checksum(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def accepted_performance(waiver_ref, matched_ref):
    assert base.check(matched_ref) == MATCHED and matched_ref["sha256"] == MATCHED_SHA
    assert base.read(MATCHED)["ticks"] == [ACTUAL]
    assert base.check(waiver_ref) == WAIVER and waiver_ref["sha256"] == WAIVER_SHA
    waiver = base.read(WAIVER)
    expected = {
        "schema": "p6-user-waiver-v1", "status": "EXPLICIT_USER_AUTHORIZATION", "candidate": "P/r6",
        "source_manifest_sha256": base.SOURCE_SHA, "compiled_index_sha256": base.COMPILED_SHA,
        "measurement": matched_ref, "original_cap": CAP, "observed_raw_ticks": ACTUAL,
        "waived_delta_ticks": ACTUAL - CAP, "normalization": "NONE", "matched_frozen03_raw_ticks": 63654,
        "raw_cap_met": False, "disposition": "USER_WAIVED_MODERN_DELTA", "historical_failures_preserved": True,
        "P7": "ISOLATED_NOT_PROMOTED",
    }
    assert all(waiver.get(key) == value for key, value in expected.items())
    assert isinstance(waiver["user_request"], str) and waiver["user_request"].strip()
    audit_ref = waiver["modern_audit"]
    assert base.check(audit_ref) == MODERN_AUDIT and audit_ref["sha256"] == MODERN_AUDIT_SHA
    raw = base.read(MODERN_AUDIT)["raw_gate"]
    assert raw["cap"] == CAP and raw["p6"] == ACTUAL and raw["matched_frozen03"] == 63654
    assert raw["delta"] == ACTUAL - CAP and raw["passed"] is False
    return {
        "basis": "user_waiver_p6_guarded_modern_20_ticks", "disposition": "USER_WAIVED_MODERN_DELTA",
        "observed_raw_ticks": ACTUAL, "original_raw_cap": CAP, "delta_ticks": ACTUAL - CAP,
        "matched_frozen03_raw_ticks": raw["matched_frozen03"], "normalization": "NONE", "raw_cap_met": False,
        "measurement": matched_ref, "modern_audit": audit_ref, "user_waiver": waiver_ref,
    }


def authorize(release_path, digest):
    release_ref = base.rec(release_path)
    assert release_ref["sha256"] == digest
    gate = base.read(base.check(release_ref))
    assert gate["status"] == "PASS_P6_USER_WAIVED_RELEASE" and gate["candidate"] == "P/r6"
    assert gate["source_manifest_sha256"] == base.SOURCE_SHA and gate["compiled_index_sha256"] == base.COMPILED_SHA
    assert base.rec(base.COMPILED)["sha256"] == base.COMPILED_SHA
    compiled = base.read(base.COMPILED)
    cases = gate["functional"]
    assert set(cases) == set(base.FIXTURES)
    for name in base.FIXTURES:
        case = cases[name]
        path = base.check(case["run"])
        assert path == RUNS / name / "result.json"
        base.model_pass(path, name, compiled)
        assert case["parity_guards_and_trace"] == "PASS"
        base.check(case["audit"])
    timing = accepted_performance(gate["performance"]["user_waiver"], cases[base.FIXTURES[0]]["run"])
    assert cases[base.FIXTURES[0]]["audit"] == timing["modern_audit"]
    assert gate["performance"] == timing
    review = gate["winning_review"]
    assert review["status"] == "PASS" and review["review_kind"] == "fresh_independent_sol_final"
    assert review["source_manifest_sha256"] == base.SOURCE_SHA and review["compiled_index_sha256"] == base.COMPILED_SHA
    assert review["deployed_bytes_plan_sha256"] == base.PLAN_SHA
    assert review["adapter"] == base.rec(Path(__file__).resolve())
    assert review["functional_sha256"] == digest_value(cases)
    assert review["performance_sha256"] == digest_value(timing)
    base.check(review["receipt"])
    return release_ref, timing


def deploy(destination, release_ref, timing):
    output, new_core, report = base.plan_bytes()
    ledger = base.plan(output, new_core, report)
    assert base.checksum(base.json_data(ledger)) == base.PLAN_SHA
    destination.mkdir(parents=True)
    base.write_new(destination / "PROMOTION_INTENT.json", base.json_data(ledger))
    installed = []

    def install(label, table):
        for name, data in table.items():
            path = base.ROOTS[label] / name
            before = path.read_bytes() if path.is_file() else None
            expected = ledger["exact_product_paths"][label][name]["before"]
            assert (None if before is None else base.checksum(before)) == (
                None if expected is None else expected["sha256"]
            )
            base.replace_bytes(path, data)
            installed.append((path, before))

    try:
        install("core", output["core"])
        mirror = output["plugin"]
        install("plugin", {name: data for name, data in mirror.items() if name not in base.PLUGIN_ONLY})
        assert base.sync_checker().verify(base.PLUGIN / base.PLUGIN_TQ, base.CORE / base.CORE_TQ) == []
        install("plugin", {name: mirror[name] for name in base.PLUGIN_ONLY})
        assert base.rec(SETUP)["sha256"] == SETUP_SHA
        for name, digest in ledger["expected_core_checkout_bytes"].items():
            assert base.rec(base.CORE / name)["sha256"] == digest
        base.check(ledger["unrelated_report_retained_verbatim"])
        receipt = {**ledger, "status": "APPLIED_EXACT_BYTES_PENDING_FINAL_HOST", "release": release_ref,
                   "accepted_performance": timing}
        path = destination / "PROMOTION_APPLIED.json"
        base.write_new(path, base.json_data(receipt))
        return base.rec(path)
    except Exception:
        for path, before in reversed(installed):
            if before is None:
                path.unlink()
            else:
                base.replace_bytes(path, before)
        raise


def main():
    if not __debug__:
        raise SystemExit("The evidence checks require regular Python execution")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("verify-release", "apply"))
    parser.add_argument("--winning-release", type=Path, required=True)
    parser.add_argument("--winning-release-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert (args.phase == "apply") == (args.output is not None)
    assert base.rec(HERE / "promote_p6.py")["sha256"] == BASE_SCRIPT_SHA
    assert base.rec(HERE / "PROMOTION_BYTES.json")["sha256"] == base.PLAN_SHA
    assert base.rec(SETUP)["sha256"] == SETUP_SHA
    release_ref, timing = authorize(args.winning_release, args.winning_release_sha256)
    if args.phase == "apply":
        assert args.output.is_absolute() and args.output.resolve() == args.output and not args.output.exists()
        release_ref = deploy(args.output, release_ref, timing)
    print(json.dumps(release_ref), flush=True)


if __name__ == "__main__":
    main()
