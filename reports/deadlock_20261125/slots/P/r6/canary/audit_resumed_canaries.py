"""Audit only the three completed supplemental P6 canaries from the sealed plan."""

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPORTS = HERE.parents[3]
sys.path.insert(0, str(REPORTS / "infra/r11-batch-p-r6-pair"))
from p_batch_contract import COMPILED, SLOT, check_ref, read, record

sys.path.insert(0, str(REPORTS / "d1/ingress"))
sys.path.insert(0, str(REPORTS / "slots/P/r2/canary"))
from audit_ingress import CROSS, LOCAL, KERNEL_ROLES, instructions, parse_footers
from audit_p_r2_timing import function


PLAN = HERE / "P6_WAIVER_RESUMED_REMAINING_CANARY_PLAN.json"
PLAN_SHA = "c92ebd6f1601a110b75cf9fff5f885abb7ee55daf5516bbe3a8898e7affb259d"
TRACE = SLOT.parent.parent / "d1-baseline-strict/audit_run_rendezvous.py"
MODELS = {
    "batch-p-strip-wide-lloyd": "strip-wide-lloyd",
    "batch-p-nz-narrow-lloyd": "nz-narrow-lloyd",
    "batch-p-nz-wide-uniform": "nz-wide-uniform",
}
ROLE_FILES = dict(zip(("AIC", "AIV0", "AIV1"), KERNEL_ROLES))


def check_local_and_cross(run, idx, role):
    path = run / f"core{idx}.{role}.instr_popped_log.dump"
    local = defaultdict(list)
    sends, waits = defaultdict(list), defaultdict(list)
    pending = []
    census = Counter()
    for row in instructions(path):
        op, body = row["op"], row["body"]
        census[op] += 1
        if op in ("SET_CROSS_CORE", "WAIT_FLAG_DEV"):
            pipe, actual = CROSS.search(body).groups()
            value = int(actual, 0)
            if op == "SET_CROSS_CORE":
                assert (value >> 4) & 3 == 2
                sends[(value >> 8) & 15].append(row["tick"])
            else:
                wait = {"tick": row["tick"], "pipe": pipe, "drain": None}
                waits[value].append(wait)
                pending.append(wait)
        elif op in ("SET_FLAG", "WAIT_FLAG"):
            src, dst, flag = LOCAL.search(body).groups()
            local[(src, dst, int(flag, 0))].append(op)
            if op == "WAIT_FLAG" and row["engine"] == dst == "SCALAR":
                for wait in pending[:]:
                    if wait["pipe"] == src:
                        wait["drain"] = row["tick"]
                        pending.remove(wait)
    assert not pending, (idx, role, pending)
    assert all(v == ["SET_FLAG", "WAIT_FLAG"] * (len(v) // 2) for v in local.values()), (idx, role)
    return {"popped_trace": record(path), "local_pairs": sum(len(v) // 2 for v in local.values()),
            "sends": sends, "waits": waits, "popped_counts": dict(census)}


def protocol(run, core, strips):
    c, a, b = (check_local_and_cross(run, core, role) for role in KERNEL_ROLES)
    assert c["popped_counts"]["MMAD"] == 512 * strips
    assert set(c["sends"]) == {8, 10} and set(c["waits"]) == {9}
    assert set(a["sends"]) == set(b["sends"]) == {9}
    assert set(a["waits"]) == set(b["waits"]) == {8, 10}
    matched = 0
    for flag, count, producers, consumers in ((8, 4*strips, [c], [a, b]), (9, 4*strips+2, [a, b], [c]),
                                                (10, 1, [c], [a, b])):
        produced = [x["sends"][flag] for x in producers]
        consumed = [x["waits"][flag] for x in consumers]
        assert all(len(x) == count for x in produced + consumed)
        for i in range(count):
            assert all(x[i]["drain"] >= max(y[i] for y in produced) for x in consumed), (core, flag, i)
        matched += count * len(consumed)
    compact = [{k: v for k, v in row.items() if k not in ("sends", "waits")} for row in (c, a, b)]
    return {"logical_strips": strips, "matching_source_pipe_releases": matched, "roles": compact}


def mapping_refs(run, result, admission):
    expected = result["expected_mappings"]
    for name in ("runner", "device_library", "private_companion"):
        p = check_ref(admission[name])
        assert expected[p.name] == [str(p), admission[name]["sha256"]]
    assert len(expected) == 8
    checked = {}
    for row in read(run / "loaded_model_libraries.json"):
        p = Path(row["path"])
        stat = p.stat()
        device, inode = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}", stat.st_ino
        assert row["device"] == device and row["inode"] == inode
        assert row["file_stamp"] == [stat.st_dev, inode, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        fields = row["mapping_line"].split(maxsplit=5)
        assert fields[1] == "r-xp" and fields[3:6] == [device, str(inode), str(p)]
        if p.name in expected:
            assert p.name not in checked and expected[p.name] == [str(p), row["sha256"]]
            checked[p.name] = record(p, row["sha256"])
    assert set(checked) == set(expected)
    at = read(run / "mapping-attestations.json")
    runner = Path(result["command"][0]).stat()
    for marker in ("ready", "done"):
        assert set(at[marker]["simultaneous"]) == set(expected)
        assert (at[marker]["executable_device"], at[marker]["executable_inode"]) == (runner.st_dev, runner.st_ino)
    return checked


def executed_code(path, merged, load_base):
    image = merged.read_bytes()
    shoff, = struct.unpack_from("<Q", image, 40)
    stride, count, names_idx = struct.unpack_from("<HHH", image, 58)
    sections = [struct.unpack_from("<IIQQQQIIQQ", image, shoff+stride*i) for i in range(count)]
    string_at, string_bytes = sections[names_idx][4:6]
    strings = image[string_at:string_at+string_bytes]
    text, = [row for row in sections if strings[row[0]:strings.index(0,row[0])] == b".text"]
    vma, offset, nbytes = text[3:6]
    checked = 0
    pseudo = 0
    for row in instructions(path):
        pc = int(row["pc"], 16)
        word = int(row["word"], 16)
        if (row["op"], pc, word) == ("END_LABEL", 0x11111111, 0x1e00):
            pseudo += 1
            continue
        width = (len(row["word"])-2)//2
        within = pc-load_base-vma
        assert width in (4, 8) and 0 <= within <= nbytes-width, (path, row)
        begin = offset+within
        assert word == int.from_bytes(image[begin:begin+width], "little"), (path, row)
        checked += 1
    return {"executed_binary_records_equal": checked, "synthetic_end_labels": pseudo}


def audit(case):
    plan = read(PLAN)
    planned, = [row for row in plan["cases"] if row["case"] == case]
    run = Path(planned["run_dir"])
    result = read(run / "result.json")
    admission = read(run / "admission.json")
    assert check_ref(planned["receipt"]).read_bytes() == (run / "admission.json").read_bytes()
    assert result["admission_sha256"] == planned["receipt"]["sha256"]
    assert result["child_exit_code"] == 0 and result["status"] == "numerical_pass_trace_pending"
    assert result["markers"] == 1 and result["mappings_verified"] and not result["mapping_conflicts"]
    geometry = admission["geometry"]
    assert result["geometry_verified"] and len(result["actual_geometry"]) == 1
    assert json.loads(result["actual_geometry"][0]) == geometry == planned["geometry"]
    assert result["command"] == [admission["runner"]["path"], *admission["argv_tail"]]
    assert not result["model_exceptions"] and 0 < result["device_seconds"] < result["watchdog_seconds"] == 900
    files = read(run / "file-state-final.json")
    for relative, (size, stamp) in files.items():
        actual = (run / relative).stat()
        assert (size, stamp) == (actual.st_size, actual.st_mtime_ns), relative
    exceptions = set(run.glob("*.excp_log.dump"))
    assert exceptions == {run / f"core{n}.excp_log.dump" for n in range(32)}
    assert all(not path.stat().st_size for path in exceptions)
    log = (run / "run.log").read_text()
    raw, model = parse_footers(log)
    assert result["ticks"] == [raw]
    rows = geometry["num_blocks"]*geometry["block"]*geometry["heads"]
    plane = rows*128
    scale = rows*2
    pass_line = f"PASS bounded batch entry: {MODELS[case]}, {plane} bytes/packed plane, {scale} FP32 scalars, exact labels, scales, unowned storage, read-only inputs and guards"
    assert log.splitlines().count(pass_line) == 1
    images = {}
    for name, size in (("key", plane), ("value", plane), ("scale", scale*4)):
        actual = run / f"{name}.actual.bin"
        golden = run / f"{name}.gold.bin"
        assert actual.stat().st_size == size and actual.read_bytes() == golden.read_bytes()
        images[name] = {"actual": record(actual), "golden": record(golden)}
    mappings = mapping_refs(run, result, admission)
    host = read(check_ref(admission["host_link"]))
    fixture, = [ref for ref in host["dependencies"] if ref["path"].endswith("batch_supplement.cpp")]
    content = check_ref(fixture).read_text()
    assert "inputs[i]->assert_guards();" in content and "inputs[i]->host == snapshots[i]" in content
    assert "cache->assert_guards();" in content
    physical = admission["selected_physical_module"]
    assert physical["function_key"] == 0
    assert check_ref(admission["device_compiled_index"]) == COMPILED
    module = read(COMPILED)["targets"][Path(physical["library"]["path"]).stem.removeprefix("lib")]
    assert module["library"] == physical["library"]
    merged = check_ref(module["objects"]["merge_obj"]["object"])
    argv = [sys.executable, "-B", str(TRACE), str(run), "--expected-collectives", "0"]
    trace = json.loads(subprocess.check_output(argv, text=True))
    assert not trace["collective_count_deltas"]
    groups = defaultdict(dict)
    for role in trace["role_traces"]:
        phys = role["physical_core"]
        name = role["role"]
        assert name not in groups[phys] and len(role["done"]) == 1 and role["last_ccu_issue"]["opcode"] == "END"
        assert role["done"][0][5] == geometry["grid"]
        symbol = physical["symbols"]["aic" if name == "AIC" else "aiv"]
        arch, code = function(merged, symbol)
        items = role["first"]
        pc = int(items[0]["pc"], 16)
        assert len(items) == 8 and all(int(x["word"], 16) == struct.unpack_from("<I", code, int(x["pc"], 16)-pc)[0] for x in items)
        instr_trace = run / f"core{phys}.{ROLE_FILES[name]}.instr_log.dump"
        groups[phys][name] = {"symbol": symbol, "first_pc": pc, "payload_base": pc-arch, "initial_matching_words": len(items),
                              "terminal": role["done"][0], "instr_trace": record(instr_trace),
                              **executed_code(instr_trace, merged, pc-arch)}
    assert len(groups) == geometry["grid"]
    assert {x["AIC"]["terminal"][4] for x in groups.values()} == set(range(geometry["grid"]))
    for triplet in groups.values():
        assert set(triplet) == {"AIC", "AIV0", "AIV1"} and len({x["payload_base"] for x in triplet.values()}) == 1
        assert len({x["terminal"][4] for x in triplet.values()}) == 1
    strips = geometry["share"]//16
    assert strips*len(groups) == geometry["strips"]
    return {"schema": "p6-functional-supplement-actual-audit-v1", "case": case, "parity_guards_and_trace": "PASS",
            "natural_exit": 0, "raw_ticks": raw, "model_log_seconds": model, "device_seconds": result["device_seconds"],
            "geometry": geometry, "physical_cache_byte_parity": images, "owned_and_unowned_matches": True, "guarded_fixture": fixture,
            "runtime_mapped_images": mappings, "exception_count": 0, "exception_files": len(exceptions),
            "selected_module_payload": record(merged), "role_capsule_bindings": groups, "rendezvous_tool": record(TRACE),
            "mode2_and_local_drain": {i: protocol(run, i, strips) for i in groups},
            "trace_stream_scope": "Mode-2 and local pipe counts use full popped traces. The execution-only trace omits issue-only events.",
            "global_collectives": 0, "run": record(run/"result.json"),
            "evidence": [record(run/n) for n in ("run.log", "admission.json", "mapping-attestations.json",
                        "loaded_model_libraries.json", "file-state-final.json")],
            "scope": "Fixed functional probe, raw footer reported unadjusted with no timing cap."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=MODELS)
    args = parser.parse_args()
    record(PLAN, PLAN_SHA)
    proof = audit(args.case)
    path = HERE / (args.case + "_ACTUAL_AUDIT.json")
    with path.open("x") as file:
        json.dump(proof, file, sort_keys=True, indent=2)
        file.write("\n")
    print(json.dumps(record(path)))


if __name__ == "__main__":
    main()
