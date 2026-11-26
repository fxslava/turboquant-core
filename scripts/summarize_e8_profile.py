"""Combine actual ACL-event samples with E8 msprof op_summary rows."""

import argparse
import csv
import json
import math
import re
from pathlib import Path


def normalized(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def event_cases(path):
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = dict(token.split("=", 1) for token in line.split() if "=" in token)
        if fields.get("mode") != "e8" or fields.get("stage") != "decode":
            continue
        if fields.get("timing_source") != "aclrtEventElapsedTime":
            continue
        samples = [float(value) for value in fields.get("device_samples_us", "").split(",") if value]
        if len(samples) != int(fields["samples"]) or not samples:
            raise ValueError("event log must contain all samples; run with --samples")
        if not all(math.isfinite(value) and value > 0 for value in samples):
            raise ValueError("event samples must be positive finite device measurements")
        if fields.get("status") != "MEASURED":
            raise ValueError("event row did not finish successfully")
        cases.append({"fields": fields, "aclrt_event_samples_us": samples})
    if not cases:
        raise ValueError("no measured E8 decode ACL-event rows found")
    return cases


def numeric(value):
    if value is None or value.strip().lower() in {"", "n/a", "na", "nan", "none", "--"}:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def op_rows(directory):
    results = []
    for path in sorted(directory.rglob("op_summary_*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as source:
            for number, raw in enumerate(csv.DictReader(source), 2):
                row = {normalized(key): value for key, value in raw.items() if key is not None}
                names = [row.get(key, "") for key in ("opname", "kernelname", "name")]
                if not any("e8_fused_decode" in name or "e8_splitk_reduce" in name for name in names):
                    continue
                duration = row.get("taskdurationus", row.get("durationus"))
                results.append({
                    "source_csv": str(path), "source_row": number,
                    "task_duration_us": numeric(duration),
                    "block_dim": row.get("blockdim"), "mix_block_dim": row.get("mixblockdim"),
                    "aiv_vec_ratio": numeric(row.get("aivvecratio")),
                    "aic_mte2_ratio": numeric(row.get("aicmte2ratio")),
                    "raw": raw,
                })
    if not results:
        raise ValueError("no E8 reader/reduction rows found in op_summary_*.csv")
    for key in ("task_duration_us", "aiv_vec_ratio", "aic_mte2_ratio"):
        if not any(row[key] is not None for row in results):
            raise ValueError(f"msprof did not export {key}; collect task time and PipeUtilization")
    if not any(row["block_dim"] or row["mix_block_dim"] for row in results):
        raise ValueError("msprof did not export Block Dim or Mix Block Dim")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-log", type=Path, required=True)
    parser.add_argument("--baseline-events-log", type=Path)
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "event_cases": event_cases(args.events_log),
        "unprofiled_event_cases": event_cases(args.baseline_events_log) if args.baseline_events_log else None,
        "msprof_kernel_rows": op_rows(args.profile_dir),
        "interpretation": "Event spans cover reader plus optional reduction on the device stream. "
        "msprof rows are kernel tasks and may include warmups; they are not paired by sample index. "
        "Pipe ratios are per-core cycle fractions and must not be summed. Missing fields remain null.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
