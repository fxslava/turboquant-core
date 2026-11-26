"""Record host gate receipts and the exact sources tested, without device imports."""

import argparse
import ast
import hashlib
import importlib.util
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def record(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def source_record(repo, source, label, name, ref):
    path = source / name
    working = path.read_bytes()
    indexed = subprocess.check_output(["git", "-C", str(repo), "show", f"{ref}:{name}"])
    if working != indexed and working.replace(b"\r\n", b"\n") != indexed.replace(
        b"\r\n", b"\n"
    ):
        raise ValueError(
            f"Tested source differs from its frozen Git object: {label}/{name}"
        )
    return {
        "repository": label,
        **record(path),
        "git_object_sha256": hashlib.sha256(indexed).hexdigest(),
        "git_ref": ref,
        "line_ending_only_delta": working != indexed,
    }


def junit(path):
    cases = list(ET.parse(path).getroot().iter("testcase"))
    failures = sum(
        any(case.find(tag) is not None for tag in ("failure", "error"))
        for case in cases
    )
    skipped = sum(case.find("skipped") is not None for case in cases)
    if not cases or failures or skipped:
        raise ValueError(
            f"Incomplete host gate: {path}: {len(cases)=}, {failures=}, {skipped=}"
        )
    return {
        **record(path),
        "passed": len(cases),
        "failures": failures,
        "skipped": skipped,
    }


def case_names(path):
    return {
        (case.get("classname"), case.get("name"))
        for case in ET.parse(path).getroot().iter("testcase")
    }


def frozen_sources(trees):
    inventories = []
    counts = {}
    for label, repo, source, ref, base in trees:
        prefix = "src/turboquant" if label == "core" else "csrc/attention/turboquant"
        inventory_bytes = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{ref}:{prefix}/SOURCE_SYNC.json"]
        )
        if inventory_bytes != (source / prefix / "SOURCE_SYNC.json").read_bytes():
            raise ValueError(f"Frozen {label} inventory differs from its Git object")
        inventories.append(inventory_bytes)
        inventory = json.loads(inventory_bytes)
        records = {**inventory["shared"], **inventory[f"{label}_only"]}
        for name, expected in records.items():
            contents = subprocess.check_output(
                ["git", "-C", str(repo), "show", f"{ref}:{prefix}/{name}"]
            )
            if hashlib.sha256(contents).hexdigest() != expected["sha256"]:
                raise ValueError(f"Staged source hash differs: {label}/{name}")
        counts[label] = len(records)
    if inventories[0] != inventories[1]:
        raise ValueError("Staged source inventories differ")
    return {"source_counts": counts, "byte_identical_manifests": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", required=True, type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    core = here.parents[1]
    plugin = args.plugin.resolve()
    state = json.loads((here / "FROZEN_SOURCES.json").read_bytes())
    core_source = Path(state["core_snapshot"])
    plugin_source = Path(state["plugin_snapshot"])
    trees = (
        ("core", core, core_source, state["core_tree"], state["core_base_head"]),
        (
            "plugin",
            plugin,
            plugin_source,
            state["plugin_commit"],
            state["plugin_base_head"],
        ),
    )
    spec = importlib.util.spec_from_file_location(
        "turboquant_sync", plugin_source / "scripts/check_turboquant_sync.py"
    )
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)
    plugin_sources = plugin_source / "csrc/attention/turboquant"
    core_sources = core_source / "src/turboquant"
    diagnostics = sync.verify(plugin_sources, core_sources)
    if diagnostics:
        raise ValueError(diagnostics)
    source_inventory = json.loads((core_sources / sync.MANIFEST).read_bytes())
    sources = []
    for label, repo, source, ref, base in trees:
        tracked_changes = subprocess.check_output(
            [
                "git",
                "-C",
                str(repo),
                "diff",
                base,
                ref,
                "--name-only",
                "--diff-filter=ACMR",
            ],
            text=True,
        ).splitlines()
        sources.extend(
            source_record(repo, source, label, name, ref)
            for name in tracked_changes
            if Path(name).parts[0]
            in {"src", "csrc", "vllm_ascend", "tests", "benchmarks", "cmake"}
            or name == "CMakeLists.txt"
        )
    baseline = case_names(here / "attention-baseline.xml")
    missing = baseline - case_names(here / "attention-final.xml")
    if missing:
        raise ValueError(
            f"Baseline regression tests absent from final suite: {sorted(missing)}"
        )
    original_envs = subprocess.check_output(
        [
            "git",
            "-C",
            str(plugin),
            "show",
            f"{state['plugin_base_head']}:vllm_ascend/envs.py",
        ],
        text=True,
    )
    current_envs = (plugin_source / "vllm_ascend/envs.py").read_text(encoding="utf-8")
    same_env_ast = ast.dump(ast.parse(original_envs)) == ast.dump(
        ast.parse(current_envs)
    )
    if not same_env_ast:
        raise ValueError("The environment configuration declaration changed")
    receipt = {
        "scope": "Frozen-source host gates and exact source synchronization; concurrent worktree edits excluded. No NPU execution or performance claim.",
        "base_heads": {
            "core": state["core_base_head"],
            "plugin": state["plugin_base_head"],
        },
        "frozen_git_refs": {label: ref for label, repo, source, ref, base in trees},
        "attention": junit(here / "attention-final.xml"),
        "baseline_attention_cases_retained": len(baseline),
        "env_configuration_ast_unchanged": same_env_ast,
        "sync_and_build_selection": junit(here / "host14-final.xml"),
        "cpp_contracts": junit(here / "contracts.xml"),
        "source_inventory": record(core_sources / sync.MANIFEST),
        "shared_cpp": sum(
            not name.endswith(".aka") for name in source_inventory["shared"]
        ),
        "shared_sidecars": sum(
            name.endswith(".aka") for name in source_inventory["shared"]
        ),
        "sync_diagnostics": diagnostics,
        "frozen_source_verification": frozen_sources(trees),
        "tracked_changes": sources,
    }
    output = here / "HOST_VALIDATION.json"
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record(output)))


if __name__ == "__main__":
    main()
