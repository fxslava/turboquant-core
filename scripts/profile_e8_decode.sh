#!/usr/bin/env bash
set -euo pipefail
device=${1:-0}
build=${2:-build-npu}
results=${3:-results/e8-splitk}
[[ "$device" =~ ^[0-9]+$ ]] || { echo 'Device ID must be an integer' >&2; exit 2; }
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
bench=$(realpath -- "$build/bin/bench_turboquant_production_npu")
accuracy=$(realpath -- "$build/bin/test_turboquant_production_npu")
command -v msprof >/dev/null
[[ ! -e "$results" ]] || { echo 'Choose a fresh results directory' >&2; exit 2; }
mkdir -p -- "$results"
{
    date -u
    git -C "$root" rev-parse HEAD || true
    sha256sum "$bench" "$accuracy" "$root/src/turboquant/common/e8_decode.hpp" \
        "$root/src/turboquant/decode/e8/e8_fused_decode.cpp"
} > "$results/provenance.txt"
for profile in split8 deepseek-v4-flash-64k qwen35-gqa6; do
    for block in 64 128; do
        case_dir="$results/${profile}-s8-b${block}"
        # Each export is bound to one application/shape, without mixing A/B runs.
        [[ ! -e "$case_dir" ]] || {
            echo "Existing result: $case_dir. Select a fresh results directory." >&2
            exit 2
        }
        mkdir -p -- "$case_dir"
        common=(--device "$device" --mode e8 --stage decode --profile "$profile" --split-k 8 --block-size "$block")
        "$accuracy" "${common[@]}" | tee "$case_dir/accuracy.log"
        args=("$bench" "${common[@]}" --warmup 20 --iterations 100 --timing graph --samples)
        "${args[@]}" | tee "$case_dir/unprofiled.events.log"
        printf -v application '%q ' "${args[@]}"
        msprof --ai-core=on --aic-mode=task-based --aic-metrics=PipeUtilization --task-time=on \
            --output="$case_dir/msprof" --application="$application" 2>&1 | tee "$case_dir/profiled.events.log"
        mapfile -d '' captures < <(find "$case_dir/msprof" -type d -name 'PROF_*' -print0)
        (( ${#captures[@]} == 1 )) || { echo 'Expected exactly one fresh PROF capture' >&2; exit 1; }
        msprof --parse=on --output="${captures[0]}" > "$case_dir/msprof-parse.log" 2>&1
        msprof --export=on --summary-format=csv --output="${captures[0]}" > "$case_dir/msprof-export.log" 2>&1
        python3 "$root/scripts/summarize_e8_profile.py" \
            --events-log "$case_dir/profiled.events.log" \
            --baseline-events-log "$case_dir/unprofiled.events.log" \
            --profile-dir "$case_dir/msprof" --output "$case_dir/measurement.json"
    done
done
