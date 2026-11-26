#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
image=quay.io/ascend/vllm-ascend:v0.30.0-a5
build_root=${TQ_BUILD_ROOT:-"$HOME/.cache/turboquant/cann921"}
mkdir -p -- "$build_root"
build_root=$(cd -- "$build_root" && pwd)
if (( $# == 0 )); then
    echo 'Usage: bash scripts/wsl_docker.sh COMMAND [ARG...]' >&2
    exit 2
fi
exec docker run --pull never --rm --network none \
    --user "$(id -u):$(id -g)" \
    --env TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    --mount "type=bind,src=$root,dst=/workspace/turboquant-core,readonly" \
    --mount "type=bind,src=$build_root,dst=/workspace/build" \
    --workdir /workspace/build --entrypoint bash "$image" \
    /workspace/turboquant-core/scripts/container_entrypoint.sh "$@"
