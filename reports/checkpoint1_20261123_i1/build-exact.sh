#!/usr/bin/env bash
set -eo pipefail
root=/home/dmin/work/checkpoint1_20261123_i1
source_dir="$root/device-source"
build="$root/device-build"
sdk=/home/dmin/Ascend-9.2.1/cann-9.2.1
guard=/mnt/c/ascend-micro-harness/scripts/camodel_build_phase_guard.sh
lock=/home/dmin/ascend-sync-schedule-ablation/queue/checkpoint1-exclusive.lock
model="$sdk/x86_64-linux/simulator/dav_3510/lib"
case "${1:-}" in
  stage)
    test ! -e "$source_dir"
    mkdir -p "$source_dir"
    tar -xf /mnt/c/tmp/wave-v11-integration-5231e2c.tar -C "$source_dir"
    find "$source_dir" -type f -print0 | sort -z | xargs -0 sha256sum >"$root/source.sha256"
    ;;
  build)
    exec 9>"$lock"
    flock -x 9
    bash "$guard" compile bash "$0" compile-inside
    ;;
  compile-inside)
    source "$sdk/set_env.sh"
    cmake -S "$source_dir" -B "$build" -DRUN_MODE=sim -DTQ_BUILD_KERNELS=ON \
      -DTQ_COMPILER_CACHE=OFF -DASCEND_HOME_PATH="$sdk" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    cmake --build "$build" --target test_wave_v11_exact_sim -j 2
    ;;
  run)
    exec 9>"$lock"
    flock -x 9
    bash "$guard" camodel bash "$0" run-inside
    ;;
  run-inside)
    source "$sdk/set_env.sh"
    unset LD_PRELOAD LD_AUDIT CAMODEL_CONFIG_PATH
    export LD_LIBRARY_PATH="$build/lib:$model:$sdk/lib64:$sdk/x86_64-linux/devlib:$sdk/lib64/stub"
    sha256sum --quiet -c "$root/source.sha256"
    sha256sum "$build/tests/test_wave_v11_exact_sim" "$build/lib/libtq_decode_sim.so" \
      "$build/tq_decode_sim_aic_device_dir/device.o" "$build/tq_decode_sim_aiv_device_dir/device.o" \
      >"$root/device-artifacts.sha256"
    timeout --signal=TERM --kill-after=15s 240s python3 \
      /home/dmin/work/e8_coordinator/scripts/run_verified_benchmark.py \
      --run-dir "$root/exact-run" --model-lib "$model" \
      --expected-pem-sha256 1a6420fe8123e88a120f0bbfc76710968e6708599c77d80a7b1097aca06eed33 \
      -- "$build/tests/test_wave_v11_exact_sim"
    ;;
  *) exit 2 ;;
esac

