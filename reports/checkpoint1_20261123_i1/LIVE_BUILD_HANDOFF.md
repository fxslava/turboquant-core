# I1 live build handoff at Checkpoint 1

At 2026-11-23T08:20:49Z, the compiler is live. It was not terminated or
restarted. **No I1 CAModel launch has occurred.** The coordinator requested a
stable pre-simulation checkpoint; verification must wait for coordinator resume.

## Exact ownership and state

- Branch: `codex/wave-v11-integration`, CPU/AKA checkpoint commit `aefc132`.
- Frozen device source: commit `5231e2c`, extracted on ext4 at
  `/home/dmin/work/checkpoint1_20261123_i1/device-source`.
- Build: `/home/dmin/work/checkpoint1_20261123_i1/device-build`.
- Unified exec session: `91942`, owned by the Slot01 agent. Poll that session or
  resume the Slot01 agent; do not launch another build while it is live.
- Build shell PID `804913`; phase guard PID `804915`; compile-inside shell
  PID `804923`; CMake build PID `804981`.
- Active BiSheng PID `806156`, PPID `806150`, generated AIV device wrapper;
  elapsed `02:03`, CPU `104%`, RSS `804396 KiB` at the snapshot.
- The precompile and both preprocess objects finished. The AIC device object
  and host object finished. The generated AIV wrapper is still compiling; the
  shared library and exact runner have not yet linked.
- Full process arguments and snapshot UTC are in `live_build_handoff.json`.

The build holds an exclusive descriptor on
`/home/dmin/ascend-sync-schedule-ablation/queue/checkpoint1-exclusive.lock`
and runs through
`/mnt/c/ascend-micro-harness/scripts/camodel_build_phase_guard.sh compile`.
Only `test_wave_v11_exact_sim` was requested, in Release mode with CANN 9.2.1
at `/home/dmin/Ascend-9.2.1/cann-9.2.1`. No other slot's source was used.

## Evidence and hashes

Full immutable source manifest:
`/home/dmin/work/checkpoint1_20261123_i1/source.sha256`, SHA-256
`add2f41af87382d746168f3ff6bd3e9ba4f24b1836fb580d57c85208c1ce94f0`.

Completed AIC `device.o`: SHA-256
`87d41bb03c12d7909d07f454d73974f67e5e2cc437f43ba11ffb7bf557281e2a`.
Completed installed host object: SHA-256
`9bfcf0a7ac5d27ef85d37040b79478fefe374d309bdcb4a3345175c380b3fe5c`.
These are partial build artifacts, not a complete verified executable.

Compiler output is in the live unified exec stream `91942`; a full compiler
stdout log was not redirected when this initial build started. CMake configure
logs remain in the build's `CMakeFiles/CMakeOutput.log` and the corresponding
ExternalProject build directories. Strict compile flags are preserved in the
build's `compile_commands.json` and each phase's `flags.make`. No C++ compiler
warning/error has appeared. The CANN generator emitted its recommendation to
use one kernel per file; this production TU intentionally defines four kernels.
There are also CMake notices about unused variables in generated projects.

Completed CPU evidence: `CPU_GATE.md`, `cpu-ctest.log` (8/8 pass).
Completed scoped AKA differential: `AKA_GATE.json` (0 fatal / 0 warning after
the exact fingerprint baseline from verified tag 7c31a112; raw backlog retained).

## Resume after the checkpoint

Let the current build finish naturally and record its exit status. The build
script performs no automatic simulation. If compilation succeeds, inspect the
runner and library before running the sole four-tile gate. After coordinator
resume, use:

```powershell
wsl -d Ubuntu-22.04 -- bash /mnt/c/Users/Admin/.codex/worktrees/wave-v11-integration/turboquant-core/reports/checkpoint1_20261123_i1/build-exact.sh run
```

That command reacquires the exclusive checkpoint lock and CAModel phase guard,
verifies the frozen source manifest, hashes the executable/library/device
objects, and uses `run_verified_benchmark.py` with a 240-second cap. It requires
the live `dav_3510/lib` libpem mapping to match SHA-256
`1a6420fe8123e88a120f0bbfc76710968e6708599c77d80a7b1097aca06eed33`.
The run directory will be
`/home/dmin/work/checkpoint1_20261123_i1/exact-run`; it does not exist yet.
Capture stdout and the final exit status when invoking the resumed run.

No device parity, timing, fully strict device build, or yellow-zone readiness
is claimed at this checkpoint. No Wave-1 reshape candidate is authorized here.
