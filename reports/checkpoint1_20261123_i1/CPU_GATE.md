# Complete CPU gate

The final CPU suite passes **8/8 CTest entries** in 3.48 seconds, including the
Wave-v11 exact fixture's host audit and all eight decode golden cases. Sources
and runners compile under GCC 11.4 with `-std=c++17 -Wall -Wextra -Werror`.

The coordinator generated the complete golden dataset with the repository's
existing `golden/generate_dumps.py`, using CPU Torch in the registry Docker
image. It lives separately at
`/home/dmin/work/checkpoint1_20261123/golden_validation/dumps`. Generation and
standalone oracle logs are `generate.log` and `oracle.log` beside that root.
No checked-in or verified fixture was overwritten. This closes the earlier
six-missing-fixtures limitation recorded in INTEGRATION.md.

CMake now accepts `TQ_GOLDEN_DUMP_ROOT`, defaulting to the checkout's
`golden/dumps`. CTest passes this path to the oracle explicitly. Reproduce:

```powershell
wsl -d Ubuntu-22.04 -- cmake -S /mnt/c/Users/Admin/.codex/worktrees/wave-v11-integration/turboquant-core -B /home/dmin/work/checkpoint1_20261123_i1/cpu-build -DRUN_MODE=sim -DTQ_BUILD_KERNELS=OFF -DTQ_COMPILER_CACHE=OFF -DTQ_GOLDEN_DUMP_ROOT=/home/dmin/work/checkpoint1_20261123/golden_validation/dumps
wsl -d Ubuntu-22.04 -- ctest --test-dir /home/dmin/work/checkpoint1_20261123_i1/cpu-build --output-on-failure
```

The captured log is `cpu-ctest.log`. These are CPU results, not device parity.
The active device build uses a separate immutable ext4 snapshot of 5231e2c;
the later CMake-only golden-root setting does not alter it.
