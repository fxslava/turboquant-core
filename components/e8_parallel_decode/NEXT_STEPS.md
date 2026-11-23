# Frozen decoder handoff

Component and uniform six-tile gates passed. Native src/build-r3 remain frozen.
Root controls all compile/model phases. Next data-only control after root grant:

```
wsl -d Ubuntu-22.04 -u dmin -- bash /mnt/c/ascend-micro-harness/reports/checkpoint2_20261123/parallel_decode/run_parallel.sh nonuniform
```

After completion:

```
python3 /mnt/c/ascend-micro-harness/reports/checkpoint2_20261123/parallel_decode/audit_reader_trace.py /home/dmin/work/checkpoint2_20261123/parallel_decode/run-nonuniform --out /home/dmin/work/checkpoint2_20261123/parallel_decode/nonuniform_trace_audit.json
python3 /mnt/c/ascend-micro-harness/reports/checkpoint2_20261123/parallel_decode/collect_reader.py nonuniform
```

Audit filters each role's last frame: two prior scatter launches share files
and reset IDs. It checks47 waits, exact mode2 quorum/broadcast counts and zero
terminal credits, balanced HardEvents, unchanged SDK V_S7 provenance,12 actual
FP32 MMADs and the static VAG alias word witnessed as MMAD. Following pipe-marker
completions are bounded observations, not exact transport-service latencies.

For raw composition, reuse slot09/raw_composition.cpp,raw_host.CMakeLists.txt
and RAW_COMPOSITION_PLAN.md, linking this build-r3/lib/libe8_fused_decode.so
and the final approved parallel-encoder raw library. Do not link rejected raw
cost prototypes or retain old288/7168 resource assumptions; derive requirements
from the final writer ELF before binding the host-only runtime guard.

ABI unchanged: separate K/V, D256,4352B packets (4096label+256scale bytes), outer
page/tile/head order. Dense composition maps96 raw tokens to page11rows0..63
andpage3rows0..31, H4/head1. Reader offsets for pages0,2,4,6,11,3 are
`(page*4+1)*4352`; context352 includes four encoded history tiles plus two real
raw tail tiles. Same device cache allocations, no host repack/reupload between
writer and reader. All4heads remain packet-compared. Revised underflow witness
is value/token0/head0; selectedhead1's independent attention oracle is unchanged.
Regenerate/bind all4head packet afterimages for the final writer anyway.

Scope remains pre-rotated FP32 query→BF16 cache-basis context. Raw-query rotation
and inverse output-basis production integration are separate; no physical NPU
gate or universal BF16 attention accuracy claim is implied.
