import json,re
from pathlib import Path
root=Path('/home/dmin/work/checkpoint2_20261123/slot07')
lines=(root/'run/core0.veccore0.ccu_log.dump').read_text().splitlines()
pat=re.compile(r'\[info\] (\d+): \[(.*?)\].*?instr.name=(\S+) instr.pc=(0x[0-9a-f]+) instr.id=(\d+)')
events=[]
for no,line in enumerate(lines,1):
 m=pat.search(line)
 if m: events.append(dict(line=no,tick=int(m[1]),stage=m[2],name=m[3],pc=m[4],id=int(m[5])))
records=[]
for target,previous in [(141,139),(269,268)]:
 target_events=[e for e in events if e['id']==target]
 prev_events=[e for e in events if e['id']==previous]
 record=dict(target_id=target,previous_scalar_id=previous,target_events=target_events,previous_events=prev_events,
             target_issue_attempts=len([e for e in target_events if e['stage']=='ISSUE INSTR']),
             target_general_hazard_retries=0)
 record['ccu_no_events_after_previous_execute_before_target_d1']=[e for e in events if
  max(x['tick'] for x in prev_events)<e['tick']<min(x['tick'] for x in target_events)]
 records.append(record)
result=dict(status='PERFORMANCE_GATE_INCOMPLETE',component_transport='PASS_CAMODEL_PACKET',
 observable_classification='Pre-D1 instruction-delivery gap. No repeated issue attempt or explicit scalar instruction hazard at either target is evidenced.',
 exact_cause='Not established: no instruction-fetch/cache miss or decode-block reason is present in retained logs.',
 no_baseline_claim='This new source lacks a matched compact-grinder control. The gaps cannot be labelled explained inherited baseline stalls.',
 isa_dependencies=[dict(pc='0x200',isa='ADD.s64 X12, X1, X23',note='Sources are existing packet base and constant4096; X12 formerly serves DMA descriptor at0x1fc. ADD has one issue/execute and no recorded general-hazard retry.'),
                   dict(pc='0x400',isa='MOVI X28, #4512',note='Immediate scalar write has no source data operand. X28 earlier names a UB scale source; no target issue retry is recorded. One separate DMA id270 retry occurs at2300..2301 after this delivery gap.')],
 candidate_hypothesis='Cold instruction fetch at512-byte code boundaries is consistent with0x200/0x400 positions and absent D1 delivery, but is not proved by these traces.',
 records=records)
(root/'ccu_gap_classification.json').write_text(json.dumps(result,indent=2))
for record in records:
 print('id',record['target_id'],'target_events=',record['target_events'],'issue_attempts=',record['target_issue_attempts'])
