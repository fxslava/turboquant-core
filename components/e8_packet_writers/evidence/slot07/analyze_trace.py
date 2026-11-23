import json,re
from pathlib import Path
root=Path('/home/dmin/work/checkpoint2_20261123/slot07')
pattern=re.compile(r'\[info\] \[(\d+)\] \(PC: (0x[0-9a-f]+)\) (\w+)\s*:.*?\(ID: (\d+)\) (.*)')
def read(name):
 result=[]
 for line in (root/'run'/name).read_text().splitlines():
  m=pattern.match(line)
  if m: result.append(dict(tick=int(m[1]),pc=int(m[2],16),pipe=m[3],id=int(m[4]),op=m[5]))
 return result
issue=read('core0.veccore0.instr_popped_log.dump')
retire=read('core0.veccore0.instr_log.dump')
by_id={e['id']:e for e in issue}
inputs=[e for e in retire if e['pipe']=='MTE2' and 'Src:OUT, Dst:UB' in e['op']]
outputs=[e for e in retire if e['pipe']=='MTE3' and 'Src:UB, Dst:OUT' in e['op']]
assert len(inputs)==len(outputs)==24,(len(inputs),len(outputs))
inputs.sort(key=lambda e:e['id']);outputs.sort(key=lambda e:e['id'])
strips=[]
for i in range(6):
 ins=inputs[i*4:i*4+4];outs=outputs[i*4:i*4+4]
 strips.append(dict(strip=i,ring_slot=i%4,first_input_popped=min(by_id[e['id']]['tick'] for e in ins),
  all_inputs_retired=max(e['tick'] for e in ins),first_output_popped=min(by_id[e['id']]['tick'] for e in outs),
  all_outputs_retired=max(e['tick'] for e in outs),output_retire_ticks=[e['tick'] for e in outs],
  input_instructions=[dict(pc=hex(e['pc']),id=e['id'],popped=by_id[e['id']]['tick'],retired=e['tick']) for e in ins],
  output_instructions=[dict(pc=hex(e['pc']),id=e['id'],popped=by_id[e['id']]['tick'],retired=e['tick']) for e in outs]))
for i,s in enumerate(strips): s['output_completion_delta']=None if not i else s['all_outputs_retired']-strips[i-1]['all_outputs_retired']
scalar=[e for e in issue if e['pipe']=='SCALAR' and 0x10d0d184<=e['pc']<=0x10d0d60c]
scalar_gaps=[dict(previous_pc=hex(a['pc']),next_pc=hex(b['pc']),start=a['tick'],end=b['tick'],gap=b['tick']-a['tick']) for a,b in zip(scalar,scalar[1:]) if b['tick']-a['tick']>16]
scalar_loads=[e for e in scalar if re.match(r'(LD|LDP|LDK|LD_IO)\b',e['op'])]
scalar_waits=[e for e in scalar if e['op'].startswith('WAIT')]
result=dict(scope='one sixstrip encoded E8 K/V packet-transport probe, not BF16 quantization',
 model_total_ticks=4576,strips=strips,completion_deltas=[s['output_completion_delta'] for s in strips[1:]],
 completion_interval_mean=sum(s['output_completion_delta'] for s in strips[1:])/5,
 post_fill_intervals=[strips[4]['output_completion_delta'],strips[5]['output_completion_delta']],
 post_fill_mean=(strips[5]['all_outputs_retired']-strips[3]['all_outputs_retired'])/2,
 stationary_steady_state_proven=False,scalar_loads_in_grinder=len(scalar_loads),scalar_explicit_waits_in_grinder=len(scalar_waits),
 scalar_popped_gaps_over16=scalar_gaps,
 timing_note='Popped and retired timestamps are queue observations. Popped-to-retired span may include queueing; it is not an isolated hardware latency. Postfill has only two samples and alternating completion gaps; no stationary throughput claim.')
(root/'strip_trace.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['strips']},indent=2))
print('strip completions:',[s['all_outputs_retired'] for s in strips])
