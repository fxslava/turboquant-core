"""Read-only six-tile role/fence census and bounded release observations."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,re
LINE=re.compile(r'\[(\d+)\]\s+\(PC: (0x[0-9a-f]+)\)\s+(\w+)\s*:.*?Binary:\s*(0x[0-9a-f]+).*?\(ID: (\d+)\)\s+(\w+)(.*)$')
FRAME=re.compile(r'core:\s*(\d+), subcore:\s*(\d+), start:\s*(\d+), tick:\s*(\d+).*done')
def read(path):
    rows=[];frames=[]
    for line in path.read_text(errors='replace').splitlines():
        m=LINE.search(line)
        if m:rows.append(dict(tick=int(m[1]),pc=m[2],engine=m[3],word=int(m[4],16),id=int(m[5]),op=m[6],tail=m[7]))
        f=FRAME.search(line)
        if f:frames.append(dict(core=int(f[1]),subcore=int(f[2]),start=int(f[3]),end=int(f[4])))
    return rows,frames
def route(r):
    m=re.search(r'PIPE:(\w+), TRIGGER PIPE:(\w+), FLAG ID:(0x[0-9a-f]+|\d+)',r['tail'])
    return (m[1],m[2],int(m[3],0)) if m else None
def value(r):
    m=re.search(r'XT:X\d+=(0x[0-9a-f]+|\d+)',r['tail'])
    return int(m[1],0) if m else None
def main(trace,out):
    popped={};retired={};sets={};waits={};frames={};violations=[];census={}
    old=Path('/mnt/c/ascend-micro-harness/reports/checkpoint2_20261123/slot09/aiv.disasm.txt')
    new=Path('/home/dmin/work/checkpoint2_20261123/parallel_decode/reader_aiv.disasm.txt')
    def sdk7(path):
        return re.findall(r'^\s*([0-9a-f]+):\s+(15e78000|40c40403)\s+(SET_FLAGI_V.S|WAIT_FLAGI.V.S)\s+#7',path.read_text(),re.M)
    assert sdk7(old)==sdk7(new) and len(sdk7(new))==10
    sdk7proof={'scope':'inherited SDK-reserved VEC/SCALAR event7; exact static words/offsets unchanged',
        'baseline_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),
        'candidate_sha256':hashlib.sha256(new.read_bytes()).hexdigest(),'static_entries':sdk7(new)}
    for role in ('cubecore0','veccore0','veccore1'):
        rows,fl=read(trace/f'core0.{role}.instr_popped_log.dump')
        done,_=read(trace/f'core0.{role}.instr_log.dump')
        # The two preceding encoded-scatter launches share files and reset IDs.
        # Keep only this role's last (fused-reader) frame, by global tick range.
        assert fl,role
        frame=fl[-1];frames[role]=frame
        rows=[r for r in rows if frame['start']<=r['tick']<=frame['end']]
        done=[r for r in done if frame['start']<=r['tick']<=frame['end']]
        popped[role]=rows;retired[role]=done;sets[role]={};waits[role]={}
        for r in rows:
            if r['op']=='SET_CROSS_CORE':
                v=value(r);assert v is not None and ((v>>4)&3)==2
                sets[role].setdefault((v>>8)&15,[]).append(r)
            elif r['op']=='WAIT_FLAG_DEV':waits[role].setdefault(value(r),[]).append(r)
        expectedSets={4:6,6:6,2:2,3:2,7:1} if role=='cubecore0' else {0:3,1:3,5:6,8:1}
        expectedWaits={0:3,1:3,5:6,8:1} if role=='cubecore0' else {4:6,6:6,2:2,3:2,7:1}
        actualSets={k:len(v) for k,v in sets[role].items()};actualWaits={k:len(v) for k,v in waits[role].items()}
        assert actualSets==expectedSets,(role,actualSets,expectedSets)
        assert actualWaits==expectedWaits,(role,actualWaits,expectedWaits)
        hard=Counter()
        for r in rows:
            k=route(r)
            if k and r['op'] in ('SET_FLAG','WAIT_FLAG'):
                assert 0<=k[2]<=3 or (role!='cubecore0' and k==('VEC','SCALAR',7) and
                    r['word']==(0x15e78000 if r['op']=='SET_FLAG' else 0x40c40403)),(role,k,r)
                hard[k,r['op']]+=1
        routes=sorted({k for k,op in hard})
        assert all(hard[k,'SET_FLAG']==hard[k,'WAIT_FLAG'] for k in routes)
        census[role]={'sets':actualSets,'waits':actualWaits,
            'hard_events':[{'route':k,'sets':hard[k,'SET_FLAG'],'waits':hard[k,'WAIT_FLAG']} for k in routes]}
    observations=[]
    for role,channels in waits.items():
        for channel,entries in channels.items():
            for ordinal,w in enumerate(entries):
                if role=='cubecore0':
                    required=max(sets[v][channel][ordinal]['tick'] for v in ('veccore0','veccore1'))
                    wanted=('FIXP','SCALAR',0) if channel==8 else ('MTE1','CUBE',0)
                    producer='dual-AIV quorum'
                else:
                    required=sets['cubecore0'][channel][ordinal]['tick']
                    wanted=('MTE3','SCALAR',0) if channel in (2,3) else ('VEC','MTE3',0)
                    producer='AIC broadcast'
                candidates=[r for r in popped[role] if r['id']>w['id'] and r['op']=='SET_FLAG' and route(r)==wanted]
                marker=min(candidates,key=lambda r:r['id']) if candidates else None
                done=[r for r in retired[role] if marker and r['id']==marker['id'] and r['pc']==marker['pc'] and r['op']=='SET_FLAG']
                release=done[0]['tick'] if len(done)==1 else None
                if release is None or release<required:violations.append({'role':role,'channel':channel,'ordinal':ordinal,'required':required,'observation':release})
                observations.append({'role':role,'channel':channel,'ordinal':ordinal,'producer':producer,
                    'wait_issue':w['tick'],'required_publication':required,'following_pipe_marker_completion':release,
                    'marker_route':wanted,'scope':'completion lower bound, not an exact transport latency'})
    cube=[r for r in popped['cubecore0'] if r['op']=='MMAD']
    assert len(cube)==12,len(cube)
    assert all('F32F32' in r['tail'] for r in cube),[r['tail'] for r in cube]
    alias=[r for r in cube if r['word']==0xf08b181d]
    assert alias,'Static decoder alias word not witnessed as MMAD'
    context=[r['tick'] for r in sets['cubecore0'][6]]
    result={'scope':'six-tile fused-reader exact mode2 census; source-contract and following-pipe-marker observations',
        'frames':frames,'census':census,'inherited_sdk_event7':sdk7proof,'observations':observations,'violations':violations,
        'cross_waits':len(observations),'runtime_FP32_MMAD':len(cube),'static_alias_word_actual_MMAD':alias,
        'context_publication_ticks':context,'context_intervals':[b-a for a,b in zip(context,context[1:])],
        'limitations':['Following marker completions bound release; they are not physical transport service latencies.',
            'Source/compiled per-pipe contracts and exact data witnesses remain separate mandatory gates.',
            'One six-tile run is not an exhaustive hardware interleaving proof.']}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'cross_waits':len(observations),'violations':len(violations),'FP32_MMAD':len(cube),'alias_MMAD':len(alias)}))
    if violations:raise SystemExit(1)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('trace',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();main(args.trace,args.out)
