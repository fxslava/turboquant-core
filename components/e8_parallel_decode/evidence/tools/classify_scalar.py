"""Read-only scalar retry classification; baseline is a terminated partial run."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,re
root=Path('/home/dmin/work/checkpoint2_20261123/parallel_decode')
report=Path('/mnt/c/ascend-micro-harness/reports/checkpoint2_20261123/parallel_decode')
oldroot=root.parent/'slot09'
FRAME=re.compile(r'start:\s*(\d+), tick:\s*(\d+).*done')
LINE=re.compile(r'\[(\d+)\]\s+\(PC: (0x[0-9a-f]+)\)\s+(\w+)\s*:.*?\(ID: (\d+)\)\s+(\w+)(.*)$')
ISSUE=re.compile(r'\[info\]\s+(\d+): \[ISSUE INSTR\] instr.name=(\w+) instr.pc=(0x[0-9a-f]+) instr.id=(\d+)')
MISS=re.compile(r'\[info\]\s+(\d+): lookup_tag is MISS\. req pc\s*:(0x[0-9a-f]+)')
ROUTE=re.compile(r'PIPE:(\w+), TRIGGER PIPE:(\w+), FLAG ID:(0x[0-9a-f]+|\d+)')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def logs(run,role,complete):
    path=run/f'core0.{role}.instr_popped_log.dump';rows=[];frames=[]
    for line in path.open(errors='replace'):
        f=FRAME.search(line)
        if f:frames.append({'start':int(f[1]),'end':int(f[2])})
        m=LINE.search(line)
        if m:rows.append({'tick':int(m[1]),'pc':int(m[2],16),'engine':m[3],'id':int(m[4]),'op':m[5],'tail':m[6]})
    if complete:
        window=frames[-1];rows=[r for r in rows if window['start']<=r['tick']<=window['end']]
    else:
        after=max((f['end'] for f in frames),default=0)
        rows=[r for r in rows if r['tick']>after]
        window={'start':min(r['tick'] for r in rows),'end':max(r['tick'] for r in rows)}
    entry=next(r['pc'] for r in rows if r['engine']=='SCALAR')
    isa=(root/('reader_aic.disasm.txt' if role=='cubecore0' else 'reader_aiv.disasm.txt')) if complete else \
        (oldroot/('aic.disasm.txt' if role=='cubecore0' else 'aiv.disasm.txt'))
    static={}
    for line in isa.open(errors='replace'):
        m=re.match(r'\s*([0-9a-f]+):\s+[0-9a-f]+\s+(\S+)(.*)',line)
        if m:static[int(m[1],16)]={'op':m[2],'operands':m[3]}
    bykey={(r['id'],r['pc']):r for r in rows};runs=[]
    ccu=run/f'core0.{role}.ccu_log.dump'
    for line in ccu.open(errors='replace'):
        m=ISSUE.search(line)
        if not m:continue
        tick=int(m[1]);pc=int(m[3],16);index=int(m[4])
        if not window['start']<=tick<=window['end']:continue
        key=(m[2],pc,index)
        if runs and runs[-1]['key']==key:
            runs[-1]['last']=tick;runs[-1]['attempts']+=1
        else:runs.append({'key':key,'name':m[2],'pc':pc,'id':index,'first':tick,'last':tick,'attempts':1})
    misses=[];dcache=run/f'core0.{role}.dcache_log.dump'
    for line in dcache.open(errors='replace'):
        m=MISS.search(line)
        if m and window['start']<=int(m[1])<=window['end']:misses.append({'tick':int(m[1]),'pc':int(m[2],16)})
    retries=[]
    for r in runs:
        if r['attempts']<2:continue
        row=bykey.get((r['id'],r['pc']))
        decoded=static.get(r['pc']-entry)
        if row is None and decoded and decoded['op']=='DSB':
            category='DSB_'+decoded['operands'].split('<')[0].strip()
        elif row is None and decoded and decoded['op']=='WAIT_FLAG.MTE3.S':
            category='explicit_MTE3_SCALAR_0_censored_before_pop'
        elif row is None and decoded and decoded['op']=='BAR.ALL':
            category='explicit_BAR_ALL'
        elif row is None:
            # A killed baseline can stop before an instruction is popped.
            category='unresolved_partial_or_not_popped'
        elif row['op']=='WAIT_FLAG':
            m=ROUTE.search(row['tail']);assert m
            route=(m[1],m[2],int(m[3],0))
            category='inherited_SDK_VEC_SCALAR_7' if route==('VEC','SCALAR',7) else 'explicit_'+','.join(map(str,route))
        elif row['op'].startswith(('ST_','ST_IO','LD_IO','LDP','LD_')):
            category='ABI_or_scalar_memory_dependency'
        elif row['op'] in ('OR','CMP','CMPN','ZEROEXT','MOV_SPR_XN','MOV_XD_SPR'):
            category='scalar_control_or_register_dependency'
        else:category='other_typed_'+row['op']
        retries.append({'name':r['name'],'op':row['op'] if row else (decoded['op'] if decoded else None),'relative_pc':hex(r['pc']-entry),
            'id':r['id'],'first':r['first'],'last':r['last'],'elapsed':r['last']-r['first'],
            'attempts':r['attempts'],'every_tick':r['attempts']==r['last']-r['first']+1,'category':category,
            'operands':row['tail'] if row else (decoded['operands'] if decoded else None),
            'typing_evidence':'popped record' if row else 'static ISA at exact relative PC'})
    groups=defaultdict(list)
    for r in retries:groups[r['category']].append(r)
    summary={k:{'count':len(v),'elapsed_sum':sum(r['elapsed'] for r in v),
        'min':min(r['elapsed'] for r in v),'max':max(r['elapsed'] for r in v)} for k,v in groups.items()}
    return {'window':window,'complete_fused_frame':complete,'entry_pc':hex(entry),
        'files':{p.name:sha(p) for p in (path,ccu,dcache)},'categories':summary,'retries':retries,'misses':misses,
        'scalar_popped':[{**r,'relative_pc':hex(r['pc']-entry)} for r in rows if r['engine']=='SCALAR']}
out={}
for label,run,complete in [('candidate',root/'run-uniform',True),('baseline_partial',oldroot/'run-uniform',False)]:
    out[label]={role:logs(run,role,complete) for role in ('cubecore0','veccore0','veccore1')}
out['scope']={'baseline':'terminated partial old reader; same data fixture, not completed timing/correctness baseline',
 'matching':'relative static PC and typed operands, not wall time or raw instruction ID across launches',
 'retry_meaning':'Repeated CCU issues bound scalar blockage; sums overlap roles and are not kernel savings.',
 'gaps':'No cause is assigned to CCU-silent fetch/delivery intervals from gaps alone.'}
for dest in (root,report):(dest/'SCALAR_RETRY_CLASSIFICATION.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({label:{role:data['categories'] for role,data in v.items()} for label,v in out.items() if label!='scope'},indent=2))
