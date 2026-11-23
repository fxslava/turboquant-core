from pathlib import Path
from collections import Counter
import hashlib,json,re,shutil
root=Path('/home/dmin/work/checkpoint2_20261123/parallel_decode')
report=Path('/mnt/c/ascend-micro-harness/reports/checkpoint2_20261123/parallel_decode')
run=root/'run-component';log=(root/'run-component.log').read_text()
assert (root/'run-component-exit.txt').read_text().strip()=='0'
assert '4096/4096 BF16words0ULP;4096/4096 INT8bytesexact' in log
mapping=json.loads((run/'loaded_model_libraries.json').read_text())
assert any(x['path'].endswith('/libpem_davinci.so') and x['sha256']=='1a6420fe8123e88a120f0bbfc76710968e6708599c77d80a7b1097aca06eed33' for x in mapping)
exceptions=[str(p) for p in run.rglob('*excp*.dump') if p.stat().st_size]
assert not exceptions
counts=Counter();rows=[]
for p in run.glob('core*.instr_popped_log.dump'):
    if not p.stat().st_size:continue
    text=p.read_text(errors='replace')
    for line in text.splitlines():
        m=re.search(r'\b(SET_FLAG|WAIT_FLAG)\b.*PIPE:(\w+), TRIGGER PIPE:(\w+), FLAG ID:(0x[0-9a-f]+|\d+)',line)
        if m:
            kind,a,b,flag=m.groups();counts[(p.name,a,b,int(flag,0),kind)]+=1
    rows.append({'file':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
pairs={}
for (file,a,b,flag,kind),n in counts.items():pairs.setdefault((file,a,b,flag),Counter())[kind]=n
assert all(v['SET_FLAG']==v['WAIT_FLAG'] for v in pairs.values()),pairs
result={'status':'PASS_BOUNDED_COMPONENT','model':'CAModel dav_3510/CANN9.2.1','total_ticks':int(re.search(r'Total tick:\s*(\d+)',log)[1]),
 'scope':'one16-row D256 component;bothnewpublicBF16/INT8Arithmetic APIs;notfusedattentiontiming',
 'BF16_0ULP_words':4096,'INT8_exact_bytes':4096,'guard_bytes':512,'inputs_preserved':True,
 'boundary_scales':['+/-FP32maxsubnormal','+/-FP32minnormal','+/-2^-132','signedzero','normalcontrols'],
 'runtime_private_warp_bytes':36864,'runtime_private_perthread_bytes':1152,'runtime_divergence_warp_bytes':1024,
 'nonempty_exception_dumps':exceptions,'canonical_mapping':mapping,
 'hard_events':[{'file':k[0],'source':k[1],'target':k[2],'id':k[3],'sets':v['SET_FLAG'],'waits':v['WAIT_FLAG']} for k,v in sorted(pairs.items())],
 'trace_files':rows,'fused_reader_gate':'pendingseparategrant','hardware_NPU':False}
for dest in (root,report):(dest/'COMPONENT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
for name in ['run-component.log','run-component-exit.txt']:
    shutil.copyfile(root/name,report/name)
shutil.copyfile(run/'loaded_model_libraries.json',report/'component_loaded_model_libraries.json')
print(json.dumps({'ticks':result['total_ticks'],'HardEvent_routes':len(pairs),'exceptions':0,'scope':result['scope']}))
