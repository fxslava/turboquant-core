from pathlib import Path
from collections import Counter
import hashlib,json,re,struct,subprocess
root=Path('/home/dmin/work/checkpoint2_20261123/parallel_decode')
report=Path('/mnt/c/ascend-micro-harness/reports/checkpoint2_20261123/parallel_decode')
build=root/'build-r3'
dump=Path('/home/dmin/Ascend-9.2.1/cann-9.2.1/tools/bisheng_compiler/bin/llvm-objdump')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sections(path):
    data=path.read_bytes();assert data[:6]==b'\x7fELF\x02\x01'
    off=struct.unpack_from('<Q',data,40)[0];size,count,idx=struct.unpack_from('<HHH',data,58)
    headers=[struct.unpack_from('<IIQQQQIIQQ',data,off+i*size) for i in range(count)]
    names=data[headers[idx][4]:headers[idx][4]+headers[idx][5]]
    result={}
    for h in headers:
        name=names[h[0]:].split(b'\0')[0].decode()
        result[name]={'address':h[3],'bytes':h[5],'alignment':h[8],'data':data[h[4]:h[4]+h[5]]}
    return result
objects={'component':build/'e8_parallel_dequant_aiv_device_dir/device_aiv.o',
    'reader_aiv':build/'e8_fused_decode_aiv_device_dir/device.o',
    'reader_aic':build/'e8_fused_decode_aic_device_dir/device.o'}
result={}
for kind,obj in objects.items():
    sect=sections(obj);text=subprocess.check_output([str(dump),'-d',str(obj)],text=True)
    (root/f'{kind}.disasm.txt').write_text(text);(report/f'{kind}.disasm.txt').write_text(text)
    allocations={k:{a:v[a] for a in ('address','bytes','alignment')} for k,v in sect.items()
        if k.startswith(('.ubuf.','.ca.','.cb.','.cc.','.cbuf.'))}
    meta={}
    for name,v in sect.items():
        if not name.startswith('.ascend.meta.'):continue
        raw=v['data'];pos=0;records=[]
        while pos<len(raw):
            tag,n=struct.unpack_from('<HH',raw,pos);pos+=4
            records.append({'tag':tag,'bytes':n,'value':int.from_bytes(raw[pos:pos+n],'little')});pos+=n
        meta[name]=records
    symbols=subprocess.check_output(['readelf','-sW',str(obj)],text=True)
    instructions=[]
    for line in text.splitlines():
        m=re.match(r'\s*([0-9a-f]+):\s+([0-9a-f]+)\s+(\S+)(.*)',line)
        if m:instructions.append({'pc':int(m[1],16),'word':m[2],'op':m[3],'operands':m[4]})
    functions={}
    for line in symbols.splitlines():
        p=line.split(maxsplit=7)
        if len(p)!=8 or p[3]!='FUNC' or not p[7].endswith('_simt_entry'):continue
        start=int(p[1],16);n=int(p[2]);body=[i for i in instructions if start<=i['pc']<start+n]
        ops=Counter(i['op'] for i in body)
        backedges=[]
        for i in body:
            if i['op'].startswith(('BRANCH','JUMP')):
                backedges.append(i)
        source='\n'.join(i['op']+' '+i['operands'] for i in body)
        functions[p[7]]={'start':start,'bytes':n,'instructions':len(body),'opcodes':dict(ops),
            'max_R':max(map(int,re.findall(r'\bR(\d+)\b',source)),default=-1),
            'private_frames':list(map(int,re.findall(r'IADD\s+R2, R3, #-(\d+)',source))),
            'branch_sites':backedges,
            'STK':sum(v for k,v in ops.items() if k.startswith('STK')),
            'LDK':sum(v for k,v in ops.items() if k.startswith('LDK'))}
    result[kind]={'object_sha256':sha(obj),'allocations':allocations,'metadata':meta,'simt_functions':functions,
        'opcodes':dict(Counter(i['op'] for i in instructions))}
    arenas=[v for k,v in allocations.items() if k.startswith('.ubuf.')]
    assert len(arenas)==1 and arenas[0]['address']==0
    assert arenas[0]['bytes']==(14400 if kind=='component' else 156416)
    assert arenas[0]['alignment']==32
    assert all(f['max_R']<127 for f in functions.values())
    if kind=='reader_aic':
        old=sections(Path('/home/dmin/work/checkpoint2_20261123/slot09/build/e8_fused_decode_aic_device_dir/device.o'))
        assert sect['.text']['data']==old['.text']['data'],'AIC math/order changed'
        result[kind]['AIC_text_byte_identical_to_frozen_reader']=True
for name in ('build-r3.log',):
    log=(root/name).read_text();assert not re.search(r'warning:|error:',log)
result['source_manifest_sha256']=sha(root/'source_manifest.json')
result['status']='ELF/strict build checked; branch dataflow and runtime gates pending'
for target in (root,report):(target/'compiled_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:{'allocations':v['allocations'],'functions':v['simt_functions']} for k,v in result.items() if isinstance(v,dict)},indent=2))
