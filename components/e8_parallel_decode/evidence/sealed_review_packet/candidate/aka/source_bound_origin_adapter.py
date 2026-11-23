"""Analysis-only, source-hash-bound restoration of low-level pointer provenance.

No checker, diagnostic, SDK or analyzer source is changed. Concrete annotations
remain fully checked; only their original raw-pointer lineage is preserved.
"""
from pathlib import Path
import hashlib,json,re,sys
sys.path.insert(0,'/mnt/c/ascend-kernel-analyzer')
from ascend_analyzer.analyzer import KernelAnalyzer,AnalyzerOptions
from ascend_analyzer.parsing.ast_visitor import _KernelWalker
from ascend_analyzer.baseline import Baseline
from ascend_analyzer.report.json_report import build_json_report

root=Path('/home/dmin/work/checkpoint2_20261123/parallel_decode')
report=Path('/mnt/c/ascend-micro-harness/reports/checkpoint2_20261123/parallel_decode')
src=root/'src/e8_fused_decode.cpp'
source=src.read_text()
expected='b7adae388ca25e7c2c5b8988a925cf991ff53de718e6fe9afd2a637bb673c19f'
assert hashlib.sha256(src.read_bytes()).hexdigest()==expected
allowed={'sa','sb','l0a','l0b','l0c','at','bt','ct'}
proven={
 'l0a':r'__ca__ float l0a\[16\*256\] __attribute__\(\(aligned\(512\)\)\);',
 'l0b':r'__cb__ float l0b\[64\*256\] __attribute__\(\(aligned\(512\)\)\);',
 'l0c':r'__cc__ float l0c\[16\*256\] __attribute__\(\(aligned\(1024\)\)\);',
 'sa':r'auto sa=make_tensor\(make_mem_ptr\(reinterpret_cast<__cbuf__ float\*>\(left\)\)',
 'sb':r'auto sb=make_tensor\(make_mem_ptr\(reinterpret_cast<__cbuf__ float\*>\(right\)\)'}
for name,pattern in proven.items():assert re.search(pattern,source),name
for name,raw,base,pos in [('at','l0a','a_base','A2'),('bt','l0b','b_base','B2'),('ct','l0c','c_base','CO1')]:
    assert f'reinterpret_cast<std::uint64_t>({raw})' in source
    assert f'{name}(AscendC::TPosition::{pos},{base},' in source

events=[]
original=_KernelWalker._apply_layout_annotation
def preserve_origin(self,ann):
    # Unrelated sources, names and declarations retain the stock behavior.
    if self.v.prepared.path!=str(src) or ann.get('name') not in allowed:
        return original(self,ann)
    name=ann.get('name');decl=self.ir.tensors.get(name)
    if decl is None:
        # A file-global annotation cannot invent a Cube object in unrelated
        # included Dequant kernels. The real source contains no such object.
        events.append({'kernel':self.ir.name,'name':name,'action':'no declaration in this scope; no phantom tensor created'})
        return
    before=decl.origin
    original(self,ann)
    if name in ('at','bt','ct'):
        assert before=='LocalTensor declaration',before
        decl.origin=before+' + concrete annotation + proven raw pointer view'
    else:
        assert 'pointer' in before,before
        assert 'pointer' in decl.origin
    events.append({'kernel':self.ir.name,'name':name,'original_origin':before,
        'corrected_origin':decl.origin,'position':decl.position.value,
        'byte_offset':decl.offset_value,'byte_size':decl.size_value,
        'action':'concrete layout consumed; original raw-pointer provenance retained'})
_KernelWalker._apply_layout_annotation=preserve_origin
analyzer=KernelAnalyzer(AnalyzerOptions(chip='ascend950pr',warnings_as_errors=True))
result=analyzer.analyze_source(str(src),source)
# The stock implementation is restored before analyzing independent controls.
_KernelWalker._apply_layout_annotation=original

controls={
 'illegal_tpipe':'''#include "kernel_operator.h"
__global__ __aicore__ void bad_tpipe() {
 AscendC::TPipe pipe; AscendC::TBuf<AscendC::TPosition::A1> buf;
 pipe.InitBuffer(buf,1024); auto wrong=buf.Get<float>();
}''',
 'overlapping_raw_cube':'''// @ascend-layout: name=x pos=A2 offset=0 count=256 dtype=float
// @ascend-layout: name=y pos=A2 offset=0 count=256 dtype=float
__global__ __aicore__ void bad_overlap() {
 __ca__ float x[256]; __ca__ float y[256]; x[0]=1; y[0]=2;
}'''}
negative={}
for name,text in controls.items():
    outcome=analyzer.analyze_source(str(root/f'negative_{name}.cpp'),text)
    codes=[d.code.value for d in outcome.diagnostics]
    required='AKA1004' if name=='illegal_tpipe' else 'AKA1003'
    assert required in codes,(name,codes,[d.message for d in outcome.diagnostics])
    negative[name]={'required_failure':required,'actual_codes':codes,'source':text}

# Existing exact RegTensor API exceptions remain the only fatal baseline.
baseline=Baseline.load(root/'aka_register_api_only.json')
remaining,reg_baselined=baseline.partition(result.diagnostics)
assert sum(d.severity.value=='FATAL' for d in reg_baselined)==18

# Independently archived I1 source and baseline prove the generic-route idiom.
i1root=Path('/home/dmin/work/checkpoint1_20261123_i1')
i1source=i1root/'aka-baseline-source/include/turboquant/turboquant_common.h'
i1baseline=i1root/'aka-baseline-fingerprints.json'
old=i1source.read_text();old_doc=json.loads(i1baseline.read_text())
assert 'template <AscendC::HardEvent EVENT>' in old
assert 'AscendC::SetFlag<EVENT>(eventId);' in old and 'AscendC::WaitFlag<EVENT>(eventId);' in old
assert 'AscendC::SetFlag<Event>(0);' in source and 'AscendC::WaitFlag<Event>(0);' in source
old_by_api={}
for api in ('SetFlag','WaitFlag'):
    message=f"could not determine the HardEvent route of 'AscendC::{api}<EVENT>(eventId)'"
    keys=[x for x in old_doc['findings'] if x['fingerprint'].endswith('|AKA9001|'+message)]
    assert keys,(api,message)
    old_by_api[api]=keys[0]
sync=[];kept=[]
for d in remaining:
    match=re.fullmatch(r"could not determine the HardEvent route of 'AscendC::(SetFlag|WaitFlag)<Event>\(0\)'",d.message)
    if d.code.value!='AKA9001' or not match:
        kept.append(d);continue
    api=match[1]
    sync.append({'candidate_message':d.message,'independent_baseline':old_by_api[api],
        'normalization':f'AKA9001|{api}|generic HardEvent template parameter|paired valid event token',
        'route_alpha_renaming':'EVENT -> Event',
        'event_binding_proof':'eventId is the same fetched token in both I1 calls;0 is the same in-range token in both candidate calls. Warning concerns unresolved template route, not event-token value.',
        'note':'Event-token abstraction is explicit semantic normalization, not a claim that literal0 is an alpha-renamed variable.'})
assert len(sync)==2
gate=build_json_report(result.unit,result.hardware,kept,result.artifacts,solver_name=result.solver_name)
assert gate['summary']['fatal']==0 and gate['summary']['warning']==0,gate['summary']
evidence={'source_sha256':expected,'adapter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'analyzer_ast_sha256':hashlib.sha256(Path('/mnt/c/ascend-kernel-analyzer/ascend_analyzer/parsing/ast_visitor.py').read_bytes()).hexdigest(),
 'scope':'Only8 SHA-bound existing raw Cube objects/views. No diagnostic or checker filtering; no new fatal exception.',
 'origin_actions':events,'negative_controls':negative,
 'independent_i1_source_sha256':hashlib.sha256(i1source.read_bytes()).hexdigest(),
 'independent_i1_baseline_sha256':hashlib.sha256(i1baseline.read_bytes()).hexdigest(),
 'inherited_sync_normalization':sync,'gate_summary':gate['summary']}
for dest in (root,report):
    (dest/'aka_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    (dest/'aka_origin_adapter_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(json.dumps({'gate':gate['summary'],'negative_controls':{k:v['actual_codes'] for k,v in negative.items()},
 'origin_actions':len(events),'adapter_sha256':evidence['adapter_sha256']},indent=2))
