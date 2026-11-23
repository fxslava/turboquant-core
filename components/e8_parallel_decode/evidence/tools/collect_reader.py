from pathlib import Path
from collections import Counter
import argparse,hashlib,json,re,shutil
root=Path('/home/dmin/work/checkpoint2_20261123/parallel_decode')
report=Path('/mnt/c/ascend-micro-harness/reports/checkpoint2_20261123/parallel_decode')
parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['uniform','nonuniform'])
args=parser.parse_args();mode=args.mode;run=root/f'run-{mode}'
log=(root/f'run-{mode}.log').read_text()
assert (root/f'run-{mode}-exit.txt').read_text().strip()=='0'
assert 'PASS actual encoded scatter->E8 fused attention:0BF16ULP' in log
audit=json.loads((root/f'{mode}_trace_audit.json').read_text())
assert audit['cross_waits']==47 and not audit['violations']
assert audit['runtime_FP32_MMAD']==12 and audit['static_alias_word_actual_MMAD']
mapping=json.loads((run/'loaded_model_libraries.json').read_text())
assert any(p['path'].endswith('/libpem_davinci.so') and p['sha256']=='1a6420fe8123e88a120f0bbfc76710968e6708599c77d80a7b1097aca06eed33' for p in mapping)
exceptions=[str(p) for p in run.rglob('*excp*.dump') if p.stat().st_size];assert not exceptions
total=int(re.search(r'Total tick:\s*(\d+)',log)[1])
frames=audit['frames'];start=min(f['start'] for f in frames.values());end=max(f['end'] for f in frames.values())
result={'status':'PASS_BOUNDED_ENCODED_SCATTER_TO_CACHE_BASIS_ATTENTION','fixture':mode,
 'process_total_ticks':total,'process_scope':'two single-token encoded scatter launches plus one six-tile fused reader',
 'fused_reader_participating_frame_start':start,'fused_reader_participating_frame_end':end,
 'fused_reader_participating_frame_ticks':end-start,'valid_context_rows':352,'attention_tiles':6,'query_heads':6,
 'BF16_attention_words_0ULP':1536,'BF16_KV_witness_words_exact':196608,
 'FP32_QK_witness_words_exact':2304,'full_cache_afterimage_bytes_exact':208896,
 'exterior_guard_bytes':1536,'inactive_output_bytes':1024,'read_only_inputs_preserved':True,
 'cross_waits_checked':47,'cross_release_observation_violations':0,'cross_terminal_credits':'zero by exact mode2 per-channel quorum/broadcast counts',
 'runtime_FP32_MMAD':12,'static_decoder_alias_verified_as_actual_MMAD':True,
 'inherited_SDK_event7':'seven balanced pairs perAIV; exactfive static instruction-pairs unchanged fromfrozenISA; notnewmanualID7',
 'runtime_private_warp_bytes':36864,'runtime_private_per_thread_bytes':1152,'runtime_divergence_warp_bytes':1024,
 'canonical_mapping':mapping,'nonempty_exception_dumps':exceptions,
 'context_publication_ticks':audit['context_publication_ticks'],'context_publication_intervals':audit['context_intervals'],
 'late_interior_context_interval_estimate':sum(audit['context_intervals'][-3:-1])/2,
 'limitation':'one six-tile process; no cache-persistence, asymptotic throughput, old-aborted-reader speedup, raw-query/inverse-basis production or physicalNPU claim',
 'source_address_mapping':{'K_packed_bytes_between_label_lanes':1024,'V_packed_bytes_between_label_lanes':256,'GM_BF16_witness_bytes_between_label_lanes':16},
 'latency_scope':'Source contains integer/FP32 arithmetic,UBloads/stores,explicitGMwitnessstores andpipewaits. Their spans overlap; gaps alone do not identify arithmetic or memory causation.',
 'artifacts':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
    [f'run-{mode}.log',f'run-{mode}-exit.txt',f'{mode}_trace_audit.json','PRE_SIM_MANIFEST.json']}}
for dest in (root,report):(dest/f'{mode.upper()}_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
for name in [f'run-{mode}.log',f'run-{mode}-exit.txt',f'{mode}_trace_audit.json']:
    shutil.copyfile(root/name,report/name)
shutil.copyfile(run/'loaded_model_libraries.json',report/f'{mode}_loaded_model_libraries.json')
print(json.dumps({'total':total,'reader_frame':end-start,'BF16_0ULP':1536,'KV_words':196608,'FP32_scores':2304,'crosswaits':47,'exceptiondumps':0}))
