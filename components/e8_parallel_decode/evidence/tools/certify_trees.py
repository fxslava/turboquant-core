from pathlib import Path
import json,re
root=Path('/home/dmin/work/checkpoint2_20261123/parallel_decode')
report=Path('/mnt/c/ascend-micro-harness/reports/checkpoint2_20261123/parallel_decode')
audit=json.loads((root/'compiled_audit.json').read_text());result={}
for kind in ('component','reader_aiv'):
    text=(root/f'{kind}.disasm.txt').read_text()
    rows=[]
    for line in text.splitlines():
        m=re.match(r'\s*([0-9a-f]+):\s+[0-9a-f]+\s+(\S+)(.*)',line)
        if m:rows.append((int(m[1],16),m[2],m[3]))
    for name,f in audit[kind]['simt_functions'].items():
        current={};nodes={};depth={};branches=[]
        body=[r for r in rows if f['start']<=r[0]<f['start']+f['bytes']]
        for pc,op,args in body:
            if op.startswith('BRANCH'):branches.append(pc)
            m=re.match(r'\s*R(\d+),',args)
            if not m or op.startswith(('ST','ISETP','PLOP')):continue
            dst=int(m[1])
            if op=='IMNMX.u32':
                regs=list(map(int,re.findall(r'\bR(\d+)\b',args)))
                assert len(regs)==3
                parents={current[r] for r in regs[1:] if current.get(r) is not None}
                nodes[pc]=parents;depth[pc]=1+max((depth[p] for p in parents),default=0)
                current[dst]=pc
            else:current[dst]=None
        groups=[];pending=set(nodes)
        while pending:
            todo=[min(pending)];group=set()
            while todo:
                n=todo.pop()
                if n in group:continue
                group.add(n);todo.extend(nodes[n])
                todo.extend(k for k,v in nodes.items() if n in v)
            pending-=group
            groups.append({'pcs':[hex(x) for x in sorted(group)],
                           'max_reduction_depth':max(depth[x] for x in group),
                           'max_operations':len(group)})
        expected=0 if 'PackQuery' in name else (2 if kind=='component' else 8)
        assert len(groups)==expected,(name,groups)
        assert all(x['max_operations']==7 and x['max_reduction_depth']==3 for x in groups),(name,groups)
        if kind=='reader_aiv':assert not branches,(name,branches)
        else:
            assert len(branches)==2
            # Entry/whole-block stride loop only; no branch inside argmax trees.
            assert all(branches[0]<min(int(p,16) for p in x['pcs']) and
                       branches[-1]>max(int(p,16) for p in x['pcs']) for x in groups)
        result[kind+':'+name]={'argmax_trees':groups,'branch_pcs':[hex(x) for x in branches],
            'scope':'direct integer-max producer dependency graph; not a one-cycle claim',
            'scalar_tensor_loops':False,'coordinate_backedges':False}
for dest in (root,report):(dest/'tree_isa_certificate.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'functions':len(result),'depth3_argmax_trees':sum(len(v['argmax_trees']) for v in result.values()),
                  'coordinate_backedges':0}))
