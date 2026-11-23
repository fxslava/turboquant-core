import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
def verify(path,digest,size=None):
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=digest: raise RuntimeError(f'Hash changed: {path}')
    if size is not None and len(data)!=size: raise RuntimeError(f'Size changed: {path}')
records=json.loads((root/'source_provenance.json').read_text())['byte_identical_native_sources']
for record in records: verify(root/record['packaged'],record['sha256'])
export=json.loads((root/'reference/e8_wave1_export.json').read_text())
for header in export['headers']: verify(root/'reference'/header['file'],header['sha256'])
manifest=json.loads((root/'fixtures/manifest.json').read_text())
for payload in manifest['files']: verify(root/'fixtures'/payload['file'],payload['sha256'],payload['bytes'])
print(f'PASS {len(records)} native source snapshots, {len(export["headers"])} frozen E8 leaves, {len(manifest["files"])} fixture payloads')
