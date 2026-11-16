"""Verify the delivered file manifest and frozen development inputs."""
from pathlib import Path
import hashlib
import json
from src.common import load_development_data
ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'CHECKSUMS_SUBMISSION.json').read_text(encoding='utf-8'))
for name,expected in manifest.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
train,val,_=load_development_data(ROOT)
print(json.dumps({'status':'passed','files':len(manifest),'train':len(train),'validation':len(val)}))
