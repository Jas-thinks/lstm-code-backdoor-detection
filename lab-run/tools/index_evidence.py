"""Refresh the project's file inventory for presentation/evidence handoff."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[2]
target=root/'results/project-evidence-index.json'
paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root).decode().split('\0')
entries=[]
for name in sorted(set(filter(None,paths))):
    path=root/name
    if path==target or not path.is_file(): continue
    sha=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): sha.update(block)
    entries.append(dict(path=name,bytes=path.stat().st_size,sha256=sha.hexdigest()))
record=dict(schema_version='project-evidence-index.v1',created_utc=datetime.now(timezone.utc).isoformat(),
            snapshot='Tracked and unignored project files; index excludes itself',
            base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            priorities=['UniXcoder benchmark','new-model detector integration','success-rate evaluation'],
            project_record='docs/PROJECT-RECORD.md',files=entries)
target.write_text(json.dumps(record,indent=2)+'\n')
print('EVIDENCE_INDEX_SAVED',len(entries),target)
