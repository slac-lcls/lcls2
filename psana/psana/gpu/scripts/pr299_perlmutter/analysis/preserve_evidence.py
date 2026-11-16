"""Refresh compact durable PR299 evidence; leave build trees and inputs on scratch."""
import hashlib
from pathlib import Path
import shutil

TASK = Path('/pscratch/sd/m/monarin/agent-work/lcls2/pr299-perlmutter-screen-20261115TKJ0du6')
DEST = Path(__file__).resolve().parent
snapshot = DEST/'snapshot'
snapshot.mkdir(exist_ok=True)
for name in ('harness', 'evidence', 'logs'):
    shutil.copytree(TASK/name, snapshot/name, dirs_exist_ok=True)
for name in ('manifest.json', 'scope.txt', 'review-approval-pilot.json', 'review-approval-campaign.json', 'pilot-estimate.json'):
    if (TASK/name).exists():
        shutil.copy2(TASK/name, snapshot/name)
    elif name == 'pilot-estimate.json' and (snapshot/name).exists():
        (snapshot/name).unlink()  # Prior estimate remains archived under evidence.
shutil.copy2(TASK/'results/reference-51.json', snapshot/'reference-51.json')
for directory in (TASK/'results').iterdir():
    if directory.is_dir() and (directory/'status.json').exists():
        # Terminal result directories only: never snapshot a live writer.
        import json
        state = json.loads((directory/'status.json').read_text())
        if state.get('complete') or 'error' in state:
            target = snapshot/directory.name
            target.mkdir(exist_ok=True)
            index = {}
            for original in sorted(directory.rglob('*')):
                if not original.is_file(): continue
                relative = original.relative_to(directory)
                index[str(relative)] = dict(bytes=original.stat().st_size, sha256=hashlib.sha256(original.read_bytes()).hexdigest())
                destination = target/relative
                destination.parent.mkdir(parents=True,exist_ok=True)
                if original.name == 'sample.json':
                    sample = json.loads(original.read_text())
                    sample.pop('nic_loop_window', None)
                    destination.with_name('sample-summary.json').write_text(json.dumps(sample,indent=2)+'\n')
                elif original.name not in ('worker.json','storage.jsonl') and original.stat().st_size <= 1024*1024:
                    shutil.copy2(original,destination)
            (target/'scratch-source-index.json').write_text(json.dumps(dict(source=str(directory),files=index),indent=2)+'\n')
for directory in (TASK/'review').iterdir():
    if directory.is_file() and directory.suffix in ('.json','.txt'):
        shutil.copy2(directory, DEST/'reviews'/directory.name)
    if directory.is_dir() and (directory/'review.json').exists():
        shutil.copytree(directory, DEST/'reviews'/directory.name, dirs_exist_ok=True)
lines=[]
for path in sorted(DEST.rglob('*')):
    if path.is_file() and path.name!='SHA256SUMS' and '__pycache__' not in path.parts:
        lines.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+str(path.relative_to(DEST)))
(DEST/'SHA256SUMS').write_text('\n'.join(lines)+'\n')
print(f'Preserved {len(lines)} files in {DEST}')
