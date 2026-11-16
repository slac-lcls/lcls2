"""Read completed campaign diagnostics; write derived evidence outside raw results."""
import json
from pathlib import Path
from analyze_storage_progress import analyze
from summarize_campaign import summarize

ROOT = Path('/pscratch/sd/m/monarin/agent-work/lcls2/pr299-perlmutter-screen-20261115TKJ0du6')
DEST = Path(__file__).resolve().parent
directory = ROOT/'results/campaign-job-61375111'
summary = summarize(directory)
(DEST/'campaign-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
results = {}
for index, sample in enumerate(sorted(directory.glob('timed-*/sample.json')), 1):
    results[sample.parent.name] = analyze(sample.parent)
    print(f'Analyzed {index}/48: {sample.parent.name}', flush=True)
(DEST/'campaign-storage-progress.json').write_text(json.dumps(results, indent=2)+'\n')
print('All diagnostic summaries saved', flush=True)
