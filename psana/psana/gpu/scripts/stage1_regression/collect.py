"""Preserve compact accepted evidence and render paired scaling plots."""
import argparse, csv, hashlib, json, statistics
from pathlib import Path
from collections import Counter
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--stage1-job', type=int, required=True)
parser.add_argument('--stage1b-job', type=int, required=True)
parser.add_argument('--output', type=Path, required=True)
args=parser.parse_args()
root=args.root
output=args.output
report={'runtime_commits':json.loads((root/'commits.json').read_text()),
        'frozen_manifest':str(root/'hashes.json'),
        'frozen_manifest_sha256':hashlib.sha256((root/'hashes.json').read_bytes()).hexdigest(),
        'reference':json.loads((root/'reference.json').read_text())['10000'],
        'comparisons':{}}
for comparison,job in [('stage1',args.stage1_job),('stage1b',args.stage1b_job)]:
 p=root/f'job-{job}'
 provenance=json.loads((p/'provenance.json').read_text())
 assert provenance.get('complete') is True, (comparison,'incomplete')
 rows=json.loads((p/'results.json').read_text())
 assert len(rows)==80 and sum(not r['diagnostic'] for r in rows)==64
 variants=provenance['settings']['variants']
 samples=[]
 for row in rows:
  active=[r for r in row['ranks'] if r['is_bd']]
  tag=f"{row['variant']}-g1-bd{row['nbds']}-{row['bulk']}-{row['cache']}-r{row['repetition']}"+('-pixels' if row['diagnostic'] else '')
  gpu_peak=0.
  for fields in csv.reader((p/(tag+'-gpu.csv')).read_text().splitlines()):
   if len(fields)==5 and fields[1].strip()=='0': gpu_peak=max(gpu_peak,float(fields[2]))
  sample={k:row[k] for k in ('variant','workload','nbds','bulk','cache','repetition','diagnostic','events','loop_s','events_per_s','steady_events_per_s','bytes','requests','pixel_samples','log_sha256')}
  sample.update(gpu_peak_mib=gpu_peak,
     charged_peak_sum=sum(r['counts']['peak_owned_and_held'] for r in active),
     pinned_bytes_sum=sum(r['pinned_bytes'] for r in active),
     bd_events=[len(r['timestamps']) for r in active],
     bd_loop_s=[r['loop_s'] for r in active],
     bd_setup_s=[r['setup_s'] for r in active],
     bd_first_event_s=[r['first_event_s'] for r in active],
     bd_read_wait_s=[r['counts']['read_wait_s'] for r in active],
     network_bytes=row.get('network_bytes'))
  # Sum startup and steady rather than overwriting equal launch names.
  counts=Counter()
  for rank in active:
   for key,stat in (rank['diagnostic_stats'] or {}).items():
    if '/launch.' in key: counts[key.split('/',1)[1]]+=stat['calls']
  sample['launch_counts']=dict(counts)
  if row['diagnostic']:sample['bd_host_phases']=[r['diagnostic_stats'] for r in active]
  samples.append(sample)
 cells=[]
 for bds in (1,2,3,4):
  for cache in ('cold','warm'):
   for bulk in ('off','on'):
    cell=dict(bds=bds,cache=cache,bulk=bulk)
    for name in variants:
     ss=[s for s in samples if not s['diagnostic'] and (s['nbds'],s['cache'],s['bulk'],s['variant'])==(bds,cache,bulk,name)]
     assert len(ss)==2
     cell[name]=dict(events_per_s=10000/statistics.median(s['loop_s'] for s in ss),
                    repetitions_hz=[s['events_per_s'] for s in ss],
                    gpu_peak_mib=max(s['gpu_peak_mib'] for s in ss),
                    charged_peak_sum=max(s['charged_peak_sum'] for s in ss))
    cell['change_percent']=(cell[variants[1]]['events_per_s']/cell[variants[0]]['events_per_s']-1)*100
    cells.append(cell)
 report['comparisons'][comparison]=dict(job=job,provenance=provenance,variants=variants,cells=cells,samples=samples)
output.write_text(json.dumps(report,indent=2)+'\n')
print(output)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'svg.fonttype':'none','font.size':10})
fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
for row_index,(name,comp) in enumerate(report['comparisons'].items()):
 for col,cache in enumerate(('cold','warm')):
  ax=axes[row_index,col]
  for mode,color in [('off','#2166ac'),('on','#c65d16')]:
   for i,variant in enumerate(comp['variants']):
    cells=[c for c in comp['cells'] if c['cache']==cache and c['bulk']==mode]
    ys=[c[variant]['events_per_s'] for c in cells]
    lows=[max(0,y-min(c[variant]['repetitions_hz'])) for y,c in zip(ys,cells)]
    highs=[max(0,max(c[variant]['repetitions_hz'])-y) for y,c in zip(ys,cells)]
    ax.errorbar([c['bds'] for c in cells],ys,yerr=[lows,highs],
        color=color,linestyle='--' if i==0 else '-',marker='o' if i==0 else 's',
        capsize=3,label=f"Bulk {mode}, {'before' if i==0 else 'after'}")
  ax.set_title(f"{'Stage 1: calibration' if name=='stage1' else 'Stage 1b: dense inputs'} — {cache}")
  ax.set_ylabel('Events / second');ax.set_xticks([1,2,3,4]);ax.grid(alpha=.2)
  if row_index==1: ax.set_xlabel('BD processes sharing one A100')
fig.suptitle('Jungfrau matched regression checks · 10,000 events · batch 20 · depth 1\nRanges show two repetitions; rows use separate nodes and workloads',fontsize=12)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,.005))
fig.tight_layout(rect=(0,.045,1,.92))
fig.savefig(output.with_suffix('.svg'))
fig.savefig(output.with_suffix('.png'),dpi=130)
