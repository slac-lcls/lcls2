"""Render descriptive screening results from the accepted audit receipt."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
cells = json.loads((root / 'sweep/audit.json').read_text())['cells']
fig, ax = plt.subplots(figsize=(8, 4.8), layout='constrained')
for threads in [5, 20, 40]:
    rows = [r for r in cells if r['threads'] == threads]
    med = [r['median_gib_s'] for r in rows]
    ax.errorbar([r['count'] for r in rows], med,
                yerr=[[r['median_gib_s'] - r['min_gib_s'] for r in rows],
                      [r['max_gib_s'] - r['median_gib_s'] for r in rows]],
                fmt='o-', capsize=4, label=f'{threads} threads')
ax.set_xscale('log', base=2)
ax.set_xticks([1, 2, 4, 8, 16, 32], labels=['1', '2', '4', '8', '16', '32'])
ax.set(xlabel='Stripe count per file (1 MiB stripe size)',
       ylabel='Aggregate read throughput (GiB/s)', ylim=(0, 14),
       title='sim10 standalone reads: screening results')
ax.grid(alpha=.25)
ax.legend(title='Median and min–max, 3 trials')
fig.text(.5, -.025, 'One placement per count; local page cache cold; backend caches uncontrolled.',
         ha='center', fontsize=9)
for ext in ['png', 'pdf']:
    fig.savefig(root / ('screening-throughput.' + ext), dpi=160, bbox_inches='tight')
