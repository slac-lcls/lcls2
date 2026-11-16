"""Export completed campaign pairs and show individual timings in execution order."""
import argparse
import csv
import json
import os
from pathlib import Path
from summarize_campaign import summarize


def export(directory, output):
    report=summarize(directory)  # Requires all48 comparable samples and complete status.
    output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR','/tmp/pr299-matplotlib-config')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pairs=json.loads((directory/'paired-results.json').read_text())
    rows=[]
    grouped={}
    for pair in pairs:
        left=json.loads((directory/pair['left']/'sample.json').read_text())
        right=json.loads((directory/pair['right']/'sample.json').read_text())
        round_number=int(left['label'].rsplit('-pair',1)[1].split('-')[0])
        rows.append(dict(case=left['case'],round=round_number,order='base/head' if round_number%2==0 else 'head/base',base_loop_s=left['loop_seconds'],head_loop_s=right['loop_seconds'],head_throughput_change_percent=100*(right['events_per_second']/left['events_per_second']-1),base_mpi_wall_s=left['mpi_process_wall_seconds'],head_mpi_wall_s=right['mpi_process_wall_seconds']))
        grouped.setdefault(left['case'],{})[round_number]=(left,right)
    with (output/'paired-timings.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fig,axes=plt.subplots(3,2,figsize=(11,9),constrained_layout=True)
    for row_index,(mode,iterator,title) in enumerate([('gpu','events','GPU events'),('cpu','events','CPU events'),('cpu','steps','CPU nested steps')]):
        for col,bds in enumerate([1,2]):
            ax=axes[row_index,col];case=f'r51-{mode}-{iterator}-bd{bds}'
            ordered=[]
            for n in range(4):
                left,right=grouped[case][n]
                ordered.extend((left,right) if n%2==0 else (right,left))
            values=[sample['loop_seconds'] for sample in ordered]
            ax.plot(range(1,9),values,color='0.65',linewidth=1,zorder=1)
            for runtime,color,label in [('base','#2166ac','Baseline'),('head','#d95f02','PR head')]:
                positions=[i+1 for i,sample in enumerate(ordered) if sample['runtime']==runtime]
                ax.scatter(positions,[values[i-1] for i in positions],color=color,s=42,label=label,zorder=2)
            for boundary in [2.5,4.5,6.5]:ax.axvline(boundary,color='0.85',linestyle='--',linewidth=.8)
            ax.set_title(f'{title}, BD{bds}');ax.set_ylabel('Event-loop time (seconds)')
            ax.set_xticks(range(1,9),['B','H','H','B','B','H','H','B'])
            ax.set_xlabel('Execution order; dashed lines separate pairs')
            ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(loc='best',frameon=False)
    fig.suptitle('PR299 Run51: 10,000 events per sample\nUncontrolled cache; individual timings, not isolated PR effects',fontsize=13)
    fig.savefig(output/'event-loop-order.png',dpi=160)
    fig.savefig(output/'event-loop-order.pdf')
    plt.close(fig)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args();export(args.directory,args.output)
