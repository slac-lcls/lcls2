"""Scheduled completion review; submit authorized scaling only after all gates pass."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def save(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temp.replace(path)


def state(job):
    text=subprocess.check_output(['sacct','-X','-n','-P','-j',job,
        '--format=JobID,State,ExitCode'],text=True)
    rows=[s.split('|') for s in text.splitlines() if s.split('|')[0]==job]
    assert len(rows)==1, ('missing scheduler record',job,text)
    return dict(job=job,state=rows[0][1],exit_code=rows[0][2])


def verify(root):
    for name,expected in json.loads((root/'hashes.json').read_text()).items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--job',required=True)
    p.add_argument('--kernel-root',type=Path,required=True)
    p.add_argument('--kernel-job',required=True)
    p.add_argument('--jf',type=Path,required=True)
    p.add_argument('--mixed',type=Path,required=True)
    a=p.parse_args();root=a.root
    verify(root)
    history=[];deadline=time.monotonic()+2.5*3600
    while True:
        current=state(a.job);current['checked_at']=datetime.datetime.now().astimezone().isoformat()
        history.append(current);save(root/'checks.json',history)
        print('CHECK '+json.dumps(current),flush=True)
        if current['state']=='COMPLETED':
            assert current['exit_code']=='0:0',current
            break
        assert current['state'] in ('RUNNING','PENDING','CONFIGURING','COMPLETING'),current
        assert time.monotonic()<deadline,'completion review deadline reached; scaling not submitted'
        time.sleep(60)
    kernel_state=state(a.kernel_job)
    assert kernel_state['state']=='COMPLETED' and kernel_state['exit_code']=='0:0',kernel_state
    verify(a.kernel_root)
    with (root/'review.json').open('w') as output:
        subprocess.run([sys.executable,str(root/'review.py'),'--root',str(a.campaign),'--job',a.job],
                       stdout=output,check=True)
    with (root/'plausibility.json').open('w') as output:
        subprocess.run([sys.executable,str(root/'plausibility.py'),'--review',str(root/'review.json'),
            '--kernel-log',str(a.kernel_root/('job-'+a.kernel_job+'.log'))],stdout=output,check=True)
    review=json.loads((root/'review.json').read_text())
    lines=['# Stage 5c completion review','',
        'All 16 diagnostics and 38 matched pairs passed; frozen sources and sample log hashes verified.',
        'Kernel times passed the independent real-input hot-buffer consistency checks. These checks do not require batching to be faster.',
        '', '| GPUs / BDs / batch / depth | Cache | Event-loop rate | Batched rate | Median paired loop change |',
        '| --- | --- | ---: | ---: | ---: |']
    for row in review['results']:
        variants=row['variants']
        lines.append('| '+' / '.join(map(str,row['point']))+' | '+row['cache']+
            ' | %.2f | %.2f | %+.2f%% |'%(variants['event_loop']['median_rate'],variants['batched_task']['median_rate'],row['median_paired_percent']))
    lines+=['','Rates are events/s from median loop duration; startup is recorded separately in review.json.',
        'Paired changes are medians of within-pair percentage changes, not ratios of separate medians.',
        'The kernel sanity check repeats the first real event with hot buffers and excludes I/O, allocation and D2H.',
        'It supports timer plausibility without identifying every cause of the batching benefit.','']
    (root/'report.md').write_text('\n'.join(lines))
    launches=json.loads((root/'launches.json').read_text()) if (root/'launches.json').exists() else {}
    for kind,prepared in (('jf',a.jf),('mixed',a.mixed)):
        if kind in launches:continue
        verify(prepared)
        response=subprocess.check_output(['sbatch','--parsable',str(prepared/'run.sbatch')],text=True).strip()
        job=response.split(';')[0];assert job.isdigit(),response
        launches[kind]=dict(job=job,root=str(prepared),submitted_at=datetime.datetime.now().astimezone().isoformat())
        save(root/'launches.json',launches)
        print('SCALING_SUBMITTED '+json.dumps(dict(kind=kind,**launches[kind])),flush=True)
    save(root/'decision.json',dict(passed=True,launches=launches,review=str(root/'review.json'),
        plausibility=str(root/'plausibility.json'),report=str(root/'report.md')))
    print('REVIEW_AND_LAUNCH_COMPLETE',flush=True)


if __name__=='__main__':main()
