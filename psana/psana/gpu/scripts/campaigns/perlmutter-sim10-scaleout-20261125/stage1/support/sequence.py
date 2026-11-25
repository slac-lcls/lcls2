"""One-shot successor launches after saved audits and terminal Slurm success."""
from pathlib import Path
import datetime
import fcntl
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time

ROOT = Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-baseline-20261115')
EXECUTION = ROOT/'grouped-replay-execution-20261124'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)


def audit(attempt, repeat, exit_record, console, accounting):
    require(exit_record['returncode'] == 0, 'Allocation command failed')
    require(exit_record['repeat'] == repeat and exit_record['attempt'] == str(attempt), 'Launch identity')
    granted = re.findall(r'Granted job allocation (\d+)', console)
    require(len(granted) == 1, 'Exactly one granted allocation')
    job = granted[0]
    rows = [line.split('|') for line in accounting.splitlines() if line.strip()]
    require(any(row[0] == job for row in rows), 'Scheduler allocation record required')
    require(all(len(row) >= 4 and (row[0] == job or row[0].startswith(job+'.')) and row[1:3] == ['COMPLETED', '0:0'] for row in rows), 'All scheduler records COMPLETED/0:0')
    plan = read(attempt/'plan.json')
    manifest = read(attempt/'bundle/bundle.json')
    require(digest(attempt/'bundle/bundle.json') == plan['bundle_sha256'], 'Bundle manifest pin')
    require(all(digest(attempt/'bundle'/name) == sha for name, sha in manifest['files'].items()), 'Bundle file pins')
    require(all(digest(Path(name)) == sha for name, sha in manifest['external'].items()), 'External pins')
    state = read(attempt/'results/complete.json')
    require(state == read(attempt/'results/status.json'), 'Final state matches completion')
    require(state['status'] == 'passed_pending_slurm_and_independent_result_review', 'Campaign completion status')
    require(state['job'] == job and state['source_commit'] == plan['source_commit'], 'Completion provenance')
    require(len(state['completed']) == 13 and [r['case'] for r in state['completed']] == plan['cases'], 'All 13 planned cases')
    nodes = set()
    for entry in state['completed']:
        name = entry['case']['name']
        path = attempt/'results'/(name+'.json')
        require(digest(path) == entry['result_sha256'], 'Result hash: '+name)
        verdict = read(attempt/'results'/(name+'.audit.json'))
        require(verdict == entry['audit'] and verdict['status'] == 'pass', 'Case audit: '+name)
        report = read(path)
        require(report['job'] == job, 'Case job: '+name)
        nodes.update(r['host'] for r in report['rows'])
    summary_path = attempt/'results/comparison.json'
    require(digest(summary_path) == state['summary_sha256'], 'Summary hash')
    summary = read(summary_path)
    require(summary['status'] == 'pass_pending_scheduler_and_independent_result_review', 'Summary status')
    require(summary['job'] == job and summary['repeat'] == repeat and nodes == {summary['node']}, 'Same job/node/repeat')
    require(all(row[3] == summary['node'] for row in rows), 'Scheduler node matches')
    require(len(summary['staging_drift']) == 2, 'Two staging drift checks')
    # Summary stores elapsed-time ratios; the plan screens RATE after/before.
    require(all(abs(1.0/r['after_over_before']-1.0) <= 0.05 for r in summary['staging_drift']), 'Staging rate drift exceeds 5%')
    return dict(status='pass_for_successor',job=job,node=summary['node'],repeat=repeat,cases=13,
                limitations=['Final independent performance review and cross-group reproduction screen remain pending.'])


def main():
    require(Path(__file__).resolve().parent == EXECUTION, 'Run deployed reviewed controller only')
    lock=(EXECUTION/'sequence.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save(EXECUTION/'sequence-start.json',dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname(),pid=os.getpid()))
    try:
        for repeat in (1,2,3):
            attempt=ROOT/('grouped-replay-20261123-r%d'%repeat)
            if repeat > 1:
                balance_text=subprocess.check_output(['iris','user',os.environ['USER'],'--json'],text=True,timeout=30)
                (EXECUTION/('iris-before-r%d.json'%repeat)).write_text(balance_text)
                balance=next(r for r in json.loads(balance_text) if r['Project']=='lcls_g')
                percentages={k:100*(balance['Allocated'+k]-balance['Charged'+k])/balance['Allocated'+k] for k in ('(user)','')}
                print('ALLOCATION BALANCE',percentages,flush=True)
                require(min(percentages.values()) >= 20, 'Balance below 20%; stop to report alert before next submission')
                # Cooperative dataset lock is also acquired by the on-node campaign.
                with (ROOT/'count2-use.lock').open('a') as dataset_lock:
                    fcntl.flock(dataset_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                subprocess.run([sys.executable,str(EXECUTION/'launch-one.py'),str(repeat)],check=True,timeout=9780)
            launch=read(EXECUTION/('launch-r%d.json'%repeat))
            deadline=datetime.datetime.fromisoformat(launch['started_utc']).timestamp()+9780
            exit_path=EXECUTION/('exit-r%d.json'%repeat)
            while not exit_path.exists():
                require(time.time() < deadline, 'Waiting for allocation exceeded bounded deadline')
                time.sleep(15)
            exit_record=read(exit_path)
            require(exit_record['returncode']==0, 'Allocation failed; successors stopped')
            console=(EXECUTION/('salloc-r%d.log'%repeat)).read_text()
            granted=re.findall(r'Granted job allocation (\d+)',console)
            require(len(granted)==1, 'Exactly one granted job')
            job=granted[0]
            accounting=''
            for poll in range(9):
                accounting=subprocess.check_output(['sacct','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode,NodeList'],text=True,timeout=30)
                rows=[line.split('|') for line in accounting.splitlines() if line.strip()]
                if rows and all(row[1] not in ('RUNNING','PENDING','COMPLETING') for row in rows):
                    break
                if poll < 8:time.sleep(15)
            (EXECUTION/('sacct-r%d.txt'%repeat)).write_text(accounting)
            verdict=audit(attempt,repeat,exit_record,console,accounting)
            save(EXECUTION/('successor-gate-r%d.json'%repeat),verdict)
            print(json.dumps(verdict),flush=True)
        save(EXECUTION/'sequence-complete.json',dict(status='all_three_completed_pending_final_result_review'))
    except BaseException as error:
        save(EXECUTION/'sequence-failed.json',dict(error=repr(error),repeat=repeat,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        raise


if __name__ == '__main__':
    main()
