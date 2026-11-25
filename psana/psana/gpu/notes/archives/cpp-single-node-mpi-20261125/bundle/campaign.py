"""One node, forty fixed workers, disjoint POSIX reads over 1/2/4/8 MPI ranks."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
import baseline as base
import cxi

HERE = Path(__file__).resolve().parent
PYTHON = base.PYTHON
REQUEST = 1048576
require, save = base.require, base.save


def cpus_for_rank(cpus, rank, ranks, total=40):
    require(total % (5*ranks) == 0 and len(cpus) >= total, 'invalid CPU partition')
    require(0 <= rank < ranks, 'invalid rank')
    per_file = total // (5*ranks)
    # Preserve CPU-to-file-and-offset assignment as process boundaries change.
    return [cpus[f*(total//5)+rank*per_file+j]
            for f in range(5) for j in range(per_file)]


def schedule():
    cases = []
    rng = random.Random(20261125)
    for dataset in ['probe40', 'sim400k-c08']:
        for repeat in range(3):
            ranks = [1, 2, 4, 8]
            rng.shuffle(ranks)
            for n in ranks:
                cases.append({'id': '%02d-%s-r%d-p%d' % (len(cases), dataset, repeat, n),
                              'dataset': dataset, 'repeat': repeat, 'ranks': n,
                              'total_workers': 40, 'threads': 40//n})
    return cases


def node(path):
    require(socket.gethostname().startswith('nid'), 'compute node required')
    base.verify_bundle()
    path = Path(path)
    case = json.loads((path/'case.json').read_text())
    rank = int(os.environ['SLURM_PROCID'])
    require(int(os.environ['SLURM_LOCALID']) == rank, 'ranks must share one node')
    cpus = cpus_for_rank(base.physical_cpus(), rank, case['ranks'])
    command = [str(HERE/('reader-check' if case.get('fixture') else 'reader')), str(case['threads']), str(REQUEST),
               ','.join(map(str, cpus)), '0', *[r['path'] for r in case['inputs']]]
    env = dict(os.environ, MPICH_GPU_SUPPORT_ENABLED='0', MPICH_OFI_CXI_COUNTER_REPORT='0')
    env.pop('CXI_HANDSHAKE', None)
    stdout, stderr = path/('rank-%d.stdout'%rank), path/('rank-%d.stderr'%rank)
    if rank == 0:
        result, counters = cxi.run_reader(command, stdout, stderr, env=env, timeout=600)
    else:
        with stderr.open('x') as f:
            p = subprocess.run(command, env=env, stdout=subprocess.PIPE,
                               stderr=f, text=True, close_fds=False, timeout=600)
        stdout.write_text(p.stdout)
        require(p.returncode == 0, 'reader failed; inspect rank logs')
        result, counters = json.loads(p.stdout), None
    save(path/('rank-%d.json'%rank), {'host':socket.gethostname(), 'rank':rank,
         'cpus':cpus, 'command':command, 'reader':result, 'cxi':counters})


def audit(case, records, testing=False):
    n, rows = case['ranks'], case['inputs']
    require(len(records) == n and len({r['host'] for r in records}) == 1, 'one node required')
    require(len({cpu for rec in records for cpu in rec['cpus']}) == 40,
            'forty distinct worker cores required')
    spans = [[] for _ in rows]
    durations, total = [], 0
    for rank, rec in enumerate(records):
        r = rec['reader']
        require(rec['rank'] == r['rank'] == rank and r['ranks'] == n, 'rank mismatch')
        require(r['state'] == 'complete' and r['threads'] == case['threads']
                and r['request_bytes'] == REQUEST, 'reader controls')
        require(r['confirmed_zero_scans'] == 2 and 1 <= r['eviction_attempts'] <= 3, 'cache gates')
        require(len(r['files']) == 5 and len(r['workers']) == case['threads'], 'file/worker count')
        require(r['cxi_handshake'] == (rank == 0), 'only rank zero samples the node')
        for got, exp in zip(r['files'], rows):
            require(got['path'] == exp['path'] and got['identity'] == exp['identity'], 'input identity')
            require(got['cold_resident_pages'] == 0 and not r['warm_probe'], 'cold cache required')
        subtotal = 0
        per_file = case['threads']//5
        for i, w in enumerate(r['workers']):
            f, part = divmod(i, per_file)
            size = rows[f]['bytes']
            chunks = (size+REQUEST-1)//REQUEST
            index, parts = rank*per_file+part, n*per_file
            begin = min(size, chunks*index//parts*REQUEST)
            end = min(size, chunks*(index+1)//parts*REQUEST)
            require((w['file'], w['begin'], w['end'], w['bytes']) == (f,begin,end,end-begin), 'range partition')
            require(w['cpu'] == w['actual_start_cpu'] == w['actual_end_cpu'] == rec['cpus'][i], 'affinity')
            require(0 <= w['start_seconds'] <= w['end_seconds'] <= r['seconds'], 'worker time')
            require(testing or w['test_checksum'] == 0, 'wrong binary')
            if case.get('fixture'):
                require(w['test_checksum'] == (end-begin)*(f+1), 'fixture checksum')
            spans[f].append((begin,end)); subtotal += end-begin
        require(subtotal == r['bytes'], 'rank byte sum')
        require(math.isfinite(r['seconds']) and r['seconds'] > 0
                and r['seconds'] == max(w['end_seconds'] for w in r['workers']), 'read timing')
        total += subtotal; durations.append(r['seconds'])
        require(rank == 0 or rec['cxi'] is None, 'duplicate node counters')
    for row, ranges in zip(rows, spans):
        previous = 0
        for begin, end in sorted(ranges):
            require(begin == previous and end >= begin, 'overlap or hole')
            previous = end
        require(previous == row['bytes'], 'incomplete file coverage')
    elapsed = max(durations)
    require(total == sum(r['bytes'] for r in rows), 'global bytes')
    for rec in records:
        require(rec['reader']['global_bytes'] == total and rec['reader']['global_seconds'] == elapsed, 'MPI reductions')
    counters = cxi.audit(records[0]['cxi'], dict(records[0]['reader'], bytes=total, seconds=elapsed))
    return {'audit':'pass', 'bytes':total, 'seconds':elapsed,
            'gib_s':total/elapsed/2**30, 'rank_seconds':durations, 'cxi':counters}


def execute(case, out):
    path = out/case['id']; path.mkdir()
    save(path/'case.json', case)
    command = ['srun', '--mpi=cray_shasta', '--exact', '--exclusive', '--nodes=1',
               '--ntasks='+str(case['ranks']), '--ntasks-per-node='+str(case['ranks']),
               '--cpus-per-task='+str(128//case['ranks']), '--cpu-bind=none',
               '--gpus-per-node=4', '--gpu-bind=none', '--kill-on-bad-exit=1',
               PYTHON, str(HERE/'campaign.py'), 'node', str(path)]
    save(path/'command.json', command)
    with (path/'srun.log').open('x') as f:
        subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=700)
    records = [json.loads((path/('rank-%d.json'%i)).read_text()) for i in range(case['ranks'])]
    summary = {k:v for k,v in case.items() if k != 'inputs'}
    summary.update(audit(case, records, testing=case.get('fixture',False)))
    save(path/'accepted.json',summary)
    return summary


def run(attempt):
    base.verify_bundle()
    require(os.environ.get('SLURM_JOB_ID') and int(os.environ['SLURM_JOB_NUM_NODES']) == 1, 'one-node allocation required')
    plan = json.loads((HERE/'plan.json').read_text()); base.validate_inputs(plan)
    out = Path(plan['output_root'])/attempt; out.mkdir(parents=True, exist_ok=False)
    save(out/'plan.json',plan); save(out/'pins.json',json.loads((HERE/'pins.json').read_text()))
    state = {'state':'running','job':os.environ['SLURM_JOB_ID'],'started_unix':time.time(),
             'completed':[], 'case_count':24, 'scope':'one node; fixed 40 workers; one CXI sampler; buffered POSIX pread, MPI synchronization only'}
    save(out/'status.json',state)
    try:
        fixture = out/'fixtures'; fixture.mkdir()
        rows = []
        for i in range(5):
            p = fixture/('file%d'%i)
            with p.open('xb') as f:
                f.write(bytes([i+1])*(REQUEST*(i+1)+17)); f.flush(); os.fsync(f.fileno())
            rows.append({'path':str(p),'bytes':p.stat().st_size,'identity':base.identity(p)})
        # Same MPI binary with checksum instrumentation for tiny validation only.
        for n in [1,2,4,8]:
            execute({'id':'smoke-p%d'%n,'ranks':n,'threads':40//n,'total_workers':40,
                     'inputs':rows,'fixture':True},out)
        state['mpi_smoke'] = 'passed at 1/2/4/8 ranks, exact range/affinity/byte/checksum/node-counter audits'
        save(out/'schedule.json',schedule())
        for case in schedule():
            require(time.time()-state['started_unix'] < 80*60, 'time budget reached')
            case['inputs'] = plan['inputs'][case['dataset']]
            state['current_case'] = case['id']; save(out/'status.json',state)
            state['completed'].append(execute(case,out)); save(out/'status.json',state)
        base.validate_inputs(plan)
        state.update(state='complete',finished_unix=time.time())
        save(out/'status.json',state); save(out/'complete.json',state)
    except BaseException as exc:
        state.update(state='failed',error=repr(exc));save(out/'status.json',state)
        raise


if __name__ == '__main__':
    if sys.argv[1] == 'node':
        node(sys.argv[2])
    elif sys.argv[1] == 'run':
        p = argparse.ArgumentParser();p.add_argument('--attempt',default='attempt01')
        args=p.parse_args(sys.argv[2:]);run(args.attempt)
    else:
        raise ValueError('unknown mode')
