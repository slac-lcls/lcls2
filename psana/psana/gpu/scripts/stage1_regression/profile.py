"""Short native traces separate from throughput; use already frozen runtimes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

from common import records, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.root, args.output
    output.mkdir(exist_ok=False)
    stage = Path('/lscratch/monarin') / ('jf-stage1-profile-'+os.environ['SLURM_JOB_ID'])
    stage.mkdir(parents=True, exist_ok=False)
    reference = json.loads((root/'reference.json').read_text())['200']
    env = dict(os.environ, SLURM_GPUS_ON_NODE='1',
               BENCH_CPU_AFFINITY=','.join(map(str, sorted(os.sched_getaffinity(0)))))
    assert '0' in os.environ['SLURM_JOB_GPUS'].split(',')
    results = []
    try:
        def stage_file(item):
            name, length = item
            with (args.source/name).open('rb') as src, (stage/name).open('xb') as dst:
                while length:
                    chunk = src.read(min(length, 16*1024**2))
                    assert chunk
                    dst.write(chunk)
                    length -= len(chunk)
            return dict(name=name, sha256=sha(stage/name))
        with ThreadPoolExecutor(max_workers=5) as pool:
            staged = list(pool.map(stage_file, reference['prefixes'].items()))
        (stage/'smalldata').mkdir()
        for name in reference['smd_hashes']:
            shutil.copy2(args.source/'smalldata'/name, stage/'smalldata'/name)
            assert sha(stage/'smalldata'/name) == reference['smd_hashes'][name]
        (output/'provenance.json').write_text(json.dumps(dict(
            job=os.environ['SLURM_JOB_ID'], host=os.uname().nodename,
            commits=json.loads((root/'commits.json').read_text()), staged=staged,
            nsys=subprocess.check_output([env['BENCH_NSYS'], '--version'], text=True),
            description='200-event bulk-on native traces; full capture; steady NVTX range begins after first delivered event per BD'), indent=2))
        for bds in (1, 4):
            for variant, workload in (('parent', 'calib'), ('stage1', 'calib'),
                                      ('stage1', 'input'), ('stage1b', 'input')):
                tag = f'{variant}-{workload}-bd{bds}'
                trace = output/tag
                trace.mkdir()
                runtime = root/'runtimes'/variant/'python'
                call_env = dict(env, BENCH_TRACE=str(trace), BENCH_PYTHON=str(runtime),
                                PYTHONPATH=str(runtime)+os.pathsep+env['PYTHONPATH'])
                cmd = ['mpirun', '-n', str(bds+2), '--oversubscribe', '--bind-to', 'none',
                       sys.executable, str(Path(__file__).with_name('profile_rank.py')),
                       '-u', str(Path(__file__).with_name('bench.py')),
                       '--directory', str(stage), '--reference', str(root/'reference.json'),
                       '--pixels', str(root/'pixels.json'), '--constants', str(root/'constants.pkl.gz'),
                       '--bulk', 'on', '--events', '200', '--workload', workload,
                       '--variant', variant, '--profile']
                print('PROFILE_BEGIN', tag, flush=True)
                with (trace/'sample.log').open('w') as log:
                    subprocess.run(cmd, env=call_env, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=600, check=True)
                row, = records((trace/'sample.log').read_text(), 'JF_SCALE_RESULT ')
                reports = sorted(trace.glob('rank-*.nsys-rep'))
                assert len(reports) == bds, (tag, reports)
                for report in reports:
                    with report.with_suffix('.stats.log').open('w') as log:
                        subprocess.run([env['BENCH_NSYS'], 'stats', '--report',
                            'cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum',
                            '--format', 'csv', '--output', str(report.with_suffix('')),
                            str(report)], stdout=log, stderr=subprocess.STDOUT, check=True)
                    with sqlite3.connect('file:'+str(report.with_suffix('.sqlite'))+'?mode=ro', uri=True) as db:
                        tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
                        if 'CUPTI_ACTIVITY_KIND_KERNEL' not in tables:
                            errors = list(db.execute('select text from DIAGNOSTIC_EVENT where severity >= 2'))
                            raise RuntimeError(f'No CUDA kernel trace in {report}: {errors}')
                        assert db.execute('select count(*) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0] > 0
                results.append(dict(tag=tag, result=row, reports=[str(r) for r in reports]))
                (output/'results.json').write_text(json.dumps(results, indent=2))
                print('PROFILE_PASS', tag, flush=True)
        (output/'complete.json').write_text(json.dumps(dict(complete=True, samples=len(results))))
    finally:
        shutil.rmtree(stage)


if __name__ == '__main__':
    main()
