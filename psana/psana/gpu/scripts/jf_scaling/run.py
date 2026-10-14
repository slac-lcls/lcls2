"""Stage private local input, validate scaling samples, and retain provenance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from common import cache_inputs, records, sha
from contract import KEY_POINTS, FULL_POINTS, FEESPEC_POINTS, matrix


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def verify(root):
    hashes = json.loads((root/'hashes.json').read_text())
    for name in ('common.py', 'warm_cache.py', 'memory_state.py'):
        helper = root/'scripts/feespec_bulk_benchmark'/name
        if str(helper) not in hashes:
            raise RuntimeError(f'Cache helper missing from frozen manifest: {helper}')
    for path, expected in hashes.items():
        assert sha(path) == expected, path


def cache_preflight(root, log_path):
    """Exercise the warm-cache subprocess before expensive staging or GPU work."""
    with log_path.open('w') as log:
        subprocess.run(['numactl', '--interleave=all', sys.executable,
                        str(root/'scripts/feespec_bulk_benchmark/warm_cache.py'),
                        '--prefixes', '[]'], stdout=log, stderr=subprocess.STDOUT,
                       check=True, timeout=60)
    print('CACHE_PREFLIGHT_PASS', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--full', action='store_true')
    p.add_argument('--include-feespec', action='store_true')
    p.add_argument('--modes', nargs='+', choices=('on', 'off'), default=['off', 'on'])
    a = p.parse_args()
    if a.include_feespec and a.full:
        p.error('--include-feespec uses the fixed one-GPU 1/2/4-BD matrix')
    root = a.root.resolve()
    verify(root)
    job = os.environ['SLURM_JOB_ID']
    output = root/('job-'+job)
    output.mkdir(exist_ok=False)
    cache_preflight(root, output/'cache-preflight.log')
    stage = Path('/lscratch/monarin')/('jf-current-scale-'+job)
    stage.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root/'reference.json').read_text())['10000']
    prefixes = manifest['prefixes']
    sizes = manifest['stage_bytes']
    assert len(prefixes) == len(sizes) == (6 if a.include_feespec else 5)
    assert shutil.disk_usage(stage).free > sum(sizes.values()) + 20*1024**3
    affinity = sorted(os.sched_getaffinity(0))
    env = dict(os.environ, BENCH_CPU_AFFINITY=','.join(map(str, affinity)))
    gpu_text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,name,memory.total',
                                        '--format=csv,noheader'], text=True)
    gpus = [[x.strip() for x in line.split(',')] for line in gpu_text.splitlines()]
    if a.include_feespec:
        assert '0' in os.environ['SLURM_JOB_GPUS'].split(','), 'GPU 0 must belong to this allocation'
        assert gpus[0][0] == '0', gpus
    else:
        assert [g[0] for g in gpus] == ['0', '1', '2', '3'], gpus
    provenance = dict(job=job, host=os.uname().nodename, affinity=affinity, gpus=gpus,
        source=str(a.source), stage=str(stage), source_commit=(root/'source-commit.txt').read_text().strip(),
        topology=subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True),
        filesystem=subprocess.check_output(['findmnt', '-T', str(stage)], text=True),
        block_devices=subprocess.check_output(['lsblk', '-o', 'NAME,MODEL,SIZE,TYPE,MOUNTPOINT'], text=True),
        settings=dict(events=10000, batch=20, depth=1, workers=8, task_mib=1,
                      bulk_target_mib=1, d2h=0, budget='automatic device_total/BD peers',
                      modes=a.modes, repetitions=2, include_feespec=a.include_feespec,
                      consumer='feespec GPU int64 sum' if a.include_feespec else 'timestamp only'),
        points=FEESPEC_POINTS if a.include_feespec else (FULL_POINTS if a.full else KEY_POINTS))
    save(output/'provenance.json', provenance)
    result_rows = []
    try:
        def copy_file(name):
            source, dest = a.source/name, stage/name
            before = source.stat()
            h = hashlib.sha256()
            remaining = sizes[name]
            with source.open('rb', buffering=0) as src, dest.open('xb', buffering=0) as dst:
                while remaining:
                    block = src.read(min(16*1024**2, remaining))
                    if not block:
                        raise RuntimeError('short stage source: '+str(source))
                    written = dst.write(block)
                    assert written == len(block)
                    h.update(block)
                    remaining -= len(block)
                os.fsync(dst.fileno())
            after = source.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            assert dest.stat().st_size == sizes[name]
            print('STAGED '+name, flush=True)
            return dict(name=name, bytes=sizes[name], sha256=h.hexdigest(),
                        source_size=before.st_size, source_mtime_ns=before.st_mtime_ns)
        with ThreadPoolExecutor(max_workers=5) as pool:
            provenance['staged'] = list(pool.map(copy_file, sizes))
        (stage/'smalldata').mkdir()
        for name in sizes:
            smd = name.replace('.xtc2', '.smd.xtc2')
            shutil.copy2(a.source/'smalldata'/smd, stage/'smalldata'/smd)
            assert sha(stage/'smalldata'/smd) == manifest['smd_hashes'][smd]
        save(output/'provenance.json', provenance)

        def sample(g, b, bulk, cache, rep, diagnostic=False):
            tag = f'g{g}-bd{b}-{bulk}-{cache}-r{rep}' + ('-pixels' if diagnostic else '')
            before = None if diagnostic else cache_inputs(stage, cache, ranges=prefixes)
            call_env = dict(env, SLURM_GPUS_ON_NODE=str(g))
            cmd = ['mpirun', '-n', str(b+2), '--oversubscribe', '--bind-to', 'none',
                   sys.executable, '-u', str(root/'scripts/jf_scaling/bench.py'),
                   '--directory', str(stage), '--reference', str(root/'reference.json'),
                   '--pixels', str(root/'pixels.json'), '--constants', str(root/'constants.pkl.gz'),
                   '--bulk', bulk, '--events', '200' if diagnostic else '10000']
            if diagnostic:
                cmd += ['--check-pixels']
            if a.include_feespec:
                cmd += ['--include-feespec']
            print('BEGIN '+tag, flush=True)
            with (output/(tag+'-gpu.csv')).open('w') as monitor_log:
                monitor = subprocess.Popen(['nvidia-smi',
                    '--query-gpu=timestamp,index,memory.used,utilization.gpu,utilization.memory',
                    '--format=csv,noheader,nounits', '-lms', '250'], stdout=monitor_log)
                try:
                    with (output/(tag+'.log')).open('w') as log:
                        status = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                                env=call_env, timeout=900)
                    assert status.returncode == 0, f'{tag}: returncode {status.returncode}'
                finally:
                    monitor.terminate()
                    monitor.wait(timeout=10)
            rows = records((output/(tag+'.log')).read_text(), 'JF_SCALE_RESULT ')
            assert len(rows) == 1
            row = rows[0]
            assert (row['ngpus'], row['nbds'], row['bulk']) == (g, b, bulk)
            assert row['include_feespec'] == a.include_feespec
            for gpu, bus in row['gpu_buses'].items():
                assert bus.lower().split(':', 1)[-1] == gpus[int(gpu)][2].lower().split(':', 1)[-1]
            row.update(cache=cache, repetition=rep, before=before,
                after=None if diagnostic else cache_inputs(stage, cache, prepare=False, ranges=prefixes),
                log_sha256=sha(output/(tag+'.log')))
            result_rows.append(row)
            save(output/'results.json', result_rows)
            print(f'PASS {tag}: {row["events_per_s"]:.2f} events/s, {row["pixel_samples"]} pixel checks', flush=True)

        for g, b in provenance['points']:
            for mode in a.modes:
                sample(g, b, mode, 'warm', 0, diagnostic=True)
        schedule = matrix(tuple(map(tuple, provenance['points'])), tuple(a.modes))
        for case in schedule:
            sample(*case)
        verify(root)
        summaries = []
        for g, b in provenance['points']:
            for cache in ('cold', 'warm'):
                for bulk in a.modes:
                    rows = [r for r in result_rows if not r['diagnostic'] and
                            (r['ngpus'],r['nbds'],r['cache'],r['bulk']) == (g,b,cache,bulk)]
                    assert len(rows) == 2
                    elapsed = statistics.median(r['loop_s'] for r in rows)
                    summaries.append(dict(gpus=g, bds=b, cache=cache, bulk=bulk,
                        loop_s=elapsed, events_per_s=10000/elapsed,
                        payload_gbps=manifest['payload_bytes']/elapsed/1e9))
        save(output/'summary.json', summaries)
        provenance['complete'] = True
        print('CAMPAIGN_COMPLETE', flush=True)
    finally:
        # Only remove this invocation's newly created node-local stage.
        assert stage == Path('/lscratch/monarin')/('jf-current-scale-'+job)
        shutil.rmtree(stage)
        provenance['removed_stage'] = str(stage)
        save(output/'provenance.json', provenance)


if __name__ == '__main__':
    main()
