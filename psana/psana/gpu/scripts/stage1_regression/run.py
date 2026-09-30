"""Matched staged-runtime comparisons with the established JF cache controls."""
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


def network_counters():
    return {f'{device.name}/{counter}': int((device/'statistics'/counter).read_text())
            for device in Path('/sys/class/net').iterdir() if device.name != 'lo'
            for counter in ('rx_bytes', 'tx_bytes')}


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


def case_variants(variants, bds, cache, control_bds=None):
    if control_bds is not None and (bds not in control_bds or cache != 'warm'):
        return tuple(v for v in variants if not v.startswith('control_'))
    return variants


def timed_cases(points, caches, modes, variants, repetitions, cold_bds=None,
                control_bds=None):
    """Keep matched versions adjacent and reverse every order on even rounds."""
    for rep in range(1, repetitions + 1):
        for g, b in (points if rep % 2 else tuple(reversed(points))):
            for cache in (caches if rep % 2 else tuple(reversed(caches))):
                if cache == 'cold' and cold_bds is not None and b not in cold_bds:
                    continue
                for mode in (modes if rep % 2 else tuple(reversed(modes))):
                    selected = case_variants(variants, b, cache, control_bds)
                    for variant in (selected if rep % 2 else tuple(reversed(selected))):
                        yield g, b, mode, cache, rep, variant


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--comparison', choices=('stage1', 'stage1b', 'stage2', 'stage2-control', 'stage3', 'stage4', 'stage4-public'), required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--modes', nargs='+', choices=('on', 'off'), default=['off', 'on'])
    p.add_argument('--bds', nargs='+', type=int, choices=(1, 2, 3, 4), default=[1, 2, 3, 4])
    p.add_argument('--caches', nargs='+', choices=('cold', 'warm'), default=['cold', 'warm'])
    p.add_argument('--cold-bds', nargs='+', type=int, choices=(1, 2, 3, 4),
                   help='restrict cold-cache samples to these selected BD counts')
    p.add_argument('--repetitions', type=int, default=2)
    a = p.parse_args()
    if a.repetitions < 1:
        p.error('--repetitions must be positive')
    for values in (a.bds, a.caches, a.modes):
        if len(values) != len(set(values)):
            p.error('matrix selections must not contain duplicates')
    if a.cold_bds is not None and ('cold' not in a.caches or
            len(a.cold_bds) != len(set(a.cold_bds)) or not set(a.cold_bds) <= set(a.bds)):
        p.error('--cold-bds requires cold cache and unique BD counts selected by --bds')
    points = tuple((1, b) for b in a.bds)
    variants = {'stage1': ('parent', 'stage1'), 'stage1b': ('stage1', 'stage1b'),
                'stage2': ('stage1b', 'stage2'),
                'stage2-control': ('control_a', 'control_b', 'stage1b', 'stage2'),
                'stage3': ('control_a', 'control_b', 'stage2', 'stage3'),
                'stage4': ('control_a', 'control_b', 'stage3c', 'stage4'),
                'stage4-public': ('event_loop', 'batched_task')}[a.comparison]
    if a.comparison == 'stage4-public' and (a.bds != [1] or a.caches != ['warm'] or a.repetitions % 2):
        p.error('stage4-public requires warm 1-BD coverage and balanced even repetitions')
    control_bds = (1,) if a.comparison in ('stage3', 'stage4') else None
    workload = 'calib' if a.comparison == 'stage1' else 'input'
    root = a.root.resolve()
    verify(root)
    commits = json.loads((root/'commits.json').read_text())
    if a.comparison in ('stage2-control', 'stage3', 'stage4'):
        # Label-only controls must load precisely the same frozen installation
        # as the A side of A/B. Resolve aliases before accepting this campaign.
        baseline = {'stage3':'stage2', 'stage4':'stage3c', 'stage2-control':'stage1b'}[a.comparison]
        labels = ('control_a', 'control_b', baseline)
        paths = [(root/'runtimes'/v/'python').resolve() for v in labels]
        if len(set(paths)) != 1 or len({commits[v] for v in labels}) != 1:
            raise ValueError('A/A controls must alias the same baseline runtime and commit')
        if a.repetitions % 2:
            p.error('controlled comparisons require an even repetition count for balanced order')
        if a.comparison in ('stage3', 'stage4') and (1 not in a.bds or 'warm' not in a.caches):
            p.error('controlled stage comparison requires warm 1-BD coverage for the A/A control')
    job = os.environ['SLURM_JOB_ID']
    output = root/('job-'+job)
    output.mkdir(exist_ok=False)
    cache_preflight(root, output/'cache-preflight.log')
    stage = Path('/lscratch/monarin')/('jf-stage1-regression-'+job)
    stage.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root/'reference.json').read_text())['10000']
    prefixes = manifest['prefixes']
    sizes = manifest['stage_bytes'] if not a.smoke else json.loads((root/'reference.json').read_text())['200']['prefixes']
    assert len(prefixes) == len(sizes) == 5
    assert shutil.disk_usage(stage).free > sum(sizes.values()) + 20*1024**3
    affinity = sorted(os.sched_getaffinity(0))
    env = dict(os.environ, BENCH_CPU_AFFINITY=','.join(map(str, affinity)))
    gpu_text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,name,memory.total',
                                        '--format=csv,noheader'], text=True)
    gpus = [[x.strip() for x in line.split(',')] for line in gpu_text.splitlines()]
    assert '0' in os.environ['SLURM_JOB_GPUS'].split(','), 'GPU 0 must belong to allocation'
    assert gpus[0][0] == '0'
    provenance = dict(job=job, host=os.uname().nodename, affinity=affinity, gpus=gpus,
        source=str(a.source), stage=str(stage), source_commits=commits,
        topology=subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True),
        filesystem=subprocess.check_output(['findmnt', '-T', str(stage)], text=True),
        block_devices=subprocess.check_output(['lsblk', '-o', 'NAME,MODEL,SIZE,TYPE,MOUNTPOINT'], text=True),
        settings=dict(events=10000, batch=20, depth=2 if a.comparison=='stage4-public' else 1, workers=8, task_mib=1,
                      bulk_target_mib=1, d2h=0, budget='automatic device_total/BD peers',
                      modes=a.modes, caches=a.caches, cold_bds=a.cold_bds, control_bds=control_bds,
                      repetitions=a.repetitions,
                      workload=workload, variants=variants, comparison=a.comparison,
                      consumer='compact output on_cpu' if a.comparison=='stage4-public' else 'timestamp only', smoke=a.smoke), points=points)
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

        def sample(g, b, bulk, cache, rep, variant, diagnostic=False):
            tag = f'{variant}-g{g}-bd{b}-{bulk}-{cache}-r{rep}' + ('-pixels' if diagnostic else '')
            before = None if diagnostic else cache_inputs(stage, cache, ranges=prefixes)
            runtime = root/'runtimes'/variant/'python'
            call_env = dict(env, SLURM_GPUS_ON_NODE=str(g), BENCH_PYTHON=str(runtime),
                            PYTHONPATH=str(runtime)+os.pathsep+env['PYTHONPATH'])
            bench_script = 'public_bench.py' if a.comparison=='stage4-public' else 'bench.py'
            cmd = ['mpirun', '-n', str(b+2), '--oversubscribe', '--bind-to', 'none',
                   sys.executable, '-u', str(root/'scripts/stage1_regression'/bench_script),
                   '--directory', str(stage), '--reference', str(root/'reference.json'),
                   '--pixels', str(root/'pixels.json'), '--constants', str(root/'constants.pkl.gz'),
                   '--bulk', bulk, '--events', '200' if diagnostic else '10000',
                   '--workload', workload, '--variant', variant]
            if diagnostic:
                cmd += ['--check-pixels']

            print('BEGIN '+tag, flush=True)
            network_before = network_counters()
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
            assert (row['workload'], row['variant']) == (workload, variant)
            for gpu, bus in row['gpu_buses'].items():
                assert bus.lower().split(':', 1)[-1] == gpus[int(gpu)][2].lower().split(':', 1)[-1]
            if a.comparison=='stage4-public':
                previous = [r for r in result_rows if r['diagnostic']==diagnostic]
                if previous:
                    assert row['public_result']['output_sha256']==previous[0]['public_result']['output_sha256']
            network_after = network_counters()
            row.update(cache=cache, repetition=rep, before=before,
                network_bytes={k: network_after[k]-v for k,v in network_before.items()},
                after=None if diagnostic else cache_inputs(stage, cache, prepare=False, ranges=prefixes),
                log_sha256=sha(output/(tag+'.log')))
            result_rows.append(row)
            save(output/'results.json', result_rows)
            print(f'PASS {tag}: {row["events_per_s"]:.2f} events/s, {row["pixel_samples"]} pixel checks', flush=True)

        for g, b in provenance['points']:
            for mode in a.modes:
                for variant in case_variants(variants, b, 'warm', control_bds):
                    sample(g, b, mode, 'warm', 0, variant, diagnostic=True)
        if not a.smoke:
            for case in timed_cases(points, a.caches, a.modes, variants,
                                    a.repetitions, a.cold_bds, control_bds):
                sample(*case)
        verify(root)
        summaries = []
        if not a.smoke:
            for _, b in points:
                for cache in a.caches:
                    if cache == 'cold' and a.cold_bds is not None and b not in a.cold_bds:
                        continue
                    for bulk in a.modes:
                        for variant in case_variants(variants, b, cache, control_bds):
                            rows = [r for r in result_rows if not r['diagnostic'] and
                                    (r['nbds'],r['cache'],r['bulk'],r['variant']) == (b,cache,bulk,variant)]
                            assert len(rows) == a.repetitions
                            elapsed = statistics.median(r['loop_s'] for r in rows)
                            summaries.append(dict(gpus=1, bds=b, cache=cache, bulk=bulk,
                                variant=variant, workload=workload, loop_s=elapsed,
                                events_per_s=10000/elapsed,
                                rates=[r['events_per_s'] for r in rows],
                                payload_gbps=manifest['payload_bytes']/elapsed/1e9))
        save(output/'summary.json', summaries)
        provenance['complete'] = True
        print('CAMPAIGN_COMPLETE', flush=True)
    finally:
        # Only remove this invocation's newly created node-local stage.
        assert stage == Path('/lscratch/monarin')/('jf-stage1-regression-'+job)
        shutil.rmtree(stage)
        provenance['removed_stage'] = str(stage)
        save(output/'provenance.json', provenance)


if __name__ == '__main__':
    main()
