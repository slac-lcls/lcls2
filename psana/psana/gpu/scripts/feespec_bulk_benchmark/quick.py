"""Short cold E-off/on baseline, reusing private FFB data and frozen E build."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from common import records, sha, tier
from run import cpu_reference, execute, verify

HERE = Path(__file__).resolve().parent


def matrix():
    return [('E-off', 1), ('E-on', 1), ('E-on', 2), ('E-off', 2)]


def input_manifest(directory, events):
    from stage import inspect_smd
    streams = [inspect_smd(directory/f'mfx101210926-r0387-s{s:03d}-c000.xtc2', events)
               for s in (0, 5, 6, 7, 8, 9)]
    assert len({r['timestamp_sha256'] for r in streams}) == 1
    return dict(events=events, streams=streams,
                payload_bytes=sum(r['payload_bytes'] for r in streams),
                timestamp_sha256=streams[0]['timestamp_sha256'])


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def summarize(root, results, provenance):
    assert [(r['variant'], r['repetition']) for r in results] == matrix()
    for r in results:
        assert r['events'] == provenance['events'] and not r['diagnostic']
        assert r['cache'] == 'cold' and r['cache_before']['measured_prefixes']
        assert len(r['cache_before']['files']) == 6
        assert all(f['resident_fraction'] <= .01 for f in r['cache_before']['files'])
        assert r['physical_rx_bytes'] >= r['payload_bytes'] * .98
        assert r['timestamp_sha256'] == provenance['reference']['timestamp_sha256']
        assert r['sums_sha256'] == provenance['reference']['sums_sha256']
        assert all(x['affinity'] == provenance['affinity'] for x in r['ranks'])
    lines = ['# Short JF + feespec cold baseline', '',
             f"Job {provenance['job_id']} on {provenance['node']}; {provenance['events']} events, batch 100, depth 1, 8 GiB.", '',
             '| Variant | R1 / R2 events/s | Rate from median time | Median loop s | Request-to-ready R1 / R2 s | Requests R1 / R2 |',
             '|---|---:|---:|---:|---:|---:|']
    groups = {}
    for variant in ('E-off', 'E-on'):
        rows = sorted([r for r in results if r['variant'] == variant], key=lambda r:r['repetition'])
        times = [r['loop_s'] for r in rows]
        rate = provenance['events']/statistics.median(times)
        counters = [r['ranks'][2]['counts'] for r in rows]
        groups[variant] = dict(seconds=times, events_per_s=rate,
                               request_to_ready_s=[c['request_to_ready_s'] for c in counters])
        lines.append(f"| {variant} | {rows[0]['events_per_s']:.2f} / {rows[1]['events_per_s']:.2f} | {rate:.2f} | {statistics.median(times):.4f} | {counters[0]['request_to_ready_s']:.4f} / {counters[1]['request_to_ready_s']:.4f} | {counters[0]['requests']} / {counters[1]['requests']} |")
    ratios = [groups['E-on']['seconds'][i]/groups['E-off']['seconds'][i] for i in (0, 1)]
    lines += ['', f'Paired on/off elapsed-time ratios: {ratios[0]:.3f}, {ratios[1]:.3f}.',
              'Cold means evicted node page cache on SSD-backed Weka FFB; server caches are not flushed.',
              'Each sample validates 200 warmup feespec arrays and three JF raw/calibrated samples before eviction.',
              'Measured loops verify timestamps and GPU sum hashes, with no full-array D2H.',
              'A lightweight wait wrapper accumulates existing reader counters across batch resets; no native-operation trace.',
              'Request-to-ready sums may overlap other pipeline work and must not be added to loop time.',
              'NIC counters include filesystem overhead and background traffic. Setup and teardown are excluded from loop rates.',
              'No A or warm timing cases. This tests short-run reproducibility; the 10,000-event run remains the acceptance baseline.', '']
    (root/'summary.md').write_text('\n'.join(lines))
    save(root/'summary.json', dict(job_id=provenance['job_id'], audit_pass=True, groups=groups,
                                  paired_on_off_time_ratios=ratios))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', required=True)
    p.add_argument('--constants', required=True)
    p.add_argument('--jf-cpu-log', required=True)
    p.add_argument('--events', type=int, default=1000)
    p.add_argument('--fallback-trace', action='store_true')
    a = p.parse_args()
    assert a.events >= 200 and a.events % 100 == 0
    directory = Path(a.directory).resolve()
    assert Path('/sdf/data/lcls/drpsrcf/ffb/users/monarin') in directory.parents
    original = json.loads((directory/'manifest.json').read_text())
    assert original['exp'] == 'mfx101210926' and original['run'] == 387 and original['events'] >= a.events
    for row in original['streams']:
        assert (directory/row['name']).stat().st_size == row['stage_bytes']
    builds = {'E':json.loads((HERE/'builds.json').read_text())['E']}
    verify(builds)
    job = os.environ['SLURM_JOB_ID']
    root = HERE/f'job-{job}'
    root.mkdir()
    affinity = sorted(os.sched_getaffinity(0))
    os.environ['BENCH_CPU_AFFINITY'] = ','.join(map(str, affinity))
    import cupy as cp
    bus = cp.cuda.runtime.deviceGetPCIBusId(0)
    bus = bus.decode() if isinstance(bus, bytes) else bus
    uuid = subprocess.check_output(['nvidia-smi', '-i', bus, '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    files = sorted(directory.glob('*.xtc2')) + sorted((directory/'smalldata').glob('*.xtc2'))
    provenance = dict(job_id=job, node=os.uname().nodename, gpu_uuid=uuid, affinity=affinity,
                      directory=str(directory), events=a.events, builds=builds,
                      scripts={f.name:sha(f) for f in HERE.glob('*.py')},
                      calibration={f:sha(f) for f in (a.constants, a.jf_cpu_log)},
                      tier_before=tier(files))
    save(root/'provenance.json', provenance)
    for n, label in ((a.events, 'measured'), (200, 'warmup')):
        manifest = input_manifest(directory, n)
        reference = cpu_reference(directory, events=n, manifest=manifest)
        save(root/(label+'-manifest.json'), manifest)
        save(root/(label+'-reference.json'), reference)
        if label == 'measured':provenance['reference'] = reference
    jf = records(Path(a.jf_cpu_log).read_text(), 'CPU_CHECK ')[0]
    assert len(jf) == 3
    save(root/'jf-reference.json', jf)
    save(root/'provenance.json', provenance)
    results = []
    cases = [(v, r, mode) for v, r in matrix()
             for mode in (('control', 'trace') if r == 1 else ('trace', 'control'))] if a.fallback_trace else [(v, r, 'control') for v, r in matrix()]
    for variant, repetition, mode in cases:
        tag = f'{variant}-cold-r{repetition}' + ('-'+mode if a.fallback_trace else '')
        env = os.environ.copy()
        prefix = Path(builds['E']['prefix'])
        env['PYTHONPATH'] = str(prefix/'lib/python3.9/site-packages')
        env['LD_LIBRARY_PATH'] = str(prefix/'lib')+':'+env.get('LD_LIBRARY_PATH', '')
        command = ['mpirun', '-n', '3', '--oversubscribe', '--bind-to', 'none', sys.executable, '-u', str(HERE/'bench.py'),
                   '--directory', str(directory), '--reference', str(root/'measured-reference.json'),
                   '--variant', variant, '--cache', 'cold', '--include-jf', '--constants', a.constants,
                   '--jf-reference', str(root/'jf-reference.json'), '--events', str(a.events),
                   '--warmup-events', '200', '--warmup-reference', str(root/'warmup-reference.json'),
                   '--range-manifest', str(root/'measured-manifest.json'), '--read-stats']
        if mode == 'trace':
            env['LD_PRELOAD'] = str(HERE/'fallback.so')
            command += ['--fallback-library', str(HERE/'fallback.so'),
                        '--fallback-output', str(root/(tag+'.bin'))]
        else:
            assert not env.get('LD_PRELOAD'), 'control must not preload tracer'
        with (root/(tag+'-gpu.csv')).open('x') as output:
            monitor = subprocess.Popen(['nvidia-smi', '-i', uuid, '--query-gpu=timestamp,uuid,memory.used,utilization.gpu,clocks.sm',
                                        '--format=csv,noheader,nounits', '-lms', '250'], stdout=output, stderr=subprocess.STDOUT)
            try:output_text = execute(command, env, root/(tag+'.log'), timeout=1200)
            finally:
                monitor.terminate()
                monitor.wait(timeout=15)
        warmup = records(output_text, 'WARMUP_CHECK ')
        assert len(warmup) == 1 and warmup[0]['events'] == 200 and warmup[0]['jf_samples'] == 3
        runtime = records(output_text, 'RUNTIME ')
        assert len(runtime) == 1 and runtime[0]['psana'].startswith(str(prefix)+'/')
        rows = records(output_text, 'RESULT ')
        assert len(rows) == 1
        r = rows[0]
        r.update(job_id=job, gpu_uuid=uuid, repetition=repetition, warmup_check=warmup[0], trace_mode=mode)
        if mode == 'trace':
            from summarize_kvikio_fallback import audit_trace
            audit = audit_trace(r['ranks'][2]['fallback_trace']['metadata'])
            assert audit['bytes'] == r['payload_bytes'] and audit['pool_workers'] == 8
            assert audit['file_concurrency']['files'] == 6
            assert audit['api_requests'] == r['ranks'][2]['counts']['requests']
            r['fallback_audit'] = audit
        results.append(r)
        save(root/'results.json', results)
        print('ACCEPTED', tag, round(r['events_per_s'], 2), flush=True)
    provenance['tier_after'] = tier(files)
    verify(builds)
    for name, digest in provenance['calibration'].items():assert sha(name) == digest
    for name, digest in provenance['scripts'].items():assert sha(HERE/name) == digest
    save(root/'provenance.json', provenance)
    if a.fallback_trace:
        from trace_summary import summarize_trace
        summarize_trace(root, results, provenance)
    else:
        summarize(root, results, provenance)
    print('COMPLETE '+str(root/'summary.md'), flush=True)


if __name__ == '__main__':
    main()
