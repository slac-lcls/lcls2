"""Two-round previous/current E comparison on one allocation and bounded inputs."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from common import records, sha, tier
from quick import input_manifest, save
from run import cpu_reference, execute, verify

HERE = Path(__file__).resolve().parent


def matrix(study='versions'):
    if study == 'current':
        from current_summary import matrix as current_matrix
        return current_matrix()
    if study in ('slots', 'slots-warm', 'files', 'files-warm', 'groups', 'groups-warm', 'groups-cold'):
        from slots_summary import matrix as slots_matrix
        return slots_matrix(warm_only=study.endswith('-warm'), cold_only=study.endswith('-cold'))
    if study == 'profile':
        from profile_summary import matrix as profile_matrix
        return profile_matrix()
    if study == 'size4':
        from size_summary import matrix as size_matrix
        return size_matrix()
    if study == 'depth':
        from depth_summary import matrix as depth_matrix
        return depth_matrix()
    variants = [('previous', 'E-off'), ('previous', 'E-on'),
                ('current', 'E-off'), ('current', 'E-on')]
    return [(build, variant, cache, repetition, 'control')
            for repetition in (1, 2)
            for cache in (('cold', 'warm') if repetition == 1 else ('warm', 'cold'))
            for build, variant in (variants if repetition == 1 else variants[::-1])
            ] + [('current', variant, 'cold', repetition, 'trace')
                 for repetition in (1, 2)
                 for variant in (('E-off', 'E-on') if repetition == 1 else ('E-on', 'E-off'))]


def summarize(root, results, *, complete=False, study='versions'):
    if study == 'current':
        from current_summary import summarize as current_summarize
        return current_summarize(root, results, complete=complete)
    if study in ('slots', 'slots-warm', 'files', 'files-warm', 'groups', 'groups-warm', 'groups-cold'):
        from slots_summary import summarize as slots_summarize
        return slots_summarize(root, results, complete=complete,
                               warm_only=study.endswith('-warm'), cold_only=study.endswith('-cold'),
                               focus=study.split('-')[0])
    if study == 'profile':
        from profile_summary import summarize as profile_summarize
        return profile_summarize(root, results, complete=complete)
    if study == 'size4':
        from size_summary import summarize as size_summarize
        return size_summarize(root, results, complete=complete)
    if study == 'depth':
        from depth_summary import summarize as depth_summarize
        return depth_summarize(root, results, complete=complete)
    if complete:
        assert [(r['build'], r['variant'], r['cache'], r['repetition'], r['trace_mode'])
                for r in results] == matrix()
    controls = [r for r in results if r['trace_mode'] == 'control']
    lines = ['# Stage 4 stream-read performance acceptance', '',
             f'{len(results)}/{len(matrix())} samples collected; '
             + ('complete.' if complete else 'partial results.'), '',
             '| Build | Bulk | Cache | R1 / R2 events/s | Rate from median time | Median seconds |',
             '|---|---|---|---:|---:|---:|']
    medians = {}
    for build in ('previous', 'current'):
        for variant in ('E-off', 'E-on'):
            for cache in ('cold', 'warm'):
                rows = sorted((r for r in controls if (r['build'], r['variant'], r['cache'])
                               == (build, variant, cache)), key=lambda r: r['repetition'])
                if not rows:
                    continue
                seconds = statistics.median(r['loop_s'] for r in rows)
                medians[build, variant, cache] = seconds
                rates = ' / '.join(f"{r['events_per_s']:.2f}" for r in rows)
                lines.append(f'| {build} | {variant} | {cache} | {rates} | '
                             f"{rows[0]['events']/seconds:.2f} | {seconds:.4f} |")
    comparisons = {}
    for cache in ('cold', 'warm'):
        keys = [(b, v, cache) for b, v in (('current', 'E-on'), ('current', 'E-off'),
                                          ('previous', 'E-on'), ('previous', 'E-off'))]
        if all(k in medians for k in keys):
            on, off, old_on, old_off = [medians[k] for k in keys]
            comparisons[cache] = dict(current_on_off_time_ratio=on/off,
                current_previous_on_time_ratio=on/old_on,
                current_previous_off_time_ratio=off/old_off,
                on_no_slower_than_off=on <= off)
    traces = [r for r in results if r['trace_mode'] == 'trace']
    if traces:
        lines += ['', '| Current bulk | Round | One-file wall s | POSIX-active wall s | One-file % | Native reads |',
                  '|---|---:|---:|---:|---:|---:|']
        for row in traces:
            audit = row['fallback_audit']
            f = audit['file_concurrency']
            lines.append(f"| {row['variant']} | {row['repetition']} | {f['single_file_s']:.6f} | "
                         f"{f['posix_active_s']:.6f} | {f['single_file_percent']:.2f} | "
                         f"{audit['operations']['POSIX']['calls']} |")
    lines += ['', 'Throughput uses untraced controls; each case has two rounds when complete.',
              'Cold: every prefix <=1% resident before timing and physical NIC RX >=98% of payload.',
              'Warm: every prefix >=99% resident before and after timing.',
              'Weka SSD placement is checked before and after; server caches are not flushed.',
              'The requested on/off target is current bulk-on median loop time <= bulk-off for both caches.',
              'This is the 1,000-event minimum reproducer, not a long-run/scaling acceptance.', '']
    for cache, comparison in comparisons.items():
        lines.append(f"{cache}: current on/off time ratio {comparison['current_on_off_time_ratio']:.4f}; "
                     f"current/previous on {comparison['current_previous_on_time_ratio']:.4f}; "
                     f"current/previous off {comparison['current_previous_off_time_ratio']:.4f}.")
    met = (complete and len(comparisons) == 2
           and all(c['on_no_slower_than_off'] for c in comparisons.values()))
    lines += ['', f'On/off performance target met: {met if complete else "pending"}.', '']
    (root/'summary.md').write_text('\n'.join(lines))
    save(root/'summary.json', dict(complete=complete, accepted_samples=len(results),
                                  comparisons=comparisons, performance_target_met=met if complete else None))


def validate(row, build, provenance):
    reference = provenance['reference']
    assert row['events'] == reference['events'] and not row['diagnostic'] and row['include_jf']
    assert row['timestamp_sha256'] == reference['timestamp_sha256']
    assert row['sums_sha256'] == reference['sums_sha256']
    assert row['payload_bytes'] == reference['payload_bytes']
    before, after = row['cache_before'], row['cache_after']
    assert before['measured_prefixes'] and len(before['files']) == 6
    if row['cache'] == 'cold':
        assert all(f['resident_fraction'] <= .01 for f in before['files'])
        assert row['physical_rx_bytes'] >= .98 * row['payload_bytes']
    else:
        assert all(f['resident_fraction'] >= .99 for state in (before, after) for f in state['files'])
    counts = row['ranks'][2]['counts']
    expected = 6000 if row['variant'] == 'E-off' else (114 if build == 'previous' else 5019)
    expected = provenance.get('builds', {}).get(build, {}).get('expected_requests', {}).get(row['variant'], expected)
    assert counts['requests'] == expected and counts['bytes'] == row['payload_bytes']
    assert all(r['affinity'] == provenance['affinity'] for r in row['ranks'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--references', type=Path, required=True)
    p.add_argument('--constants', required=True)
    p.add_argument('--builds', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fallback-library', type=Path, required=True)
    p.add_argument('--events', type=int, default=1000)
    p.add_argument('--study', choices=('versions', 'depth', 'size4', 'profile', 'slots', 'slots-warm',
                                      'files', 'files-warm', 'groups', 'groups-warm', 'groups-cold', 'current'), default='versions')
    a = p.parse_args()
    if a.events < 200 or a.events % 100:
        p.error('events must be a multiple of 100, at least 200')
    if a.events != 1000 and a.study != 'current':
        p.error('non-default event counts require --study current')
    root = a.output
    root.mkdir(parents=True, exist_ok=False)
    builds = json.loads(a.builds.read_text())
    if a.events != 1000:
        assert set(builds['current']['expected_requests']) == {'E-off', 'E-on'}
    verify(builds)
    files = sorted(a.directory.glob('*.xtc2')) + sorted((a.directory/'smalldata').glob('*.xtc2'))
    affinity = sorted(os.sched_getaffinity(0))
    os.environ['BENCH_CPU_AFFINITY'] = ','.join(map(str, affinity))
    import cupy as cp
    bus = cp.cuda.runtime.deviceGetPCIBusId(0)
    bus = bus.decode() if isinstance(bus, bytes) else bus
    uuid = subprocess.check_output(['nvidia-smi', '-i', bus, '--query-gpu=uuid',
                                    '--format=csv,noheader'], text=True).strip()
    manifest = input_manifest(a.directory, a.events)
    reference = cpu_reference(a.directory, events=a.events, manifest=manifest)
    assert reference == json.loads((a.references/'measured-reference.json').read_text())
    assert manifest == json.loads((a.references/'measured-manifest.json').read_text())
    inputs = [a.constants, a.builds, a.fallback_library] + list(a.references.glob('*.json'))
    provenance = dict(job_id=os.environ['SLURM_JOB_ID'], node=os.uname().nodename,
        gpu_uuid=uuid, gpu_bus=bus, affinity=affinity, builds=builds, reference=reference,
        manifest=manifest, study=a.study, inputs={str(f):sha(f) for f in inputs},
        scripts={str(f):sha(f) for f in HERE.glob('*.py')}, tier_before=tier(files))
    save(root/'provenance.json', provenance)
    results = []
    save(root/'results.json', results)
    summarize(root, results, study=a.study)
    for build, variant, cache, repetition, mode in matrix(a.study):
        tag = f'{build}-{variant}-{cache}-r{repetition}-{mode}'
        env = os.environ.copy()
        entry = builds[build]
        env['PYTHONPATH'] = entry['python']
        env['LD_LIBRARY_PATH'] = entry['native_prefix']+'/lib:'+env.get('LD_LIBRARY_PATH', '')
        cmd = ['mpirun', '-n', '3', '--oversubscribe', '--bind-to', 'none', sys.executable,
               '-u', str(HERE/'bench.py'), '--directory', str(a.directory),
               '--reference', str(a.references/'measured-reference.json'),
               '--variant', variant, '--cache', cache, '--include-jf', '--constants', a.constants,
               '--jf-reference', str(a.references/'jf-reference.json'), '--events', str(a.events),
               '--warmup-events', '200', '--warmup-reference', str(a.references/'warmup-reference.json'),
               '--range-manifest', str(a.references/'measured-manifest.json'), '--read-stats',
               '--pool-depth', str(entry.get('depth', 1))]
        task_size = entry.get('task_size', 1 << 20)
        env['KVIKIO_TASK_SIZE'] = str(task_size)
        cmd += ['--task-size', str(task_size)]
        if 'bulk_target_bytes' in entry:
            cmd += ['--bulk-target-bytes', str(entry['bulk_target_bytes'])]
        if mode == 'pipeline':
            cmd += ['--pipeline-stats']
        if mode == 'profile':
            cmd += ['--python-profile', str(root/(tag+'.pstats'))]
        if mode == 'trace':
            env['LD_PRELOAD'] = str(a.fallback_library)
            cmd += ['--fallback-library', str(a.fallback_library), '--fallback-output', str(root/(tag+'.bin'))]
        else:
            assert not env.get('LD_PRELOAD')
        with (root/(tag+'-gpu.csv')).open('x') as output:
            monitor = subprocess.Popen(['nvidia-smi', '-i', uuid,
                '--query-gpu=timestamp,uuid,memory.used,utilization.gpu,clocks.sm',
                '--format=csv,noheader,nounits', '-lms', '250'], stdout=output, stderr=subprocess.STDOUT)
            try:
                text = execute(cmd, env, root/(tag+'.log'), timeout=max(600, a.events * .12))
            finally:
                monitor.terminate()
                monitor.wait(timeout=15)
        runtime, = records(text, 'RUNTIME ')
        assert runtime['psana'].startswith(entry['python']+'/') and runtime['gpu'] == bus
        warmup, = records(text, 'WARMUP_CHECK ')
        assert warmup == dict(events=200, feespec_arrays_pass=True, jf_samples=3)
        row, = records(text, 'RESULT ')
        validate(row, build, provenance)
        assert row['pool_depth'] == entry.get('depth', 1)
        assert row['task_size'] == runtime['task_size'] == task_size
        assert row['bulk_target_bytes'] == entry.get('bulk_target_bytes')
        if mode == 'pipeline':
            stats = row['ranks'][2]['pipeline_stats']
            assert sum(stats['subbatches']) == a.events
            assert 0 < stats['peak_charged_bytes'] <= 8*1024**3
            assert stats['launches']['gather'] == len(stats['subbatches'])
            assert stats['launches']['calibration'] == a.events
        row.update(build=build, repetition=repetition, trace_mode=mode, warmup_check=warmup)
        if mode == 'profile':
            from profile_summary import read_profile
            path = root/(tag+'.pstats')
            assert row['ranks'][2]['python_profile'] == str(path)
            row['python_profile'] = read_profile(path)
            row['python_profile']['sha256'] = sha(path)
        if mode == 'trace':
            from summarize_kvikio_fallback import audit_trace
            audit = audit_trace(row['ranks'][2]['fallback_trace']['metadata'])
            assert audit['bytes'] == row['payload_bytes'] and audit['pool_workers'] == 8
            assert audit['file_concurrency']['files'] == 6
            assert audit['api_requests'] == row['ranks'][2]['counts']['requests']
            loop_s = row['ranks'][2]['elapsed']
            active_s = audit['file_concurrency']['posix_active_s']
            assert 0 <= active_s <= loop_s
            audit['bd_loop_s'] = loop_s
            audit['no_posix_read_s'] = loop_s-active_s
            audit['no_posix_read_percent'] = 100*(loop_s-active_s)/loop_s
            row['fallback_audit'] = audit
        results.append(row)
        save(root/'results.json', results)
        summarize(root, results, study=a.study)
        print('ACCEPTED', tag, round(row['events_per_s'], 2), flush=True)
    provenance['tier_after'] = tier(files)
    verify(builds)
    for path, digest in {**provenance['inputs'], **provenance['scripts']}.items():
        assert sha(path) == digest, path
    save(root/'provenance.json', provenance)
    summarize(root, results, complete=True, study=a.study)
    print('COMPLETE', root/'summary.md', flush=True)


if __name__ == '__main__':
    main()
