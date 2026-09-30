"""Bulk-on-only baseline/candidate comparison for isolated CPU optimizations."""
import json
import statistics


def matrix(*, warm_only=False, cold_only=False):
    assert not (warm_only and cold_only)
    if warm_only or cold_only:
        return [(build, 'E-on', 'cold' if cold_only else 'warm', repetition, 'control')
                for repetition in range(1, 5)
                for build in (('baseline', 'candidate') if repetition % 2 else ('candidate', 'baseline'))]
    return ([(build, 'E-on', cache, repetition, 'control')
             for repetition in (1, 2)
             for cache in (('cold', 'warm') if repetition == 1 else ('warm', 'cold'))
             for build in (('baseline', 'candidate') if repetition == 1 else ('candidate', 'baseline'))]
            + [(build, 'E-on', cache, 1, 'profile')
               for cache in ('cold', 'warm') for build in ('baseline', 'candidate')]
            + [(build, 'E-on', 'warm', 1, 'pipeline') for build in ('baseline', 'candidate')])


def summarize(root, results, *, complete=False, warm_only=False, cold_only=False, focus='slots'):
    title, filename, target = {
        'slots': ('Slot selection', '/gpu_input_group.py', 'plan_slots'),
        'groups': ('Direct group submission', '/gpu_kvikio_read.py', 'issue_group'),
        'files': ('Pending-file cleanup', '/gpu_kvikio_read.py', '_prune_files'),
    }[focus]
    if complete:
        assert [(r['build'], r['variant'], r['cache'], r['repetition'], r['trace_mode'])
                for r in results] == matrix(warm_only=warm_only, cold_only=cold_only)
    lines = [f'# {title}: bulk-on-only comparison', '',
             f'{len(results)}/{len(matrix(warm_only=warm_only, cold_only=cold_only))} validated samples; complete={complete}.', '',
             '| Build | Cache | Control rates by round (events/s) | Median-time events/s | Median loop s |',
             '|---|---|---:|---:|---:|']
    medians = {}
    for build in ('baseline', 'candidate'):
        for cache in ('cold', 'warm'):
            rows = sorted((r for r in results if (r['build'], r['cache'], r['trace_mode'])
                           == (build, cache, 'control')), key=lambda r: r['repetition'])
            if not rows:
                continue
            seconds = statistics.median(r['loop_s'] for r in rows)
            medians[build, cache] = seconds
            rates = ' / '.join(f"{r['events_per_s']:.2f}" for r in rows)
            lines.append(f'| {build} | {cache} | {rates} | {1000/seconds:.2f} | {seconds:.4f} |')
    ratios = {cache: medians['candidate', cache]/medians['baseline', cache]
              for cache in ('cold', 'warm')
              if all((b, cache) in medians for b in ('baseline', 'candidate'))}
    lines += ['', f'Candidate/baseline control loop-time ratios: {ratios}', '',
              f'| Build | Cache | Profile loop s | {target} calls | Self s | Cumulative s |',
              '|---|---|---:|---:|---:|---:|']
    profiles = []
    for row in results:
        if row['trace_mode'] != 'profile':
            continue
        function, = [f for f in row['python_profile']['functions']
                     if f['file'].endswith(filename) and f['function'] == target]
        profiles.append(dict(build=row['build'], cache=row['cache'], **function))
        lines.append(f"| {row['build']} | {row['cache']} | {row['loop_s']:.4f} | "
                     f"{function['calls']} | {function['self_s']:.6f} | {function['cumulative_s']:.6f} |")
    pipelines = {r['build']: r['ranks'][2]['pipeline_stats'] for r in results
                 if r['trace_mode'] == 'pipeline'}
    equal = (pipelines['baseline'] == pipelines['candidate'] if len(pipelines) == 2 else None)
    if complete and not (warm_only or cold_only):
        assert equal, 'CPU optimization changed pipeline counts or charged-memory peak'
    lines += ['', '| Build | Subbatches | Launch counts | Charged peak MiB | Reserve calls |',
              '|---|---:|---|---:|---:|']
    for build, stats in pipelines.items():
        lines.append(f"| {build} | {len(stats['subbatches'])} | {stats['launches']} | "
                     f"{stats['peak_charged_bytes']/1024**2:.3f} | {stats['allocation_reserve_calls']} |")
    lines += ['', f'Pipeline diagnostics identical: {equal}.',
              'All samples use bulk on, 1,000 events, batch 100, depth 1, 8 GiB, and 4 MiB targets/tasks.',
              'Only controls determine throughput; profiles and pipeline samples are separate.',
              'Profiles include blocking calls on the BD Python thread; cumulative times overlap.',
              'Cold/warm residency, cold NIC bytes, correctness and source/placement checks remain enabled.', '']
    (root/'summary.md').write_text('\n'.join(lines))
    (root/'summary.json').write_text(json.dumps(dict(
        complete=complete, accepted_samples=len(results), time_ratios=ratios,
        pipeline_equivalent=equal, profiles=profiles), indent=2)+'\n')
