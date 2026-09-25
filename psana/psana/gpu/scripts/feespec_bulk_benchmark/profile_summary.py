"""Measured-loop CPU call attribution; diagnostic timings are not controls.

cProfile observes the BD Python thread, including blocking native calls made
by that thread. It does not time asynchronous CUDA kernels or KvikIO workers.
Self times are additive; cumulative times overlap and must not be summed.
"""
import json
import pstats
import statistics


def matrix():
    # Reverse cache, variant, and instrumentation order in the second round.
    return [('current', variant, cache, repetition, mode)
            for repetition in (1, 2)
            for cache in (('cold', 'warm') if repetition == 1 else ('warm', 'cold'))
            for variant in (('E-off', 'E-on') if repetition == 1 else ('E-on', 'E-off'))
            for mode in (('control', 'profile') if repetition == 1 else ('profile', 'control'))]


def read_profile(path):
    stats = pstats.Stats(str(path))
    rows = []
    for (filename, line, name), (primitive, calls, own, cumulative, callers) in stats.stats.items():
        rows.append(dict(file=filename, line=line, function=name, calls=calls,
                         primitive_calls=primitive, self_s=own, cumulative_s=cumulative,
                         callers=[dict(file=f, line=n, function=label,
                                       primitive_calls=value[0], calls=value[1],
                                       self_s=value[2], cumulative_s=value[3])
                                  for (f, n, label), value in callers.items()]))
    return dict(path=str(path), total_self_s=stats.total_tt,
                functions=sorted(rows, key=lambda r: r['self_s'], reverse=True))


def summarize(root, results, *, complete=False):
    if complete:
        assert [(r['build'], r['variant'], r['cache'], r['repetition'], r['trace_mode'])
                for r in results] == matrix()
    lines = ['# Group scheduling CPU profile', '',
             f'{len(results)}/{len(matrix())} validated samples; complete={complete}.', '',
             '| Bulk | Cache | Control R1 / R2 events/s | Control median loop s | Profile median loop s |',
             '|---|---|---:|---:|---:|']
    medians = {}
    for variant in ('E-off', 'E-on'):
        for cache in ('cold', 'warm'):
            times = {}
            for mode in ('control', 'profile'):
                rows = sorted((r for r in results if (r['variant'], r['cache'], r['trace_mode'])
                               == (variant, cache, mode)), key=lambda r: r['repetition'])
                times[mode] = statistics.median(r['loop_s'] for r in rows) if rows else None
                if mode == 'control':
                    rates = ' / '.join(f"{r['events_per_s']:.2f}" for r in rows) or 'pending'
            medians[f'{variant}/{cache}'] = times
            rendered = [f'{times[m]:.4f}' if times[m] is not None else 'pending'
                        for m in ('control', 'profile')]
            lines.append(f'| {variant} | {cache} | {rates} | {rendered[0]} | {rendered[1]} |')
    lines += ['', 'Profiles contain the BD measured loop including its final synchronization.',
              'Warmup, cache preparation, and profile serialization are excluded.',
              'Self time includes blocking native calls on the Python thread; it is not GPU kernel time.',
              'Cumulative times overlap. Instrumented times must not be used as control throughput.', '']
    for row in results:
        if row['trace_mode'] != 'profile':
            continue
        lines += [f"## {row['variant']} {row['cache']} round {row['repetition']}", '',
                  '| Function | Calls | Self s | Cumulative s |', '|---|---:|---:|---:|']
        profile = row['python_profile']
        for item in profile['functions'][:20]:
            label = f"{item['file']}:{item['line']}({item['function']})".replace('|', '/')
            lines.append(f"| {label} | {item['calls']} | {item['self_s']:.6f} | {item['cumulative_s']:.6f} |")
        lines += ['', f"Full call graph: `{profile['path']}`; structured functions/callers in results.json.", '']
    (root/'summary.md').write_text('\n'.join(lines))
    (root/'summary.json').write_text(json.dumps(dict(
        complete=complete, accepted_samples=len(results), medians=medians), indent=2)+'\n')
