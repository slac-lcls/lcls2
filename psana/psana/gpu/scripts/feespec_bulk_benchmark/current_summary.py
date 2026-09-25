"""Current runtime, two rounds of bulk off/on with cold and warm caches."""
import statistics

from quick import save


def matrix():
    return [('current', variant, cache, repetition, 'control')
            for repetition in (1, 2)
            for cache in (('cold', 'warm') if repetition == 1 else ('warm', 'cold'))
            for variant in (('E-off', 'E-on') if repetition == 1 else ('E-on', 'E-off'))]


def summarize(root, results, *, complete=False):
    assert [(r['build'], r['variant'], r['cache'], r['repetition'], r['trace_mode'])
            for r in results] == matrix()[:len(results)]
    if complete:
        assert len(results) == len(matrix())
    assert len({r['events'] for r in results}) <= 1
    groups, ratios = {}, {}
    lines = ['# Current-code bulk off/on comparison', '',
             f'{len(results)}/8 samples; ' + ('complete.' if complete else 'partial.'), '',
             '| Bulk | Cache | Events | R1 / R2 events/s | Median loop s | Rate from median time |',
             '|---|---|---:|---:|---:|---:|']
    for cache in ('cold', 'warm'):
        for variant in ('E-off', 'E-on'):
            rows = [r for r in results if (r['variant'], r['cache']) == (variant, cache)]
            if not rows:
                continue
            seconds = statistics.median(r['loop_s'] for r in rows)
            rates = ' / '.join(f"{r['events_per_s']:.2f}" for r in rows)
            requests = [r['ranks'][2]['counts']['requests'] for r in rows]
            assert len(set(requests)) == 1, 'request counts changed between repetitions'
            groups[f'{variant}/{cache}'] = dict(median_loop_s=seconds,
                events=rows[0]['events'], events_per_s=rows[0]['events']/seconds,
                seconds=[r['loop_s'] for r in rows], requests=requests[0])
            lines.append(f"| {variant} | {cache} | {rows[0]['events']} | {rates} | "
                         f"{seconds:.4f} | {rows[0]['events']/seconds:.2f} |")
        if all(f'{v}/{cache}' in groups for v in ('E-off', 'E-on')):
            ratios[cache] = groups[f'E-on/{cache}']['median_loop_s']/groups[f'E-off/{cache}']['median_loop_s']
    lines += ['', 'Both rounds use the same frozen runtime, with variant and cache order reversed.',
              'Controls have native reader counters but no profiles or native-operation tracing.',
              'Cold: every measured prefix <=1% resident before timing; physical NIC RX >=98% of payload.',
              'Warm: every measured prefix >=99% resident before and after timing.',
              'Cold refers to node page cache; Weka server caches are not flushed.', '']
    lines += [f'{cache}: bulk-on/off median time ratio {ratio:.4f}.' for cache, ratio in ratios.items()]
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    save(root/'summary.json', dict(complete=complete, accepted_samples=len(results),
                                  groups=groups, bulk_on_off_time_ratios=ratios))
