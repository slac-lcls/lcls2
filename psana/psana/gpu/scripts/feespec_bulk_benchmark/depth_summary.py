"""Current-runtime depth 1/2 comparison; diagnostics never determine rates."""
import statistics
from quick import save


def matrix():
    variants = [('depth1', 'E-off'), ('depth1', 'E-on'),
                ('depth2', 'E-off'), ('depth2', 'E-on')]
    return ([(b, v, 'warm', 1, 'pipeline') for b, v in variants]
            + [(b, v, cache, repetition, 'control')
               for repetition in (1, 2)
               for cache in (('cold', 'warm') if repetition == 1 else ('warm', 'cold'))
               for b, v in (variants if repetition == 1 else variants[::-1])]
            + [(b, v, 'cold', repetition, 'trace') for repetition in (1, 2)
               for b, v in (variants if repetition == 1 else variants[::-1])])


def summarize(root, results, *, complete=False, builds=('depth1', 'depth2'),
              title='Current runtime: pool depth 1 versus 2', task_mib=1, schedule=None):
    schedule = matrix() if schedule is None else schedule
    if complete:
        assert [(r['build'], r['variant'], r['cache'], r['repetition'], r['trace_mode'])
                for r in results] == schedule
    lines = ['# '+title, '',
             f'{len(results)}/{len(schedule)} validated samples; complete={complete}.', '',
             '| Build | Bulk | Cache | R1 / R2 events/s | Median-time rate | Median loop s | Future-get block s |',
             '|---|---|---|---:|---:|---:|---:|']
    medians = {}
    for build in builds:
        for variant in ('E-off', 'E-on'):
            for cache in ('cold', 'warm'):
                rows = sorted((r for r in results if r['trace_mode'] == 'control'
                               and (r['build'], r['variant'], r['cache']) == (build, variant, cache)),
                              key=lambda r: r['repetition'])
                if not rows:
                    continue
                seconds = statistics.median(r['loop_s'] for r in rows)
                medians[build, variant, cache] = seconds
                rates = ' / '.join(f"{r['events_per_s']:.2f}" for r in rows)
                wait = statistics.median(r['ranks'][2]['counts']['read_wait_s'] for r in rows)
                lines.append(f'| {build} | {variant} | {cache} | {rates} | {1000/seconds:.2f} | {seconds:.4f} | {wait:.4f} |')
    ratios = {}
    for variant in ('E-off', 'E-on'):
        for cache in ('cold', 'warm'):
            if all((b, variant, cache) in medians for b in ('depth1', 'depth2')):
                ratios[variant+'-'+cache] = medians['depth2', variant, cache]/medians['depth1', variant, cache]
    lines += ['', '| Build | Bulk | Cold trace round | Single-file % | POSIX-active s | No POSIX reads s | No POSIX reads % of BD loop |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in results:
        if r['trace_mode'] != 'trace':
            continue
        a = r['fallback_audit']; f = a['file_concurrency']
        lines.append(f"| {r['build']} | {r['variant']} | {r['repetition']} | {f['single_file_percent']:.2f} | "
                     f"{f['posix_active_s']:.4f} | {a['no_posix_read_s']:.4f} | {a['no_posix_read_percent']:.2f} |")
    lines += ['', '| Build | Bulk | Execution subbatches | Walk/init/locate/gather/calib launches | Peak charged MiB | Reserve calls |',
              '|---|---|---:|---|---:|---:|']
    for r in results:
        if r['trace_mode'] != 'pipeline':
            continue
        s = r['ranks'][2]['pipeline_stats']
        launches = '/'.join(str(s['launches'].get(k, 0)) for k in ('walk', 'init', 'locate', 'gather', 'calibration'))
        lines.append(f"| {r['build']} | {r['variant']} | {len(s['subbatches'])} | {launches} | "
                     f"{s['peak_charged_bytes']/1024**2:.3f} | {s['allocation_reserve_calls']} |")
    on_off = {cache: medians['size4', 'E-on', cache]/medians['size4', 'E-off', cache]
              for cache in ('cold', 'warm')
              if all(('size4', v, cache) in medians for v in ('E-off', 'E-on'))}
    comparisons = ('Bulk on/off median loop-time ratios: '+str(on_off) if builds == ('size4',)
                   else 'Depth2/depth1 median loop-time ratios: '+str(ratios))
    lines += ['', comparisons, '',
              'Rates use controls only; read counters remain enabled. Traces and pipeline diagnostics are separate.',
              'No-POSIX-read time includes useful GPU compute/H2D and setup/tail; it does not mean GPU idle.',
              'Single-file % excludes intervals without POSIX reads. Both are union wall-time measures.',
              'Peak charged memory includes live/cached owned backing, not total CUDA process memory.',
              f'Same 1,000 events, batch 100, 8 GiB budget, eight KvikIO workers, {task_mib} MiB tasks.',
              'Cold prefixes <=1% and NIC RX >=98% payload; warm prefixes >=99% before/after.', '']
    (root/'summary.md').write_text('\n'.join(lines))
    save(root/'summary.json', dict(complete=complete, accepted_samples=len(results),
                                  depth2_depth1_time_ratios=ratios, bulk_on_off_time_ratios=on_off))
