"""Report the short cold sweep's native file overlap and control timings."""
import json


def summarize_trace(root, results, provenance):
    from quick import summarize
    controls = [r for r in results if r['trace_mode'] == 'control']
    traces = [r for r in results if r['trace_mode'] == 'trace']
    assert len(controls) == len(traces) == 4
    # Apply all cold-cache, placement, byte/hash checks to both sets.
    summarize(root, traces, provenance)
    summarize(root, controls, provenance)
    for suffix in ('md', 'json'):
        (root/('summary.'+suffix)).rename(root/('control-summary.'+suffix))
    lines = ['# Short JF + feespec cold Weka FFB fallback trace', '',
             f"Job {provenance['job_id']} on {provenance['node']}; {provenance['events']} events.", '',
             '| Bulk | Round | One-file wall s | POSIX-active wall s | One-file % | Control loop s | Trace loop s | Trace read-ready s |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for t in sorted(traces, key=lambda r:(r['variant'], r['repetition'])):
        c = next(r for r in controls if (r['variant'],r['repetition']) == (t['variant'],t['repetition']))
        a = t['fallback_audit']
        f = a['file_concurrency']
        lines.append(f"| {t['variant']} | {t['repetition']} | {f['single_file_s']:.6f} | {f['posix_active_s']:.6f} | {f['single_file_percent']:.2f} | {c['loop_s']:.4f} | {t['loop_s']:.4f} | {a['read_interval_s']:.4f} |")
    lines += ['', 'One-file % = 100 × wall time with exactly one file in POSIX reads / wall time with any POSIX reads.',
              'Overlapping workers count once; intervals with no POSIX reads are excluded. H2D and existing CUDA waits are audited separately.',
              'All six tested file prefixes must be <=1% resident immediately before each loop, with physical NIC RX >=98% of requested bytes.',
              'Weka files must remain fully SSD-backed, with no object or remote backing. Server-side caches are not flushed.',
              'Each sample validates 200 feespec arrays and three JF samples during warmup; measured timestamp and GPU-sum hashes must match.',
              'The tracer buffers native pread/copy/existing-wait timestamps in RAM and adds no CUDA synchronization.',
              'Trace vs control differences include run variability. POSIX duration includes storage, kernel and scheduling time; it is not device-only latency.', '']
    (root/'summary.md').write_text('\n'.join(lines))
    (root/'summary.json').write_text(json.dumps(dict(audit_pass=True, job_id=provenance['job_id'],
                                                    controls=controls, traces=traces), indent=2)+'\n')
