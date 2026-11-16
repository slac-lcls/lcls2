"""Validate six native reports and summarize completed-rate measurements."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize(root):
    cases = []
    reports = []
    receipt = json.loads((Path(__file__).resolve().parents[1] /
                          'mfx_expand10_10k_20261106/files.json').read_text())
    expected_counts = [sum(i % 99 == b for i in range(10000)) for b in range(100)]
    for repeat in (1, 2, 3):
        for streams in (1, 2):
            name = f'clean-s{streams}-r{repeat}'
            path = root / name / 'report.json'
            r = json.loads(path.read_text())
            require(r['status'] == 'pass' and r['mode'] == 'throughput', name + ': incomplete/incorrect mode')
            require(r['delivered_events'] == 10000 and r['checks']['events'] == 10000, name + ': event count')
            require(r['checks']['status'] == 'pass' and r['checks']['counts'] == expected_counts, name + ': bins')
            require((r['streams'], r['bins'], r['active_bins'], r['batch_size'], r['requested_warmup']) ==
                    (streams, 100, 99, 7, 1000), name + ': settings')
            require(r['admission']['bulk_target_bytes'] == 524288 and
                    r['admission']['framework_budget_gib'] == 4, name + ': admission settings')
            require(r['admission_pressure_trims'] == {'warmup': 0, 'steady': 0}, name + ': admission drains')
            env = r['runtime']['environment']
            require(env['PS_PARALLEL'] == 'none' and env['PS_FAKESTEP_FLAG'] == '0' and
                    env['KVIKIO_COMPAT_MODE'] == 'ON', name + ': runtime mode')
            require(r['read_stats'] is None and 'export' not in r, name + ': profiled/exported case')
            require(Path(r['fixture']).resolve() == Path(receipt['root']).resolve() and
                    r['fixture_check']['generation_sha256'] == receipt['generation_sha256'], name + ': fixture')
            window = r['window']
            callbacks = r['callbacks']
            require(all(c['size'] > 0 and c['phase'] in ('warmup', 'steady') for c in callbacks), name + ': callbacks')
            require(sum(c['size'] for c in callbacks) == 10000, name + ': callback total')
            warm = sum(c['size'] for c in callbacks if c['phase'] == 'warmup')
            require(window['warmup_events'] == warm and 1000 <= warm < 1000 + max(c['size'] for c in callbacks)
                    and window['steady_events'] == 10000 - warm, name + ': warmup accounting')
            by_stream = {}
            ordinal = 0
            for c in callbacks:
                counts = by_stream.setdefault(c['stream'], [0] * 100)
                for i in range(ordinal, ordinal + c['size']):
                    counts[i % 99] += 1
                ordinal += c['size']
            require(len(by_stream) == streams and list(by_stream.values()) == r['checks']['per_bank_counts']
                    and [sum(v) for v in by_stream.values()] == r['bank_events'], name + ': bank ownership')
            seconds = window['completed_window_seconds']
            whole = r['iteration_completed_seconds']
            require(math.isfinite(seconds) and seconds > 0 and math.isfinite(whole) and whole >= seconds,
                    name + ': timing')
            rate = window['steady_events'] / seconds
            require(math.isclose(rate, window['completed_events_per_second'], rel_tol=1e-12), name + ': rate arithmetic')
            if reports:
                for key in ('host', 'job', 'runtime', 'runtime_files', 'code_sha256', 'calibration', 'fixture_check', 'timing_contract'):
                    require(r[key] == reports[0][key], name + ': changed ' + key)
                require(r['memory']['samples'][0]['monotonic_seconds'] >
                        reports[-1]['memory']['samples'][-1]['monotonic_seconds'], name + ': execution order')
            reports.append(r)
            cases.append(dict(name=name, streams=streams, repeat=repeat,
                report_path=str(path), report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                window=window, whole_iteration_seconds=whole, whole_iteration_events_per_second=10000 / whole,
                callback_sizes=dict(Counter(c['size'] for c in callbacks)), callbacks=len(callbacks),
                bank_events=r['bank_events'], admission_pressure_trims=r['admission_pressure_trims'],
                state_bytes=r['admission']['state_bytes'], memory=r['memory']))
    statistics_by_stream = {}
    for streams in (1, 2):
        rows = [r for r in cases if r['streams'] == streams]
        stats = {}
        for label, values in (
            ('window_events_per_second', [r['window']['completed_events_per_second'] for r in rows]),
            ('whole_iteration_events_per_second', [r['whole_iteration_events_per_second'] for r in rows])):
            stats[label] = dict(median=statistics.median(values), minimum=min(values), maximum=max(values),
                                sample_stdev=statistics.stdev(values), repeats=len(values))
        statistics_by_stream[streams] = stats
    return dict(status='pass', topology='serial single worker / one GPU; not an MPI BD benchmark',
                host=reports[0]['host'], job=reports[0]['job'], runtime=reports[0]['runtime'],
                fixture_check=reports[0]['fixture_check'], cases=cases, statistics=statistics_by_stream,
                two_stream_to_one_window_median_ratio=
                statistics_by_stream[2]['window_events_per_second']['median'] /
                statistics_by_stream[1]['window_events_per_second']['median'],
                limitations='Cache uncontrolled; sequential repeats in one allocation. Callback-boundary warmed window may exclude first measured batch input preparation. Whole iteration includes startup and intentional warmup join. Memory is sampled outside iteration, not continuous device peaks. No kernel-overlap, GDS, MPI, per-event D2H, full-cube numerical or 35kHz claim.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    with args.output.open('x') as f:
        f.write(json.dumps(result, indent=2) + '\n')
