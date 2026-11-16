"""Offline validation of the completed 100k Sim10 application record.

Reads archived receipts only. Does not import/execute frozen campaign scripts,
query Slurm, touch external raw files, or claim numerical CPU-reference validation.
"""
import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

EVENT_BYTES = 3358200


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def near(a, b, label):
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and
            math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-12), label)


def stats(values):
    return dict(median=statistics.median(values), minimum=min(values),
                maximum=max(values), samples=values)


def expected_cases():
    rows = []

    def add(name, phase, stage, bd, gpu, repeat, events):
        rows.append(dict(name=name, cohort=phase, stage=stage, bds=bd, gpus=gpu,
                         repeat=repeat, events=events, batch={1: 50, 4: 200}[bd]))

    for bd, events in ((1, 201), (4, 1003)):
        add(f'accept-bd{bd}-ECC', 'acceptance', 'ECC', bd, bd, 0, events)
    stages = ['E', 'EC', 'ECC']
    for repeat in (1, 2, 3):
        add(f'bridge-r{repeat}', 'bridge', 'S', 4, 1, repeat, 100000)
        for bd in ((1, 4) if repeat % 2 else (4, 1)):
            for stage in ['S'] + stages[repeat-1:] + stages[:repeat-1]:
                add(f'full-r{repeat}-bd{bd}-{stage}', 'comparison', stage, bd, bd, repeat, 100000)
            add(f'anchor-r{repeat}-bd{bd}-S', 'anchor', 'S', bd, bd, repeat, 100000)
    return rows


def summarize(receipts):
    require(len(receipts) == 35, 'Expected 35 receipts')
    groups, paired, drift = [], [], []
    for bd in (1, 4):
        for stage in ('S', 'E', 'EC', 'ECC'):
            times = [r['seconds'] for r in receipts if r['case']['cohort'] == 'comparison'
                     and r['case']['bds'] == bd and r['case']['stage'] == stage]
            require(len(times) == 3, 'Expected three stage/profile repeats')
            groups.append(dict(bds=bd, gpus=bd, batch={1: 50, 4: 200}[bd], stage=stage,
                               events=100000, seconds=stats(times),
                               events_per_second=stats([100000/t for t in times]),
                               logical_payload_gib_s=stats([100000*EVENT_BYTES/2**30/t for t in times])))
        for repeat in (1, 2, 3):
            selected = [r for r in receipts if r['case']['repeat'] == repeat
                        and r['case']['bds'] == r['case']['gpus'] == bd]
            full = {r['case']['stage']: r['seconds'] for r in selected
                    if r['case']['cohort'] == 'comparison'}
            anchor = next(r['seconds'] for r in selected if r['case']['cohort'] == 'anchor')
            drift.append(dict(bds=bd, repeat=repeat, staging_before=full['S'], staging_after=anchor,
                              after_over_before=anchor/full['S']))
            for stage in ('E', 'EC', 'ECC'):
                paired.append(dict(bds=bd, repeat=repeat, stage=stage,
                                   seconds_over_staging=full[stage]/full['S'],
                                   seconds_minus_staging=full[stage]-full['S']))
    return dict(comparison=groups, paired=paired, staging_drift=drift,
                staging_bridge_seconds=stats([r['seconds'] for r in receipts if r['case']['cohort'] == 'bridge']))


def table(summary):
    lines = ['| BDs / GPUs / batch | Stage | Median seconds (range) | Events/s | Input GiB/s |',
             '|---|---|---:|---:|---:|']
    for r in summary['comparison']:
        s = r['seconds']
        lines.append(f"| {r['bds']} / {r['gpus']} / {r['batch']} | {r['stage']} | "
                     f"{s['median']:.3f} ({s['minimum']:.3f}–{s['maximum']:.3f}) | "
                     f"{r['events_per_second']['median']:.3f} | {r['logical_payload_gib_s']['median']:.3f} |")
    return '\n'.join(lines)


def validate(record, report, repository):
    repo = Path(repository)
    # Catalog validation resolves and bounds these repository-relative paths first.
    archive = repo / record['archive_root']
    directory = report.parent
    receipts = read(archive/'case-receipts.json')
    original = read(archive/'evidence/job61409171-results.json')
    summary = read(directory/'summary.json')
    bundle = read(archive/'bundle/bundle.json')
    require(record['status'] == 'complete' and record['jobs'] == ['61409171'] and
            record['full_cases'] == 24 and record['total_cases'] == 35 and record['replicates'] == 3,
            'Completed campaign metadata')
    require(record['framework_commit'] == '67c327a148a08b2aaf972e67e1e02e367c2b298f' and
            record['source_commit'] == bundle['commit'] == original['source_commit'], 'Source provenance')
    require(record['configuration']['workers_per_bd'] == 4 and
            record['storage']['stripe_count'] == 2 and record['storage']['stripe_size_bytes'] == 1048576,
            'Worker/storage configuration')
    require(len(bundle['files']) == 49 and len(bundle['external']) == 7, 'Bundle population')
    for name, expected in bundle['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe bundle member')
        require(hashlib.sha256((archive/'bundle'/name).read_bytes()).hexdigest() == expected,
                'Frozen bundle changed: ' + name)
    external = read(archive/'external-index.json')
    require({r['path']: r['sha256'] for r in external} == bundle['external'], 'External pin inventory')
    require(read(archive/'history/status.json')['result_review_verdict'] == 'pass', 'Results review state')
    review = read(archive/'reviews/comparison02-results-round01.json')
    require(review['verdict'] == 'pass' and review['reviewed_commit'] ==
            '890761707f117c515125fac60ba7f86ddfd0e940', 'Independent results-review receipt')
    require(original['scheduler']['state'] == 'COMPLETED' and original['scheduler']['exit_code'] == '0:0',
            'Terminal scheduler state')
    native = read(archive/'execution/results/complete.json')
    require(native['summary_sha256'] == original['summary_sha256'] ==
            hashlib.sha256((directory/'summary.json').read_bytes()).hexdigest(), 'Native summary hash')
    require(summary == original['comparison'] == read(archive/'execution/results/comparison.json'),
            'Original summary correspondence')
    raw_index = {r['path']: r for r in read(archive/'raw-index.json')['files']}
    with (directory/'runs.csv').open(newline='') as f:
        runs = list(csv.DictReader(f))
    expected = expected_cases()
    require(len(receipts) == len(runs) == len(original['records']) == len(native['completed']) == 35,
            'Case population')
    cubes = 0
    for r, row, old, want, done in zip(receipts, runs, original['records'], expected, native['completed']):
        c = r['case']
        require(all(c[k] == v for k, v in want.items()), 'Case order/identity')
        require(c == old['case'] == done['case'], 'Native matrix correspondence')
        require(r['job'] == '61409171' and r['node'] == 'nid001020', 'Same allocation/node')
        require(r['audit'] == dict(status='pass', events=c['events'], logical_payload_bytes=c['events']*EVENT_BYTES),
                'Case acceptance')
        require(r['raw_sha256'] == old['result_sha256'] == done['result_sha256'] ==
                raw_index[r['raw_path']]['sha256'], 'Raw receipt hash')
        require(r['seconds'] > 0 and r['events'] == c['events'], 'Event/timing denominator')
        near(r['seconds'], old['seconds'], 'Original timing')
        near(r['hz'], c['events']/r['seconds'], 'Event rate')
        near(r['payload_gib_s'], c['events']*EVENT_BYTES/2**30/r['seconds'], 'Payload rate')
        strings = dict(case_id=c['name'], job_id=r['job'], phase=c['cohort'], stage=c['stage'],
                       node=r['node'], site='perlmutter', storage='lustre', cache_state='client-page-cold',
                       raw_path=r['raw_path'], raw_sha256=r['raw_sha256'], nic_rx_bytes='', nic_tx_bytes='')
        require(all(row[k] == v for k, v in strings.items()), 'CSV labels/raw mapping')
        numbers = dict(block=1, repeat=c['repeat'], gpus_allocated=4, gpus_active=c['gpus'], bds=c['bds'],
                       batch=c['batch'], workers_per_bd=4, stripe_count=2, events_global=c['events'],
                       requested_bytes=c['events']*EVENT_BYTES, loop_seconds=r['seconds'],
                       whole_case_seconds=r['end_to_end_seconds'], whole_step_seconds=r['whole_step_seconds'],
                       payload_gib_s=r['payload_gib_s'], cache_attempts=len(r['cache_gate']['attempts']),
                       conditioning_seconds=r['cache_gate']['elapsed_seconds'])
        for k, v in numbers.items():
            near(row[k], v, 'CSV ' + k)
        gate = r['cache_gate']
        require(gate['status'] == 'pass' and 1 <= len(gate['attempts']) <= 3 and
                len(gate['attempts']) == old['cache_attempts'], 'Cache retry population')
        final = gate['attempts'][-1]['scans']
        require([s['phase'] for s in final] == ['evict', 'verify'] and
                all(len(s['files']) == 5 and all(f['resident_pages'] == 0 for f in s['files']) for s in final),
                'Client-cold final scans')
        require(r['cube_exported'] is False and r['numerical_reference_checked'] is False, 'No export/reference')
        require(sum(x['event_count'] for x in r['ranks']) == c['events'] and
                len(r['ranks']) == c['bds']+2, 'Rank/event coverage')
        for rank in r['ranks'][2:]:
            cube = rank['cube_verification']
            if c['stage'] != 'ECC':
                require(cube is None, 'Unexpected cube')
                continue
            cubes += 1
            require(cube['status'] == 'pass' and cube['shape'] == [100,32,512,1024] and
                    cube['dtype'] == 'float32' and cube['bytes'] == 6710886400 and cube['drained'], 'Cube layout/drain')
            require(len(cube['counts']) == len(cube['bins']) == 100 and
                    sum(cube['counts']) == rank['event_count'] and cube['counts'][99] == 0, 'Cube counts')
            require(all(b['bin'] == i and b['finite'] is True and
                        b['nonzero'] == (cube['counts'][i] > 0) for i, b in enumerate(cube['bins'])), 'Cube content')
            require(cube['cube_payload_host_bytes'] == 0 and cube['exported'] is False and
                    cube['reference_checked'] is False, 'Cube policy')
        if c['stage'] == 'ECC':
            totals = [sum(rank['cube_verification']['counts'][i] for rank in r['ranks'][2:]) for i in range(100)]
            require(totals == [c['events']//99 + (i < c['events']%99) for i in range(99)]+[0],
                    'Global ordinal bin counts')
    memory = read(archive/'memory-summary.json')
    ranks = [rank for receipt in receipts for rank in receipt['ranks']]
    host_peak = max(sample['VmHWM'] for rank in ranks for sample in rank['host_memory'].values()
                    if 'VmHWM' in sample)
    pinned_capacity = max(rank['task_copy']['pinned_capacity_high'] for rank in ranks if rank['task_copy'])
    require(memory['max_recorded_per_process_rss_high_water_bytes'] == host_peak and
            memory['max_task_d2h_destination_capacity_bytes'] == pinned_capacity, 'Memory summary')
    require(f'**{host_peak/2**30:.3f} GiB**' in report.read_text() and
            f'({host_peak} bytes)' in report.read_text() and f'**{pinned_capacity} bytes**' in report.read_text(),
            'Memory report values')
    for rank in ranks:
        sample = rank['sampled_memory_summary']
        require(sample['samples'] >= 0, 'Memory sample count')
        if rank['rank'] >= 2:
            require(sample['samples'] > 0 and sample['minimum_free_bytes'] >= 0 and
                    sample['maximum_pool_reserved_bytes'] >= sample['maximum_pool_used_bytes'] >= 0,
                    'Memory sample extrema')
    require(cubes == 20, 'Twenty cube banks')
    require(Counter(len(r['cache_gate']['attempts']) for r in receipts) == {1: 32, 2: 3}, 'Cache histogram')
    recomputed = summarize(receipts)
    for k, v in recomputed.items():
        require(v == summary[k], 'Summary recomputation: ' + k)
    for r, cache in zip(receipts, summary['cache_conditioning']):
        require(cache['name'] == r['case']['name'] and cache['attempts'] == len(r['cache_gate']['attempts']),
                'Cache summary')
        near(cache['seconds'], r['cache_gate']['elapsed_seconds'], 'Conditioning time')
    body = report.read_text().split('<!-- results-table -->')[1].split('<!-- /results-table -->')[0].strip()
    require(body == table(recomputed), 'Report table differs from receipts')
    return 35
