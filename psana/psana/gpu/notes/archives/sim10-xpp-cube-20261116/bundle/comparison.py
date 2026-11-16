"""Re-audit one complete allocation, then summarize matched 100k cases only."""
import json
import math
from pathlib import Path
import statistics
from audit import audit_case
from memory_watch import validate_memory
from plan import cases, placement
from support import require, sha, write, EVENT_BYTES
from storage import CACHE_MAX_ATTEMPTS, CACHE_SECONDS


def stats(values):
    require(values and all(math.isfinite(x) and x > 0 for x in values), 'Finite positive timings')
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values), samples=values)


def summarize(records):
    require([r['case'] for r in records] == cases(), 'Complete ordered comparison matrix')
    require(len({r['job'] for r in records}) == len({r['node'] for r in records}) == 1, 'Same job and node')
    for row in records:
        require(row['audit'] == 'pass', 'All case audits must pass')
        stats([row['seconds']])
        require(type(row['cache_attempts']) is int and 1 <= row['cache_attempts'] <= CACHE_MAX_ATTEMPTS, 'Cache summary attempts')
        require(math.isfinite(row['conditioning_seconds']) and
                0 <= row['conditioning_seconds'] < CACHE_SECONDS, 'Cache summary duration')
    groups, paired, drift = [], [], []
    for bd in (1, 4):
        for stage in ('S', 'E', 'EC', 'ECC'):
            rows = [r for r in records if r['case']['cohort'] == 'comparison'
                    and r['case']['bds'] == bd and r['case']['stage'] == stage]
            require(len(rows) == 3 and {r['case']['repeat'] for r in rows} == {1,2,3}, 'Three matched repeats')
            times = [r['seconds'] for r in rows]
            groups.append(dict(bds=bd, gpus=bd, batch={1:50,4:200}[bd], stage=stage, events=100000,
                               seconds=stats(times), events_per_second=stats([100000/t for t in times]),
                               logical_payload_gib_s=stats([100000*EVENT_BYTES/2**30/t for t in times])))
        for repeat in (1,2,3):
            rows = [r for r in records if r['case']['repeat'] == repeat
                    and r['case']['bds'] == r['case']['gpus'] == bd]
            full = {r['case']['stage']: r['seconds'] for r in rows if r['case']['cohort'] == 'comparison'}
            anchor = next(r['seconds'] for r in rows if r['case']['cohort'] == 'anchor')
            drift.append(dict(bds=bd, repeat=repeat, staging_before=full['S'], staging_after=anchor,
                              after_over_before=anchor/full['S']))
            for stage in ('E','EC','ECC'):
                paired.append(dict(bds=bd, repeat=repeat, stage=stage,
                                   seconds_over_staging=full[stage]/full['S'],
                                   seconds_minus_staging=full[stage]-full['S']))
    bridge = [r['seconds'] for r in records if r['case']['cohort'] == 'bridge']
    return dict(status='pass_pending_scheduler_and_independent_result_review', job=records[0]['job'],
                node=records[0]['node'], comparison=groups, paired=paired, staging_drift=drift,
                staging_bridge_seconds=stats(bridge),
                cache_conditioning=[dict(name=r['case']['name'], cohort=r['case']['cohort'],
                                         stage=r['case']['stage'], bds=r['case']['bds'],
                                         gpus=r['case']['gpus'], repeat=r['case']['repeat'],
                                         attempts=r['cache_attempts'], seconds=r['conditioning_seconds'])
                                    for r in records],
                interpretation='Processing includes delivery/drain and existing memory queries. Setup, cache conditioning and cube checks are outside it. Differences are total pipeline costs, not isolated kernel times. Basic GPU cube content checks are not numerical-reference agreement. Logical payload bandwidth is not measured NIC/OST traffic. Cache-conditioning attempts and durations are reported per case for assessing fairness; conditioning is excluded from processing. No pooling with the prior node.')


def build(attempt):
    output = attempt/'results'
    state = json.loads((output/'status.json').read_text())
    require([x['case'] for x in state['completed']] == cases(), 'All cases completed before summary')
    info = json.loads((attempt/'inputs.json').read_text())
    ranges = json.loads((attempt/'ranges.json').read_text())
    hardware = json.loads((output/'hardware.json').read_text())
    mapping = json.loads((output/'affinity.json').read_text())['mapping']
    records = []
    for entry in state['completed']:
        case = entry['case']; name = case['name']
        path = output/(name+'.json')
        require(sha(path) == entry['result_sha256'], 'Case receipt changed')
        report = json.loads(path.read_text())
        require(report['job'] == state['job'], 'Case belongs to this job')
        require(report['rate_is_clean'] == (case['kind'] in ('full','anchor')), 'Timing category')
        require(all(row['trace_window']['tracer_pid']==0 for row in report['rows'][2:]), 'Untraced comparison')
        gpu_plan = placement(hardware['uuids'], case)
        gpu_plan['pci_by_uuid'] = hardware['pci_by_uuid']
        verdict = audit_case(report, case, info, ranges, mapping[:case['bds']+2], gpu_plan)
        require(verdict == entry['audit'] == json.loads((output/(name+'.audit.json')).read_text()), 'Audit receipt changed')
        memory = json.loads((output/(name+'.memory.json')).read_text())
        require(len(memory) == case['gpus'] and {m['uuid'] for m in memory} == set(gpu_plan['rank_uuids'][2:]), 'Memory device coverage')
        for receipt in memory:
            validate_memory(receipt)
        records.append(dict(case=case, seconds=report['seconds'], audit=verdict['status'],
                            job=report['job'], node=report['rows'][0]['host'],
                            cache_attempts=len(report['observation']['cache_gate']['attempts']),
                            conditioning_seconds=report['observation']['conditioning_seconds']))
    result = summarize(records)
    result['source_commit'] = state['source_commit']
    write(output/'comparison.json', result)
    return result


if __name__ == '__main__':
    import sys
    build(Path(sys.argv[1]))
