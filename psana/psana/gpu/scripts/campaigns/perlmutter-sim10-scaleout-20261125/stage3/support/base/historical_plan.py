"""Frozen full comparison; one allocation, no retuning or scheduler side effects."""
from support import require
from storage import CACHE_SECONDS

ATTEMPT = 'comparison02'
WALLTIME = 21300
TIME_LIMIT = '05:55:00'
OUTPUT_RESERVE = 4 << 30
BUDGET = dict(preflight=300, device_checks=480, acceptance=600,
              staging=3600, compute=8640, final_audit=700, summary=120, safety=560, cache_conditioning=35*CACHE_SECONDS)


def cell(name, kind, stage, bds, gpus, batch, events, timeout, repeat=0, cohort='comparison'):
    return dict(name=name, kind=kind, stage=stage, bds=bds, gpus=gpus,
                batch=batch, events=events, timeout=timeout+CACHE_SECONDS, repeat=repeat, cohort=cohort)


def cases():
    result = [cell('accept-bd%d-ECC'%bd, 'acceptance', 'ECC', bd, bd,
                   {1:50,4:200}[bd], {1:201,4:1003}[bd], 300, cohort='acceptance')
              for bd in (1,4)]
    compute = ['E','EC','ECC']
    for repeat in range(1,4):
        result.append(cell('bridge-r%d'%repeat, 'full', 'S', 4, 1, 200, 100000, 240,
                           repeat, 'bridge'))
        for bd in ((1,4) if repeat % 2 else (4,1)):
            batch = {1:50,4:200}[bd]
            order = ['S'] + compute[repeat-1:] + compute[:repeat-1]
            for stage in order:
                timeout = 240 if stage=='S' else {1:600,4:360}[bd]
                result.append(cell('full-r%d-bd%d-%s'%(repeat,bd,stage), 'full', stage,
                                   bd, bd, batch, 100000, timeout, repeat))
            result.append(cell('anchor-r%d-bd%d-S'%(repeat,bd), 'anchor', 'S', bd, bd,
                               batch, 100000, 240, repeat, 'anchor'))
    return result


def placement(uuids, case):
    require(len(uuids)==len(set(uuids))==4 and all(u.startswith('GPU-') for u in uuids), 'Four physical GPUs')
    require(case['gpus']==case['bds'] or (case['stage']=='S' and case['bds']==4 and case['gpus']==1),
            'Only the staging bridge shares a GPU')
    return dict(rank_uuids=['','']+(uuids[:case['bds']] if case['gpus']==case['bds'] else [uuids[0]]*4))


def validate():
    matrix = cases()
    require(sum(BUDGET.values())==WALLTIME, 'Whole-allocation budget')
    require(len(matrix)==35 and len({c['name'] for c in matrix})==35, 'Unique bounded schedule')
    require(sum(c['timeout'] for c in matrix)==BUDGET['acceptance']+BUDGET['staging']+BUDGET['compute']+BUDGET['cache_conditioning'], 'Case budget')
    require(len([c for c in matrix if c['cohort']=='comparison'])==24, 'Three repeats/two profiles/four stages')
    require(all(c['events']==100000 for c in matrix if c['kind']!='acceptance'), 'Matched full event count')

validate()
