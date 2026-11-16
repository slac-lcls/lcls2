"""Bounded bootstrap, not the full campaign; no scheduler side effects."""
from support import require

WALLTIME = 4200
OUTPUT_RESERVE = 2 << 30
BUDGET = dict(preflight=300, device_checks=480, acceptance=600,
              staging=2160, final_audit=220, safety=440)

def cases():
    result = [dict(name='accept-bd%d-ECC'%bd, kind='acceptance', stage='ECC', bds=bd,
                   gpus=bd, batch={1:50, 4:200}[bd], events={1:201, 4:1003}[bd], timeout=300)
              for bd in (1,4)]
    profiles = [(4,1,200), (1,1,50), (4,4,200)]
    for repeat in range(3):
        for bd,gpus,batch in profiles[repeat:]+profiles[:repeat]:
            result.append(dict(name='staging-r%d-bd%d-gpu%d'%(repeat+1,bd,gpus), kind='full',
                               stage='S', bds=bd, gpus=gpus, batch=batch, events=100000, timeout=240))
    return result

def placement(uuids, case):
    require(len(uuids)==len(set(uuids))==4 and all(u.startswith('GPU-') for u in uuids), 'Four physical GPUs')
    require(case['gpus']==case['bds'] or (case['stage']=='S' and case['bds']==4 and case['gpus']==1),
            'Only the staging bridge shares a GPU')
    return dict(rank_uuids=['','']+(uuids[:case['bds']] if case['gpus']==case['bds'] else [uuids[0]]*4))

def validate():
    require(sum(BUDGET.values())==WALLTIME, 'Whole-allocation budget')
    require(sum(c['timeout'] for c in cases())==BUDGET['acceptance']+BUDGET['staging'], 'Case budget')
    require(len(cases())==11, 'Bounded schedule only')

validate()
