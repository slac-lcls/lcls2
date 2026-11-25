"""Host-side rejection tests for the new launch boundary."""
import datetime
import time
import pytest
import plan
from campaign import allocation, launch

UUIDS=['GPU-'+str(i) for i in range(4)]

def test_comparison_matrix_and_bracket_controls():
    matrix=plan.cases()
    assert len(matrix)==35
    assert sum(plan.BUDGET.values())==21300
    assert [c['events'] for c in matrix[:2]]==[201,1003]
    assert len([c for c in matrix if c['cohort']=='bridge'])==3
    assert len([c for c in matrix if c['cohort']=='anchor'])==6
    for repeat in (1,2,3):
        for bd,batch in [(1,50),(4,200)]:
            rows=[c for c in matrix if c['repeat']==repeat and c['bds']==bd and c['gpus']==bd]
            assert len(rows)==5
            assert rows[0]['stage']=='S' and rows[0]['cohort']=='comparison'
            assert rows[-1]['stage']=='S' and rows[-1]['cohort']=='anchor'
            assert {c['stage'] for c in rows[:-1]}=={'S','E','EC','ECC'}
            assert all(c['events']==100000 and c['batch']==batch for c in rows)
    assert len({tuple(c['stage'] for c in matrix if c['repeat']==r and c['cohort']=='comparison' and c['bds']==1) for r in (1,2,3)})==3

@pytest.mark.parametrize('stage',['E','EC','ECC'])
def test_compute_sharing_is_rejected(stage):
    with pytest.raises(ValueError):plan.placement(UUIDS,dict(bds=4,gpus=1,stage=stage))

def test_gpu_map_masks_helpers():
    assert plan.placement(UUIDS,dict(bds=4,gpus=4,stage='ECC'))['rank_uuids']==['','']+UUIDS
    assert plan.placement(UUIDS,dict(bds=1,gpus=1,stage='S'))['rank_uuids']==['','',UUIDS[0]]
    with pytest.raises(ValueError):plan.placement([UUIDS[0]]*4,dict(bds=4,gpus=4,stage='ECC'))

def record():
    end=datetime.datetime.fromtimestamp(time.time()+plan.WALLTIME,datetime.timezone.utc).replace(tzinfo=None).isoformat()
    return 'Account=lcls_g QOS=gpu_premium NumNodes=1 TimeLimit=05:55:00 EndTime='+end

def test_deadline_is_scheduler_absolute():
    assert plan.WALLTIME-10<allocation(record())-time.time()<plan.WALLTIME+10

@pytest.mark.parametrize('before,after',[('lcls_g','wrong'),('NumNodes=1','NumNodes=2'),
    ('05:55:00','24:00:00'),('gpu_premium','regular')])
def test_wrong_allocation_rejected(before,after):
    with pytest.raises(ValueError):allocation(record().replace(before,after))

def test_step_has_explicit_time_and_cray_resources():
    cmd=launch(6,'runner.py',[],240)
    for expected in ['--mpi=cray_shasta','--time=4','--ntasks=6','--ntasks-per-node=6','--cpus-per-task=20',
                     '--gpus-per-node=4','--gpu-bind=none','--cpu-bind=none']:
        assert expected in cmd


def test_original_guard_oracle_and_global_dependencies():
    import builtins
    import dis
    import numpy as np
    from types import SimpleNamespace
    from device_check import load_guards
    guard = load_guards(SimpleNamespace(WORDS=52429, PIXELS=512*1024))
    oracle = guard.__globals__['expand']
    for function in (guard, oracle):
        globals_used = {op.argval for op in dis.get_instructions(function)
                        if op.opname in ('LOAD_GLOBAL', 'LOAD_NAME')}
        assert not (globals_used - function.__globals__.keys() - vars(builtins).keys())
    seed = np.arange(52429, dtype=np.uint16)
    raw = oracle(seed)
    assert raw.shape == (512, 1024) and raw.dtype == np.uint16
    indices = np.array([0, 9, 10, 524279, 524280, 524287])
    assert np.array_equal(raw.reshape(-1)[indices], seed[indices//10])
    with pytest.raises(ValueError):oracle(seed[:-1])
    with pytest.raises(ValueError):oracle(seed.astype(np.uint32))


def unresolved_globals(function):
    """Resolve LOAD_GLOBAL in this function and nested code, including late branches."""
    import builtins
    import dis
    from types import CodeType
    def names(code):
        found = {op.argval for op in dis.get_instructions(code) if op.opname == 'LOAD_GLOBAL'}
        for value in code.co_consts:
            if isinstance(value, CodeType): found.update(names(value))
        return found
    return names(function.__code__) - function.__globals__.keys() - vars(builtins).keys()


def test_runner_execution_globals_resolve_including_post_loop():
    import runner
    assert not unresolved_globals(runner.execute)


def test_runner_global_check_detects_missing_post_loop_cache(monkeypatch):
    import runner
    monkeypatch.delitem(runner.__dict__, 'cache')
    assert 'cache' in unresolved_globals(runner.execute)
