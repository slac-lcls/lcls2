"""Host-side rejection tests for the new launch boundary."""
import datetime
import time
import pytest
import plan
from campaign import allocation, launch

UUIDS=['GPU-'+str(i) for i in range(4)]

def test_bootstrap_has_three_profiles_repeated():
    matrix=plan.cases()
    assert len(matrix)==11
    triples=[(c['bds'],c['gpus'],c['batch']) for c in matrix if c['stage']=='S']
    for profile in [(1,1,50),(4,1,200),(4,4,200)]:assert triples.count(profile)==3
    assert sum(plan.BUDGET.values())==4200

@pytest.mark.parametrize('stage',['E','EC','ECC'])
def test_compute_sharing_is_rejected(stage):
    with pytest.raises(ValueError):plan.placement(UUIDS,dict(bds=4,gpus=1,stage=stage))

def test_gpu_map_masks_helpers():
    assert plan.placement(UUIDS,dict(bds=4,gpus=4,stage='ECC'))['rank_uuids']==['','']+UUIDS
    assert plan.placement(UUIDS,dict(bds=1,gpus=1,stage='S'))['rank_uuids']==['','',UUIDS[0]]
    with pytest.raises(ValueError):plan.placement([UUIDS[0]]*4,dict(bds=4,gpus=4,stage='ECC'))

def record():
    end=datetime.datetime.fromtimestamp(time.time()+4200,datetime.timezone.utc).replace(tzinfo=None).isoformat()
    return 'Account=lcls_g QOS=gpu_premium NumNodes=1 TimeLimit=01:10:00 EndTime='+end

def test_deadline_is_scheduler_absolute():
    assert 4190<allocation(record())-time.time()<4210

@pytest.mark.parametrize('before,after',[('lcls_g','wrong'),('NumNodes=1','NumNodes=2'),
    ('01:10:00','24:00:00'),('gpu_premium','regular')])
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
