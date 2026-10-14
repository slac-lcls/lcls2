"""Reject scientifically or operationally unmatched performance pairs."""
from pathlib import Path
import importlib.util
import pytest

path=Path(__file__).resolve().parents[3]/'gpu/scripts/stage5c/pairs.py'
spec=importlib.util.spec_from_file_location('stage5c_pairs',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def sample(variant,seconds):
    return dict(variant=variant,events=200,ngpus=1,nbds=1,batch_size=20,depth=2,
        cache='warm',bulk='on',gpu_buses={'0':'bus'},pipeline_budget_gb=10,
        diagnostic=False,output_sha256='same',loop_s=seconds)


def test_same_work_accepts_both_improvement_and_regression():
    for seconds in (8,12):
        result=module.validate_pair(sample('event_loop',10),sample('batched_task',seconds))
        assert result['delta_s']==seconds-10


@pytest.mark.parametrize('key,value',[('output_sha256','different'),('batch_size',1),('cache','cold'),
    ('events',199),('pipeline_budget_gb',12),('gpu_buses',{'0':'another-bus'}),('diagnostic',True)])
def test_reject_unmatched_pair(key,value):
    after=sample('batched_task',8);after[key]=value
    with pytest.raises(AssertionError):module.validate_pair(sample('event_loop',10),after)
