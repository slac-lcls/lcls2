"""Keep follow-up launches gated on physically consistent completed evidence."""
from pathlib import Path
import importlib.util
import pytest

path=Path(__file__).resolve().parents[3]/'gpu/scripts/stage5c/plausibility.py'
spec=importlib.util.spec_from_file_location('stage5c_plausibility',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def evidence():
    kernel=dict(total_valid_pixels=100,nonzero_sums=64,shape=[32,512,1024],
        device=dict(memoryClockRate=1215000,memoryBusWidth=5120),
        median_ms={'1':dict(calibrate=4.,integrate=2.5),'20':dict(calibrate=7.,integrate=5.)})
    diagnostics=[]
    for depth in (1,2):
        for variant,n in (('event_loop',1),('batched_task',20)):
            diagnostics.append(dict(point=[1,1,20,depth],variant=variant,
                calibration_ms_per_event=kernel['median_ms'][str(n)]['calibrate']/n,
                integration_ms_per_event=kernel['median_ms'][str(n)]['integrate']/n))
    review=dict(verified=True,diagnostics=diagnostics,results=[dict(point=[1,1,20,2],cache='warm',
        variants={'event_loop':dict(median_loop_s=100.),'batched_task':dict(median_loop_s=35.)})])
    return review,kernel


def test_accepts_consistent_work_even_when_batching_loses():
    review,kernel=evidence()
    review['results'][0]['variants']['batched_task']['median_loop_s']=120.
    assert module.check(review,kernel)['passed']


@pytest.mark.parametrize('change',['incomplete','no_valid_pixels','timer_mismatch','impossible_loop','unexplained_slowdown'])
def test_rejects_unresolved_measurements(change):
    review,kernel=evidence()
    if change=='incomplete':review['verified']=False
    elif change=='no_valid_pixels':kernel['total_valid_pixels']=0
    elif change=='timer_mismatch':review['diagnostics'][0]['calibration_ms_per_event']=.01
    else:review['results'][0]['variants']['event_loop']['median_loop_s']=1. if change=='impossible_loop' else 1000.
    with pytest.raises(AssertionError):module.check(review,kernel)
