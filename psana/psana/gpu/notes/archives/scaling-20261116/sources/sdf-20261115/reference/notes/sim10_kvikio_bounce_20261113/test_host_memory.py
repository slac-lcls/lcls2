import copy
import pytest
from host_memory import demand, parse_kib, snapshot, validate_host_memory


def test_actual_procfs_and_estimate():
    sample = snapshot()
    assert sample['VmRSS'] > 0 and sample['VmHWM'] >= sample['VmRSS']
    assert demand(1, 8, 16)['aggregate_estimate_bytes'] == 128 << 20
    assert demand(1, 8, 64)['aggregate_estimate_bytes'] == 512 << 20
    assert demand(4, 4, 16)['aggregate_estimate_bytes'] == 256 << 20
    assert demand(4, 4, 64)['aggregate_estimate_bytes'] == 1024 << 20


@pytest.mark.parametrize('text', ['VmRSS: 2 MB', 'VmRSS: -2 kB', 'VmHWM: 2 kB'])
def test_reject_unusable_procfs(text):
    with pytest.raises(ValueError):
        parse_kib(text, ('VmRSS',))


@pytest.mark.parametrize('kind', ['negative', 'rss', 'highwater', 'demand', 'headroom'])
def test_reject_invalid_memory_evidence(kind):
    sample = dict(VmRSS=1024, VmHWM=2048, VmLck=0, VmPin=0, mem_available_bytes=100 << 30)
    r = dict(startup=copy.deepcopy(sample), before_loop=copy.deepcopy(sample),
             after_loop=copy.deepcopy(sample), demand=demand(4, 4, 64))
    validate_host_memory(r, 4, 4, 64)
    if kind == 'negative':
        r['after_loop']['VmLck'] = -1
    elif kind == 'rss':
        r['after_loop']['VmRSS'] = 3000
    elif kind == 'highwater':
        r['after_loop']['VmHWM'] = 1024
    elif kind == 'demand':
        r['demand']['per_bd_estimate_bytes'] = 0
    else:
        r['before_loop']['mem_available_bytes'] = 1
    with pytest.raises(ValueError):
        validate_host_memory(r, 4, 4, 64)
