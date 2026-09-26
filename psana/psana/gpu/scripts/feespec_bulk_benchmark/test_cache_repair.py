"""Warm preparation may repair residency; post-timing acceptance must not."""
import ctypes
import io
import mmap

import pytest

from common import cache_inputs, prefix_residency, rewarm_prefixes
from warm_cache import warm_missing_prefix


def states(monkeypatch, fractions):
    values = iter(fractions)
    def measured(*args, **kwargs):
        fraction = next(values)
        return dict(pages=100, resident_pages=int(100*fraction), resident_fraction=fraction)
    monkeypatch.setattr('common.prefix_residency', measured)


def files(tmp_path, count=6):
    paths = [tmp_path/f's{i}.xtc2' for i in range(count)]
    for path in paths:
        path.write_bytes(bytes(4096))
    return paths, {p.name:4096 for p in paths}


def test_repair_rechecks_all_six_files_and_detects_new_eviction(tmp_path, monkeypatch):
    paths, ranges = files(tmp_path)
    states(monkeypatch, [.9139,1,1,1,1,1] + [1,1,.98,1,1,1] + [1]*6)
    calls = []
    monkeypatch.setattr('common.rewarm_prefixes', lambda paths, interleave: calls.append(paths))
    result = cache_inputs(tmp_path, 'warm', ranges=ranges)
    assert calls == [[(paths[0],4096)], [(paths[2],4096)]]
    assert result['warm_retries'] == 2 and result['resident_fraction'] == 1
    assert len(result['files']) == 6


def test_persistent_page_loss_fails_after_three_repairs(tmp_path, monkeypatch):
    paths, ranges = files(tmp_path, 1)
    states(monkeypatch, [.91]*4)
    calls = []
    monkeypatch.setattr('common.rewarm_prefixes', lambda *a: calls.append(a))
    with pytest.raises(RuntimeError, match='Warm prefix residency too low'):
        cache_inputs(tmp_path, 'warm', ranges=ranges)
    assert len(calls) == 3


@pytest.mark.parametrize('mode,prepare,fraction', [('warm',False,.98),('cold',True,.02)])
def test_failed_postcheck_and_cold_gate_never_rewarm(tmp_path, monkeypatch, mode, prepare, fraction):
    _, ranges = files(tmp_path, 1)
    states(monkeypatch, [fraction])
    monkeypatch.setattr('common.rewarm_prefixes', lambda *a: pytest.fail('unexpected repair'))
    with pytest.raises(RuntimeError, match='prefix residency'):
        cache_inputs(tmp_path, mode, prepare=prepare, ranges=ranges)


def test_repair_of_small_file_keeps_large_campaign_numa_policy(monkeypatch, tmp_path):
    import json
    calls = []
    monkeypatch.setattr('common.subprocess.run', lambda cmd, **kw: calls.append((cmd,kw)))
    paths = [(tmp_path/'s000.xtc2', 97414048)]
    rewarm_prefixes(paths, interleave=True)
    cmd, kwargs = calls[0]
    assert cmd[:2] == ['numactl', '--interleave=all']
    assert cmd[-2] == '--missing-prefixes'
    assert json.loads(cmd[-1]) == [[str(paths[0][0]), paths[0][1]]]
    assert kwargs['check']


@pytest.mark.parametrize('bits,expected', [
    ([1,0,0,1,0], [(4096,8192),(16384,17)]),
    ([0,0,0,0,0], [(0,16401)]),
    ([1,1,1,1,1], []),
])
def test_missing_ranges_coalesce_and_clip_final_page(tmp_path, monkeypatch, bits, expected):
    assert mmap.PAGESIZE == 4096
    path = tmp_path/'data'
    path.write_bytes(bytes(5*4096))
    class Libc:
        def mincore(self, address, length, vector):
            out = ctypes.cast(vector, ctypes.POINTER(ctypes.c_ubyte))
            for i, bit in enumerate(bits): out[i] = bit
            return 0
    monkeypatch.setattr(ctypes, 'CDLL', lambda *a, **kw: Libc())
    result = prefix_residency(path, 16401, missing_ranges=True)
    assert result['missing_ranges'] == expected
    assert result['resident_pages'] == sum(bits)


def test_repair_reads_only_missing_ranges_and_rejects_short_reads(monkeypatch):
    import builtins
    class Source(io.BytesIO):
        def close(self): pass
        def read(self, n):
            reads.append((self.tell(), n))
            return super().read(n)
    reads = []
    source = Source(bytes(16384))
    monkeypatch.setattr(builtins, 'open', lambda *a, **kw: source)
    monkeypatch.setattr('common.prefix_residency', lambda *a, **kw:
                        dict(missing_ranges=[(4096,4096),(12288,17)]))
    assert warm_missing_prefix('file', 12305) == 4113
    assert reads == [(4096,4096),(12288,17)]
    monkeypatch.setattr('common.prefix_residency', lambda *a, **kw:
                        dict(missing_ranges=[(16380,10)]))
    with pytest.raises(RuntimeError, match='short warm-prefix repair'):
        warm_missing_prefix('file',16390)
