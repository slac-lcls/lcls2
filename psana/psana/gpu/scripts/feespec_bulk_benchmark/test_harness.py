import json
import pytest
from common import tier
from run import matrix
from common import prefix_residency


def test_two_rounds_reverse_variant_and_cache_order():
    rows=list(matrix())
    assert len(rows)==len(set(rows))==12
    assert rows[:3]==[('A','cold',1),('E-off','cold',1),('E-on','cold',1)]
    assert rows[6:9]==[('E-on','warm',2),('E-off','warm',2),('A','warm',2)]
    assert all(sum(v==variant and c==cache for v,c,_ in rows)==2
               for variant in ('A','E-off','E-on') for cache in ('cold','warm'))


@pytest.mark.parametrize('ssd,object_bytes,remote,accepted',[
    (4096,0,0,True),(4096,4096,0,False),(2048,0,0,False),(4096,0,4096,False)])
def test_tier_requires_full_ssd_and_no_object_or_remote_backing(monkeypatch,ssd,object_bytes,remote,accepted):
    value=[dict(path='/private/ffb/input',file_size=4096,ssd_write_cache_bytes=ssd,
                ssd_read_cache_bytes=0,object_storage_bytes=object_bytes,remote_storage_bytes=remote)]
    monkeypatch.setattr('subprocess.check_output',lambda *a,**k:json.dumps(value))
    if accepted:
        assert tier(['/private/ffb/input'])==value
    else:
        with pytest.raises(RuntimeError):tier(['/private/ffb/input'])


def test_short_run_measures_only_selected_prefix(tmp_path):
    import mmap
    path=tmp_path/'input'
    path.write_bytes(b'x'*(mmap.PAGESIZE*4))
    row=prefix_residency(path,mmap.PAGESIZE)
    assert row['bytes']==mmap.PAGESIZE and row['pages']==1
    assert row['resident_pages']==1
    with pytest.raises(ValueError):prefix_residency(path,mmap.PAGESIZE*5)
    with pytest.raises(ValueError):prefix_residency(path,0)


def test_short_baseline_balances_order():
    from quick import matrix as quick_matrix
    assert quick_matrix()==[('E-off',1),('E-on',1),('E-on',2),('E-off',2)]
