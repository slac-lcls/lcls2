"""Corruption tests for independent staging-result acceptance."""
import copy
import hashlib
import struct
import pytest
from audit import audit_case
from support import EVENT_BYTES, histogram

def fixture():
    stamps=list(range(50))
    files=['/qualified/file%d.xtc2'%i for i in range(5)]
    sizes=[524724,629664,734604,734604,734604]
    ranges={path.rsplit('/',1)[1]:dict(size=size,offsets=[i*size+100 for i in stamps]) for path,size in zip(files,sizes)}
    wanted={}
    for path,size in zip(files,sizes):
        wanted[path]=hashlib.sha256(b''.join(struct.pack('<QQQ',i,i*size+100,size) for i in stamps)).hexdigest()
    masks=[[0,1,2],[3,4,5],[6,7,8,9,10,11,12]]
    gpu=dict(rank_uuids=['','','GPU-0'],pci_by_uuid={'GPU-0':'0000:03:00.0'})
    helpers=[dict(rank=i,host='node',affinity=masks[i],timestamps=[],device=None,cube_verification=None) for i in range(2)]
    bd=dict(rank=2,host='node',affinity=masks[2],timestamps=stamps,device=dict(mask='GPU-0',pci='0000:03:00.0'),
            granted_uuids=['GPU-0'],cube_verification=None,errors=0,cpu_reads=0,
            io=dict(completed_requests=250,requests=250,completed_bytes=50*EVENT_BYTES,requested_bytes=50*EVENT_BYTES,
                    read_size_histogram=histogram(50),pressure_drains=0,range_sha256=wanted),
            callbacks=[dict(size=50,timestamps=stamps,stream=7)],sampled_memory=[dict(free=5<<30)],
            drain=dict(closed=True,pending=0,active=0,held=0,peers=1),
            task_copy=dict(bytes=0,ordinary_host_fallbacks=0,pinned_capacity_high=0),
            array_weakref_probe=dict(released=True),optional_shared_resources_absent=dict(derived=True),
            budget_peaks=[dict(peak_committed_plus_held=1<<30)])
    cache_rows = [dict(path=p, pages=100, resident=0, resident_pages=0) for p in files]
    cache_gate = dict(status='pass', max_attempts=3, budget_seconds=180, elapsed_seconds=2.,
                      attempts=[dict(number=1, scans=[dict(phase=phase, files=copy.deepcopy(cache_rows), elapsed_seconds=t)
                                for phase,t in [('evict',1.),('verify',2.)]])])
    report=dict(cube_policy='gpu_shape_finite_nonzero_v1',cube_exported=False,numerical_reference_checked=False,status='pass',stage='S',kind='full',events=50,active_gpus=1,seconds=1.,hz=50.,
                payload_gib_s=50*EVENT_BYTES/2**30,rows=helpers+[bd],
                observation=dict(cache_gate=cache_gate, conditioning=copy.deepcopy(cache_rows),
                                 initial_cache=copy.deepcopy(cache_rows), conditioning_seconds=2.))
    case=dict(stage='S',kind='full',events=50,bds=1,gpus=1,batch=50)
    return report,case,dict(timestamps=stamps,files=files),ranges,masks,gpu

def test_consistent_staging_receipt_passes():
    assert audit_case(*fixture())['status']=='pass'

@pytest.mark.parametrize('corruption', ['bytes','range','uuid','pci','duplicate','drain','headroom','publication','cache'])
def test_corrupted_staging_receipt_is_rejected(corruption):
    args=list(fixture());report=args[0];bd=report['rows'][2]
    if corruption=='bytes':bd['io']['completed_bytes']-=1
    if corruption=='range':bd['io']['range_sha256']['/qualified/file0.xtc2']='0'*64
    if corruption=='uuid':bd['device']['mask']='GPU-other'
    if corruption=='pci':bd['device']['pci']='0000:04:00.0'
    if corruption=='duplicate':bd['timestamps']=bd['timestamps'][:-1]+[0]
    if corruption=='drain':bd['drain']['held']=1
    if corruption=='headroom':bd['sampled_memory'][0]['free']=(4<<30)-1
    if corruption=='publication':bd['task_copy']['bytes']=64
    if corruption=='cache':report['observation']['initial_cache'][0]['resident_pages']=1
    with pytest.raises(ValueError):audit_case(*args)


def ecc_fixture():
    args = list(fixture())
    report, case = args[:2]
    report['stage'] = case['stage'] = 'ECC'
    bd = report['rows'][2]
    bd['task_copy']['bytes'] = case['events'] * 128
    counts = [int(i < 50) for i in range(100)]
    bd['cube_verification'] = dict(status='pass', shape=[100,32,512,1024],
        dtype='float32', bytes=100*32*512*1024*4, counts_shape=[100], counts_dtype='uint64',
        counts=counts, bins=[dict(bin=i,finite=True,nonzero=bool(c)) for i,c in enumerate(counts)],
        drained=True, cube_payload_host_bytes=0, exported=False, reference_checked=False, seconds=1.)
    return args


def test_gpu_ecc_receipt_passes_without_cube_files():
    assert audit_case(*ecc_fixture())['status'] == 'pass'


@pytest.mark.parametrize('corruption', ['shape','finite','populated','empty','counts','export','copy','drain'])
def test_bad_gpu_cube_receipt_rejected(corruption):
    args = ecc_fixture()
    receipt = args[0]['rows'][2]['cube_verification']
    if corruption == 'shape': receipt['shape'][-1] -= 1
    if corruption == 'finite': receipt['bins'][0]['finite'] = False
    if corruption == 'populated': receipt['bins'][0]['nonzero'] = False
    if corruption == 'empty': receipt['bins'][99]['nonzero'] = True
    if corruption == 'counts': receipt['counts'][0] += 1
    if corruption == 'export': receipt['exported'] = True
    if corruption == 'copy': receipt['cube_payload_host_bytes'] = 64
    if corruption == 'drain': receipt['drained'] = False
    with pytest.raises(ValueError): audit_case(*args)
