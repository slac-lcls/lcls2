"""Host rejection tests using tiny files; no GPU, Slurm, or payload access."""
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pytest
import audit
import campaign
import plan
import storage
import support
from qualified_support import select_affinity


def probes():
    # Two hardware threads for each of 40 physical cores.
    return [dict(rank=r,host='node',cpus=list(range(80)),
                 topology={str(c):[0,c%40] for c in range(80)}) for r in range(6)]


def test_physical_affinity_prefix():
    mapping=select_affinity(probes())
    assert [mapping[str(i)] for i in range(3)]==[list(range(3)),list(range(3,6)),list(range(6,13))]
    flat=[c for group in mapping.values() for c in group]
    assert len(flat)==len(set(c%40 for c in flat))==34


@pytest.mark.parametrize('damage',['small_mask','topology','host'])
def test_bad_probe_mapping(damage):
    rows=probes()
    if damage=='small_mask':rows[0]['cpus']=list(range(33))
    if damage=='topology':rows[1]['topology']['0']=[0,100]
    if damage=='host':rows[1]['host']='different-node'
    with pytest.raises(ValueError):select_affinity(rows)


@pytest.mark.parametrize('damage',['overlap','smt','outside'])
def test_rank_affinity_rejection(monkeypatch,damage):
    mapping=[list(range(3)),list(range(3,6)),list(range(6,13))]
    if damage=='overlap':mapping[1][0]=0
    monkeypatch.setenv('SIM10_CPU_PLAN',json.dumps(mapping));monkeypatch.setenv('SLURM_PROCID','2')
    def topology(path,*args,**kwargs):
        cpu=int(path.parent.parent.name[3:])
        return '0' if path.name=='physical_package_id' else str(cpu%12 if damage=='smt' else cpu)
    monkeypatch.setattr(Path,'read_text',topology)
    monkeypatch.setattr(os,'sched_getaffinity',lambda _:set(range(10 if damage=='outside' else 80)))
    monkeypatch.setattr(os,'sched_setaffinity',lambda *_:pytest.fail('Must reject before binding'))
    with pytest.raises(ValueError):support.affinity()


@pytest.mark.parametrize('damage',['files','directory','media'])
def test_cache_allowlist_rejected_before_io(monkeypatch,damage):
    info=dict(files=['/qualified/a'],directory='/qualified',payload_records=[])
    monkeypatch.setattr(storage,'input_info',lambda:info)
    monkeypatch.setattr(storage,'scan',lambda *_:pytest.fail('No I/O outside allowlist'))
    values=dict(files=info['files'],directory='/qualified',media='lustre',evict=True)
    values[damage]={'files':['/other/a'],'directory':'/other','media':'weka'}[damage]
    with pytest.raises(ValueError):storage.cache(**values)


@pytest.mark.parametrize('damage',['file','manifest','external','plan'])
def test_bundle_tamper_rejected(tmp_path,monkeypatch,damage):
    bundle=tmp_path/'bundle';bundle.mkdir();monkeypatch.setattr(campaign,'HERE',bundle)
    payload=bundle/'code.py';payload.write_text('pass\n')
    external=tmp_path/'external.json';external.write_text('{}')
    manifest=dict(commit='test-commit',files={'code.py':support.sha(payload)},external={str(external):support.sha(external)})
    support.write(bundle/'bundle.json',manifest)
    frozen=dict(walltime_seconds=plan.WALLTIME,budget=plan.BUDGET,cases=plan.cases(),source_commit='test-commit',bundle_sha256=support.sha(bundle/'bundle.json'))
    support.write(tmp_path/'plan.json',frozen)
    campaign.verify_bundle(tmp_path)
    if damage=='file':payload.write_text('bad\n')
    if damage=='manifest':support.write(bundle/'bundle.json',{})
    if damage=='external':external.write_text('{"changed":true}')
    if damage=='plan':
        frozen['walltime_seconds']+=1;support.write(tmp_path/'plan.json',frozen)
    with pytest.raises(ValueError):campaign.verify_bundle(tmp_path)


def test_step_timeout_terminates_process(tmp_path):
    begin=time.monotonic()
    with pytest.raises(campaign.subprocess.TimeoutExpired):
        campaign.runstep(tmp_path,'timeout',[sys.executable,'-c','import time; time.sleep(30)'],.05,time.time()+200)
    assert time.monotonic()-begin<5
    assert json.loads((tmp_path/'timeout.timing.json').read_text())['seconds']<5


def test_inventory_normalizes_cuda_and_smi_bus_ids():
    assert support.gpu_inventory('GPU-a, 00000000:AB:00.0\n')=={'GPU-a':'0000:ab:00.0'}
    assert support.normalize_pci(b'0000:ab:00.0')=='0000:ab:00.0'
    with pytest.raises(ValueError):support.gpu_inventory('GPU-a, 0000:ab:00.0\nGPU-b, 0000:ab:00.0')


def test_utc_deadline_independent_of_local_timezone(monkeypatch):
    import datetime
    monkeypatch.setenv('TZ','America/Los_Angeles');time.tzset()
    try:
        end=time.time()+4100
        utc=datetime.datetime.fromtimestamp(end,datetime.timezone.utc).replace(tzinfo=None).isoformat()
        text='Account=lcls_g QOS=premium NumNodes=1 TimeLimit=01:10:00 EndTime='
        assert abs(campaign.allocation(text+utc)-end)<.01
        local=datetime.datetime.fromtimestamp(end).isoformat()
        with pytest.raises(ValueError):campaign.allocation(text+local)
        future=datetime.datetime.fromtimestamp(end+3600,datetime.timezone.utc).replace(tzinfo=None).isoformat()
        with pytest.raises(ValueError):campaign.allocation(text+future)
    finally:
        monkeypatch.undo();time.tzset()
