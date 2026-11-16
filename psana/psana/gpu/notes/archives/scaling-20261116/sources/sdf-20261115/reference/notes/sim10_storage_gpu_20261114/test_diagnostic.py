"""CPU-only contracts; never import the MPI/CUDA runner on a login node."""
from collections import Counter
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import hashlib
import csv

import pytest

from plan import HERE, BASE, CONTROLS, RESOURCES, BUDGET_SECONDS, matrix, placement, source_hashes
from cpu_plan import plan as cpu_plan
from host_memory import demand
from support import EVENT_BYTES, histogram
from storage import disk_delta, evidence, leaf_devices, cache
from probe import requests, SIZES
from audit import validate_case
from launch import run_owned, verify_manifest
import lifecycle
import launch

UUIDS = ['GPU-test-%d' % index for index in range(4)]


def counters(timestamp, rx=0, read_bytes=0):
    return dict(nic={'nic0': dict(device='/pci/nic0', monotonic=timestamp, rx_bytes_phy=rx, tx_bytes_phy=0)},
                disks={name: dict(device='/pci/' + name, monotonic=timestamp, reads=int(read_bytes > 0),
                        sectors=read_bytes // 512, read_ms=int(read_bytes > 0), inflight=0,
                        busy_ms=int(read_bytes > 0), weighted_ms=int(read_bytes > 0)) for name in ('nvme1n1', 'nvme2n1')})


def measurement(media, payload):
    disk_bytes = ((payload + 1023) // 1024) * 512 if media == 'nvme' else 0
    return dict(idle_before=counters(1), idle_after=counters(3), before=counters(4),
                after=counters(16, rx=payload if media == 'weka' else 0, read_bytes=disk_bytes))


@pytest.mark.parametrize('gpus,counts', [(1, [4]), (2, [2, 2]), (4, [1, 1, 1, 1])])
def test_gpu_groups(gpus, counts):
    result = placement(UUIDS, gpus)
    assert result['rank_uuids'][:2] == ['', '']
    assert list(result['peers'].values()) == counts
    assert result['rank_uuids'][2] == UUIDS[0]


@pytest.mark.parametrize('uuids,gpus', [(UUIDS[:3], 1), ([UUIDS[0]] * 4, 2), (UUIDS, 3), (['0', '1', '2', '3'], 4)])
def test_bad_gpu_groups(uuids, gpus):
    with pytest.raises(ValueError):
        placement(uuids, gpus)


def test_matrix_and_cpu_demand():
    cases = matrix()
    assert len(cases) == len({case['name'] for case in cases}) == 30
    assert Counter(case['kind'] for case in cases) == dict(pilot=6, full=18, anchor=6)
    assert Counter((case['media'], case['gpus']) for case in cases if case['kind'] == 'full') == {
        (media, gpus): 3 for media in ('weka', 'nvme') for gpus in (1, 2, 4)}
    assert all(case['events'] == (4000 if case['kind'] == 'pilot' else 100000) for case in cases)
    topology = [dict(cpu=index, socket=index % 2, core=index // 2) for index in range(36)]
    assignment = cpu_plan(topology, 4, 4)
    assert assignment['required_physical_cores'] == RESOURCES['physical_cores_required'] == 34
    assert len({cpu for rank in assignment['ranks'] for cpu in rank}) == 34
    with pytest.raises(ValueError):
        cpu_plan(topology[:33], 4, 4)


@pytest.mark.parametrize('media', ['weka', 'nvme'])
def test_physical_evidence(media):
    payload = EVENT_BYTES * 4000
    record = measurement(media, payload)
    result = evidence(record, media, payload, 10)
    assert result['idle_adjusted_' + ('rx' if media == 'weka' else 'disk') + '_payload_ratio'] >= 0.90


@pytest.mark.parametrize('fault', ['short', 'reset', 'identity', 'idle', 'network', 'missing', 'bracket'])
def test_reject_invalid_nvme_evidence(fault):
    payload = EVENT_BYTES * 4000
    record = measurement('nvme', payload)
    if fault == 'short':
        record['after'] = counters(16)
    elif fault == 'reset':
        record['before']['disks']['nvme1n1']['reads'] = 100
    elif fault == 'identity':
        record['after']['disks']['nvme1n1']['device'] = '/other'
    elif fault == 'idle':
        record['idle_after'] = counters(3, read_bytes=payload)
    elif fault == 'network':
        record['after']['nic']['nic0']['rx_bytes_phy'] = payload
    elif fault == 'missing':
        del record['after']['disks']['nvme1n1']
    elif fault == 'bracket':
        record['after']['disks']['nvme1n1']['monotonic'] = 5
    with pytest.raises(ValueError):
        evidence(record, 'nvme', payload, 10)


def test_weka_needs_network():
    with pytest.raises(ValueError):
        evidence(measurement('nvme', EVENT_BYTES * 4000), 'weka', EVENT_BYTES * 4000, 10)


def test_leaf_only_topology(tmp_path):
    for name in ('md0', 'nvme1n1', 'nvme2n1', 'sda'):
        (tmp_path / name / 'slaves').mkdir(parents=True)
    for name in ('nvme1n1', 'nvme2n1'):
        (tmp_path / 'md0/slaves' / name).symlink_to(tmp_path / name)
    assert leaf_devices('md0', tmp_path) == ['nvme1n1', 'nvme2n1']
    with pytest.raises(ValueError):
        leaf_devices('sda', tmp_path)
    (tmp_path / 'nvme1n1/slaves/md0').symlink_to(tmp_path / 'md0')
    with pytest.raises(ValueError):
        leaf_devices('md0', tmp_path)


def test_owned_cache_scope(tmp_path, monkeypatch):
    import storage
    monkeypatch.setattr(storage, 'ROOTS', dict(nvme=tmp_path))
    stage = tmp_path / 'sim10-storage-gpu-test'
    stage.mkdir(mode=0o700)
    target = stage / 'payload'
    target.write_bytes(b'payload')
    monkeypatch.setattr(storage.raw, 'prefix_residency', lambda path, length: dict(path=str(path), resident_pages=0))
    assert cache([target], stage, 'nvme')[0]['resident_pages'] == 0
    outside = tmp_path / 'foreign'
    outside.write_bytes(b'foreign')
    link = stage / 'escape'
    link.symlink_to(outside)
    with pytest.raises(ValueError):
        cache([link], stage, 'nvme')
    with pytest.raises(ValueError):
        cache([], stage, 'nvme')
    stage.chmod(0o755)
    with pytest.raises(ValueError):
        cache([target], stage, 'nvme')


def test_probe_sizes_and_nonoverlap():
    result = list(requests([size * 3 for size in SIZES], count=3))
    assert len(result) == 15 and sum(row[2] for row in result) == EVENT_BYTES * 3
    for index, size in enumerate(SIZES):
        assert [(offset, length) for stream, offset, length in result if stream == index] == [(step * size, size) for step in range(3)]
    with pytest.raises(ValueError):
        list(requests([1] * 5, count=3))


def example_case(media='nvme', gpus=2):
    case = next(case for case in matrix() if case['kind'] == 'pilot' and case['media'] == media and case['gpus'] == gpus)
    topology = [dict(cpu=index, socket=index % 2, core=index // 2) for index in range(36)]
    assignment = cpu_plan(topology, 4, 4)
    gpu_plan = placement(UUIDS, gpus)
    devices = [dict(uuid=uuid, pci='00000000:%02x:00.0' % index) for index, uuid in enumerate(UUIDS)]
    runtime = {'psana': dict(path='/pr/psana/psana/__init__.py', sha256='digest')}
    provenance = dict(host='node', job='1', controls=CONTROLS, topology=topology, cpu_plan=assignment, devices=devices,
                      source_sha256={'file': 'digest'}, native_receipt=dict(source='/pr', python={'__init__.py': 'digest'},
                      native={}, kvikio_identity={'kvikio': 'digest'}))
    info = dict(files=['/stage/%d' % index for index in range(5)], timestamps=list(range(1, 4001)))
    rows = []
    for rank in range(6):
        sample = dict(VmRSS=1, VmHWM=1, VmPin=0, VmLck=0, mem_available_bytes=100 << 30)
        row = dict(rank=rank, host='node', role=('smd0' if rank == 0 else 'eb' if rank == 1 else 'bd'),
                   runtime_files=runtime, cpu_plan=assignment, affinity=assignment['ranks'][rank],
                   inherited_affinity=list(range(36)), numa_policy='policy: interleave',
                   host_memory=dict(demand=demand(4, 4, 16), startup=sample, before_loop=sample, after_loop=sample),
                   timestamps=[], device=None, threads_before=dict(clock_ticks=100, threads=[1]),
                   threads_after=dict(clock_ticks=100, threads=[1]))
        if rank >= 2:
            uuid = gpu_plan['rank_uuids'][rank]
            local_stamps = info['timestamps'][(rank - 2) * 1000:(rank - 1) * 1000]
            row.update(device=dict(mask=uuid, pci=devices[UUIDS.index(uuid)]['pci'], name='A100-SXM4-40GB',
                kvikio='26.10.0', compat_mode='ON', kvikio_identity={'kvikio': 'digest'},
                effective=dict(num_threads=4, task_size=1 << 20, bounce_buffer_size=16 << 20, auto_direct_io_read=False)),
                timestamps=local_stamps, trace_window=dict(tracer_pid=1, start=1, end=2), reader_task_sizes=[None],
                callbacks=[dict(size=200, stream=123, timestamps=local_stamps[offset:offset+200]) for offset in range(0, 1000, 200)],
                io=dict(requests=5000, completed_requests=5000, requested_bytes=EVENT_BYTES * 1000,
                        completed_bytes=EVENT_BYTES * 1000, read_size_histogram=histogram(1000),
                        issue_groups_calls=5, pressure_drains=0), cpu_reads=0,
                drain=[dict(closed=True, pending=0, active=0, held=0, peers=4 // gpus)],
                memory=dict(before_free=40 << 30))
        rows.append(row)
    measured = measurement(media, EVENT_BYTES * 4000)
    measured.update(seconds=10, storage_evidence=evidence(measured, media, EVENT_BYTES * 4000, 10))
    measured.update({key: [dict(path=path, resident_pages=0) for path in info['files']] for key in ('conditioning', 'initial_cache')})
    result = dict(status='pass', host='node', job='1', source_sha256=provenance['source_sha256'],
                  settings=dict(**{key: CONTROLS[key] for key in ('bds', 'workers', 'batch', 'streams', 'bounce_mib')},
                                media=media, gpus=gpus, events=4000),
                  framework=dict(batch_size=200, streams=1, budget_gib=6, bulk_target_bytes=524288, smd_events=1000),
                  events=4000, bytes=EVENT_BYTES * 4000, seconds=10, hz=400, gib_s=EVENT_BYTES * 4000 / 2**30 / 10,
                  gpu_plan=gpu_plan, rows=rows, measurement=measured)
    return result, case, info, provenance


@pytest.mark.parametrize('media', ['weka', 'nvme'])
@pytest.mark.parametrize('gpus', [1, 2, 4])
def test_valid_case_audit(media, gpus):
    result, case, info, provenance = example_case(media, gpus)
    assert validate_case(result, case, info, provenance)['hz'] == 400


@pytest.mark.parametrize('fault', ['gpu', 'pci', 'peers', 'memory', 'affinity', 'split', 'timestamp', 'read', 'cold', 'source', 'trace', 'runtime'])
def test_case_audit_negative(fault):
    result, case, info, provenance = example_case()
    worker = result['rows'][2]
    if fault == 'gpu': worker['device']['mask'] = UUIDS[-1]
    elif fault == 'pci': worker['device']['pci'] = '0000:ff:00.0'
    elif fault == 'peers': worker['drain'][0]['peers'] = 4
    elif fault == 'memory': worker['memory']['before_free'] = 15 << 30
    elif fault == 'affinity': worker['affinity'] = []
    elif fault == 'split': worker['callbacks'][0]['size'] = 100
    elif fault == 'timestamp': worker['timestamps'][0] = 0
    elif fault == 'read': worker['io']['completed_bytes'] -= 1
    elif fault == 'cold': result['measurement']['initial_cache'][0]['resident_pages'] = 1
    elif fault == 'source': result['source_sha256'] = {}
    elif fault == 'trace': worker['trace_window']['tracer_pid'] = 0
    elif fault == 'runtime': worker['runtime_files'] = {}
    with pytest.raises(ValueError):
        validate_case(result, case, info, provenance)


def test_mask_before_runtime_and_no_submission():
    source = (HERE / 'runner.py').read_text()
    assert source.index("os.environ['CUDA_VISIBLE_DEVICES']=") < source.index('activate()') < source.index('from mpi4py')
    assert '--submit' not in (HERE / 'prepare.py').read_text()
    assert {str(HERE / filename) for filename in ('runner.py', 'storage.py', 'probe.py', 'run.sbatch')} <= source_hashes().keys()


def test_review_gate(tmp_path):
    (tmp_path / 'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='Unreviewed'):
        verify_manifest(tmp_path, 'not-reviewed')


def test_run_owned_timeout(tmp_path):
    with (tmp_path / 'process.log').open('w') as log:
        with pytest.raises(subprocess.TimeoutExpired):
            run_owned([sys.executable, '-c', 'import time; time.sleep(10)'], log, 0.05)


def test_empirical_walltime_budget():
    with (BASE / 'results/runs.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    seconds = [float(row['seconds']) for row in rows if row['phase'] != 'pilot'
               and row['bds'] == '4' and row['bounce_mib'] == '16']
    assert len(seconds) == 2
    assert 2 * max(seconds) + 50 <= BUDGET_SECONDS['full_and_anchors'] / 24
    assert sum(BUDGET_SECONDS.values()) == 9540 < 3 * 3600


@pytest.mark.parametrize('remaining,allowed', [(1200, True), (1199, False), (-1, False)])
def test_walltime_guard(remaining, allowed):
    if allowed:
        assert lifecycle.ensure_time(1000 + remaining, 'case', 900, now=1000) == remaining
    else:
        with pytest.raises(RuntimeError, match='Insufficient walltime'):
            lifecycle.ensure_time(1000 + remaining, 'case', 900, now=1000)


def test_job_deadline_from_environment(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_END_TIME', '2000')
    monkeypatch.setattr(lifecycle.time, 'time', lambda: 1000)
    assert lifecycle.job_end_time() == 2000
    monkeypatch.setenv('SLURM_JOB_END_TIME', '999')
    with pytest.raises(ValueError, match='future'):
        lifecycle.job_end_time()


def test_job_deadline_scheduler_fallback(monkeypatch):
    monkeypatch.delenv('SLURM_JOB_END_TIME', raising=False)
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    monkeypatch.setattr(lifecycle.time, 'time', lambda: 1000)
    commands = []

    def query(command, **kwargs):
        commands.append((command, kwargs))
        return '2099-01-01T00:00:00\n'

    monkeypatch.setattr(lifecycle.subprocess, 'check_output', query)
    assert lifecycle.job_end_time() > 1000
    assert commands == [(['squeue', '-h', '-j', '123', '-o', '%e'], dict(text=True, timeout=10))]
    monkeypatch.setenv('SLURM_JOB_END_TIME', 'Unknown')
    with pytest.raises(ValueError):
        lifecycle.job_end_time()


@pytest.mark.parametrize('fault', ['none', 'stop', 'deadline', 'digest', 'size'])
def test_bounded_stage_copy(tmp_path, fault):
    source = tmp_path / 'source'
    target = tmp_path / 'target'
    payload = b'bounded staging test'
    source.write_bytes(payload)
    stop = threading.Event()
    if fault == 'stop':
        stop.set()
    deadline = time.time() - 1 if fault == 'deadline' else time.time() + 60
    checksum = 'bad' if fault == 'digest' else hashlib.sha256(payload).hexdigest()
    size = len(payload) + int(fault == 'size')
    if fault == 'none':
        lifecycle.copy_checked(source, target, size, checksum, stop, deadline)
        assert target.read_bytes() == payload
    else:
        with pytest.raises((RuntimeError, ValueError)):
            lifecycle.copy_checked(source, target, size, checksum, stop, deadline)
        if fault in ('stop', 'deadline'):
            assert not target.exists()


def test_stage_copy_stop_during_copy(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    target = tmp_path / 'target'
    source.write_bytes(b'test')
    calls = []
    stop = threading.Event()

    def stopped():
        calls.append(True)
        return len(calls) >= 3

    monkeypatch.setattr(stop, 'is_set', stopped)
    with pytest.raises(RuntimeError, match='interrupted'):
        lifecycle.copy_checked(source, target, 4, 'unused', stop, time.time() + 60)
    assert len(calls) == 3


def test_cleanup_failure_preserves_other_receipts(tmp_path, monkeypatch):
    stages = {media: tmp_path / media for media in ('weka', 'nvme')}
    for stage in stages.values():
        stage.mkdir()

    def check(stage, media):
        if media == 'weka':
            raise ValueError('ownership changed')
        return stage

    monkeypatch.setattr(lifecycle, 'owned_stage', check)
    rows = lifecycle.cleanup_stages(stages, True)
    assert len(rows) == 2 and not rows[0]['removed'] and rows[1]['removed']
    assert rows[0]['error'] == 'ownership changed'
    assert stages['weka'].exists() and not stages['nvme'].exists()
    preserved = lifecycle.cleanup_stages({'weka': stages['weka']}, False)
    assert not preserved[0]['removed'] and stages['weka'].exists()


def test_main_walltime_refusal_records_partial(tmp_path, monkeypatch):
    for key, value in dict(SLURM_JOB_ID='123', SLURM_JOB_NUM_NODES='1', SIM10_EXPECTED_NODE='node',
                           SLURM_NTASKS='6', SLURM_CPUS_PER_TASK='6').items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(launch.platform, 'node', lambda: 'node')
    monkeypatch.setattr(launch, 'verify_manifest', lambda *_: dict(node='node'))
    monkeypatch.setattr(launch, 'job_end_time', lambda: time.time() + 100)
    original_handlers = {signum: launch.signal.getsignal(signum)
                         for signum in (launch.signal.SIGTERM, launch.signal.SIGINT)}
    try:
        with pytest.raises(RuntimeError, match='Insufficient walltime'):
            launch.main(tmp_path, 'unused')
    finally:
        for signum, handler in original_handlers.items():
            launch.signal.signal(signum, handler)
    status = json.loads((tmp_path / 'results/campaign-status.json').read_text())
    assert status['status'] == 'partial_not_accepted' and status['completed_cases'] == []
    assert json.loads((tmp_path / 'results/cleanup.json').read_text()) == []
    assert not (tmp_path / 'results/complete.json').exists()
