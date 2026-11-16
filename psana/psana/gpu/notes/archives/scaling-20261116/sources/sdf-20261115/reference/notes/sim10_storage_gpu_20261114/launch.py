"""Run only inside an authorized, exclusive allocation after manifest review."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import threading
import time

from plan import BASE, HERE, CONTROLS, RESOURCES, BUDGET_SECONDS, matrix, placement, source_hashes
from cpu_plan import discover, plan
from hardware import inventory
from support import fixture, require, sha, write, raw, ROOT, EVENT_BYTES, histogram
from task_evidence import pilot_evidence
from storage import ROOTS, qualify
from lifecycle import job_end_time, ensure_time, copy_checked, cleanup_stages


def stop_group(child):
    for requested_signal in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, requested_signal)
        except ProcessLookupError:
            child.wait()
            return
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
        for attempt in range(30):
            try:
                os.killpg(child.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.1)
    raise RuntimeError('Process group still exists; preserve staged inputs for inspection')


def run_owned(command, log, timeout):
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        result = child.wait(timeout=timeout)
        if result:
            raise subprocess.CalledProcessError(result, command)
    finally:
        stop_group(child)


def verify_manifest(attempt, reviewed_digest):
    manifest = attempt / 'manifest.json'
    require(len(reviewed_digest) == 64 and sha(manifest) == reviewed_digest, 'Unreviewed source manifest')
    record = json.loads(manifest.read_text())
    source = attempt / 'source'
    for relative, digest in record['files'].items():
        path = source / relative
        require(path.resolve().is_relative_to(source.resolve()), 'Manifest escape')
        require(sha(path) == digest, 'Frozen source changed: ' + relative)
    receipt_path = Path(record['native_receipt'])
    require(sha(receipt_path) == record['native_receipt_sha256'], 'Native receipt changed')
    receipt = json.loads(receipt_path.read_text())
    require(receipt['status'] == 'pass' and receipt['pr_sha'] == record['pr_sha'], 'Native compatibility')
    for relative, digest in receipt['python'].items():
        require(sha(Path(receipt['source']) / 'psana/psana' / relative) == digest, 'Pinned Python changed')
    for path, digest in receipt['native'].items():
        require(sha(Path(path)) == digest, 'Native library changed')
    for path, digest in receipt['kvikio_identity'].items():
        require(sha(Path(path)) == digest, 'KvikIO installation changed')
    os.environ['SIM10_PR_SOURCE'] = receipt['source']
    os.environ['SIM10_NATIVE_RECEIPT'] = str(receipt_path)
    return record


def gpu_inventory():
    mask = os.environ['CUDA_VISIBLE_DEVICES']
    require(len(mask.split(',')) == 4, 'Require all four allocated GPUs')
    text = subprocess.check_output(['nvidia-smi', '-i', mask,
        '--query-gpu=uuid,pci.bus_id,name,memory.total', '--format=csv,noheader,nounits'], text=True)
    rows = []
    for line in text.splitlines():
        uuid, pci, name, memory = [field.strip() for field in line.split(',')]
        require(name == 'NVIDIA A100-SXM4-40GB' or name == 'A100-SXM4-40GB', 'Expected A100 40GB')
        require(int(memory) >= 40000, 'GPU memory capacity')
        rows.append(dict(uuid=uuid, pci=pci, name=name, memory_mib=int(memory)))
    rows.sort(key=lambda row: row['pci'])
    placement([row['uuid'] for row in rows], 4)
    return mask, rows


def main(attempt, reviewed_digest):
    require(os.environ.get('SLURM_JOB_ID') and int(os.environ.get('SLURM_JOB_NUM_NODES', '0')) == 1,
            'An authorized single-node allocation is required')
    require(platform.node().split('.')[0] == os.environ['SIM10_EXPECTED_NODE'], 'Unexpected node')
    require(int(os.environ['SLURM_NTASKS']) == 6 and int(os.environ['SLURM_CPUS_PER_TASK']) == 6,
            'Require six tasks, six physical CPUs per task')
    manifest = verify_manifest(attempt, reviewed_digest)
    require(manifest['node'] == os.environ['SIM10_EXPECTED_NODE'], 'Node differs from reviewed manifest')
    output = attempt / 'results'
    output.mkdir(exist_ok=False)
    stop = threading.Event()

    def terminate(*_):
        stop.set()
        sys.exit(143)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    stages = {}
    safe_cleanup = True
    completed = []
    operation = 'qualification'
    try:
        deadline = job_end_time()
        write(output / 'walltime.json', dict(job_end_epoch=deadline, checked_at=time.time(),
              expected_budget_seconds=BUDGET_SECONDS, expected_total_seconds=sum(BUDGET_SECONDS.values()),
              cleanup_margin_seconds=300, case_ceiling_seconds=900))
        ensure_time(deadline, operation, 600)
        mask, devices = gpu_inventory()
        uuids = [row['uuid'] for row in devices]
        topology = discover()
        cpu_assignment = plan(topology, 4, 4)
        require(cpu_assignment['available_physical_cores'] >= 36, 'Physical CPU allocation too small')
        os.environ['SIM10_CPU_PLAN'] = json.dumps(cpu_assignment)
        hardware = inventory(','.join(uuids))
        hardware['storage_scope'] = 'Paired Weka FFB and local md RAID0 NVMe, same allocation'
        hardware['devices'] = devices
        hardware['strace'] = subprocess.check_output(['strace', '--version'], text=True).splitlines()[0]
        write(output / 'hardware.json', hardware)
        for index, uuid in enumerate(uuids):
            operation = 'gpu-tests-%d' % index
            ensure_time(deadline, operation, 300)
            os.environ['JF_GPU_UUID'] = uuid
            with (output / ('gpu-tests-%d.log' % index)).open('x') as log:
                run_owned([sys.executable, str(BASE / 'gpu_tests.py'),
                           str(output / ('gpu-tests-%d.json' % index)),
                           str(output / ('pytest-tmp-%d' % index))], log, 300)
        fixture_manifest, streams, stamps = fixture()
        items = [(stream['stream'], kind, record) for stream in streams
                 for kind, record in stream['files'].items()]
        required_bytes = sum(record['bytes'] for stream, kind, record in items)
        storage = {}
        for media, parent in ROOTS.items():
            if media == 'nvme' and not parent.exists():
                parent.mkdir(mode=0o700)
            require(parent.is_dir() and parent.resolve() == parent and parent.stat().st_uid == os.geteuid(),
                    'Owned staging parent missing: ' + str(parent))
            stage = parent / ('sim10-storage-gpu-' + os.environ['SLURM_JOB_ID'])
            stage.mkdir(mode=0o700, exist_ok=False)
            stages[media] = stage
            storage[media] = qualify(stage, media, required_bytes)
        write(output / 'storage-qualification.json', storage)
        expected_model = ('SAMSUNG MZQL27T6HBLA-00A07' if os.environ['SIM10_EXPECTED_NODE'] == 'sdfampere036'
                          else 'KCM6XRUL7T68')
        for device in storage['nvme']['physical_inventory']:
            require(device['model'] == expected_model and device['bytes'] >= 7_000_000_000_000,
                    'NVMe model/capacity changed; re-review hardware selection')
            require(device['current_link_width'] == '4' and device['current_link_speed'].startswith('16.0')
                    and device['numa_node'] == '5', 'NVMe PCIe/NUMA topology changed; re-review placement')
        copied = []
        for media, stage in stages.items():
            operation = 'staging-' + media
            ensure_time(deadline, operation, 1800)
            stage_deadline = time.time() + 1800
            (stage / 'smalldata').mkdir()

            def copy(item):
                stream, kind, record = item
                source = Path(record['path'])
                target = (stage if kind == 'bd' else stage / 'smalldata') / source.name
                require(not target.exists(), 'Existing stage target')
                copy_checked(source, target, record['bytes'], record['sha256'], stop, stage_deadline)
                return dict(media=media, stream=stream, kind=kind, source=str(source), target=str(target),
                            bytes=record['bytes'], sha256=record['sha256'])

            with ThreadPoolExecutor(max_workers=4) as pool:
                try:
                    media_copies = list(pool.map(copy, items))
                except BaseException:
                    stop.set()
                    raise
            copied.extend(media_copies)
            info = dict(exp='mfx101555026', run=10, media=media, directory=str(stage),
                        files=[row['target'] for row in media_copies if row['kind'] == 'bd'],
                        timestamps=stamps, event_bytes=EVENT_BYTES, storage=storage[media],
                        observed_leaves=storage['nvme']['physical_leaves'],
                        generation_sha256=sha(ROOT / 'generation.json'), timestamp_sha256=raw.ts_hash(stamps))
            write(output / ('inputs-' + media + '.json'), info)
        write(output / 'provenance.json', dict(manifest=manifest, manifest_sha256=reviewed_digest,
              controls=CONTROLS, allocation_mask=mask, devices=devices, topology=topology,
              cpu_plan=cpu_assignment, source_sha256=source_hashes(), staged=copied,
              host=platform.node(), job=os.environ['SLURM_JOB_ID'],
              hardware_sha256=sha(output / 'hardware.json'),
              native_receipt=json.loads(Path(os.environ['SIM10_NATIVE_RECEIPT']).read_text())))
        for case in matrix():
            if case['name'] == matrix()[6]['name']:
                for media in ROOTS:
                    operation = 'probe-' + media
                    ensure_time(deadline, operation, 300)
                    with (output / ('probe-' + media + '.log')).open('x') as log:
                        run_owned(['numactl', '--interleave=all', sys.executable, str(HERE / 'probe.py'),
                                   str(output / ('inputs-' + media + '.json')),
                                   str(output / ('probe-' + media + '.json'))], log, 300)
            operation = case['name']
            ensure_time(deadline, operation, 900)
            os.environ.update(SIM10_GPU_PLAN=json.dumps(placement(uuids, case['gpus'])),
                              KVIKIO_NTHREADS='4', KVIKIO_TASK_SIZE=str(1 << 20),
                              KVIKIO_BOUNCE_BUFFER_SIZE=str(16 << 20))
            path = output / (case['name'] + '.json')
            runner = ([str(BASE / 'trace_runner.py')] if case['kind'] == 'pilot' else []) + [str(HERE / 'runner.py')]
            command = ['numactl', '--interleave=all', 'mpiexec', '-n', '6', '--bind-to', 'none',
                       sys.executable, *runner, '--inputs', str(output / ('inputs-' + case['media'] + '.json')),
                       '--output', str(path), '--bds', '4', '--workers', '4', '--batch', '200',
                       '--streams', '1', '--bounce-mib', '16', '--events', str(case['events']),
                       '--media', case['media'], '--gpus', str(case['gpus'])]
            with (output / (case['name'] + '-gpu.csv')).open('x') as telemetry:
                monitor = subprocess.Popen(['nvidia-smi', '-i', ','.join(uuids),
                    '--query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,clocks.sm,clocks.mem',
                    '--format=csv', '-l', '1'], stdout=telemetry, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    with path.with_suffix('.log').open('x') as log:
                        run_owned(command, log, 900)
                    require(monitor.poll() is None, 'GPU telemetry failed')
                finally:
                    stop_group(monitor)
            result = json.loads(path.read_text())
            require(result['status'] == 'pass', 'Case failed')
            if case['kind'] == 'pilot':
                write(output / (case['name'] + '-io-evidence.json'),
                      pilot_evidence(path, result, case['events'], histogram(case['events'])))
            completed.append(dict(**case, command=command, hz=result['hz']))
            write(output / 'progress.json', completed)
            print(json.dumps(completed[-1]), flush=True)
        write(output / 'complete.json', dict(status='pass', cases=completed))
    except BaseException as error:
        stop.set()
        if 'Process group still exists' in str(error):
            safe_cleanup = False
        write(output / 'campaign-status.json', dict(status='partial_not_accepted', operation=operation,
              error=repr(error), completed_cases=[case['name'] for case in completed], at=time.time()))
        raise
    finally:
        cleanup = cleanup_stages(stages, safe_cleanup)
        write(output / 'cleanup.json', cleanup)
        if sys.exc_info()[0] is None and any(not row['removed'] for row in cleanup):
            write(output / 'campaign-status.json', dict(status='cleanup_failed_not_accepted', cleanup=cleanup))
            raise RuntimeError('Stage cleanup incomplete; see cleanup.json')
    from audit import audit
    write(output / 'audit.json', audit(output))
    write(output / 'campaign-status.json', dict(status='pass', completed_cases=len(completed), at=time.time()))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve(), sys.argv[2])
