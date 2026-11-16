"""Allocation-only orchestration. Never submits or deletes retained resources.

The manifest is the only schedule/node authority. Guards and both bounded ECC
acceptances precede the eight traced pilots; all pilot correctness, syscall and
CUDA diagnostic evidence is required before any of the 30 untraced full/anchor
cases. Missing parent interfaces produce an incomplete campaign, never a pass.
"""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import threading
import time

from plan import (HERE, BASE, PYTHON, WEKA, LOCAL, RECEIPT, CONTROLS, SHARED_RESERVE,
                  EVENT_BYTES, NATIVE_CAVEAT, EXPORT_ALLOWANCE, acceptance_cases,
                  matrix, placement, case_seconds, retained_bytes, historical, require, sha, write, slurm_seconds)
from prepare import verify_manifest
import node_selection


def remaining_gate(deadline, budget, pending, phases=(), now=None, extra_seconds=0):
    required = budget['safety'] + budget['final_audit']
    required += sum(budget[phase] for phase in phases)
    required += sum(case_seconds(budget, case) for case in pending) + extra_seconds
    available = deadline - (time.time() if now is None else now)
    require(available >= required, 'Incomplete campaign: remaining walltime %.1fs < %ds including reference/export/audit' %
            (available, required))
    return required


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
            continue
    child.wait(timeout=15)


def run_owned(command, log, timeout, env):
    require(timeout > 0, 'Operation budget exhausted')
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, env=env)
    try:
        result = child.wait(timeout=timeout)
        require(result == 0, 'Child failed: ' + str(command))
    finally:
        stop_group(child)


def exclusive_gate(fields, node_fields, occupants, job):
    """25.11 info_job.c emits job_share_string(shared); NONE prints NO.

    Current scontrol emits Exclusive=NODE separately. NO alone is not proof.
    See SchedMD/slurm slurm-25.11 src/scontrol/info_job.c and
    src/common/slurm_protocol_defs.c:job_share_string, versus master info_job.c.
    The old-format fallback also requires all effective CPUs and sole occupancy.
    """
    if 'Exclusive' in fields:
        require(fields['Exclusive'] == 'NODE', 'Require whole-node Exclusive=NODE, not USER/TOPO/NO')
        return 'Exclusive=NODE'
    require(fields.get('OverSubscribe') == 'NO', 'Missing recognizable whole-node evidence')
    effective = int(node_fields.get('CPUEfctv', '0'))
    require(effective > 0 and int(fields.get('NumCPUs', '0')) == effective
            and int(node_fields.get('CPUAlloc', '0')) == effective, 'Require entire effective CPU allocation')
    require(set(occupants) == {job}, 'Another allocation or incomplete occupancy evidence')
    return '25.11 NO plus entire effective CPU allocation and sole live-job occupancy; frozen script requests --exclusive'


def allocation(manifest):
    job = os.environ.get('SLURM_JOB_ID', '')
    require(job.isdigit(), 'Must run in an explicitly authorized Slurm allocation')
    host = platform.node().split('.')[0]
    for key, value in dict(SLURM_JOB_NUM_NODES='1', SLURM_NTASKS='6', SLURM_CPUS_PER_TASK='6').items():
        require(os.environ.get(key) == value, 'Allocation mismatch: ' + key)
    job_text = subprocess.check_output(['scontrol', 'show', 'job', '-o', job], text=True, timeout=30)
    fields = dict(re.findall(r'(\w+)=(\S+)', job_text))
    node_selection.selected_host(manifest, host, fields.get('NodeList'))
    require(fields.get('Partition') == 'ampere', 'Slurm partition mismatch')
    node_text = subprocess.check_output(['scontrol', 'show', 'node', '-o', host], text=True, timeout=30)
    node_fields = dict(re.findall(r'(\w+)=(\S+)', node_text))
    occupant_command = ['squeue', '--local', '-h', '-w', host,
                        '-t', 'RUNNING,COMPLETING,CONFIGURING,SUSPENDED', '-o', '%A']
    occupants = subprocess.check_output(occupant_command, text=True, timeout=30).split()
    exclusive = exclusive_gate(fields, node_fields, occupants, job)
    require(set(occupants) == {job}, 'Another live job on the pinned node')
    require(fields.get('Command') == str(HERE / 'run.sbatch'), 'Allocation did not use the frozen --exclusive job script')
    text = subprocess.check_output(['scontrol', 'show', 'partition', 'ampere', '-o'], text=True, timeout=30)
    match = re.search(r'\bMaxTime=(\S+)', text)
    require(match is not None and manifest['walltime_seconds'] <= slurm_seconds(match.group(1)), 'Partition walltime limit changed')
    mask = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    require(len(mask.split(',')) == 4, 'All four allocated GPUs must be visible')
    text = subprocess.check_output(['nvidia-smi', '-i', mask, '--query-gpu=uuid,pci.bus_id,name,memory.total',
                                    '--format=csv,noheader,nounits'], text=True, timeout=30)
    devices = []
    for line in text.splitlines():
        uuid, pci, name, memory = [field.strip() for field in line.split(',')]
        require(name in ('NVIDIA A100-SXM4-40GB', 'A100-SXM4-40GB') and int(memory) >= 40000, 'Expected A100-SXM4-40GB')
        devices.append(dict(uuid=uuid, pci=pci, name=name, memory_mib=int(memory)))
    devices.sort(key=lambda row: row['pci'])
    placement([row['uuid'] for row in devices], 4)
    return dict(job=job, host=platform.node(), selected_node=host, devices=devices, allocation_mask=mask, slurm=job_text,
                node_record=node_text, occupants=occupants, occupant_command=occupant_command,
                exclusive_evidence=exclusive)


def local_capacity(root, required):
    require(root.is_absolute() and root == root.resolve() and root.is_relative_to(LOCAL), 'Local cube directory only')
    require(root.stat().st_uid == os.geteuid() and os.access(root, os.W_OK), 'Local output ownership/access')
    info = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(root), '-o', 'SOURCE,TARGET,FSTYPE'],
                                              text=True, timeout=30))
    mount, = info['filesystems']
    require(mount['target'] == '/lscratch' and mount['fstype'] in ('xfs', 'ext4'), 'Outputs must be on node-local filesystem')
    free = shutil.disk_usage(root).free
    require(free >= required, 'Insufficient local retention capacity; no output deletion authorized')
    return dict(mount=info, free_bytes=free, required_bytes=required, checked_at=time.time())


def case_command(case, inputs, output, local):
    wrapper = [str(BASE / 'trace_runner.py')] if case['kind'] == 'pilot' else []
    return ['numactl', '--interleave=all', 'mpiexec', '-n', str(case['bds'] + 2), '--bind-to', 'none',
            str(PYTHON), *wrapper, str(HERE / 'runner.py'), '--inputs', str(inputs), '--output', str(output),
            '--bds', str(case['bds']), '--batch', str(case['batch']), '--stage', case['stage'],
            '--events', str(case['events']), '--kind', case['kind'], '--local-output', str(local)]


def validate_result(result, case, stamps, cpu, gpu, node, sample_indices):
    require(result['status'] == 'pass' and result['events'] == case['events']
            and result['stage'] == case['stage'] and result['kind'] == case['kind'], 'Case result identity/status')
    require(math.isfinite(result['hz']) and result['hz'] > 0 and math.isfinite(result['seconds'])
            and result['seconds'] > 0 and math.isclose(result['hz'] * result['seconds'], case['events']), 'Invalid completed throughput')
    require(result['rate_is_clean'] is (case['kind'] in ('full', 'anchor')), 'Instrumented/clean classification')
    rows = result['rows']
    require([row['rank'] for row in rows] == list(range(case['bds'] + 2)), 'Rank coverage')
    require(not rows[0]['timestamps'] and not rows[1]['timestamps'], 'Helper delivered events')
    merged = [stamp for row in rows[2:] for stamp in row['timestamps']]
    require(len(set(merged)) == case['events'] and sorted(merged) == stamps[:case['events']], 'Exact global disjoint identity')
    for row in rows:
        require(row['host'].split('.')[0] == node and sorted(row['affinity']) == sorted(cpu['ranks'][row['rank']]), 'Actual CPU/node placement')
    for row in rows[2:]:
        require(row['device']['mask'] == gpu['rank_uuids'][row['rank']], 'Actual GPU placement')
        count, accounting = len(row['timestamps']), row['io']
        require(accounting['requests'] == accounting['completed_requests'] == 5 * count
                and accounting['requested_bytes'] == accounting['completed_bytes'] == EVENT_BYTES * count
                and accounting['pressure_drains'] == row['cpu_reads'] == 0 and not row['errors'], 'Read/fallback accounting')
        expected_histogram = {'524724': count, '629664': count, '734604': 3 * count} if count else {}
        require(accounting['read_size_histogram'] == expected_histogram, 'Read histogram')
        require([stamp for callback in row['callbacks'] for stamp in callback['timestamps']] == row['timestamps'], 'Callback coverage')
        if case['kind'] != 'acceptance':
            require(all(callback['size'] == case['batch'] for callback in row['callbacks']), 'Unexpected split')
        tracer = row['trace_window']['tracer_pid']
        require((tracer > 0) == (case['kind'] == 'pilot'), 'Unexpected trace state')
        require(row['task_copy']['ordinary_host_fallbacks'] == 0, 'Task-output fallback')
        require(row['task_copy']['bytes'] == (0 if case['stage'] == 'S' else count * 128), 'Task-output copy bytes')
    if case['kind'] == 'acceptance':
        sizes = [callback['size'] for row in rows[2:] for callback in row['callbacks']]
        require(case['batch'] in sizes and any(0 < size < case['batch'] for size in sizes), 'Acceptance did not exercise tails')
        require(sum(len(row['acceptance']) for row in rows[2:]) == case['events'], 'Full-pixel acceptance missing')
    else:
        require(result['observation']['storage_evidence']['idle_adjusted_rx_payload_ratio'] >= 0.90,
                'Physical Weka input evidence missing')
    if case['stage'] == 'ECC':
        cube = result['cube']
        require(cube['status'] == 'complete' and cube['global_events'] == case['events']
                and cube['nbins'] == 100 and cube['active_bins'] == 99, 'Cube export incomplete')
        checks = result['numerical_checks']
        require([row['bin'] for row in checks] == list(range(100)), 'Independent every-bin validation missing')
        require(sum(row['count'] for row in checks) == case['events'] and checks[99]['count'] == 0, 'Exact bin counts')
        for row in checks:
            require(math.isfinite(row['max_bound_ratio']) and 0 <= row['max_bound_ratio'] <= 1, 'Numerical bound failed')
        if case['kind'] in ('acceptance', 'pilot'):
            require(all(row['pixels'] == 32 * 512 * 1024 for row in checks), 'All-pixel reference required before full runs')
        else:
            require(result['sample_seed'] == 20261115 and result['sample_indices'] == sample_indices, 'Frozen sampled reference missing')


def full_gate(pilots):
    require(len(pilots) == 8, 'All eight pilots must complete before full runs')
    for case, result in pilots:
        if case['stage'] == 'S':
            continue
        fields = ['expansion']
        if case['stage'] in ('EC', 'ECC'):
            fields.append('calibration')
        if case['stage'] == 'ECC':
            fields.append('cube_count_and_sum')
        for row in result['rows'][2:]:
            if not row['callbacks']:
                require(not row['timestamps'], 'Missing callbacks for owned events')
                continue
            diagnostic = row.get('cuda_intervals', {})
            require(diagnostic.get('interpretation'), 'Pilot CUDA diagnostic interpretation unavailable')
            intervals = diagnostic.get('intervals', [])
            require(set(fields) <= {item['name'] for item in intervals}, 'Required pilot CUDA diagnostics unavailable; full phase blocked')
            require(all(math.isfinite(item['milliseconds']) and item['milliseconds'] >= 0 for item in intervals), 'Invalid CUDA intervals')


def export_receipt(result, local):
    cube = result['cube']
    require(Path(cube['directory']) == local and local.resolve() == local, 'Cube export escaped local case directory')
    bins = cube['bins']
    require([row['bin'] for row in bins] == list(range(100)), 'Incomplete retained cube')
    files = []
    for row in bins:
        path = local / row['file']
        require(path.resolve().parent == local and not path.is_symlink() and path.stat().st_size == row['bytes'], 'Export file identity')
        files.append(dict(path=str(path), bytes=row['bytes'], sha256=sha(path)))
    manifest = local / 'manifest.json'
    require(json.loads(manifest.read_text())['status'] == 'complete', 'Export publication missing')
    return dict(directory=str(local), files=files, manifest_sha256=sha(manifest), retained=True,
                cleanup_authorized=False, durability='Regenerable node-local cubes retained; local retention is unguaranteed')


def main(attempt, review, review_sha):
    started = time.time()
    require(sha(review) == review_sha, 'Review receipt changed')
    manifest = verify_manifest(attempt, review)
    require(HERE == attempt / 'source/notes/sim10_compute_cube_20261115', 'Execute the frozen launcher only')
    observed = allocation(manifest)
    selected_node = observed['selected_node']
    output = attempt / 'results'
    output.mkdir(mode=0o700, exist_ok=False)
    budget, completed, pilots = manifest['budget_seconds'], [], []
    operation = 'qualification'
    stop = threading.Event()

    def interrupted(signum, frame):
        stop.set()
        raise RuntimeError('Allocation interrupted; retaining every resource')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with historical() as helpers:
            deadline = helpers.lifecycle.job_end_time()
            remaining_gate(deadline, budget, matrix(), ('qualification', 'staging', 'acceptance'))
            uuids = [row['uuid'] for row in observed['devices']]
            topology = helpers.cpu_plan.discover()
            cpu_plans = {bds: helpers.cpu_plan.plan(topology, bds, 4) for bds in (1, 4)}
            require(cpu_plans[4]['available_physical_cores'] >= 36, 'Need 36 allocated physical cores')
            hardware = helpers.hardware.inventory(','.join(uuids))
            write(output / 'hardware.json', dict(**observed, inventory=hardware, cpu_topology=topology,
                  cpu_plans=cpu_plans, gpu_plans={bds: placement(uuids, bds) for bds in (1, 4)},
                  native_caveat=NATIVE_CAVEAT, controls=CONTROLS))
            if manifest.get('node_allowlist'):
                write(output / 'node-qualification.json', node_selection.hardware_gate(
                    manifest['node_allowlist'], selected_node, hardware, observed['devices']))
            require(LOCAL.is_dir() and LOCAL.resolve() == LOCAL and LOCAL.stat().st_uid == os.geteuid(), 'Owned local parent missing')
            local = LOCAL / ('sim10-compute-cube-' + observed['job'])
            require(not local.exists(), 'Local output already exists; never overwrite/retry')
            write(output / 'capacity.json', local_capacity(LOCAL, manifest['local_required_bytes']))
            local.mkdir(mode=0o700)
            write(output / 'retention.json', dict(**manifest['retention'], directory=str(local), node=selected_node,
                  preserve_all=True, cleanup_authorized=False, warning='Node-local retention is unguaranteed'))
            env = dict(os.environ, SIM10_PR_SOURCE=manifest['pr_source'], SIM10_NATIVE_RECEIPT=str(RECEIPT),
                       SIM10_EXPECTED_NODE=selected_node, SIM10_MANIFEST=str(attempt / 'manifest.json'),
                       SIM10_LOCAL_ROOT=str(local), PS_PARALLEL='mpi', PS_FAKESTEP_FLAG='0',
                       PS_JF_SHARE_DERIVED='0', PS_GEO_SHARE='0', PS_SMD_N_EVENTS='1000',
                       KVIKIO_COMPAT_MODE='ON', KVIKIO_NTHREADS='4', KVIKIO_TASK_SIZE=str(1 << 20),
                       KVIKIO_BOUNCE_BUFFER_SIZE=str(16 << 20), PYTHONDONTWRITEBYTECODE='1')
            operation = 'staging'
            require(time.time() - started <= budget['qualification'], 'Qualification exceeded declared ceiling')
            remaining_gate(deadline, budget, matrix(), ('staging', 'acceptance'))
            stage = WEKA / ('sim10-storage-gpu-compute-' + observed['job'])
            require(WEKA.resolve() == WEKA and WEKA.stat().st_uid == os.geteuid(), 'Owned Weka parent required')
            stage.mkdir(mode=0o700, exist_ok=False)
            storage = helpers.storage.qualify(stage, 'weka', manifest['input']['required_bytes'])
            write(output / 'storage.json', dict(qualification=storage, stage=str(stage), preserve=True))
            (stage / 'smalldata').mkdir()
            stage_deadline = time.time() + budget['staging']
            staged = []
            for row in manifest['input']['files']:
                source = Path(row['path'])
                target = (stage if row['kind'] == 'bd' else stage / 'smalldata') / source.name
                helpers.lifecycle.copy_checked(source, target, row['bytes'], row['sha256'], stop, stage_deadline)
                staged.append(dict(**row, target=str(target)))
            generation = json.loads(Path(manifest['input']['generation']).read_text())
            stamps = [row['timestamp'] for row in generation['plan'] if row['service'] == 12]
            require(helpers.support.raw.ts_hash(stamps) == manifest['input']['timestamp_sha256'], 'Fixture identity changed')
            inputs = output / 'inputs.json'
            write(inputs, dict(exp='mfx101555026', run=10, media='weka', directory=str(stage),
                  files=[row['target'] for row in staged if row['kind'] == 'bd'], timestamps=stamps,
                  event_bytes=EVENT_BYTES, observed_leaves=[], storage=storage, staged=staged,
                  generation_sha256=manifest['input']['generation_sha256'], timestamp_sha256=manifest['input']['timestamp_sha256']))
            operation = 'acceptance'
            remaining_gate(deadline, budget, matrix(), ('acceptance',))
            acceptance_deadline = time.time() + budget['acceptance']
            for index, uuid in enumerate(uuids):
                regression = output / ('regression-%d.json' % index)
                regression_temp = attempt / ('regression-tmp-%d' % index)
                require(not regression_temp.exists(), 'Fresh regression basetemp required; no cleanup authority')
                with regression.with_suffix('.log').open('x') as log:
                    run_owned([str(PYTHON), str(BASE / 'gpu_tests.py'), str(regression), str(regression_temp)],
                              log, acceptance_deadline - time.time(), dict(env, JF_GPU_UUID=uuid))
                regression_result = json.loads(regression.read_text())
                require(regression_result['exitcode'] == regression_result['failed'] == regression_result['skipped'] == 0
                        and regression_result['passed'] == 31, 'Historical GPU regressions require 31 passed, zero skips')
                target = output / ('guards-%d.json' % index)
                with target.with_suffix('.log').open('x') as log:
                    run_owned([str(PYTHON), str(HERE / 'gpu_guards.py'), str(target)],
                              log, acceptance_deadline - time.time(), dict(env, JF_GPU_UUID=uuid))
                guard = json.loads(target.read_text())
                require(guard['status'] == 'pass' and len(guard['cases']) == 10
                        and all(row['pass'] is True for row in guard['cases']), 'GPU descriptor guards failed/unavailable')
            cases = acceptance_cases() + matrix()
            for index, case in enumerate(cases):
                operation = case['name']
                if index == 2:
                    write(output / 'acceptance.json', dict(status='pass', cases=completed[:2], scope='bounded CPU/GPU/tail acceptance'))
                if index == 10:
                    full_gate(pilots)
                    write(output / 'pilot-gate.json', dict(status='pass', cases=[item[0]['name'] for item in pilots]))
                pending = [item for item in cases[index:] if item['kind'] != 'acceptance']
                remaining_gate(deadline, budget, pending,
                               extra_seconds=max(0, acceptance_deadline - time.time()) if index < 2 else 0)
                local_capacity(local, retained_bytes(pending, manifest['retention'], include_acceptance=index < 2))
                require(shutil.disk_usage(attempt).free >= SHARED_RESERVE, 'Compact evidence capacity exhausted')
                for relative, digest in manifest['files'].items():
                    require(sha(attempt / 'source' / relative) == digest, 'Frozen source changed during campaign')
                cpu, gpu = cpu_plans[case['bds']], placement(uuids, case['bds'])
                child_env = dict(env, SIM10_CPU_PLAN=json.dumps(cpu), SIM10_GPU_PLAN=json.dumps(gpu))
                target, local_case = output / (case['name'] + '.json'), local / case['name']
                require(not local_case.exists(), 'Case output exists; no automatic retries')
                command = case_command(case, inputs, target, local_case)
                ceiling = acceptance_deadline - time.time() if index < 2 else case_seconds(budget, case)
                began = time.time()
                with (output / (case['name'] + '-gpu.csv')).open('x') as telemetry:
                    monitor = subprocess.Popen(['nvidia-smi', '-i', ','.join(uuids),
                        '--query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,clocks.sm,clocks.mem',
                        '--format=csv', '-l', '2'], stdout=telemetry, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        with target.with_suffix('.log').open('x') as log:
                            run_owned(command, log, ceiling, child_env)
                        require(monitor.poll() is None, 'GPU telemetry failed')
                    finally:
                        stop_group(monitor)
                result = json.loads(target.read_text())
                validate_result(result, case, stamps, cpu, gpu, selected_node, manifest['sampling']['indices'])
                if case['kind'] == 'pilot':
                    write(output / (case['name'] + '-io.json'), helpers.task_evidence.pilot_evidence(
                          target, result, case['events'], helpers.support.histogram(case['events'])))
                    pilots.append((case, result))
                if case['stage'] == 'ECC':
                    write(output / (case['name'] + '-retained.json'), export_receipt(result, local_case))
                require(time.time() - began <= ceiling, 'Case exceeded whole-operation reference/export/audit ceiling')
                completed.append(dict(**case, hz=result['hz'], result_sha256=sha(target), command=command,
                                      seconds_including_reference_export_audit=time.time() - began))
                write(output / (case['name'] + '-complete.json'), completed[-1])
            operation = 'final-audit'
            remaining_gate(deadline, budget, [])
            for relative, digest in manifest['files'].items():
                require(sha(attempt / 'source' / relative) == digest, 'Frozen source changed during final case')
            for filename, digest in manifest['read_only_files'].items():
                require(time.time() < deadline - budget['safety'], 'Final audit walltime exhausted')
                require(sha(filename) == digest, 'Read-only dependency changed during campaign')
            require(len(completed) == 40, 'Incomplete schedule')
            require(time.time() < deadline - budget['safety'], 'Final audit walltime exhausted')
            write(output / 'campaign-status.json', dict(status='execution_complete_pending_independent_results_review',
                  completed=completed, retained_local=str(local), retained_weka=str(stage),
                  native_caveat=NATIVE_CAVEAT, no_cleanup=True, finished_at=time.time()))
    except BaseException as error:
        stop.set()
        write(output / 'campaign-status.json', dict(status='incomplete_not_accepted', operation=operation,
              error=repr(error), completed=completed, no_cleanup=True,
              retention='All inputs, partial outputs and complete cubes retained; no retry or deletion', at=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempt', type=Path)
    parser.add_argument('review', type=Path)
    parser.add_argument('review_sha256')
    args = parser.parse_args()
    main(args.attempt.resolve(), args.review.resolve(), args.review_sha256)
