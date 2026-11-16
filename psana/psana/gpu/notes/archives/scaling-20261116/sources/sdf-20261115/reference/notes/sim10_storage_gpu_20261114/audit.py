"""Recompute diagnostic correctness and media evidence before comparing curves."""
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys

from plan import CONTROLS, RESOURCES, matrix, placement
from support import require, sha, histogram, EVENT_BYTES, raw, fixture
from cpu_plan import plan as cpu_plan
from host_memory import validate_host_memory
from storage import evidence
from task_evidence import pilot_evidence
from prepare_runtime import PR_SHA, validate_configs


def pci_key(value):
    domain, bus, device = value.upper().split(':')
    return int(domain, 16), int(bus, 16), device


def validate_case(result, case, info, provenance):
    require(result['status'] == 'pass' and result['host'] == provenance['host'] and result['job'] == provenance['job'],
            'Run identity')
    require(result['source_sha256'] == provenance['source_sha256'], 'Harness changed')
    settings = result['settings']
    for key in ('bds', 'workers', 'batch', 'streams', 'bounce_mib'):
        require(settings[key] == CONTROLS[key], 'Control mismatch: ' + key)
    require(settings['media'] == case['media'] and settings['gpus'] == case['gpus']
            and settings['events'] == case['events'], 'Case identity')
    require(result['framework'] == dict(batch_size=200, streams=1, budget_gib=6, bulk_target_bytes=524288, smd_events=1000),
            'Framework controls')
    count = case['events']
    require(result['events'] == count and result['bytes'] == EVENT_BYTES * count, 'Global totals')
    require(result['seconds'] > 0 and math.isclose(result['hz'], count / result['seconds'])
            and math.isclose(result['gib_s'], result['bytes'] / 2**30 / result['seconds']), 'Rate')
    devices = provenance['devices']
    gpu_plan = placement([device['uuid'] for device in devices], case['gpus'])
    require(result['gpu_plan'] == gpu_plan, 'Placement plan')
    cpu_assignment = cpu_plan(provenance['topology'], 4, 4)
    require(cpu_assignment == provenance['cpu_plan'], 'CPU plan')
    rows = result['rows']
    require(len(rows) == 6 and [row['rank'] for row in rows] == list(range(6)), 'MPI ranks')
    require([row['role'] for row in rows] == ['smd0', 'eb', 'bd', 'bd', 'bd', 'bd'], 'MPI roles')
    timestamps = [timestamp for row in rows[2:] for timestamp in row['timestamps']]
    require(sorted(timestamps) == info['timestamps'][:count] and len(set(timestamps)) == count, 'Exact event partition')
    require(all(not row['timestamps'] and row['device'] is None for row in rows[:2]), 'Helper payload')
    require(len({row['device']['pci'] for row in rows[2:]}) == case['gpus'], 'Distinct physical GPUs')
    receipt = provenance['native_receipt']
    runtime = rows[0]['runtime_files']
    require(bool(runtime), 'Runtime identity missing')
    for entry in runtime.values():
        path = Path(entry['path'])
        if path.suffix == '.py':
            source = Path(receipt['source']) / 'psana/psana'
            require(path.is_relative_to(source), 'Runtime Python escaped pinned source')
            digest = receipt['python'].get(str(path.relative_to(source)))
        else:
            digest = receipt['native'].get(str(path))
        require(digest is not None and entry['sha256'] == digest, 'Runtime digest')
    for rank, row in enumerate(rows):
        require(row['runtime_files'] == runtime and row['host'] == provenance['host'], 'Runtime/host mismatch')
        require(row['cpu_plan'] == cpu_assignment and row['affinity'] == cpu_assignment['ranks'][rank]
                and set(row['affinity']) <= set(row['inherited_affinity']), 'CPU placement')
        require('policy: interleave' in row['numa_policy'], 'NUMA policy')
        validate_host_memory(row['host_memory'], 4, 4, 16)
        for key in ('threads_before', 'threads_after'):
            require(row[key]['clock_ticks'] > 0 and row[key]['threads'], 'CPU thread evidence')
        if rank < 2:
            continue
        uuid = gpu_plan['rank_uuids'][rank]
        peers = gpu_plan['peers'][uuid]
        device = row['device']
        inventory = next(item for item in devices if item['uuid'] == uuid)
        require(device['mask'] == uuid and pci_key(device['pci']) == pci_key(inventory['pci']), 'Actual physical placement')
        require('A100-SXM4-40GB' in device['name'] and device['kvikio'].startswith('26.10.')
                and device['compat_mode'] == 'ON', 'GPU/runtime versions')
        require(device['effective'] == dict(num_threads=4, task_size=1 << 20, bounce_buffer_size=16 << 20, auto_direct_io_read=False),
                'KvikIO controls')
        require(device['kvikio_identity'] == receipt['kvikio_identity'], 'KvikIO binary/header identity')
        require(row['reader_task_sizes'] in ([None], [1 << 20]), 'Reader task override')
        window = row['trace_window']
        require((window['tracer_pid'] > 0) == (case['kind'] == 'pilot') and 0 < window['start'] < window['end'], 'Tracing mode')
        local_count = len(row['timestamps'])
        require(local_count > 0 and row['timestamps'] == sorted(set(row['timestamps'])), 'Local event identity')
        callbacks = row['callbacks']
        require([timestamp for callback in callbacks for timestamp in callback['timestamps']] == row['timestamps'], 'Callback identity')
        require(len(callbacks) == local_count // 200 and all(callback['size'] == len(callback['timestamps']) == 200 for callback in callbacks),
                'Unexpected batch split')
        require(len({callback['stream'] for callback in callbacks}) == 1, 'CUDA stream count')
        io = row['io']
        require(io['requests'] == io['completed_requests'] == 5 * local_count
                and io['requested_bytes'] == io['completed_bytes'] == EVENT_BYTES * local_count, 'Read totals')
        require(io['read_size_histogram'] == histogram(local_count) and io['issue_groups_calls'] == len(callbacks)
                and io['pressure_drains'] == row['cpu_reads'] == 0, 'Read shape/admission')
        require(row['drain'] == [dict(closed=True, pending=0, active=0, held=0, peers=peers)], 'GPU sharing/drain')
        require(row['memory']['before_free'] >= (peers * 6 + 4) * 2**30, 'Per-device aggregate budget')
    measurement = result['measurement']
    for key in ('conditioning', 'initial_cache'):
        require([row['path'] for row in measurement[key]] == info['files']
                and all(row['resident_pages'] == 0 for row in measurement[key]), 'Cold host cache')
    require(measurement['seconds'] == result['seconds'], 'Timing bracket')
    measured = evidence(measurement, case['media'], result['bytes'], result['seconds'])
    require(measured == measurement['storage_evidence'], 'Storage evidence changed')
    return dict(**case, hz=result['hz'], gib_s=result['gib_s'], seconds=result['seconds'],
                bd_events=[len(row['timestamps']) for row in rows[2:]], storage=measured)


def audit(root):
    def read(name):
        return json.loads((root / name).read_text())

    provenance = read('provenance.json')
    require(provenance['controls'] == CONTROLS, 'Campaign controls')
    manifest = provenance['manifest']
    require(manifest['controls'] == CONTROLS and manifest['resources'] == RESOURCES and manifest['cases'] == matrix(),
            'Reviewed manifest controls')
    require(sha(root.parent / 'manifest.json') == provenance['manifest_sha256']
            and json.loads((root.parent / 'manifest.json').read_text()) == manifest, 'Reviewed manifest identity')
    source_root = root.parent / 'source'
    for relative, digest in manifest['files'].items():
        path = source_root / relative
        require(provenance['source_sha256'].get(str(path)) == digest and sha(path) == digest, 'Frozen source mismatch')
    fixture_manifest, streams, expected_stamps = fixture()
    expected_files = {record['path']: record for stream in streams for record in stream['files'].values()}
    require(len(provenance['staged']) == 2 * len(expected_files), 'Staged file count')
    for media in ('weka', 'nvme'):
        copied = [row for row in provenance['staged'] if row['media'] == media]
        require({row['source'] for row in copied} == set(expected_files), 'Staged source set')
        for row in copied:
            expected = expected_files[row['source']]
            require(row['bytes'] == expected['bytes'] and row['sha256'] == expected['sha256'], 'Staged source identity')
    receipt = provenance['native_receipt']
    require(receipt['status'] == 'pass' and receipt['pr_sha'] == PR_SHA, 'Pinned runtime')
    validate_configs(receipt['build_configurations'])
    require(sha(root / 'hardware.json') == provenance['hardware_sha256'], 'Hardware receipt')
    require(len(provenance['devices']) == 4, 'GPU allocation inventory')
    for index in range(4):
        tests = read('gpu-tests-%d.json' % index)
        require(tests['exitcode'] == 0 and tests['passed'] == 31 and tests['failed'] == tests['skipped'] == 0, 'GPU acceptance')
    cleanup = read('cleanup.json')
    require(len(cleanup) == 2 and {row['media'] for row in cleanup} == {'weka', 'nvme'}
            and all(row['removed'] for row in cleanup), 'Owned stage cleanup')
    complete = read('complete.json')
    cases = matrix()
    require(complete['status'] == 'pass' and len(complete['cases']) == len(cases), 'Incomplete campaign')
    results = []
    for case, completed in zip(cases, complete['cases']):
        require(all(completed[key] == value for key, value in case.items()), 'Campaign order')
        result = read(case['name'] + '.json')
        info = read('inputs-' + case['media'] + '.json')
        require(raw.ts_hash(info['timestamps']) == info['timestamp_sha256'] and len(info['timestamps']) == 100000,
                'Accepted timestamp reference')
        require(info['timestamps'] == expected_stamps, 'Wrong event reference')
        require(info['media'] == case['media'] and info['event_bytes'] == EVENT_BYTES, 'Input identity')
        rows = [row for row in provenance['staged'] if row['media'] == case['media'] and row['kind'] == 'bd']
        require([row['target'] for row in rows] == info['files'], 'Staged paths')
        results.append(validate_case(result, case, info, provenance))
        if case['kind'] == 'pilot':
            observed = pilot_evidence(root / (case['name'] + '.json'), result, case['events'], histogram(case['events']))
            require(read(case['name'] + '-io-evidence.json') == observed, 'Pilot syscall evidence')
        telemetry = (root / (case['name'] + '-gpu.csv')).read_text()
        require(all(device['uuid'] in telemetry for device in provenance['devices']), 'Per-GPU telemetry coverage')
    for media in ('weka', 'nvme'):
        probe = read('probe-' + media + '.json')
        require(probe['status'] == 'pass' and probe['media'] == media and probe['depth'] == 16
                and probe['bytes'] == EVENT_BYTES * 16000 and probe['requests'] == 80000, 'Probe controls')
        info = read('inputs-' + media + '.json')
        for key in ('conditioning', 'initial_cache'):
            require([row['path'] for row in probe[key]] == info['files']
                    and all(row['resident_pages'] == 0 for row in probe[key]), 'Probe cold cache')
        require(probe['seconds'] > 0 and math.isclose(probe['gib_s'], probe['bytes'] / 2**30 / probe['seconds']), 'Probe rate')
        require(probe['evidence'] == evidence(probe['counters'], media, probe['bytes'], probe['seconds']), 'Probe physical evidence')
    samples = defaultdict(list)
    baselines = {(row['repeat'], row['media']): row['hz'] for row in results if row['kind'] == 'full' and row['gpus'] == 1}
    for row in results:
        if row['kind'] == 'full':
            samples[(row['media'], row['gpus'])].append(row)
    summary = []
    for (media, gpus), rows in sorted(samples.items()):
        rates = [row['hz'] for row in rows]
        require(len(rows) == 3, 'Repeat count')
        summary.append(dict(media=media, gpus=gpus, bds_per_gpu=4 // gpus,
                       mean_hz=statistics.mean(rates), stdev_hz=statistics.stdev(rates), samples_hz=rates,
                       within_repeat_same_media_speedup=[row['hz'] / baselines[row['repeat'], media] for row in rows]))
    return dict(status='pass', host=provenance['host'], cases=results, summary=summary,
                limitations=['Host-cache-cold is not proof of cold Weka servers or SSD internal caches.',
                             'NVMe is not assumed faster or storage-unlimited; inspect probes, anchors, variability and utilization.',
                             'GPU/CPU telemetry supports hypotheses, not proof of PCIe or DRAM saturation.',
                             'These are staging-only results, not calibrated XPP cube throughput.'])


if __name__ == '__main__':
    print(json.dumps(audit(Path(sys.argv[1])), indent=2))
