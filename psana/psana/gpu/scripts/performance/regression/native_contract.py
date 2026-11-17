"""Offline contracts and Linux observations for the Perlmutter raw adapter.

No psana/CUDA imports here. Native qualification is separate from these checks.
"""
import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import sys
import time

SCHEMA = 'psana-pr-regression-v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write('\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def artifact(request, key):
    entry = request['artifacts'][key]
    require(file_hash(entry['path']) == entry['sha256'], 'artifact changed: ' + key)
    return Path(entry['path']).resolve()


def input_ranges(entry):
    """Exact, ordered, disjoint L1 ranges; v1 single-range receipts remain valid."""
    if 'ranges' in entry:
        require('range_start' not in entry, 'ambiguous input range representation')
        ranges = entry['ranges']
    else:
        ranges = [[entry['range_start'], entry['range_bytes']]]
    require(isinstance(ranges, list) and ranges, 'empty input ranges')
    end, total = -1, 0
    for row in ranges:
        require(isinstance(row, list) and len(row) == 2 and
                all(type(v) is int for v in row), 'invalid input range')
        start, length = row
        require(start >= 0 and length > 0 and start >= end, 'overlapping/invalid input ranges')
        end, total = start + length, total + length
        require(end <= entry.get('size', end), 'range exceeds input size')
    require(total == entry['range_bytes'], 'input range byte total differs')
    return ranges


def load_contract(request):
    require(request['schema'] == SCHEMA and request['site'] == 'perlmutter', 'Perlmutter adapter only')
    c = request['contract']
    require(digest(c) == request['contract_sha256'] and c['case'] == request['case'], 'request contract mismatch')
    require(request['phase'] in ('correctness', 'aa', 'ab', 'profile'), 'unknown phase')
    case, data, settings = c['case'], c['dataset'], c['settings']
    require(data['experiment'] == 'mfx100848724' and data['run'] == 51 and data['detector'] == 'jungfrau', 'run51 Jungfrau only')
    require(data['selected_streams'] == [3, 5, 7, 8, 9], 'unexpected selected streams')
    require(0 < data['events'] <= data['available_events'] == 17872, 'invalid event prefix')
    require(case['workload'] in ('cpu-raw', 'gpu-raw') and case['bds'] in (1, 2) and case['ebs'] == 1, 'unsupported topology/workload')
    require(case['iterator'] in ('events', 'steps'), 'unknown iterator')
    require(case['gpus'] == int(case['workload'] == 'gpu-raw'), 'one shared GPU or CPU only')
    require(case['workload'] != 'gpu-raw' or case['iterator'] == 'events', 'GPU steps outside default scope')
    fixed = dict(batch_size=20, n_gpu_streams=1, gpu_bulk_read=True, gpu_bulk_target_bytes=1048576,
                 gpu_memory_budget_gb=8, kvikio_workers=4, kvikio_task_size=1048576,
                 kvikio_bounce_buffer_bytes=16777216, kvikio_compat_mode='ON', gds=False,
                 d2h_policy='none for clean raw timing', calibration=None, ps_smd_n_events=1000,
                 cpu_policy='explicit physical-core affinity per rank; freeze in site adapter',
                 metrics=['setup', 'first_batch', 'loop', 'process', 'requested_bytes', 'nic_rx',
                          'nic_tx', 'host_memory', 'gpu_memory', 'pinned_memory'])
    require(settings == fixed, 'unsupported settings; qualify an adapter change explicitly')
    require(c['environment'] == dict(PS_FAKESTEP_FLAG='0', PS_SRV_NODES='0', PS_EB_NODES='1', MPICH_GPU_SUPPORT_ENABLED='0'), 'unsupported environment')
    require(c['cache_policy'] == 'client-cold', 'only verified client-cold supported')
    require(c['warmup']['policy'] == 'separate-process-before-cache-conditioning' and
            0 < c['warmup']['events'] <= data['events'], 'invalid separate warmup')
    site = read(artifact(request, 'native_site'))
    require(site['schema'] == 'perlmutter-raw-site-v1', 'unknown site manifest')
    require(site['job_id'] == os.environ.get('SLURM_JOB_ID'), 'allocation differs from frozen site manifest')
    require(site['node'] == os.uname().nodename, 'controller must run on the frozen compute node')
    require(0 < site['step_timeout_seconds'] < 3600, 'invalid native timeout')
    require(2 * site['step_timeout_seconds'] + 120 < request['timeout_seconds'], 'controller timeout leaves no cleanup headroom')
    require(re.fullmatch(r'GPU-[0-9a-f-]{36}', site['gpu_uuid']) is not None, 'full GPU UUID required')
    cpus = site['cpu_affinity'][str(case['bds'])]
    require(len(cpus) == case['bds'] + 2 and all(cpus), 'one affinity per MPI rank required')
    reserved = site.get('step_cpus_per_task', len(cpus[0]))
    require(type(reserved) is int and len(cpus[0]) <= reserved <= 64, 'invalid step CPU reservation')
    flat = [n for group in cpus for n in group]
    require(all(type(n) is int and n >= 0 for n in flat) and len(set(flat)) == len(flat), 'overlapping/invalid CPUs')
    require(all(len(group) == len(cpus[0]) for group in cpus), 'equal CPU allowance required')
    reference = read(artifact(request, 'event_reference'))
    events = reference['events']  # timestamp, native step index, selected-stream L1 datagram bytes
    require(reference['schema'] == 'run51-raw-reference-v1' and len(events) == data['events'], 'wrong event reference')
    require(all(len(e) == 3 and all(type(x) is int for x in e) and e[0] > 0 and e[1] >= 0 and e[2] > 0 for e in events), 'invalid event reference records')
    timestamps = [e[0] for e in events]
    require(timestamps == sorted(set(timestamps)), 'duplicate/out-of-order reference')
    require(digest(timestamps) == data['event_sha256'], 'reference event digest differs')
    require(sum(e[2] for e in events) == data['requested_bytes'], 'reference byte denominator differs')
    manifest = read(artifact(request, 'dataset_manifest'))
    require(manifest['schema'] in ('run51-private-input-v1', 'run51-private-input-v2') and manifest['private_copy'] is True, 'private input manifest required')
    require(manifest['root'] == data['sites']['perlmutter'], 'input root differs')
    require(manifest['event_reference_sha256'] == request['artifacts']['event_reference']['sha256'], 'input/reference association differs')
    require([f['stream'] for f in manifest['files']] == data['selected_streams'], 'manifest stream set differs')
    require(sum(f['bigdata']['range_bytes'] for f in manifest['files']) == data['requested_bytes'], 'range denominator differs')
    for item in manifest['files']:
        require(('ranges' in item['bigdata']) == (manifest['schema'] == 'run51-private-input-v2'), 'range schema differs')
        for key in ('bigdata', 'smd'):
            entry = item[key]
            path = Path(entry['path'])
            require(path.is_absolute() and Path(manifest['root']).resolve() in path.resolve().parents, 'input outside private root')
            expected = 'mfx100848724-r0051-s%03d-c000%s.xtc2' % (item['stream'], '.smd' if key == 'smd' else '')
            expected_path = Path(manifest['root']) / ('smalldata' if key == 'smd' else '') / expected
            require(path == expected_path, 'input layout/filename differs')
            stat = path.stat()
            require(not path.is_symlink() and stat.st_nlink == 1 and stat.st_uid == os.getuid(), 'cache conditioning requires owned private copies, no links')
            for name, actual in (('size', stat.st_size), ('mtime_ns', stat.st_mtime_ns), ('device', stat.st_dev), ('inode', stat.st_ino)):
                require(entry[name] == actual, 'input stat changed: ' + str(path))
            require(re.fullmatch('[0-9a-f]{64}', entry['sha256']) is not None, 'input hash receipt missing')
            if key == 'smd':
                require(file_hash(path) == entry['sha256'], 'SMD changed')
            else:
                input_ranges(entry)
    runtimes = [read(artifact(request, k)) for k in ('baseline_runtime', 'candidate_runtime')]
    shared = ('python', 'python_version', 'environment', 'dependencies', 'native_libraries',
              'cupy_version', 'kvikio_version', 'cuda_runtime_version', 'cuda_driver_version')
    require(runtimes[0].get('path_policy', 'legacy') == runtimes[1].get('path_policy', 'legacy'), 'runtime path policies differ')
    for key in shared:
        require(runtimes[0][key] == runtimes[1][key], 'baseline/candidate runtime environment differs: ' + key)
    matches = [r for r in runtimes if r['commit'] == request['runtime_commit'] and r['install'] == request['runtime']['install']]
    require(matches and all(r == matches[0] for r in matches), 'runtime not uniquely pinned')
    runtime = matches[0]
    require(runtime['schema'] == 'psana-native-runtime-v1' and runtime['files'], 'runtime file manifest missing')
    require(runtime['api_base'] == '199575205', 'adapter API qualification differs')
    return site, reference, manifest, runtime


def check_selected_files(datasource, manifest):
    for attribute, key in (('xtc_files', 'bigdata'), ('smd_files', 'smd')):
        actual = [str(Path(p).resolve()) for p in getattr(datasource, attribute)]
        expected = [str(Path(f[key]['path']).resolve()) for f in manifest['files']]
        require(actual == expected, 'selected ' + attribute + ' differ in path/order')


def native_environment(request, runtime):
    # Retain scheduler/Cray transport environment; replace workload/runtime overrides.
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PS_', 'PSANA_', 'KVIKIO_', 'CUDA_', 'CUPY_', 'OMP_'))
           and k not in ('PYTHONPATH', 'PYTHONHOME', 'LD_PRELOAD', 'SIT_PSDM_OFFSITE')}
    permitted = {'PATH', 'LD_LIBRARY_PATH', 'CUDA_HOME', 'CUDA_PATH', 'CONDA_PREFIX', 'LCLS_CALIB_HTTP'}
    require(set(runtime['environment']) == permitted, 'runtime environment must pin all six supported keys')
    env.update(runtime['environment'])
    policy = runtime.get('path_policy', 'legacy')
    require(policy in ('legacy', 'isolated-install-v1'), 'unknown runtime path policy')
    if policy == 'isolated-install-v1':
        env['PATH'] = str(Path(runtime['install'])/'bin') + ':' + env['PATH']
        env['LD_LIBRARY_PATH'] = str(Path(runtime['install'])/'lib') + ':' + env['LD_LIBRARY_PATH']
    env.update(request['contract']['environment'])
    env.update(PS_PARALLEL='mpi', PS_SMD_N_EVENTS='1000', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=runtime['pythonpath'], OMP_NUM_THREADS='1', KVIKIO_NTHREADS='4',
               KVIKIO_TASK_SIZE='1048576', KVIKIO_BOUNCE_BUFFER_SIZE='16777216', KVIKIO_COMPAT_MODE='ON')
    return env


def runtime_evidence(runtime):
    import psana
    require(Path(sys.executable).resolve() == Path(runtime['python']).resolve(), 'wrong Python executable')
    require(sys.version.split()[0] == runtime['python_version'], 'wrong Python version')
    package = Path(psana.__file__).resolve().parent
    require(package == Path(runtime['package']).resolve() and Path(runtime['install']).resolve() in package.parents, 'psana import outside pinned install')
    if runtime.get('path_policy') == 'isolated-install-v1':
        install = Path(runtime['install']).resolve()
        expected_install = runtime['install_files']
        actual_install = {str(p.relative_to(install)) for p in install.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
        require(actual_install == set(expected_install), 'installed support file set differs')
        for relative, sha in expected_install.items():
            path = install/relative
            require(install in path.resolve().parents and file_hash(path) == sha, 'installed support file changed: ' + relative)
        for name, module in tuple(sys.modules.items()):
            if name == 'psalg' or name.startswith('psalg.'):
                path = getattr(module, '__file__', None)
                require(path and install in Path(path).resolve().parents, 'mixed psalg imports')
        owned_libraries = {Path(p).name for p in expected_install if '.so' in Path(p).name}
        for line in Path('/proc/self/maps').read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5].startswith('/') and Path(fields[5]).name in owned_libraries:
                require(install in Path(fields[5]).resolve().parents, 'mixed installed native library')
    expected = runtime['files']
    actual_files = {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file() and (p.suffix == '.py' or '.so' in p.name)}
    require(actual_files == set(expected), 'installed psana file set differs')
    for relative, sha in expected.items():
        require(file_hash(package / relative) == sha, 'installed psana file changed: ' + relative)
    loaded = {}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name == 'psana' or name.startswith('psana.'):
            if path:
                path = Path(path).resolve()
                require(package in path.parents and str(path.relative_to(package)) in expected, 'mixed psana imports')
                loaded[name] = str(path)
    dependencies = {}
    for name, module in tuple(sys.modules.items()):
        roots = ('numpy', 'cupy', 'cupyx', 'kvikio', 'mpi4py') + (('cupy_backends',) if runtime.get('path_policy') == 'isolated-install-v1' else ())
        if name.split('.')[0] in roots:
            path = getattr(module, '__file__', None)
            if path:
                path = str(Path(path).resolve())
                require(path in runtime['dependencies'] and file_hash(path) == runtime['dependencies'][path], 'unqualified dependency: ' + path)
                dependencies[path] = runtime['dependencies'][path]
    libraries = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/') and any(s in Path(fields[5]).name for s in ('libmpi', 'libcuda', 'libnvrtc', 'libcufile', 'libkvikio')):
            path = str(Path(fields[5]).resolve())
            if path in libraries:
                continue
            require(path in runtime['native_libraries'] and file_hash(path) == runtime['native_libraries'][path], 'unqualified mapped native library: ' + path)
            libraries[path] = runtime['native_libraries'][path]
    require(any('libmpi' in p for p in libraries), 'MPI library not observed')
    return dict(python=sys.version, executable=sys.executable, package=str(package), loaded=loaded,
                libraries=libraries, dependencies=dependencies, manifest_commit=runtime['commit'])


def cache_ranges(manifest, evict=False):
    """mincore only the frozen event ranges; count resident bytes intersecting pages.

    MAP_PRIVATE writable mappings allow an address without reading/faulting data.
    posix_fadvise is an attempt, never the proof. No system-wide cache operation.
    """
    page = mmap.PAGESIZE
    records = []
    for item in manifest['files']:
        entry = item['bigdata']
        for start, length in input_ranges(entry):
            aligned = start // page * page
            span = start - aligned + length
            with open(entry['path'], 'rb') as stream:
                if evict:
                    os.posix_fadvise(stream.fileno(), aligned, ((span + page - 1) // page) * page, os.POSIX_FADV_DONTNEED)
                mapping = mmap.mmap(stream.fileno(), span, flags=mmap.MAP_PRIVATE, prot=mmap.PROT_READ | mmap.PROT_WRITE, offset=aligned)
            vector = (ctypes.c_ubyte * ((span + page - 1) // page))()
            anchor = ctypes.c_char.from_buffer(mapping)
            try:
                libc = ctypes.CDLL(None, use_errno=True)
                status = libc.mincore(ctypes.c_void_p(ctypes.addressof(anchor)), ctypes.c_size_t(span), ctypes.byref(vector))
                if status != 0:
                    error = ctypes.get_errno()
                    raise OSError(error, os.strerror(error))
                resident = sum(max(0, min(aligned + (i+1)*page, start+length) - max(aligned+i*page, start))
                               for i, value in enumerate(vector) if value & 1)
            finally:
                del anchor
                mapping.close()
            records.append(dict(path=entry['path'], start=start, bytes=length, resident_bytes=resident))
    return dict(policy='client-cold', verified=True, files=records, measured_bytes=sum(r['bytes'] for r in records),
                resident_bytes_before=sum(r['resident_bytes'] for r in records), monotonic=time.monotonic())


def cold_cache(manifest):
    observations = [cache_ranges(manifest, evict=True), cache_ranges(manifest)]
    require(all(r['resident_bytes_before'] == 0 for r in observations), 'measured ranges are not client-cold')
    return dict(observations=observations, scope='selected L1 bigdata ranges after setup; SMD/server caches uncontrolled')


def nic_snapshot(names):
    values, unavailable = {}, {}
    for name in names:
        require(re.fullmatch('[a-zA-Z0-9_.-]+', name) is not None, 'invalid NIC interface')
        try:
            values[name] = {direction: int(Path('/sys/class/net', name, 'statistics', direction + '_bytes').read_text()) for direction in ('rx', 'tx')}
        except OSError as error:
            unavailable[name] = str(error)
    return dict(monotonic=time.monotonic(), interfaces=values, unavailable=unavailable)


def nic_delta(before, after):
    if before['unavailable'] or after['unavailable'] or not before['interfaces'] or before['interfaces'].keys() != after['interfaces'].keys():
        return dict(rx_bytes=None, tx_bytes=None)
    deltas = {k: [after['interfaces'][n][k] - before['interfaces'][n][k] for n in before['interfaces']] for k in ('rx', 'tx')}
    require(all(d >= 0 for values in deltas.values() for d in values), 'NIC counter reset')
    return {k+'_bytes': sum(v) for k, v in deltas.items()}


def check_rows(rows, reference, case):
    require([r['rank'] for r in rows] == list(range(case['bds'] + 2)), 'missing/duplicate ranks')
    physical, timestamps, steps = set(), [], []
    for row in rows:
        require(row['role'] == ('smd0' if row['rank'] == 0 else 'eb' if row['rank'] == 1 else 'bd'), 'wrong MPI role')
        cores = set(tuple(x) for x in row['physical_cores'])
        require(cores and not physical.intersection(cores), 'overlapping physical cores')
        physical.update(cores)
        events = row['timestamps']
        require(events == sorted(set(events)), 'duplicate/out-of-order per-rank events')
        require(bool(events) == (row['role'] == 'bd'), 'idle BD or events on service rank')
        timestamps.extend(events)
        steps.extend(row['steps'])
    require(sorted(timestamps) == [e[0] for e in reference], 'event loss/duplication/identity mismatch')
    if case['iterator'] == 'steps':
        require(sorted(steps) == [[e[0], e[1]] for e in reference], 'step identity mismatch')
    expected_bytes = sum(e[2] for e in reference)
    if case['workload'] == 'cpu-raw':
        require(sum(r['cpu_raw_bytes'] for r in rows) == expected_bytes, 'CPU datagram byte count mismatch')
    else:
        for r in rows[2:]:
            io = r['io']
            require(io and io['total_requests'] > 0 and io['fallback'] and io['bulk'], 'no successful fallback bulk I/O')
            require(io['requested_bytes'] == io['total_bytes'] == io['useful_bytes'], 'incomplete/amplified GPU reads')
            require(r['callback_events'] == len(r['timestamps']), 'GPU callback work mismatch')
        require(sum(r['io']['requested_bytes'] for r in rows[2:]) == expected_bytes, 'GPU read byte count mismatch')
    return dict(events=len(timestamps), event_sha256=digest(sorted(timestamps)), requested_bytes=expected_bytes)
