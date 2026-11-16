"""Frozen Weka-only campaign contract; importing this module performs no I/O.

Runner: --inputs JSON --output JSON --bds 1|4 --batch 50|200
--stage S|E|EC|ECC --events 201|1003|4000|100000 --kind acceptance|pilot|full|anchor
--local-output ABSLOCAL. SIM10_CPU_PLAN and SIM10_GPU_PLAN are per case.
The runner must recondition owned Weka files AFTER setup/reference reads,
before its common MPI timing barrier, using historical storage.cache/evidence.

Acceptance: historical gpu_tests.py OUTPUT BASETEMP (31 pass, zero skips),
then gpu_guards.py OUTPUT with JF_GPU_UUID on each allocated GPU,
then two untraced ECC runner cases (201/1003 events, 1/4 BDs, batches 50/200).
Results use the parent's status/hz/events, rows (affinity/device/timestamps/io),
observation, cube and numerical_checks schema. Pilot syscall evidence uses the
unchanged trace_runner and task_evidence. Before full runs, E/EC/ECC pilots
must additionally provide rows[2:].cuda_intervals.intervals with name and
milliseconds for expansion, calibration, cube_count_and_sum as applicable.
Missing diagnostics stop before full runs; no overlap claim is synthesized.

Budget JSON: positive integer qualification/staging/acceptance/final_audit/
safety seconds, plus cases[kind][setup|processing|export|reference|audit].
basis={kind: conservative_unmeasured|measured_compute, justification: TEXT}.
These are reviewed ceilings, not predicted throughput or staging extrapolation.
Optional retention JSON supplies acceptance_bytes, reference_workspace_bytes,
reserve_bytes capacity allowances only. Every cube is retained; local storage
retention is unguaranteed. Compact hashes/checks/provenance stay on shared scratch
for durable result summaries; no retention date or user approval is invented.
No file deletion, migration, submission, or numerical acceptance is implied.
"""
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = HERE.parent / 'sim10_kvikio_bounce_20261113'
STORAGE = HERE.parent / 'sim10_storage_gpu_20261114'
SCRATCH = Path('/sdf/scratch/users/m/monarin/gpu-validation')
RECEIPT = SCRATCH / 'sim10-kvikio-bounce-20261113/attempt01/native-compatibility.json'
RECEIPT_SHA = '7b39133dcec657409256ae44a71d3fad162d90d3cee4f1c8f2de8cfd2f146cca'
PR_SHA = '67c327a148a08b2aaf972e67e1e02e367c2b298f'
PYTHON = SCRATCH / 'kvikio-2610-validation-20261015-r1/venv/bin/python'
CALIBRATION = Path('/sdf/data/lcls/ds/prj/prjdatpsana/results/synthetic_mfx/r0001/stage1-20261103')
WEKA = Path('/sdf/data/lcls/drpsrcf/ffb/users/monarin')
LOCAL = Path('/lscratch/monarin')
EVENT_BYTES = 3358200
CUBE_BYTES = 100 * 32 * 512 * 1024 * 8 + 800
EXPORT_ALLOWANCE = CUBE_BYTES + (64 << 20)
SHARED_RESERVE = 2 << 30
DEFAULT_RETENTION = dict(acceptance_bytes=2 * EXPORT_ALLOWANCE,
                         reference_workspace_bytes=4 << 30, reserve_bytes=10 << 30)
STAGES = ('S', 'E', 'EC', 'ECC')
CONTROLS = dict(workers=4, streams=1, task_bytes=1 << 20, bounce_mib=16,
                bulk_bytes=512 << 10, budget_gib=6, smd_events=1000,
                compat_mode='ON', auto_direct_io_read=False, d2h_pinned_bytes=64 << 20,
                publication_limit_bytes=64 << 10, bins=100, active_bins=99,
                PS_JF_SHARE_DERIVED='0', PS_GEO_SHARE='0', media='weka')
RESOURCES = dict(ntasks=6, cpus_per_task=6, physical_cores=36, gpus=4,
                 memory='192G', partition='ampere', exclusive=True, nodes=1)
NATIVE_CAVEAT = ('Inherited PR298 receipt has different psana/meson.build hashes '
                 'from the frozen native build. Python/Git checks do not establish '
                 'native rebuild compatibility for compute; no new build is authorized. '
                 'Independent implementation review must explicitly cover this path.')
CONFIG_MISMATCHES = {'psana/meson.build': dict(
    pr='4cd078217825f316d8db0ef62782ac5d8a2a3d42283b839240bfc984efe9f1ae',
    frozen='27ae0bce919c99bfb7705747074f7d6413f23d3fffcf5aa6ba65741602149452')}
HISTORICAL = {
    'sim10_kvikio_bounce_20261113/gpu_tests.py': '23157c6b56899908124ead406c4313922fbe8344ebfb9fb0735f2414cd833d1a',
    'sim10_storage_gpu_20261114/storage.py': '79a38ff6b9fff9dda7844cc70a6360939f50b937d74b3ad8714a0bb062344c77',
    'sim10_storage_gpu_20261114/lifecycle.py': '09eeb16db83e3577ab7efcd71a3f7392cd7d2e57176840bc8e81e9e3bb6c3e2d',
    'sim10_kvikio_bounce_20261113/support.py': '57a3ae78564f5d4133ac2482e4951058b346a62af076a3d93de8ee7d84866a8b',
    'sim10_kvikio_bounce_20261113/runtime.py': 'f1ceb2a2cdd5f3868fdd86880d0dfbc925ef51cfdb47f764ee4d05607ef56af2',
    'sim10_kvikio_bounce_20261113/trace_runner.py': 'cc51a461f857b0effeedd694d0d2503c1931d20e325270e19b5dad2d2f72dfae',
    'sim10_kvikio_bounce_20261113/cpu_plan.py': 'fa6a756b67c1e29371d195e89e08264d49268374d0b270ecbf3de5fbfce510be',
    'sim10_kvikio_bounce_20261113/hardware.py': 'ddbf8acf9ddd7ee6f512d6d38b4b53fc9f456a52c1ba8f225b1ee5bd9d099856',
    'sim10_kvikio_bounce_20261113/task_evidence.py': 'd9c8108156791ed7c4a084a6d8c243507761ec71db8cc3db2b0088a34f762099',
    'sim10_kvikio_bounce_20261113/bulk_accounting.py': '160afc1dbbb4b7a85130e13585346b13cb020733f162b82a36458228802bc46e',
    'sim10_kvikio_bounce_20261113/task_runtime.py': '0157be63682cbcefebd608e0faf01b0d9fde69a8ef3b78eb993dc06b194b3a03',
    'sim10_kvikio_bounce_20261113/host_memory.py': 'e4142f35478934c917adf7c15739e840ade9764f0722b7233f1c9e51b0927c8d',
    'sim10_kvikio_bounce_20261113/prepare_runtime.py': '980744cef65fb16827f458b60cd35b996027ec2a5d5df9a465f9650951e24089',
    'jf_raw_cache_20261109/common.py': 'd4ffdde96c09c6895e350bbd6deb6860543bcbaffa927f3a3be03bb3d41f5c63',
    'jf_raw_cache_20261109/reference.json': '94625e5b7efb92ba38e3a60050d688a86f912c946a219f61315b15f890f0d9ce',
    'jf_raw_cache_20261109/runtime.sh': '4fe181ef41dfb56565956aae8ec0718658c331292c1fc0340f770b2e56cfe524',
    'jf_cold_ffb_20261109/run.py': 'a946811519d497629684169d8ca081abcae4e6ca136ec9fff06ab35c7a34fd36',
    'mfx_kvikio_read_20261108/network_watch.py': '61abd903d9b89a7caf1891dbda3eb2fd690a7f498a8d3b7325a0d06538bcfc79',
    'mfx_kvikio_read_20261108/plan_reads.py': '88055e287d050d9e99a9f9f2434ec239e463d1293e0bcdca603a5dad5e7e663a',
    'mfx_expand10_performance_20261106/fixture_check.py': '626be609a09207e37559b81f7147b37799290b0a936a2e9ed1e951482441d3ce',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, record):
    with Path(path).open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')


def matrix():
    cases = []

    def add(kind, repeat, bds, stage):
        cases.append(dict(name='%02d-%s-r%d-bd%d-%s' % (len(cases), kind, repeat, bds, stage),
                          kind=kind, repeat=repeat, bds=bds, gpus=bds,
                          batch=50 if bds == 1 else 200, stage=stage,
                          events=4000 if kind == 'pilot' else 100000, media='weka'))

    for bds in (1, 4):
        for stage in STAGES:
            add('pilot', 0, bds, stage)
    for repeat in range(1, 4):
        order = STAGES[repeat - 1:] + STAGES[:repeat - 1]
        for bds in ((1, 4) if repeat % 2 else (4, 1)):
            for stage in order:
                add('full', repeat, bds, stage)
            add('anchor', repeat, bds, 'S')
    return cases


def placement(uuids, bds):
    require(len(uuids) == len(set(uuids)) == 4 and bds in (1, 4), 'Four unique GPUs; one/four BDs')
    require(all(value.startswith('GPU-') for value in uuids), 'Physical UUIDs required')
    ranks = ['', ''] + uuids[:bds]
    return dict(gpus=bds, rank_uuids=ranks, peers=dict(Counter(ranks[2:])), allocation_uuids=uuids)


def acceptance_cases():
    return [dict(name='acceptance-bd%d-ECC' % bds, kind='acceptance', repeat=0,
                 bds=bds, gpus=bds, batch=50 if bds == 1 else 200, stage='ECC',
                 events=201 if bds == 1 else 1003, media='weka') for bds in (1, 4)]


def case_seconds(budget, case):
    return sum(budget['cases'][case['kind']].values())


def slurm_seconds(value):
    require(re.fullmatch(r'(\d+-)?\d+:\d{2}:\d{2}', value) is not None,
            'Finite Slurm MaxTime evidence required: ' + value)
    days, clock = value.split('-') if '-' in value else ('0', value)
    hours, minutes, seconds = map(int, clock.split(':'))
    require(minutes < 60 and seconds < 60, 'Invalid Slurm time')
    return int(days) * 86400 + hours * 3600 + minutes * 60 + seconds


def validate_budget(budget, walltime, partition_max):
    require(set(budget) == {'qualification', 'staging', 'acceptance', 'final_audit', 'safety', 'cases', 'basis'},
            'Incomplete walltime budget')
    require(set(budget['cases']) == {'pilot', 'full', 'anchor'}, 'Missing case budgets')
    for parts in budget['cases'].values():
        require(set(parts) == {'setup', 'processing', 'export', 'reference', 'audit'}, 'Budget must include reference/export')
    values = [budget[key] for key in ('qualification', 'staging', 'acceptance', 'final_audit', 'safety')]
    values += [value for parts in budget['cases'].values() for value in parts.values()]
    require(all(type(value) is int and value > 0 for value in values), 'Positive integer ceilings required')
    require(budget['basis']['kind'] in ('conservative_unmeasured', 'measured_compute')
            and bool(budget['basis']['justification'].strip()), 'No staging-only extrapolation; explain compute/reference estimates')
    total = sum(budget[key] for key in ('qualification', 'staging', 'acceptance', 'final_audit', 'safety'))
    total += sum(case_seconds(budget, case) for case in matrix())
    require(type(walltime) is int and type(partition_max) is int and total <= walltime <= partition_max,
            'Budget exceeds requested walltime or observed partition maximum')
    return total


def validate_retention(record):
    require(set(record) == {'acceptance_bytes', 'reference_workspace_bytes', 'reserve_bytes'}, 'Capacity allowances only')
    for key in ('acceptance_bytes', 'reference_workspace_bytes', 'reserve_bytes'):
        require(type(record[key]) is int and record[key] > 0, 'Explicit local capacity allowances required')
    require(record['reserve_bytes'] >= 10 << 30, 'At least 10 GiB local reserve')
    require(record['acceptance_bytes'] >= 2 * EXPORT_ALLOWANCE, 'Retain both acceptance cubes too')


def retained_bytes(cases, retention, include_acceptance=True):
    exports = sum(case['stage'] == 'ECC' for case in cases)
    return (exports * EXPORT_ALLOWANCE + retention['reference_workspace_bytes']
            + retention['reserve_bytes'] + (retention['acceptance_bytes'] if include_acceptance else 0))


def source_hashes():
    paths = {HERE.parent / relative for relative in HISTORICAL}
    for relative, digest in HISTORICAL.items():
        require(sha(HERE.parent / relative) == digest, 'Historical helper changed: ' + relative)
    for filename in ('runner.py', 'gpu_guards.py', 'plan.py', 'prepare.py', 'launch.py', 'run.sbatch', 'test_campaign.py',
                     'stages.py', 'reference.py', 'mpi_cube.py'):
        require((HERE / filename).is_file(), 'Parent integration incomplete: ' + filename)
    paths.update(path for path in HERE.rglob('*') if path.is_file()
                 and path.suffix in ('.py', '.sbatch', '.json', '.md') and '__pycache__' not in path.parts)
    paths.update((HERE.parent / 'mfx_expand10_stage2_20261105').glob('*.py'))
    for relative in (
        'sim10_compute_cube_plan_20261115/PLAN.md', 'sim10_compute_cube_plan_20261115/response-02.md',
        'sim10_kvikio_bounce_20261113/kvikio-source-contract.json',
        'mfx_expand10_100k_20261109/acceptance.json', 'mfx_expand10_100k_20261109/files.json',
        'mfx_expand10_100k_20261109/verification.json', 'mfx_expand10_stage2_20261105/adapter.py',
        'mfx_expand10_stage2_20261105/fixture.py', 'mfx_expand10_stage2_20261105/verify.py',
        'mfx_expand10_stage2_20261105/gpu_accept.py', 'synthetic_mfx_stage3_20261104/xtc_fixture.py',
        'mfx_expand10_stage3a_20261105/cube.py'):
        paths.add(HERE.parent / relative)
    paths.add(REPO / 'psana/psana/gpu/examples/jungfrau_calibration.py')
    require(all(not path.is_symlink() and path.resolve().is_relative_to(REPO) for path in paths), 'Source escape')
    return {str(path.relative_to(REPO)): sha(path) for path in sorted(paths)}


@contextmanager
def historical():
    """Task-local aliases for unchanged helpers; never import another campaign's plan.

    Keep this context active while using helpers (some import lazily). The parent
    runner may use the same context, including storage.cache and cpu.plan.
    """
    aliases = ('plan', 'support', 'common', 'plan_reads', 'network_watch', 'fixture_check',
               'cpu_plan', 'storage', 'lifecycle', 'runtime', 'hardware', 'task_evidence',
               'bulk_accounting', 'host_memory', 'task_runtime')
    original = {name: sys.modules.get(name) for name in aliases}
    search = sys.path[:]
    try:
        for relative, digest in HISTORICAL.items():
            require(sha(HERE.parent / relative) == digest, 'Historical helper changed: ' + relative)
        for name in aliases:
            sys.modules.pop(name, None)
        shim = ModuleType('plan')
        shim.BASE = BASE
        sys.modules['plan'] = shim
        sys.path[:0] = [str(BASE), str(HERE.parent / 'mfx_expand10_performance_20261106')]
        loaded = {}
        for name in ('support', 'cpu_plan', 'storage', 'lifecycle', 'runtime', 'hardware',
                     'task_evidence', 'bulk_accounting', 'host_memory', 'task_runtime'):
            path = (STORAGE if name in ('storage', 'lifecycle') else BASE) / (name + '.py')
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
        yield SimpleNamespace(**loaded)
    finally:
        sys.path[:] = search
        for name, module in original.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


if __name__ == '__main__':
    print(json.dumps(dict(controls=CONTROLS, resources=RESOURCES, cases=matrix(),
                         native_caveat=NATIVE_CAVEAT, availability='unconfirmed; historical target sdfampere036'), indent=2))
