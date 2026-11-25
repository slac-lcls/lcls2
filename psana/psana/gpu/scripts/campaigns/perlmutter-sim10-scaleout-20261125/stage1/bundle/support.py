"""Perlmutter contracts; no SDF runtime or storage imports."""
import json
import os
from pathlib import Path
from qualified_support import require, sha, write, identity, fd_record, histogram, EVENT_BYTES, runtime

HERE = Path(__file__).resolve().parent
ROOT = Path('/pscratch/sd/m/monarin/psana2-gpu/pr298-striping-20261111')
SOURCE = ROOT/'source'
PREFIX = SOURCE/'install_psana'
PACKAGE = Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-baseline-transfer-20261115/package')
PR_SHA = '67c327a148a08b2aaf972e67e1e02e367c2b298f'
TASK_ROOT = Path('/pscratch/sd/m/monarin/psana2-gpu/sim10-baseline-20261115')

def affinity():
    mapping = json.loads(os.environ['SIM10_CPU_PLAN'])
    rank = int(os.environ['SLURM_PROCID'])
    require(len(mapping) in (3, 6), 'Three or six MPI ranks')
    all_cpus = [cpu for row in mapping for cpu in row]
    require(len(set(all_cpus)) == len(all_cpus), 'Rank CPU overlap')
    physical = set()
    for cpu in all_cpus:
        root = Path('/sys/devices/system/cpu/cpu%d/topology' % cpu)
        physical.add(((root/'physical_package_id').read_text(), (root/'core_id').read_text()))
    require(len(physical) == len(all_cpus), 'SMT sibling overlap')
    require([len(row) for row in mapping] == [3, 3]+[7]*(len(mapping)-2), 'Frozen physical-core demand')
    require(set(mapping[rank]) <= set(os.sched_getaffinity(0)), 'Outside step affinity')
    os.sched_setaffinity(0, mapping[rank])
    require(set(os.sched_getaffinity(0)) == set(mapping[rank]), 'Affinity readback')
    return mapping[rank]

def input_info():
    raw = json.loads((ROOT/'inputs.json').read_text())
    selected = next(row for row in raw['sets'] if row['count'] == 2)
    return dict(exp=raw['exp'], run=raw['run'], timestamps=raw['timestamps'],
                directory=selected['directory'], files=[row['path'] for row in selected['files']],
                payload_records=selected['files'], media='lustre', stripe_count=2,
                stripe_size_bytes=1048576)

def verify_inputs(info):
    import subprocess
    expected = input_info()
    require(info == expected, 'Selected input manifest changed')
    for row in info['payload_records']:
        require(Path(row['view']).resolve() == Path(row['path']), 'Payload view relocation changed')
        require(sha(row['smd']) == row['smd_sha256'], 'SMD identity changed')
        require(identity(row['path']) == row['identity'], 'Qualified input identity changed')
        require(subprocess.check_output(['lfs', 'getstripe', row['path']], text=True) == row['layout_raw'],
                'Qualified Lustre layout changed')

def runtime_identity(names):
    import importlib
    from qualified_support import runtime
    for name in names:
        importlib.import_module(name)
    return runtime()


def normalize_pci(value):
    import re
    if isinstance(value, bytes):value=value.decode('ascii')
    match=re.fullmatch(r'([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])',value.strip())
    require(match is not None, 'Invalid PCI bus ID')
    domain,bus,device,function=(int(part,16) for part in match.groups())
    return '%04x:%02x:%02x.%x'%(domain,bus,device,function)


def gpu_inventory(text):
    result={}
    for line in text.strip().splitlines():
        uuid,pci=[part.strip() for part in line.split(',')]
        require(uuid.startswith('GPU-') and uuid not in result, 'Unique GPU inventory UUID')
        result[uuid]=normalize_pci(pci)
    require(len(set(result.values()))==len(result), 'Unique inventory PCI IDs')
    return result
