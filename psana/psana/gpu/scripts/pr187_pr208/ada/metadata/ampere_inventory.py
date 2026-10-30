#!/usr/bin/env python3
"""Read-only node inventory. No CUDA contexts, benchmarks, or settings changes.

Run inside a Slurm allocation; stdout is one JSON document. Python 3.6+.
Command failures and restricted device visibility are retained as evidence.
"""
import datetime
import hashlib
import json
import os
import re
from pathlib import Path
import socket
import subprocess
import xml.etree.ElementTree as ET


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError as exc:
        return {"error": str(exc)}


def attrs(path, names):
    return {name: read(path / name) for name in names if (path / name).exists()}


def command(args):
    try:
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           universal_newlines=True, timeout=12)
        return {"argv": args, "returncode": p.returncode,
                "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": args, "error": str(exc)}


def pci_chain(path):
    result = []
    for p in [path] + list(path.parents):
        if (p / "vendor").exists():
            result.append(dict(address=p.name, path=str(p), **attrs(p, (
                "vendor", "device", "class", "numa_node", "local_cpulist",
                "current_link_speed", "current_link_width", "max_link_speed",
                "max_link_width"))))
    return result


def collect():
    result = {
        "schema": 2, "host": socket.gethostname().split('.')[0],
        "collected_utc": datetime.datetime.utcnow().isoformat() + 'Z',
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "job": os.environ.get('SLURM_JOB_ID'),
        "affinity": sorted(os.sched_getaffinity(0)),
        "environment": {k: os.environ.get(k) for k in (
            'SLURM_JOB_NODELIST', 'SLURM_CPUS_PER_TASK', 'SLURM_MEM_PER_NODE',
            'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'CUDA_VISIBLE_DEVICES')},
        "commands": {}, "blocks": {}, "nvme": {}, "pci": {}, "numa": {},
        "gpu_query": [], "cgroup": {},
    }
    commands = {
        "mount": ['findmnt', '-J', '-T', '/lscratch', '-o', 'SOURCE,TARGET,FSTYPE,OPTIONS'],
        "blocks": ['lsblk', '-J', '-b', '-o', 'NAME,TYPE,SIZE,ROTA,MOUNTPOINT,MODEL'],
        "pci_tree": ['lspci', '-Dtv'], "pci_devices": ['lspci', '-Dnn'],
        "cpu": ['lscpu', '-J'], "cpu_map": ['lscpu', '-e=CPU,NODE,SOCKET,ONLINE'],
        "numa": ['numactl', '--hardware'], "space": ['df', '-B1', '/lscratch'],
        "kernel": ['uname', '-r'], "gpu_xml": ['nvidia-smi', '-q', '-x'],
        "gpu_topology": ['nvidia-smi', 'topo', '-m'],
    }
    for key, argv in commands.items():
        result['commands'][key] = command(argv)
    for name, path in [('meminfo', '/proc/meminfo'), ('mdstat', '/proc/mdstat'),
                       ('self_cgroup', '/proc/self/cgroup'),
                       ('diskstats', '/proc/diskstats')]:
        result[name] = read(path)
    for p in sorted(Path('/sys/block').iterdir()):
        if not p.name.startswith(('nvme', 'md')):
            continue
        result['blocks'][p.name] = {
            'path': str(p.resolve()),
            'slaves': sorted(q.name for q in (p / 'slaves').iterdir()),
            'attributes': attrs(p, ('size', 'queue/rotational', 'queue/read_ahead_kb',
                'queue/scheduler', 'queue/max_sectors_kb', 'queue/max_hw_sectors_kb',
                'device/model', 'device/firmware_rev', 'device/numa_node',
                'md/level', 'md/chunk_size', 'md/raid_disks', 'md/degraded', 'md/array_state')),
            'pci_chain': pci_chain(p.resolve()),
        }
    for p in sorted(Path('/sys/class/nvme').glob('nvme*')):
        result['nvme'][p.name] = dict(path=str(p.resolve()),
            attributes=attrs(p, ('model', 'firmware_rev', 'state', 'transport')),
            pci_chain=pci_chain(p.resolve()))
    for p in sorted(Path('/sys/bus/pci/devices').iterdir()):
        kind = read(p / 'class')
        if isinstance(kind, str) and kind.startswith(('0x0108', '0x030')):
            result['pci'][p.name] = dict(class_code=kind, chain=pci_chain(p.resolve()))
    for p in sorted(Path('/sys/devices/system/node').glob('node[0-9]*')):
        result['numa'][p.name] = attrs(p, ('cpulist', 'distance', 'meminfo'))
    result['nvidia_proc_version'] = read('/proc/driver/nvidia/version')
    result['nvidia_proc_gpus'] = {}
    for p in sorted(Path('/proc/driver/nvidia/gpus').glob('*/information')):
        info = read(p)
        result['nvidia_proc_gpus'][p.parent.name] = info
        if isinstance(info, str):
            fields = dict(line.split(':', 1) for line in info.splitlines() if ':' in line)
            fields = {k.strip(): v.strip() for k, v in fields.items()}
            result['gpu_query'].append(dict(name=fields.get('Model'),
                uuid=fields.get('GPU UUID'), pci=fields.get('Bus Location'),
                memory=None, source='proc/driver/nvidia (NVML not required)'))
    version = result['nvidia_proc_version']
    if isinstance(version, str):
        match = re.search(r'\b\d{3}\.\d+\.\d+\b', version)
        if match:
            result['driver_version'] = match.group(0)
    cg = result['self_cgroup']
    if isinstance(cg, str):
        for line in cg.splitlines():
            _, controllers, relative = line.split(':', 2)
            if controllers == '':  # cgroup v2; walk inherited limits as well
                root = Path('/sys/fs/cgroup')
                p = root / relative.lstrip('/')
                while p == root or root in p.parents:
                    result['cgroup'][str(p)] = attrs(p, (
                        'io.max', 'io.weight', 'memory.max', 'memory.high',
                        'cpuset.cpus.effective', 'cpuset.mems.effective', 'cpu.max'))
                    if p == root:
                        break
                    p = p.parent
    xml = result['commands']['gpu_xml']
    if xml.get('returncode') == 0:
        try:
            tree = ET.fromstring(xml['stdout'])
            result['driver_version'] = tree.findtext('driver_version')
            result['gpu_query'] = []
            for gpu in tree.findall('gpu'):
                fields = {name: gpu.findtext(path) for name, path in {
                    'name': 'product_name', 'uuid': 'uuid', 'pci': 'pci/pci_bus_id',
                    'memory': 'fb_memory_usage/total',
                    'link_gen_max': 'pci/pci_gpu_link_info/pcie_gen/max_link_gen',
                    'link_gen_current': 'pci/pci_gpu_link_info/pcie_gen/current_link_gen',
                    'link_width_max': 'pci/pci_gpu_link_info/link_widths/max_link_width',
                    'link_width_current': 'pci/pci_gpu_link_info/link_widths/current_link_width',
                    'temperature': 'temperature/gpu_temp',
                    'power_limit': 'gpu_power_readings/power_limit',
                }.items()}
                result['gpu_query'].append(fields)
        except ET.ParseError as exc:
            result['gpu_xml_error'] = str(exc)
    result['completed_utc'] = datetime.datetime.utcnow().isoformat() + 'Z'
    return result


if __name__ == '__main__':
    print(json.dumps(collect(), indent=2, sort_keys=True))
