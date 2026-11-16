"""Evidence-backed cold-Weka candidates; selection never allocates resources."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

from plan import require, sha, write


CPU = {'Model name:': 'AMD EPYC 7542 32-Core Processor', 'Socket(s):': '2',
       'Core(s) per socket:': '32', 'NUMA node(s):': '8'}
GPU_NUMA = {'0000:01:00.0': '3', '0000:41:00.0': '1',
            '0000:81:00.0': '7', '0000:c1:00.0': '5'}
INVENTORY = Path('/sdf/scratch/users/m/monarin/gpu-validation/ampere-inventory-20261026-r1/nodes')


def cpu_fields(text):
    return {row['field']: str(row['data']) for row in json.loads(text)['lscpu']
            if row['field'] in CPU}


def nic_lines(text):
    return sorted(line.strip() for line in text.splitlines() if 'Ethernet controller [0200]' in line)


def profile(data):
    require(data['schema'] == 2 and data['completed_utc'], 'Incomplete inventory')
    commands = data['commands']
    require(all(commands[key]['returncode'] == 0 for key in ('cpu', 'mount', 'pci_devices')), 'Missing host evidence')
    require(cpu_fields(commands['cpu']['stdout']) == CPU, 'CPU topology mismatch')
    devices = data['gpu_query']
    require(len(devices) == 4 and {row['pci'] for row in devices} == set(GPU_NUMA), 'GPU PCI inventory mismatch')
    require(len({row['uuid'] for row in devices}) == 4, 'Duplicate GPU UUID inventory')
    for row in devices:
        require(row['name'] == 'NVIDIA A100-SXM4-40GB', 'GPU model mismatch')
        endpoint = data['pci'][row['pci']]['chain'][0]
        require(endpoint['numa_node'] == GPU_NUMA[row['pci']]
                and endpoint['max_link_speed'] == '16.0 GT/s PCIe'
                and endpoint['max_link_width'] == '16', 'GPU topology mismatch')
    mount, = json.loads(commands['mount']['stdout'])['filesystems']
    require(mount['target'] == '/lscratch' and mount['fstype'] in ('xfs', 'ext4'), 'Local scratch filesystem missing')
    array = data['blocks'][Path(mount['source']).name]
    require(array['attributes']['md/level'] == 'raid0' and len(array['slaves']) == 2, 'Scratch RAID mismatch')
    drives = []
    for namespace in array['slaves']:
        controller = data['nvme'][namespace.split('n1')[0]]
        endpoint = controller['pci_chain'][0]
        require(endpoint['current_link_speed'] == '16.0 GT/s PCIe'
                and endpoint['current_link_width'] == '4', 'Scratch is not inventoried Gen4 x4')
        drives.append(dict(model=controller['attributes']['model'],
                           firmware=controller['attributes']['firmware_rev'],
                           numa=endpoint['numa_node'], link='Gen4 x4'))
    nics = nic_lines(commands['pci_devices']['stdout'])
    require(len(nics) == 4 and sum('ConnectX-6' in line for line in nics) == 2, 'NIC inventory mismatch')
    require(data['driver_version'] == '575.57.08', 'Driver differs from frozen baseline')
    return dict(cpu=CPU, gpu_numa=GPU_NUMA, driver=data['driver_version'], nics=nics,
                local_scratch=dict(mount=mount, drives=drives),
                observed_at=data['completed_utc'], gpu_uuids=sorted(row['uuid'] for row in devices))


def validate(record):
    require(record['schema_version'] == 1 and record['partition'] == 'ampere', 'Allowlist schema/partition')
    names, universe = record['nodes'], record['partition_nodes']
    require(names and names == sorted(set(names)) and universe == sorted(set(universe)), 'Nonempty canonical allowlist required')
    require(all(re.fullmatch(r'sdfampere\d{3}', name) for name in universe)
            and set(names) < set(universe), 'Invalid allowlist universe')
    require(set(record['candidates']) == set(names), 'Candidate evidence coverage')
    for name, row in record['candidates'].items():
        require(row['profile']['cpu'] == CPU and row['profile']['gpu_numa'] == GPU_NUMA,
                'Candidate CPU/GPU contract')
        require(row['profile']['driver'] == '575.57.08' and len(set(row['profile']['gpu_uuids'])) == 4,
                'Candidate device/driver inventory')
        require(record['evidence_files'].get(row['source']) == row['sha256'], 'Missing inventory hash')
    return names


def scheduler_flags(record, live_nodes):
    validate(record)
    require(sorted(set(live_nodes)) == record['partition_nodes'], 'Partition membership changed; prepare and review again')
    excluded = sorted(set(live_nodes) - set(record['nodes']))
    return ['--nodes=1', '--exclude=' + ','.join(excluded)]


def selected_host(manifest, host, allocation_nodes):
    allowed = validate(manifest['node_allowlist']) if manifest.get('node_allowlist') else [manifest['node']]
    require(host in allowed, 'Host outside reviewed node allowlist')
    require(allocation_nodes == host, 'Slurm must allocate exactly the selected host')
    return host


def hardware_gate(record, host, hardware, devices):
    validate(record)
    require(host in record['nodes'] and hardware['host'].split('.')[0] == host, 'Hardware host mismatch')
    expected = record['candidates'][host]['profile']
    require(cpu_fields(hardware['commands']['lscpu']['stdout']) == expected['cpu'], 'Live CPU topology drift')
    require(nic_lines(hardware['commands']['pci']['stdout']) == expected['nics'], 'Live NIC inventory drift')
    require(set(row['uuid'] for row in devices) == set(expected['gpu_uuids']), 'Live GPU identity drift')
    pci = {row['bdf']: row for row in hardware['pci_devices']}
    for address, numa in expected['gpu_numa'].items():
        endpoint = pci[address]
        require(endpoint['numa_node'] == numa and endpoint['max_link_speed'] == '16.0 GT/s PCIe'
                and endpoint['max_link_width'] == '16', 'Live GPU PCI/NUMA drift')
    driver = re.search(r'Driver Version\s*:\s*(\S+)', hardware['commands']['gpu']['stdout'])
    require(driver is not None and driver.group(1) == expected['driver'], 'Live driver drift')
    return dict(status='pass', selected_node=host, historical_inventory_sha256=record['candidates'][host]['sha256'],
                scope='CPU/GPU/NIC identity; local capacity and cold-Weka NIC delivery checked separately')


def collect():
    observed = datetime.now(timezone.utc).isoformat()
    hosts = subprocess.check_output(['sinfo', '-p', 'ampere', '-N', '-h', '-o', '%N'], text=True, timeout=30).split()
    hosts = sorted(set(hosts))
    require(hosts and all(re.fullmatch(r'sdfampere\d{3}', host) for host in hosts), 'Invalid partition discovery')
    raw = subprocess.check_output(['scontrol', 'show', 'node', ','.join(hosts), '-o'], text=True, timeout=30)
    live = {row['NodeName']: row for line in raw.splitlines()
            if (row := dict(re.findall(r'(\w+)=(\S+)', line)))}
    require(set(live) == set(hosts), 'Incomplete Slurm node observation')
    candidates, excluded, files = {}, {}, {}
    for host in hosts:
        state = live[host]['State']
        if any(flag in state for flag in ('DOWN', 'DRAIN', 'MAINTENANCE', 'RESERVED', 'NOT_RESPONDING', 'INVALID')):
            excluded[host] = state
            continue
        if live[host]['Gres'] != 'gpu:a100:4':
            excluded[host] = 'Four GPUs not advertised'
            continue
        errors = []
        for path in sorted(INVENTORY.glob(host + '-*.json'), reverse=True):
            try:
                data = json.loads(path.read_text())
                require(data['host'].split('.')[0] == host, 'Inventory host mismatch')
                candidate = profile(data)
                require(int(live[host]['CPUEfctv']) >= 72 and int(live[host]['RealMemory']) >= 192 * 1024,
                        'Insufficient configured CPU/memory')
                candidates[host] = dict(source=str(path), sha256=sha(path), profile=candidate,
                                        slurm=live[host], qualification='candidate; runtime gates mandatory')
                files[str(path)] = sha(path)
                break
            except (ValueError, KeyError, TypeError) as error:
                errors.append(path.name + ': ' + str(error))
        if host not in candidates:
            excluded[host] = errors or ['No complete inventory']
    record = dict(schema_version=1, partition='ampere', observed_at=observed, partition_nodes=hosts,
                  nodes=sorted(candidates), candidates=candidates, excluded=excluded, evidence_files=files,
                  slurm_raw=raw, purpose='One exclusive cold-Weka campaign; not an NVMe performance-equivalence pool',
                  limitations=['Historical hardware is not current qualification or a reservation',
                               'SSD models differ; local storage is only for untimed cube export',
                               'No NIC speed or Weka bandwidth equivalence claimed before runtime probes'])
    validate(record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    record = collect()
    write(args.output, record)
    print(json.dumps(dict(nodes=record['nodes'], excluded=record['excluded']), indent=2))
