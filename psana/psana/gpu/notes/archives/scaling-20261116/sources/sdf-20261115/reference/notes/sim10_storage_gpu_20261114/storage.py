"""Owned-file cache checks and physical-leaf NVMe evidence, never raw-device I/O."""
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time

from plan import BASE
from support import raw, require, physical_counters, counter_delta

ROOTS = dict(weka=Path('/sdf/data/lcls/drpsrcf/ffb/users/monarin'),
             nvme=Path('/lscratch/monarin'))


def owned_stage(stage, media):
    stage = Path(stage)
    resolved = stage.resolve(strict=True)
    require(media in ROOTS and resolved.parent == ROOTS[media], 'Unexpected stage root')
    require(stage == resolved and stage.name.startswith('sim10-storage-gpu-'), 'Stage path/identity')
    require(stage.stat().st_uid == os.geteuid() and stat.S_IMODE(stage.stat().st_mode) == 0o700,
            'Stage must be privately owned')
    return resolved


def cache(files, stage, media, evict=False):
    stage = owned_stage(stage, media)
    rows = []
    for filename in files:
        path = Path(filename)
        require(path == path.resolve(strict=True) and path.parent == stage, 'Foreign cache path')
        require(path.stat().st_uid == os.geteuid() and path.is_file(), 'Foreign cache owner/type')
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            require(os.fstat(descriptor).st_ino == path.stat().st_ino, 'Cache file replaced')
            if evict:
                os.fsync(descriptor)
                for attempt in range(20):
                    os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
                    if raw.prefix_residency(path, path.stat().st_size)['resident_pages'] == 0:
                        break
                    time.sleep(0.1)
            rows.append(raw.prefix_residency(path, path.stat().st_size))
        finally:
            os.close(descriptor)
    require(bool(rows), 'Missing cache targets')
    if evict:
        require(all(row['resident_pages'] == 0 for row in rows), 'Cold gate failed')
    return rows


def leaf_devices(device, sysfs=Path('/sys/class/block'), seen=None):
    seen = set() if seen is None else seen
    require(device not in seen, 'Cyclic block topology')
    seen.add(device)
    root = sysfs / device
    require(root.exists(), 'Missing block device')
    if (root / 'partition').exists():
        return leaf_devices(root.resolve().parent.name, sysfs, seen)
    slaves = sorted(path.name for path in (root / 'slaves').iterdir())
    if slaves:
        leaves = [leaf for slave in slaves for leaf in leaf_devices(slave, sysfs, seen.copy())]
        require(len(leaves) == len(set(leaves)), 'Repeated physical leaf')
        return sorted(leaves)
    require(re.fullmatch(r'nvme\d+n\d+', device) is not None, 'Non-NVMe leaf')
    return [device]


def qualify(stage, media, required_bytes):
    stage = owned_stage(stage, media)
    mount = json.loads(subprocess.check_output(
        ['findmnt', '-J', '-T', str(stage), '-o', 'SOURCE,TARGET,FSTYPE,MAJ:MIN'], text=True))
    row, = mount['filesystems']
    capacity = os.statvfs(stage)
    free = capacity.f_bavail * capacity.f_frsize
    require(free >= required_bytes + (10 << 30), 'Insufficient stage free space')
    leaves = []
    raid = None
    physical_inventory = []
    if media == 'weka':
        require(row['source'] == 'drpsrcf/ffb', 'Expected Weka FFB mount')
    else:
        require(row['target'] == '/lscratch' and row['fstype'] in ('xfs', 'ext4'), 'Expected local scratch mount')
        device = (Path('/sys/dev/block') / row['maj:min']).resolve(strict=True).name
        leaves = leaf_devices(device)
        require(len(leaves) == 2, 'Expected two NVMe leaves; re-review changed topology')
        raid_root = Path('/sys/class/block') / device / 'md'
        require(raid_root.is_dir(), 'Expected verified md RAID0 scratch device')
        raid = {key: (raid_root / key).read_text().strip() for key in ('level', 'raid_disks', 'chunk_size')}
        require(raid['level'] == 'raid0' and raid['raid_disks'] == '2', 'Unexpected RAID configuration')
        for leaf in leaves:
            controller = re.match(r'nvme\d+', leaf).group()
            controller_path = Path('/sys/class/nvme') / controller
            pci = (controller_path / 'device').resolve(strict=True)
            physical_inventory.append(dict(leaf=leaf, controller=controller, pci_path=str(pci),
                model=(controller_path / 'model').read_text().strip(),
                firmware=(controller_path / 'firmware_rev').read_text().strip(),
                bytes=int((Path('/sys/class/block') / leaf / 'size').read_text()) * 512,
                **{key: (pci / key).read_text().strip() for key in
                   ('numa_node', 'current_link_speed', 'current_link_width', 'max_link_speed', 'max_link_width')}))
    return dict(media=media, stage=str(stage), mount=mount, free_bytes=free,
                required_bytes=required_bytes, physical_leaves=leaves, raid=raid,
                physical_inventory=physical_inventory, captured_at=time.time())


def disk_counters(leaves):
    result = {}
    for name in leaves:
        root = Path('/sys/class/block') / name
        fields = list(map(int, (root / 'stat').read_text().split()))
        require(len(fields) >= 11, 'Short block stat record')
        result[name] = dict(device=str(root.resolve()), monotonic=time.perf_counter(),
                            reads=fields[0], sectors=fields[2], read_ms=fields[3],
                            inflight=fields[8], busy_ms=fields[9], weighted_ms=fields[10])
    return result


def disk_delta(before, after):
    require(before.keys() == after.keys(), 'Block inventory changed')
    result = {}
    for name, first in before.items():
        last = after[name]
        elapsed = last['monotonic'] - first['monotonic']
        require(elapsed > 0 and last['device'] == first['device'], 'Block device changed')
        counts = {key: last[key] - first[key] for key in ('reads', 'sectors', 'read_ms', 'busy_ms', 'weighted_ms')}
        require(all(value >= 0 for value in counts.values()), 'Block counter reset')
        result[name] = dict(seconds=elapsed, bytes=counts['sectors'] * 512, **counts,
                            read_await_ms=counts['read_ms'] / counts['reads'] if counts['reads'] else None,
                            average_queue=counts['weighted_ms'] / (elapsed * 1000))
    return result


def counters(leaves):
    return dict(nic=physical_counters(), disks=disk_counters(leaves))


def evidence(record, media, payload, seconds):
    nic = counter_delta(record['before']['nic'], record['after']['nic'], payload)
    disks = disk_delta(record['before']['disks'], record['after']['disks'])
    idle_nic = counter_delta(record['idle_before']['nic'], record['idle_after']['nic'])
    idle_disks = disk_delta(record['idle_before']['disks'], record['idle_after']['disks'])
    require(seconds > 0 and payload > 0, 'Invalid measurement')
    require(all(row['seconds'] >= seconds for row in [*nic.values(), *disks.values()]), 'Counter bracket')
    require(nic.keys() == idle_nic.keys() and disks.keys() == idle_disks.keys(), 'Idle endpoint mismatch')
    adjusted_rx = sum(row['rx_bytes'] - idle_nic[name]['rx_bytes'] * row['seconds'] / idle_nic[name]['seconds']
                      for name, row in nic.items())
    adjusted_disk = sum(row['bytes'] - idle_disks[name]['bytes'] * row['seconds'] / idle_disks[name]['seconds']
                        for name, row in disks.items())
    if media == 'weka':
        require(adjusted_rx >= 0.90 * payload, 'Insufficient physical NIC RX for cold Weka')
    elif media == 'nvme':
        require(len(disks) == 2 and all(row['reads'] > 0 for row in disks.values()), 'Missing physical NVMe reads')
        require(adjusted_disk >= 0.90 * payload, 'Insufficient physical NVMe bytes')
        require(sum(row['rx_bytes'] for row in nic.values()) <= 0.10 * payload, 'NVMe run has excessive network traffic')
    else:
        raise ValueError('Unknown media')
    return dict(nic=nic, disks=disks, idle_nic=idle_nic, idle_disks=idle_disks,
                idle_adjusted_rx_payload_ratio=adjusted_rx / payload,
                idle_adjusted_disk_payload_ratio=adjusted_disk / payload,
                limitation='Whole-device counters, not per-process attribution; disk bytes count physical leaves only')
