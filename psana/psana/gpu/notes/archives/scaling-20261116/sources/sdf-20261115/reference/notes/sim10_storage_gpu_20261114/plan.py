"""Fixed-work, paired-storage GPU-sharing diagnostic; no submission side effects."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'sim10_kvikio_bounce_20261113'
sys.path.append(str(BASE))
from support import source_hashes as baseline_hashes

CONTROLS = dict(bds=4, workers=4, batch=200, streams=1, bounce_mib=16,
                task_bytes=1 << 20, budget_gib=6, bulk_bytes=512 << 10,
                smd_events=1000, events=100000, compat_mode='ON',
                auto_direct_io_read=False)
RESOURCES = dict(ntasks=6, cpus_per_task=6, physical_cores_required=34,
                 gpus=4, memory='192G', time='03:00:00', exclusive=True)
BUDGET_SECONDS = dict(full_and_anchors=24 * 180, pilots=6 * 120,
                      gpu_tests=4 * 300, probes=2 * 300, staging=2 * 900,
                      qualification=600, cleanup_and_audit=300)


def placement(uuids, gpu_count):
    if len(uuids) != 4 or len(set(uuids)) != 4 or gpu_count not in (1, 2, 4):
        raise ValueError('Need four distinct allocated GPUs and a 1/2/4-GPU case')
    if not all(value.startswith('GPU-') for value in uuids):
        raise ValueError('Use physical GPU UUIDs, not CUDA ordinals')
    ranks = ['', ''] + [uuids[index * gpu_count // 4] for index in range(4)]
    return dict(gpus=gpu_count, rank_uuids=ranks,
                peers=dict(Counter(ranks[2:])), allocation_uuids=uuids)


def matrix():
    cases = []

    def add(kind, repeat, media, gpus):
        cases.append(dict(name='%02d-%s-r%d-%s-gpu%d' %
                          (len(cases), kind, repeat, media, gpus),
                          kind=kind, repeat=repeat, media=media, gpus=gpus,
                          events=4000 if kind == 'pilot' else 100000))

    for media in ('weka', 'nvme'):
        for gpus in (1, 2, 4):
            add('pilot', 0, media, gpus)
    for repeat, order in enumerate(((1, 2, 4), (2, 4, 1), (4, 1, 2)), 1):
        add('anchor', repeat, 'weka', 1)
        for gpus in order:
            for media in (('weka', 'nvme') if repeat != 2 else ('nvme', 'weka')):
                add('full', repeat, media, gpus)
        add('anchor', repeat, 'weka', 1)
    return cases


def source_hashes():
    hashes = baseline_hashes()
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in ('.py', '.sbatch', '.md', '.json'):
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    contract = BASE / 'kvikio-source-contract.json'
    hashes[str(contract)] = hashlib.sha256(contract.read_bytes()).hexdigest()
    for path in (BASE / 'results/runs.csv', BASE / 'results/accounting.psv',
                 HERE.parent / 'sim10_single_node_scaling_20261113/evidence/44770441/hardware.json'):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


if __name__ == '__main__':
    print(json.dumps(dict(controls=CONTROLS, resources=RESOURCES, cases=matrix()), indent=2))
