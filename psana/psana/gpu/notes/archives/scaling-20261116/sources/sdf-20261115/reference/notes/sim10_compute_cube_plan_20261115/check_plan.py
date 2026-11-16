"""Reproduce the plan's arithmetic from saved worker evidence, without GPU I/O."""
import csv
import hashlib
import json
from pathlib import Path


def main():
    notes = Path(__file__).resolve().parents[1]
    summary = notes / 'sim10_single_node_scaling_20261113/summary.csv'
    with summary.open() as stream:
        rows = list(csv.DictReader(stream))

    def rate(bds, workers, batch):
        matches = [row for row in rows if
                   (row['cohort'], row['bds'], row['workers_per_bd'],
                    row['streams_per_bd'], row['batch']) ==
                   ('44770441', str(bds), str(workers), '1', str(batch))]
        assert len(matches) == 1
        return float(matches[0]['mean_hz'])

    worker_gain = 100 * (rate(1, 8, 50) / rate(1, 4, 50) - 1)
    assert round(worker_gain, 2) == 12.57
    pixels = 32 * 512 * 1024
    raw_bytes, calibrated_bytes = pixels * 2, pixels * 4
    bank_bytes = 100 * (calibrated_bytes + 8)
    assert raw_bytes == 32 << 20
    assert calibrated_bytes == 64 << 20
    assert bank_bytes == (6400 << 20) + 800
    memory = []
    for bds, gpus, batch, subtotal in (
            (1, 1, 50, 16.9375), (4, 1, 200, 124),
            (4, 2, 200, 62), (4, 4, 200, 31)):
        peers = bds // gpus
        images = peers * batch * (raw_bytes + calibrated_bytes)
        required = images + peers * bank_bytes + peers * (6 << 30)
        assert (required - peers * 800) / 2**30 == subtotal
        memory.append(dict(bds=bds, gpus=gpus, batch=batch,
                           images_per_gpu_bytes=images,
                           planning_subtotal_per_gpu_bytes=required,
                           excludes='uncharged state, allocator overhead and safety margin'))
    payload_bytes = 3358200
    assert payload_bytes * 100000 == 335820000000
    assert round(1400 * payload_bytes / 2**30, 6) == 4.378594
    print(json.dumps(dict(
        status='pass', scope='arithmetic and saved-row lookup only; no GPU tests',
        worker_summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest(),
        one_bd_four_worker_hz=rate(1, 4, 50),
        one_bd_eight_worker_hz=rate(1, 8, 50),
        eight_vs_four_worker_gain_percent=worker_gain,
        four_bd_four_worker_hz=rate(4, 4, 200),
        four_bd_eight_worker_hz=rate(4, 8, 200),
        memory=memory, raw_bytes_per_event=raw_bytes,
        calibrated_bytes_per_event=calibrated_bytes, bank_bytes=bank_bytes,
        payload_gib_s_at_1400_hz=1400 * payload_bytes / 2**30,
        proposed_full_runs=2 * 4 * 3, proposed_anchors=2 * 3), indent=2))


if __name__ == '__main__':
    main()
