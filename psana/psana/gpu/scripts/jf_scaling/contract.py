"""CPU-only acceptance gates for the multi-BD Jungfrau scaling harness."""
import hashlib
import struct

KEY_POINTS = ((1, 1), (1, 4), (2, 4), (4, 8))
FEESPEC_POINTS = ((1, 1), (1, 2), (1, 4))
FULL_POINTS = tuple((g, b) for g, bs in ((1, (1, 2, 4, 6, 8)),
                   (2, (2, 4, 6)), (4, (4, 8, 12))) for b in bs)


def matrix(points=KEY_POINTS, modes=('off', 'on')):
    return [(g, b, bulk, cache, rep) for rep in (1, 2)
            for g, b in (points if rep == 1 else tuple(reversed(points)))
            for cache in (('cold', 'warm') if rep == 1 else ('warm', 'cold'))
            for bulk in (modes if rep == 1 else tuple(reversed(modes)))]


def validate_result(records, reference, pixels, *, events, ngpus, check_pixels,
                    expected_requests=None):
    bd = [r for r in records if r['is_bd']]
    assert len(bd) == len(records) - 2 and len(bd) >= ngpus
    assert [r['rank'] for r in records] == list(range(len(records)))
    stamps = [ts for r in bd for ts in r['timestamps']]
    assert len(stamps) == len(set(stamps)) == events
    assert hashlib.sha256(struct.pack(f'<{events}Q', *sorted(stamps))).hexdigest() == reference['timestamp_sha256']
    assert all(not r['timestamps'] for r in records if not r['is_bd'])
    assert sum(r['counts']['bytes'] for r in bd) == reference['payload_bytes']
    # All five JF dgrams exceed the 1-MiB bulk target: one request per dgram.
    if expected_requests is None:
        expected_requests = events * 5
    assert sum(r['counts']['requests'] for r in bd) == expected_requests
    buses = {}
    for r in bd:
        gpu = (r['rank'] - 2) % ngpus
        assert r['physical_gpu'] == gpu
        d, c = r['device'], r['counts']
        assert not d['gds_available'] and d['workers'] == 8 and d['task_bytes'] == 1 << 20
        assert c['cpu_bd_reads'] == 0
        assert 0 < c['peak_owned_and_held'] <= c['budget_limit']
        peers = sum((i % ngpus) == gpu for i in range(len(bd)))
        assert r['peers'] == [peers]
        assert c['budget_limit'] <= d['total_bytes'] // peers
        assert buses.setdefault(gpu, d['bus']) == d['bus']
        assert r['affinity'] == records[0]['affinity']
        assert r['timestamps'] == sorted(r['timestamps'])
    assert len(set(buses.values())) == ngpus
    checks = [c for r in bd for c in r['checks']]
    if check_pixels:
        assert sorted(checks, key=lambda c: c['timestamp']) == [pixels[t] for t in sorted(pixels)]
    else:
        assert not checks
    elapsed = max(r['loop_s'] for r in records)
    assert elapsed > 0
    return dict(events=events, unique_timestamps=len(set(stamps)), loop_s=elapsed,
        events_per_s=events/elapsed, payload_gbps=reference['payload_bytes']/elapsed/1e9,
        bytes=reference['payload_bytes'], requests=expected_requests, ngpus=ngpus, nbds=len(bd),
        gpu_buses=buses, pixel_samples=len(checks))
