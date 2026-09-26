"""Explicit shared-stream benchmark exception and independent CPU checks."""


def exclusive_feespec(rank):
    import json
    from psana.psexp.ds_base import DsParms
    original = DsParms.resolve_gpu_stream_ids

    def resolve(self):
        assert self._detector_names(self.gpu_det) == ['jungfrau', 'feespec']
        assert not self.hybrid_det
        streams = self.det_stream_ids_table['feespec']
        assert len(streams) == 1
        original_owners = self.stream_id_to_detnames
        owners = list(original_owners[streams[0]])
        assert 'feespec' in owners and 'jungfrau' not in owners
        self.stream_id_to_detnames = dict(original_owners)
        self.stream_id_to_detnames[streams[0]] = ['feespec']
        try:
            original(self)
        finally:
            self.stream_id_to_detnames = original_owners
        print('ROUTING_OVERRIDE ' + json.dumps(dict(
            rank=rank, streams=streams, original_owners=owners)), flush=True)
    DsParms.resolve_gpu_stream_ids = resolve


def coalesced_requests(rows, batch_size=20, target=1 << 20):
    """Count adjacent SMD extents independently of the production planner."""
    count = 0
    for base in range(0, len(rows), batch_size):
        end = fence = None
        size = 0
        for row in rows[base:base + batch_size]:
            if row['offset'] != end or row['fence'] != fence or size + row['size'] > target:
                count += 1
                size = 0
            size += row['size']
            end = row['offset'] + row['size']
            fence = row['fence']
    return count


def validate_feespec(records, reference, check_pixels):
    bd = [r for r in records if r['is_bd']]
    observed = []
    arrays = []
    for r in bd:
        assert len(r['feespec_sums']) == len(r['timestamps'])
        observed.extend(zip(r['timestamps'], r['feespec_sums']))
        arrays.extend(r['feespec_arrays'])
    assert sorted(observed) == list(zip(reference['timestamps'], reference['sums']))
    if check_pixels:
        expected = reference['arrays']
        assert sorted(arrays, key=lambda r: r['timestamp']) == expected
    else:
        assert not arrays
    return dict(feespec_sums_pass=True, feespec_arrays=len(arrays))
