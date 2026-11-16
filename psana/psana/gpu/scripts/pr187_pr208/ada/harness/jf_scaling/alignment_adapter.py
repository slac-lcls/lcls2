"""Benchmark-only direct-read address shift; never install in production.

Every case reserves N+1 bytes and exposes N bytes at offset zero or one.
Diagnostic audits synchronize and download metadata; timed runs do neither.
"""


def install(shift, *, audit=False):
    if shift not in (0, 1):
        raise ValueError('shift must be zero or one')
    from psana.gpu.gpu_allocation import owned_empty, allocation_requirement, backing_capacity
    from psana.gpu.gpu_kvikio_read import KvikioGpuReader

    report = dict(shift=shift, audit=audit, reader_allocations=0,
                  allocated_capacity_bytes=0, padding_bytes_requested=0,
                  aligned_rows=0, unaligned_rows=0, audited_dgrams=0,
                  gather_calls=0, examples=[])

    def ensure(self, slot, total_nbytes):
        existing = self._slot_bufs[slot]
        if existing is not None and existing.nbytes >= total_nbytes:
            return
        backing = owned_empty(self.cp, total_nbytes + 1, self.cp.uint8,
                              self._budget, 'reader')
        view = backing[shift:shift + total_nbytes]
        assert int(view.data.ptr) % 2 == shift
        assert int(view.nbytes) == total_nbytes
        # The slice retains the allocation owner and its rounded budget charge.
        self._slot_bufs[slot] = view
        report['reader_allocations'] += 1
        report['padding_bytes_requested'] += 1
        report['allocated_capacity_bytes'] += backing_capacity(view)

    def requirements(self, nbytes, slot):
        existing = self._slot_bufs[slot]
        if existing is not None and existing.nbytes >= nbytes:
            return [(0, 0)]
        # The old generation remains charged independently during replacement.
        return [allocation_requirement(self.cp, nbytes + 1, None)]

    KvikioGpuReader._ensure_slot_buffer = ensure
    KvikioGpuReader.allocation_requirements = requirements
    if not audit:
        return report

    import numpy as np
    from psana.gpu.gpu_kvikio_read import DESC_DEVICE_OFFSET, DESC_READ_SIZE
    from psana.gpu import gpu_detector as gd
    from psana.gpu.gpudgram import parser as p

    submit = KvikioGpuReader._submit_read

    def audited_submit(self, desc_table, *args, **kwargs):
        # Even dgram extents preserve address parity under different BD/group
        # distributions. Check the actual submitted table, not nominal sizes.
        assert np.all(desc_table[:, DESC_DEVICE_OFFSET] % 2 == 0)
        assert np.all(desc_table[:, DESC_READ_SIZE] % 2 == 0)
        report['audited_dgrams'] += len(desc_table)
        return submit(self, desc_table, *args, **kwargs)

    KvikioGpuReader._submit_read = audited_submit
    gather = gd._CanonicalGatherPlan.gather

    def audited_gather(self, inputs, target, present, pixels, stream, shape=None):
        for locations in inputs.locations:
            locations.wait_on(stream)
        stream.synchronize()
        owners = inputs.owners.get().reshape(-1, gd._GATHER_OWNER_WORDS)
        rows = inputs.rows.get().reshape(-1, len(self.streams), gd._GATHER_ROW_WORDS)
        tables = [locations.backing.get() for locations in inputs.locations]
        for event in rows:
            for stream_column, handle, field_type, rank in self.host:
                owner_index, dgram = map(int, event[int(stream_column)])
                assert owner_index < len(owners), 'missing diagnostic input owner'
                owner = owners[owner_index]
                loc = tables[owner_index][int(handle), dgram]
                offset, size = int(loc[p.LOC_OFFSET]), int(loc[p.LOC_NBYTES])
                assert int(loc[p.LOC_STATUS]) == p.STATUS_FOUND
                assert (int(loc[p.LOC_TYPE]), int(loc[p.LOC_RANK])) == (int(field_type), int(rank))
                assert size == pixels * target.dtype.itemsize
                assert offset <= int(owner[1]) and size <= int(owner[1]) - offset
                if shape:
                    dim = p.LOC_DIM0 + (1 if int(rank) == 3 else 0)
                    assert tuple(map(int, loc[dim:dim + 2])) == tuple(shape)
                    assert int(rank) == 2 or (int(rank) == 3 and int(loc[p.LOC_DIM0]) == 1)
                address = int(owner[0]) + offset
                residue = address % target.dtype.itemsize
                # Abort before the old kernel could dereference an unsafe row.
                assert residue == shift, (address, offset, shift)
                report['unaligned_rows' if residue else 'aligned_rows'] += 1
                if len(report['examples']) < 8:
                    report['examples'].append(dict(base=int(owner[0]), offset=offset,
                                                   address=address, residue=residue))
        report['gather_calls'] += 1
        return gather(self, inputs, target, present, pixels, stream, shape=shape)

    gd._CanonicalGatherPlan.gather = audited_gather
    return report
