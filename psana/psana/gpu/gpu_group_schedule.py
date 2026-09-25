"""Bounded execution ranges over stream-interleaved read groups (CPU metadata)."""
from dataclasses import replace

from .gpu_admission import AdmissionEvent, plan_admission
from .gpu_budget import GpuMemoryPressureError
from .gpu_read_plan import ResolvedDgram
from .gpu_stream_read_plan import StreamReadGroup, build_stream_read_plan


class GroupReadSchedule:
    """One EB batch; a run-scoped InputGroupPool owns the actual stream credits."""

    def __init__(self, descriptors, epochs, events, *, batch_id, capacity,
                 parser_bytes, depth, target_bytes=1 << 20):
        rows, fences = [], {}
        for d in descriptors:
            epoch = epochs[d.batch_event_index, d.stream_id]
            previous = fences.setdefault(d.batch_event_index, epoch.fence)
            if previous != epoch.fence:
                raise ValueError('event streams disagree on transition fence')
            rows.append(ResolvedDgram(d.batch_event_index, d.timestamp, d.stream_id,
                                      epoch.file, d.offset, d.size, d.smd_size))
        self.plan = build_stream_read_plan(
            rows, n_events=len(events), batch_id=batch_id,
            small_target_bytes=target_bytes, input_capacity_bytes=capacity,
            fence_by_event=fences)
        # Empty valid descriptors still need parser rows and event identity,
        # although no physical request is submitted for them.
        empty = [StreamReadGroup(len(self.plan.groups) + i, d.stream_id,
                                 fences[d.batch_event_index], d.file, d.file_offset,
                                 0, (d,), False)
                 for i, d in enumerate(self.plan.empty_dgrams)]
        self.plan = replace(self.plan, groups=tuple(sorted(
            self.plan.groups + tuple(empty), key=lambda g: (g.first_event, g.stream_id))))
        self.batch_id = batch_id
        self.issued = set()
        self.groups_by_event = {}
        maxima, cuts, seen_small = {}, {0, len(events)}, set()
        small_rows = set()
        for group in self.plan.groups:
            for d in group.dgrams:
                self.groups_by_event.setdefault(d.batch_event_index, []).append(group)
            if group.small:
                small_rows.update((d.batch_event_index, d.stream_id) for d in group.dgrams)
                maxima[group.stream_id] = max(maxima.get(group.stream_id, 0),
                    group.size + len(group.dgrams) * parser_bytes)
                if group.stream_id in seen_small:
                    cuts.add(group.first_event)
                seen_small.add(group.stream_id)
        # A small group can retain its first parser arena after large raw groups
        # from that arena retire. Bound that extra metadata conservatively by
        # one complete batch arena per small stream. Runtime reserves actual
        # rounded allocations/growth before issuing anything.
        fixed = sum(maxima.values()) + len(maxima) * len(rows) * parser_bytes
        if fixed > capacity:
            raise GpuMemoryPressureError('small groups and shared parser metadata exceed input budget')
        costs = [AdmissionEvent(tuple((s, n) for s, n in e.streams
                                      if (i, s) not in small_rows), e.detector_bytes)
                 for i, e in enumerate(events)]
        admission = plan_admission(costs, capacity - fixed,
                                   parser_bytes_per_dgram=parser_bytes,
                                   max_inflight=depth)
        ranges = []
        for start, stop in admission.execution_ranges:
            edges = [start] + sorted(c for c in cuts if start < c < stop) + [stop]
            ranges.extend(zip(edges, edges[1:]))
        self.execution_ranges = tuple(ranges)

    def groups_for(self, start, stop):
        selected = {g.group_id for e in range(start, stop)
                    for g in self.groups_by_event.get(e, ())}
        return tuple(g for g in self.plan.groups if g.group_id in selected)

    def new_groups(self, start, stop):
        return tuple(g for g in self.groups_for(start, stop) if g.group_id not in self.issued)
