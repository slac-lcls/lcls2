"""Deterministic byte admission for current and future resident-input schedules.

No allocation or I/O. Stage 4 uses ordered execution ranges without residency;
Stage 5 will consume the optional resident-stream decisions. Inputs are actual
descriptor presence, parser bytes per dgram, and detector working-set bytes.
"""

from dataclasses import dataclass

from .gpu_budget import GpuMemoryPressureError


@dataclass(frozen=True)
class AdmissionEvent:
    streams: tuple  # (physical stream ID, fetched bytes), one per valid dgram
    detector_bytes: int

    def __post_init__(self):
        object.__setattr__(self, 'streams', tuple(self.streams))
        ids = [s for s, _ in self.streams]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate stream in admission event")
        if self.detector_bytes < 0 or any(s < 0 or n < 0 for s, n in self.streams):
            raise ValueError("negative admission size or stream")


@dataclass(frozen=True)
class AdmissionPlan:
    resident_streams: tuple
    resident_bytes: int
    execution_ranges: tuple
    inflight: int
    per_execution_bytes: int


def plan_admission(events, capacity_bytes, *, parser_bytes_per_dgram=0,
                   max_inflight=2, allow_residency=False):
    """Admit whole streams only with guaranteed room for minimum executions.

    Capacity excludes fixed allocations, retained unrelated results, and the
    allocator margin. Runtime additionally reserves actual allocation-growth
    peaks against cached pool capacities before issuing reads.
    """
    events = tuple(events)
    if capacity_bytes < 0 or parser_bytes_per_dgram < 0 or max_inflight < 1:
        raise ValueError("invalid admission capacity/parser size/concurrency")
    if not events:
        return AdmissionPlan((), 0, (), 0, 0)

    def cost(event, resident):
        return event.detector_bytes + sum(
            n + parser_bytes_per_dgram for s, n in event.streams if s not in resident
        )

    largest = max(cost(e, ()) for e in events)
    if largest > capacity_bytes:
        index = max(range(len(events)), key=lambda i: cost(events[i], ()))
        event = events[index]
        raise GpuMemoryPressureError(
            f"event {index} cannot fit alone: input={sum(n for _, n in event.streams)}, "
            f"parser={len(event.streams) * parser_bytes_per_dgram}, "
            f"detector={event.detector_bytes}, capacity={capacity_bytes} bytes"
        )
    depth = min(max_inflight, len(events))
    while depth > 1 and largest * depth > capacity_bytes:
        depth -= 1
    resident, resident_bytes = [], 0
    if allow_residency:
        streams = {}
        for event in events:
            for stream, nbytes in event.streams:
                streams[stream] = streams.get(stream, 0) + nbytes + parser_bytes_per_dgram
        for stream, nbytes in sorted(streams.items(), key=lambda item: (item[1], item[0])):
            selected = resident + [stream]
            working = max(cost(e, selected) for e in events)
            # Prefer the cheapest complete input over extra overlap if needed.
            # Once residency is established, additional streams must fit the
            # selected depth; do not collapse the pipeline to retain everything.
            if not resident and resident_bytes + nbytes + working <= capacity_bytes:
                while depth > 1 and resident_bytes + nbytes + depth * working > capacity_bytes:
                    depth -= 1
            if resident_bytes + nbytes + depth * working <= capacity_bytes:
                resident, resident_bytes = selected, resident_bytes + nbytes
    allowance = (capacity_bytes - resident_bytes) // depth
    ranges, start, current = [], 0, 0
    for i, event in enumerate(events):
        nbytes = cost(event, resident)
        if i > start and current + nbytes > allowance:
            ranges.append((start, i))
            start, current = i, 0
        current += nbytes
    ranges.append((start, len(events)))
    return AdmissionPlan(tuple(resident), resident_bytes, tuple(ranges), depth, allowance)
